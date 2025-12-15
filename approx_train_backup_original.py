# approx_train.py —— 监督学习三段近似（无 LUT / 硬近似 / INT8 码位学习）
# Teacher(真 INT8) 常驻 CPU；Student(QAT/FakeQuant) 在 CUDA(如可用)。
import os, json
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_vgg16_tap_quant import VGG16TapQuant
from utils import get_dataloaders, ensure_dir
from recorder import ActivationRecorder
from torch.ao.quantization import prepare_qat, get_default_qat_qconfig, convert

# ============ 调试开关 ============
DEBUG_PRINT_LAYER_PARAMS = True  # True=打印每层5个参数; False=关闭打印
# ==================================

# 需要近似的层（名字要与 VGG16TapQuant.forward 中的记录点对应：block_output.{name}）
CONV_LAYERS = [
    "conv1_1","conv1_2",
    "conv2_1","conv2_2",
    "conv3_1","conv3_2","conv3_3",
    "conv4_1","conv4_2","conv4_3",
    "conv5_1","conv5_2","conv5_3",
]

# ---------- 仅做“编辑”的 Recorder（避免在 CUDA 上调用 int_repr） ----------
class EditOnlyRecorder:
    """只执行 maybe_edit，不做 record（因此不会触发 int_repr）"""
    def __init__(self, edits=None):
        self.edits = edits or {}
    def record(self, name: str, x: torch.Tensor):
        return
    def maybe_edit(self, name: str, x: torch.Tensor) -> torch.Tensor:
        fn = self.edits.get(name)
        return fn(x) if fn is not None else x
    storage = {}
    storage_float = {}

# ------------------ Teacher（真 INT8，固定 CPU） / Student（QAT，权重冻结） ------------------
def _build_teacher_int8(args):
    torch.backends.quantized.engine = args.backend
    m = VGG16TapQuant(num_classes=10)
    m.eval(); m.fuse_model()
    m.qconfig = get_default_qat_qconfig(args.backend)
    m.train(); prepare_qat(m, inplace=True)
    sd = torch.load(os.path.join(args.out, "vgg16_qat_preconvert.pth"), map_location="cpu")
    m.load_state_dict(sd, strict=True)
    m.eval()
    qmodel = convert(m, inplace=False)  # 真 INT8
    qmodel.eval()                       # 常驻 CPU
    return qmodel

def _build_student_fq(args):
    torch.backends.quantized.engine = args.backend
    m = VGG16TapQuant(num_classes=10)
    m.eval(); m.fuse_model()
    m.qconfig = get_default_qat_qconfig(args.backend)
    m.train(); prepare_qat(m, inplace=True)
    sd = torch.load(os.path.join(args.out, "vgg16_qat_preconvert.pth"), map_location="cpu")
    m.load_state_dict(sd, strict=True)
    # 冻结主干；FakeQuant 开、Observer 关；BN 固定统计
    from torch.ao.quantization.fake_quantize import FakeQuantize
    for p in m.parameters():
        p.requires_grad_(False)
    for mod in m.modules():
        if isinstance(mod, FakeQuantize):
            mod.enable_fake_quant()
            mod.disable_observer()
        if isinstance(mod, nn.BatchNorm2d):
            mod.eval()
    m.train()
    return m

# ------------------ 工具：直方图、分位、teacher 前向拿 (s,z) ------------------
@torch.no_grad()
def _collect_histograms(teacher, loader, num_batches=8):
    """在 CPU teacher 上收集每层激活的 INT8 直方图（码位 0..255）"""
    h = {f"block_output.{lyr}": torch.zeros(256, dtype=torch.long) for lyr in CONV_LAYERS}
    it = 0
    for images, _ in loader:
        rec = ActivationRecorder(store_cpu=True)
        _ = teacher(images, recorder=rec)  # teacher 在 CPU；images 也在 CPU
        for lyr in CONV_LAYERS:
            k = f"block_output.{lyr}"
            if k in rec.storage:
                v = rec.storage[k].flatten()
                h[k] += torch.bincount(v, minlength=256).to(torch.long)
        it += 1
        if it >= num_batches:
            break
    return h

def _q_nonzero_from_hist(h: torch.Tensor, p: float) -> int:
    """非零分布的 p 分位（返回 code ∈ [1,255]；若全零则返回 1）"""
    if h.sum()==0: return 1
    h = h.clone()
    h[0] = 0
    tot = int(h.sum().item())
    if tot == 0: return 1
    k = int(max(0, min(tot-1, round(p * (tot-1)))))
    c = torch.cumsum(h, 0)
    idx = torch.searchsorted(c, torch.tensor(k, dtype=c.dtype))
    return max(1, min(255, int(idx.item())))

@torch.no_grad()
def teacher_forward_with_scales(teacher, images_cpu):
    """
    teacher：量化模型（CPU）；一次前向同时抓取各层 (scale, zp)。
    返回：logits, {layer_name: (scale, zero_point)}
    """
    scales: Dict[str, Tuple[float,float]] = {}
    def make_sniffer(name):
        def _fn(x):
            scales[name] = (float(x.q_scale()), float(x.q_zero_point()))
            return x
        return _fn
    edits = {f"block_output.{k}": make_sniffer(k) for k in CONV_LAYERS}
    rec = ActivationRecorder(store_cpu=False, edits=edits)
    logits = teacher(images_cpu, recorder=rec)
    return logits, scales

@torch.no_grad()
def _eval_top1_teacher(teacher, loader, max_batches=None) -> float:
    total, correct, seen = 0, 0, 0
    for images, targets in loader:
        logits = teacher(images, recorder=None)   # teacher 在 CPU
        pred = logits.argmax(1)
        correct += (pred==targets).sum().item()
        total   += targets.size(0)
        seen    += 1
        if max_batches is not None and seen>=max_batches:
            break
    return 100.0 * correct / max(1,total)

# ------------------ 三段近似：INT8 码位学习 + 硬近似（前向硬/反向软） ------------------
class TriApproxINT8(nn.Module):
    """
    三段硬近似：[0,t1]→0, (t1,t2]→v1, (t2,tmax]→v2, 其余不变。
    t1/t2/tmax 以及 v1/v2 都在 INT8 码位上学习：
      - 前向：round 到整数码位（0..255），再用 (s,z) 解量化；输出使用硬替换
      - 反向：阈值门用软门产生梯度（straight-through），码位 round 用 STE
    """
    def __init__(self, beta: float=1.2):
        super().__init__()
        # 阈值：无约束参数 → softplus 累加保证 t1 < t2 < tmax ，单位为“码位”
        self.t1_base_p = nn.Parameter(torch.tensor(3.0))   # t1_code ≈ 1 + softplus(3)
        self.dt12_p    = nn.Parameter(torch.tensor(2.0))   # t2_code = t1_code + 1 + softplus(dt12)
        self.dt2m_p    = nn.Parameter(torch.tensor(2.0))   # tmax_code = t2_code + 1 + softplus(dt2m)
        # 代表值（码位）
        self.v1_code_p = nn.Parameter(torch.tensor(4.0))
        self.v2_code_p = nn.Parameter(torch.tensor(8.0))
        self.beta = beta

    @staticmethod
    def _ste_round(x: torch.Tensor):
        # 前向 round，反向梯度当作恒等（straight-through）
        return (x - x.detach()) + x.detach().round()

    def _codes_int(self):
        # 连续码位（确保有序且间隔≥1）
        t1_c   = 1.0 + F.softplus(self.t1_base_p)            # >=1
        t2_c   = t1_c + 1.0 + F.softplus(self.dt12_p)        # >=t1+1
        tmax_c = t2_c + 1.0 + F.softplus(self.dt2m_p)        # >=t2+1
        # clamp到合法范围并 round（STE） -> 整数码位
        t1_code   = self._ste_round(t1_c.clamp(1.0, 253.0))
        t2_code   = self._ste_round(t2_c.clamp(2.0, 254.0))
        tmax_code = self._ste_round(tmax_c.clamp(3.0, 255.0))
        # 代表值码位（0..255）
        v1_code = self._ste_round(self.v1_code_p.clamp(0.0, 255.0))
        v2_code = self._ste_round(self.v2_code_p.clamp(0.0, 255.0))
        return t1_code, t2_code, tmax_code, v1_code, v2_code

    def forward(self, x: torch.Tensor, s: torch.Tensor, z: torch.Tensor, train_mode: bool=True):
        x = x.clamp_min(0)

        # ---- 先得到“整数码位”，再解量化到浮点域 ----
        t1_code, t2_code, tmax_code, v1_code, v2_code = self._codes_int()
        t1   = s * (t1_code   - z)
        t2   = s * (t2_code   - z)
        tmax = s * (tmax_code - z)
        v1   = s * (v1_code   - z)
        v2   = s * (v2_code   - z)

        # ---- 硬近似（前向用它）----
        m1 = (x > 0)  & (x <= t1)
        m2 = (x > t1) & (x <= t2)
        m3 = (x > t2) & (x <= tmax)
        y_hard = torch.where(m1, torch.zeros_like(x),
                  torch.where(m2, v1,
                  torch.where(m3, v2, x)))

        # ---- 软门（只给梯度用）----
        b = self.beta
        g1 = torch.sigmoid((t1 - x)/b)
        g2 = torch.sigmoid((x - t1)/b) * torch.sigmoid((t2 - x)/b)
        g3 = torch.sigmoid((x - t2)/b) * torch.sigmoid((tmax - x)/b)
        g_in = (g1 + g2 + g3).clamp(0., 1.)
        y_soft = (1.0 - g_in) * x + g1*0.0 + g2*v1 + g3*v2

        # ---- Straight-Through：前向=硬输出；反向=软门梯度 ----
        y = y_hard + (y_soft - y_hard).detach()

        # 覆盖率统计（进入三段的比例）
        cover = float(g_in.mean().detach())
        r1, r2, r3 = float(g1.mean().detach()), float(g2.mean().detach()), float(g3.mean().detach())

        return y, cover, (r1, r2, r3), (
            int(t1_code.detach().item()), int(t2_code.detach().item()), int(tmax_code.detach().item())
        ), (
            int(v1_code.detach().item()),  int(v2_code.detach().item())
        )

    # -------- 初始化：tmax=非零90%分位，其余在 [1,tmax] 随机 --------
    @torch.no_grad()
    def init_from_hist(self, h_nonzero_90_code: int):
        tmax_code = int(max(3, min(255, h_nonzero_90_code)))
        t1_code = int(torch.randint(1, max(2, tmax_code-1), (1,)).item())
        t2_code = int(torch.randint(t1_code+1, max(t1_code+2, tmax_code), (1,)).item())
        # 把连续参数设置在这些 code 的邻近（前向会 round）
        self.t1_base_p.copy_(torch.tensor(float(t1_code - 1.0)))
        self.dt12_p.copy_( torch.tensor(float((t2_code - t1_code) - 1.0)))
        self.dt2m_p.copy_( torch.tensor(float((tmax_code - t2_code) - 1.0)))
        # v1/v2 码位随机到 [0, tmax]
        self.v1_code_p.copy_(torch.tensor(float(torch.randint(0, tmax_code+1, (1,)).item())))
        self.v2_code_p.copy_(torch.tensor(float(torch.randint(0, tmax_code+1, (1,)).item())))

    @torch.no_grad()
    def export_codes(self):
        t1_code, t2_code, tmax_code, v1_code, v2_code = self._codes_int()
        return dict(
            t1_code=int(t1_code.round().item()),
            t2_code=int(t2_code.round().item()),
            tmax_code=int(tmax_code.round().item()),
            v1_code=int(v1_code.round().item()),
            v2_code=int(v2_code.round().item()),
        )

# ------------------ 训练主流程：无 LUT / 每步打印 teacher & student 精度 ------------------
def train_supervised(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_dataloaders(args.data, args.batch_size, args.workers)

    teacher = _build_teacher_int8(args)          # CPU
    student = _build_student_fq(args).to(device) # CUDA(如可用)

    # Teacher 在测试集的基线精度
    baseline = _eval_top1_teacher(teacher, test_loader, max_batches=None)
    print(f"[Teacher INT8] baseline(top1) = {baseline:.2f}%")

    # 直方图（CPU teacher）
    calib_batches = getattr(args, "calib_batches", 8)
    print(f"[hist] collecting {calib_batches} batches ...")
    hists = _collect_histograms(teacher, test_loader, num_batches=calib_batches)

    # 取一小批拿 (s,z)（仅用于日志或需要时）
    for images,_ in train_loader:
        _, scale_cache = teacher_forward_with_scales(teacher, images.cpu())
        break

    # 每层一个 TriApprox，并按规则初始化
    tri_map: Dict[str, TriApproxINT8] = {lyr: TriApproxINT8(beta=1.2).to(device) for lyr in CONV_LAYERS}
    for lyr, mod in tri_map.items():
        h = hists.get(f"block_output.{lyr}")
        if h is None or h.sum()==0: 
            continue
        tmax_code = _q_nonzero_from_hist(h, 0.90)  # 非零 90% 分位
        mod.init_from_hist(tmax_code)
        print(f"[init] {lyr} -> {mod.export_codes()}")

    # 优化器 & 超参
    optim = torch.optim.Adam([p for m in tri_map.values() for p in m.parameters()],
                             lr=getattr(args, "lr", 1e-2))

    T = getattr(args, "kd_T", 2.0)
    alpha = float(getattr(args, "alpha", 0.05))  # 覆盖奖励系数（建议从 0.02~0.05 起）
    epochs = int(getattr(args, "epochs", 3))
    beta0 = 1.2
    log_every = int(getattr(args, "log_every", 50))

    # 训练
    for ep in range(1, epochs+1):
        # 退火门温度（逐步“变硬”）
        for m in tri_map.values():
            m.beta = max(0.2, beta0 * (0.85 ** (ep-1)))

        student.train()
        avg_kd = avg_cov = avg_acc_s = avg_acc_t = 0.0
        steps = 0

        for images, targets in train_loader:
            images  = images.to(device)
            targets = targets.to(device)

            # teacher: CPU 前向（logits + scales）
            with torch.no_grad():
                t_logits_cpu, scales = teacher_forward_with_scales(teacher, images.cpu())
            t_logits = t_logits_cpu.to(device)

            # student: 组装 edits（按层传入 s,z）
            cov_sum = 0.0
            edits = {}
            layer_params = {}  # 用于收集每层的参数
            for lyr, mod in tri_map.items():
                s, z = scales.get(lyr, (1.0, 0.0))
                key = f"block_output.{lyr}"
                def make_edit(m, s_, z_, layer_name):
                    s_t = torch.tensor(s_, dtype=torch.float32, device=images.device)
                    z_t = torch.tensor(z_, dtype=torch.float32, device=images.device)
                    def _fn(x):
                        y, cov, ratios, thresh, vcode = m(x, s_t, z_t, train_mode=True)
                        nonlocal cov_sum
                        cov_sum += cov
                        # 记录参数: thresh=(t1,t2,tmax), vcode=(v1,v2)
                        layer_params[layer_name] = {
                            't1': thresh[0], 't2': thresh[1], 'tmax': thresh[2],
                            'v1': vcode[0], 'v2': vcode[1]
                        }
                        return y
                    return _fn
                edits[key] = make_edit(mod, s, z, lyr)

            rec = EditOnlyRecorder(edits=edits)
            s_logits = student(images, recorder=rec)

            # KD - alpha*coverage
            logp = F.log_softmax(s_logits / T, dim=1)
            q    = F.softmax(t_logits / T, dim=1)
            kd   = F.kl_div(logp, q, reduction="batchmean") * (T*T)
            cov_mean = cov_sum / max(1e-6, len(tri_map))
            loss = kd - alpha * cov_mean

            optim.zero_grad()
            loss.backward()
            optim.step()

            # 当前 batch 精度（student & teacher）
            pred_s = s_logits.argmax(1)
            acc_s_batch = (pred_s == targets).float().mean().item() * 100.0

            pred_t = t_logits.argmax(1)
            acc_t_batch = (pred_t == targets).float().mean().item() * 100.0

            # 统计与打印
            avg_kd  += float(kd.item())
            avg_cov += float(cov_mean)
            avg_acc_s += float(acc_s_batch)
            avg_acc_t += float(acc_t_batch)
            steps += 1

            if steps % log_every == 0:
                print(f"[Ep{ep}] step={steps} "
                      f"KD={avg_kd/steps:.4f}  cov={avg_cov/steps:.4f}  "
                      f"student_acc={acc_s_batch:.2f}% (avg {avg_acc_s/steps:.2f}%)  "
                      f"teacher_acc={acc_t_batch:.2f}% (avg {avg_acc_t/steps:.2f}%)  "
                      f"alpha={alpha:.3f}")
                
                # 根据开关决定是否打印每层的5个参数
                if DEBUG_PRINT_LAYER_PARAMS:
                    print(f"  Layer Parameters (t1, t2, tmax, v1, v2):")
                    for lyr in CONV_LAYERS:
                        if lyr in layer_params:
                            p = layer_params[lyr]
                            print(f"    {lyr:8s}: t1={p['t1']:3d}  t2={p['t2']:3d}  tmax={p['tmax']:3d}  v1={p['v1']:3d}  v2={p['v2']:3d}")
                    print()  # 空行分隔

    # ---- 仅保存学习到的各层码位参数（无 LUT） ----
    result = {"layers": {}, "backend": args.backend}
    for lyr, mod in tri_map.items():
        result["layers"][lyr] = mod.export_codes()

    save_dir = os.path.join(args.out, "tri_supervised_hard_int_codes")
    ensure_dir(save_dir)
    with open(os.path.join(save_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"[Saved params] {os.path.join(save_dir,'result.json')}")

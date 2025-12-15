# activation_tools.py
import os
import json
from typing import Dict
import torch
import pandas as pd
from torch.ao.quantization import prepare_qat, get_default_qat_qconfig, convert

from model_vgg16_tap_quant import VGG16TapQuant
from model_vgg19_tap_quant import VGG19TapQuant
from model_alexnet_tap_quant import AlexNetTapQuant
from model_lenet5_tap_quant import LeNet5TapQuant
from model_nin_tap_quant import NiNTapQuant
from model_mobilenetv1_tap_quant import MobileNetV1TapQuant
from recorder import ActivationRecorder, WeightVarianceRecorder, zero_channels, scale_value
from utils import get_dataloaders, ensure_dir

# ---- 这是你原来写死区间的版本，后面可以换成 make_int8_bucket_edit ----
def modify_int8_activation(x: torch.Tensor) -> torch.Tensor:
    """
    原脚本里的示例：把某些取值范围的 int8 激活压到一个点上。
    """
    if not hasattr(x, "int_repr"):
        return x
    x_int = x.int_repr()
    scale = x.q_scale()
    zp = x.q_zero_point()

    original = x_int
    modified = x_int.clone()

    # 按原代码的阈值来
    mask_0_to_2 = (original > 0) & (original < 5)
    modified[mask_0_to_2] = 4
    mask_0_to_2 = (original >= 5) & (original < 12)
    modified[mask_0_to_2] = 8
    mask_0_to_2 = (original >= 12) & (original < 18)
    modified[mask_0_to_2] = 14
    # mask_0_to_2 = (original >= 18) & (original < 24)
    # modified[mask_0_to_2] = 20
    # mask_0_to_2 = (original > 0) & (original < 4)
    # modified[mask_0_to_2] = 0

    # mask_0_to_3 = (original > 0) & (original < 3)
    # modified[mask_0_to_3] = 1

    # mask_3_to_8 = (original > 3) & (original < 8)
    # modified[mask_3_to_8] = 4

    # mask_8_to_16 = (original > 8) & (original < 16)
    # modified[mask_8_to_16] = 16

    # mask_16_to_32 = (original > 16) & (original < 32)
    # modified[mask_16_to_32] = 32

    return torch._make_per_tensor_quantized_tensor(modified, scale, zp)


def make_int8_triseg_edit(t1_code: int, v1_code: int, t2_code: int, v2_code: int, tmax_code: int):
    """
    基于 PPO 学到的整型阈值/取值，在 INT8 激活上做三段近似（纯推理，不涉及训练/梯度）：
    规则（以 INT8 码值为域）：
      - (0, t1]            → 0
      - (t1, t2]           → v1
      - (t2, tmax]         → v2
      - 其他（≤0 或 >tmax）→ 不变
    """
    def _fn(x: torch.Tensor) -> torch.Tensor:
        if not hasattr(x, "int_repr"):
            return x
        x_int = x.int_repr()
        scale = x.q_scale()
        zp = x.q_zero_point()

        modified = x_int.clone()
        # 三段映射（严格使用整型码值区间）
        mask1 = (x_int > 0) & (x_int <= t1_code)
        modified[mask1] = 0

        mask2 = (x_int > t1_code) & (x_int <= t2_code)
        modified[mask2] = int(v1_code)

        mask3 = (x_int > t2_code) & (x_int <= tmax_code)
        modified[mask3] = int(v2_code)

        return torch._make_per_tensor_quantized_tensor(modified, scale, zp)

    return _fn


@torch.no_grad()
def eval_int8_with_ppo_config(args, config_path: str = None, save_report: bool = True):
    """
    纯近似推理评估：
    - 读取 PPO 导出的每层参数（t1_code, v1_code, t2_code, v2_code, tmax_code）
    - 在 INT8 模型上套用分层近似，仅做推理，评估修改前后精度与差值
    - 不涉及 Student/FakeQuant 与任何训练
    """
    torch.backends.quantized.engine = args.backend

    # 1) 构建 INT8 模型
    model_name = getattr(args, "model", "vgg16")
    print(f"[Eval] Loading model: {model_name}")
    
    if model_name == "vgg19":
        float_model = VGG19TapQuant(num_classes=10)
    elif model_name == "alexnet":
        float_model = AlexNetTapQuant(num_classes=10)
    elif model_name == "lenet5":
        float_model = LeNet5TapQuant(num_classes=10)
    elif model_name == "nin":
        float_model = NiNTapQuant(num_classes=10)
    elif model_name == "mobilenetv1":
        float_model = MobileNetV1TapQuant(num_classes=10)
    else:
        float_model = VGG16TapQuant(num_classes=10)

    float_model.eval()
    float_model.fuse_model()
    float_model.qconfig = get_default_qat_qconfig(args.backend)
    float_model.train()
    prepare_qat(float_model, inplace=True)
    
    # Load checkpoint
    ckpt_path = os.path.join(args.out, f"{model_name}_qat_preconvert.pth")
    if not os.path.exists(ckpt_path):
        # Fallback for backward compatibility or if user didn't specify model but file exists
        if model_name == "vgg16" and not os.path.exists(ckpt_path):
             ckpt_path = os.path.join(args.out, "vgg16_qat_preconvert.pth")
    
    print(f"[Eval] Loading checkpoint from: {ckpt_path}")
    float_model.load_state_dict(
        torch.load(ckpt_path, map_location="cpu")
    )
    float_model.eval()
    qmodel = convert(float_model, inplace=False)

    # 2) 数据
    _, test_loader = get_dataloaders(args.data, args.batch_size, args.workers)

    # 3) 读取 PPO 配置
    if config_path is None:
        config_path = os.path.join(args.out, "tri_ppo_int_codes", "result.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r") as f:
        cfg = json.load(f)

    layer_cfgs = cfg.get("layers", {})

    # 4) 构造每层的 edit 函数
    edits = {}
    for layer_name, params in layer_cfgs.items():
        t1 = int(params["t1_code"]) 
        v1 = int(params["v1_code"]) 
        t2 = int(params["t2_code"]) 
        v2 = int(params["v2_code"]) 
        tmax = int(params["tmax_code"])
        
        # Determine key based on layer name and model type
        if layer_name.startswith('conv'):
            key = f"block_output.{layer_name}"
        elif layer_name == 'fc':
            key = "fc.out"
        elif layer_name.startswith('fc'):
            key = f"classifier.{layer_name}.out"
        else:
            key = f"block_output.{layer_name}"
            
        edits[key] = make_int8_triseg_edit(t1, v1, t2, v2, tmax)

    # 5) 评估（基线与修改后）
    correct_mod, correct_orig, total = 0, 0, 0
    for images, targets in test_loader:
        # 修改后的推理
        rec = ActivationRecorder(store_cpu=False, edits=edits)
        logits_mod = qmodel(images, recorder=rec)

        # 原始不改
        logits_orig = qmodel(images, recorder=None)

        pred_mod = logits_mod.argmax(1)
        pred_orig = logits_orig.argmax(1)

        correct_mod += (pred_mod == targets).sum().item()
        correct_orig += (pred_orig == targets).sum().item()
        total += targets.size(0)

    acc_mod = 100.0 * correct_mod / total
    acc_orig = 100.0 * correct_orig / total
    acc_drop = acc_orig - acc_mod

    print("\n[PPO Config Evaluation - INT8 pure inference]")
    print(f"  Backend         : {args.backend}")
    print(f"  Model           : {model_name}")
    print(f"  Config file     : {config_path}")
    print(f"  INT8 Baseline   : {acc_orig:.2f}%")
    print(f"  INT8 + Approx   : {acc_mod:.2f}%")
    print(f"  Accuracy drop   : {acc_drop:.2f}% (orig - approx)")

    # 6) 可选：保存简报
    if save_report:
        save_dir = os.path.join(args.out, "tri_ppo_int_codes")
        ensure_dir(save_dir)
        report = {
            "model": model_name,
            "config_path": config_path,
            "backend": args.backend,
            "acc_baseline": acc_orig,
            "acc_modified": acc_mod,
            "acc_drop": acc_drop,
            "selection": cfg.get("selection", None),
            "metrics_in_config": cfg.get("metrics", None)
        }
        with open(os.path.join(save_dir, "eval_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"[Saved] Evaluation report -> {os.path.join(save_dir, 'eval_report.json')}")
    
    return acc_orig, acc_mod, acc_drop

@torch.no_grad()
def eval_int8_modified(args):
    """评估在激活修改下的 INT8 模型的准确率"""
    torch.backends.quantized.engine = args.backend

    # 1) 构建 Python 下的量化模型
    float_model = VGG16TapQuant(num_classes=10)
    float_model.eval()
    float_model.fuse_model()
    float_model.qconfig = get_default_qat_qconfig(args.backend)
    float_model.train()
    prepare_qat(float_model, inplace=True)
    float_model.load_state_dict(
        torch.load(os.path.join(args.out, "vgg16_qat_preconvert.pth"), map_location="cpu")
    )
    float_model.eval()
    qmodel = convert(float_model, inplace=False)

    # 2) 数据
    _, test_loader = get_dataloaders(args.data, args.batch_size, args.workers)

    # 3) 把所有卷积层都绑上这个修改
    conv_layers = [
        "conv1_1", "conv1_2",
        "conv2_1", "conv2_2",
        "conv3_1", "conv3_2", "conv3_3",
        "conv4_1", "conv4_2", "conv4_3",
        "conv5_1", "conv5_2", "conv5_3",
    ]
    edits = {f"block_output.{layer}": modify_int8_activation for layer in conv_layers}

    correct_mod, correct_orig, total = 0, 0, 0
    for images, targets in test_loader:
        rec = ActivationRecorder(store_cpu=False, edits=edits)
        logits_mod = qmodel(images, recorder=rec)

        # 原始不改的
        logits_orig = qmodel(images, recorder=None)

        pred_mod = logits_mod.argmax(1)
        pred_orig = logits_orig.argmax(1)

        correct_mod += (pred_mod == targets).sum().item()
        correct_orig += (pred_orig == targets).sum().item()
        total += targets.size(0)

    acc_mod = correct_mod / total * 100.0
    acc_orig = correct_orig / total * 100.0
    print(f"[INT8 原始] {acc_orig:.2f}%")
    print(f"[INT8 修改] {acc_mod:.2f}%")
    print(f"[差值] {acc_mod - acc_orig:.2f}%")

@torch.no_grad()
def dump_acts_float(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, test_loader = get_dataloaders(args.data, args.batch_size, args.workers)

    model = VGG16TapQuant(num_classes=10).to(device).eval()
    state = torch.load(os.path.join(args.out, "vgg16_float_unfused.pth"), map_location=device)
    model.load_state_dict(state)

    images, _ = next(iter(test_loader))
    images = images.to(device)

    rec = ActivationRecorder(store_cpu=True)
    logits = model(images, recorder=rec)

    save_dir = os.path.join(args.out, "acts_float")
    ensure_dir(save_dir)
    for k, v in rec.storage.items():
        torch.save(v, os.path.join(save_dir, f"{k.replace('/','_')}.pt"))
    print(f"[Saved] {len(rec.storage)} activation tensors -> {save_dir}")
    print("Sample keys:", list(rec.storage.keys())[:10])
    print("Logits shape:", logits.shape)

@torch.no_grad()
def demo_edit_float(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, test_loader = get_dataloaders(args.data, args.batch_size, args.workers)
    model = VGG16TapQuant(num_classes=10).to(device).eval()
    state = torch.load(os.path.join(args.out, "vgg16_float_unfused.pth"), map_location=device)
    model.load_state_dict(state)

    images, _ = next(iter(test_loader))
    images = images.to(device)

    edits = {
        "features.conv1_2.relu_out": zero_channels(list(range(8))),
        # "features.conv3_1.bn_out": scale_value(0.5),
    }
    rec = ActivationRecorder(store_cpu=True, edits=edits)
    logits = model(images, recorder=rec)
    print("Edited logits (first 5 softmax):", logits[:5].softmax(-1))
    print("Captured keys:", len(rec.storage))

@torch.no_grad()
def dump_acts_int8(args):
    # 构建量化模型
    torch.backends.quantized.engine = args.backend

    float_model = VGG16TapQuant(num_classes=10)
    float_model.eval()
    float_model.fuse_model()
    float_model.qconfig = get_default_qat_qconfig(args.backend)
    float_model.train()
    prepare_qat(float_model, inplace=True)
    float_model.load_state_dict(
        torch.load(os.path.join(args.out, "vgg16_qat_preconvert.pth"), map_location="cpu")
    )
    float_model.eval()
    qmodel = convert(float_model, inplace=False)

    _, test_loader = get_dataloaders(args.data, args.batch_size, args.workers)
    images, targets = next(iter(test_loader))  # 取一批
    rec = ActivationRecorder(store_cpu=True)
    logits = qmodel(images, recorder=rec)

    save_dir = os.path.join(args.out, "acts_int8")
    ensure_dir(save_dir)

    # 量化激活
    for k, v in rec.storage.items():
        torch.save(v, os.path.join(save_dir, f"{k.replace('/','_')}_int.pt"))
    # 对应的 dequant
    for k, v in rec.storage_float.items():
        torch.save(v, os.path.join(save_dir, f"{k.replace('/','_')}_float.pt"))

    # 保存 logits
    torch.save(logits, os.path.join(save_dir, "logits.pt"))
    print(f"[INT8 dump] saved to {save_dir}")


@torch.no_grad()
def analyze_variance_distribution(args):
    """分析并记录卷积核权重和激活值的方差分布"""
    torch.backends.quantized.engine = args.backend
    
    # 1. 构建量化模型（记录权重时使用转换前的模型）
    float_model = VGG16TapQuant(num_classes=10)
    float_model.eval()
    float_model.fuse_model()
    float_model.qconfig = get_default_qat_qconfig(args.backend)
    float_model.train()
    prepare_qat(float_model, inplace=True)
    float_model.load_state_dict(
        torch.load(os.path.join(args.out, "vgg16_qat_preconvert.pth"), map_location="cpu")
    )
    float_model.eval()
    
    # 2. 记录权重方差（在转换前）
    print("[Info] Recording weight variance from QAT model (before convert)...")
    weight_recorder = WeightVarianceRecorder()
    weight_recorder.record_model_weights(float_model)
    
    save_dir = os.path.join(args.out, "variance_analysis")
    ensure_dir(save_dir)
    weight_recorder.save_to_file(os.path.join(save_dir, "weight_variance.json"))
    
    # 3. 转换为INT8模型（用于激活值统计）
    qmodel = convert(float_model, inplace=False)
    
    # 3. 记录激活值方差（使用多个batch求平均）
    print("[Info] Recording activation variance...")
    _, test_loader = get_dataloaders(args.data, args.batch_size, args.workers)
    
    # 使用多个batch来统计激活值方差
    num_batches = min(10, len(test_loader))
    all_variance_stats = {}
    
    for batch_idx, (images, targets) in enumerate(test_loader):
        if batch_idx >= num_batches:
            break
        
        rec = ActivationRecorder(store_cpu=True, record_variance=True)
        logits = qmodel(images, recorder=rec)
        
        # 累加方差统计
        for layer_name, stats in rec.variance_stats.items():
            if layer_name not in all_variance_stats:
                all_variance_stats[layer_name] = {
                    'mean_sum': 0, 'var_sum': 0, 'std_sum': 0,
                    'count': 0, 'shape': stats['shape']
                }
            all_variance_stats[layer_name]['mean_sum'] += stats['mean']
            all_variance_stats[layer_name]['var_sum'] += stats['var']
            all_variance_stats[layer_name]['std_sum'] += stats['std']
            all_variance_stats[layer_name]['count'] += 1
    
    # 4. 计算平均值
    activation_variance_stats = {}
    for layer_name, stats in all_variance_stats.items():
        count = stats['count']
        activation_variance_stats[layer_name] = {
            'mean_avg': stats['mean_sum'] / count,
            'var_avg': stats['var_sum'] / count,
            'std_avg': stats['std_sum'] / count,
            'shape': stats['shape'],
            'num_samples': count
        }
    
    # 5. 保存激活值方差统计
    with open(os.path.join(save_dir, "activation_variance.json"), 'w') as f:
        json.dump(activation_variance_stats, f, indent=2)
    print(f"[Saved] Activation variance stats to {save_dir}/activation_variance.json")
    
    # 6. 生成汇总报告
    print("\n" + "="*60)
    print("Weight Variance Summary (Top 5 layers by variance):")
    print("="*60)
    sorted_weights = sorted(weight_recorder.variance_stats.items(), 
                           key=lambda x: x[1]['var'], reverse=True)[:5]
    for name, stats in sorted_weights:
        print(f"{name:40s} | Var: {stats['var']:.6f} | Std: {stats['std']:.6f}")
    
    print("\n" + "="*60)
    print("Activation Variance Summary (Top 5 layers by variance):")
    print("="*60)
    sorted_acts = sorted(activation_variance_stats.items(), 
                        key=lambda x: x[1]['var_avg'], reverse=True)[:5]
    for name, stats in sorted_acts:
        print(f"{name:40s} | Var: {stats['var_avg']:.6f} | Std: {stats['std_avg']:.6f}")
    
    print(f"\n[Info] All results saved to {save_dir}/")
    return weight_recorder.variance_stats, activation_variance_stats


@torch.no_grad()
def collect_activation_percentiles(args):
    """收集每层INT8激活值的分位数（去除0后）"""
    torch.backends.quantized.engine = args.backend
    
    # 构建量化模型
    float_model = VGG16TapQuant(num_classes=10)
    float_model.eval()
    float_model.fuse_model()
    float_model.qconfig = get_default_qat_qconfig(args.backend)
    float_model.train()
    prepare_qat(float_model, inplace=True)
    float_model.load_state_dict(
        torch.load(os.path.join(args.out, "vgg16_qat_preconvert.pth"), map_location="cpu")
    )
    float_model.eval()
    qmodel = convert(float_model, inplace=False)
    
    print("[Info] Collecting INT8 activation values...")
    _, test_loader = get_dataloaders(args.data, args.batch_size, args.workers)
    
    # 收集所有激活值
    conv_layers = [
        "conv1_1", "conv1_2",
        "conv2_1", "conv2_2",
        "conv3_1", "conv3_2", "conv3_3",
        "conv4_1", "conv4_2", "conv4_3",
        "conv5_1", "conv5_2", "conv5_3",
    ]
    
    layer_activations = {layer: [] for layer in conv_layers}
    
    # 使用多个batch收集数据
    num_batches = min(20, len(test_loader))
    for batch_idx, (images, targets) in enumerate(test_loader):
        if batch_idx >= num_batches:
            break
        
        rec = ActivationRecorder(store_cpu=True)
        logits = qmodel(images, recorder=rec)
        
        # 收集每层的INT8激活值
        for layer in conv_layers:
            key = f"block_output.{layer}"
            if key in rec.storage:
                int8_acts = rec.storage[key]  # INT8表示
                # 去除0值
                non_zero_acts = int8_acts[int8_acts != 0]
                if len(non_zero_acts) > 0:
                    layer_activations[layer].append(non_zero_acts.cpu())
        
        # if (batch_idx + 1) % 5 == 0:
            # print(f"  Processed {batch_idx + 1}/{num_batches} batches...")
    
    # 计算每层的分位数
    percentiles = [20, 40, 60, 80]
    layer_percentile_values = {}
    
    print("\n[Info] Computing percentiles for each layer...")
    for layer in conv_layers:
        if len(layer_activations[layer]) > 0:
            # 合并所有batch的激活值
            all_acts = torch.cat(layer_activations[layer])
            
            # 如果数据量太大，进行采样
            max_samples = 10000000  # 1000万个样本
            if all_acts.numel() > max_samples:
                indices = torch.randperm(all_acts.numel())[:max_samples]
                all_acts = all_acts.flatten()[indices]
                # print(f"  {layer}: Sampled {max_samples} from {all_acts.numel()} values")
            
            # 计算分位数
            percentile_vals = {}
            for p in percentiles:
                val = torch.quantile(all_acts.float(), p / 100.0).item()
                percentile_vals[f"p{p}"] = int(round(val))
            
            # 添加统计信息
            percentile_vals['min'] = int(all_acts.min().item())
            percentile_vals['max'] = int(all_acts.max().item())
            percentile_vals['mean'] = float(all_acts.float().mean().item())
            percentile_vals['count'] = int(all_acts.numel())
            
            layer_percentile_values[layer] = percentile_vals
            
            print(f"  {layer:15s} | p20={percentile_vals['p20']:3d} , p40={percentile_vals['p40']:3d} ,"
                  f"p60={percentile_vals['p60']:3d} , p80={percentile_vals['p80']:3d} "
                  f"(min={percentile_vals['min']:3d}, max={percentile_vals['max']:3d})")
    
    # 保存结果
    save_dir = os.path.join(args.out, "sensitivity_analysis")
    ensure_dir(save_dir)
    with open(os.path.join(save_dir, "layer_percentiles.json"), 'w') as f:
        json.dump(layer_percentile_values, f, indent=2)
    print(f"\n[Saved] Percentile values to {save_dir}/layer_percentiles.json")
    
    return layer_percentile_values


def create_percentile_modifier(p20, p40, p60, p80):
    """创建基于分位数的激活值修改函数
    规则：
    - 0 < x <= p20  → 0
    - p20 < x <= p40 → p20
    - p40 < x <= p60 → p40
    - p60 < x <= p80 → p60
    - x > p80 → 不变
    """
    def modify_fn(x: torch.Tensor) -> torch.Tensor:
        if not hasattr(x, "int_repr"):
            return x
        
        x_int = x.int_repr()
        scale = x.q_scale()
        zp = x.q_zero_point()
        
        modified = x_int.clone()
        
        # 应用分段规则
        mask_0_p20 = (x_int > 0) & (x_int <= p20)
        modified[mask_0_p20] = 0
        
        mask_p20_p40 = (x_int > p20) & (x_int <= p40)
        modified[mask_p20_p40] = p20
        
        mask_p40_p60 = (x_int > p40) & (x_int <= p60)
        modified[mask_p40_p60] = p40
        
        mask_p60_p80 = (x_int > p60) & (x_int <= p80)
        modified[mask_p60_p80] = p60
        
        # 大于p80的值保持不变（不需要额外操作，因为modified是clone的）
        
        return torch._make_per_tensor_quantized_tensor(modified, scale, zp)
    
    return modify_fn


@torch.no_grad()
def analyze_layer_sensitivity(args):
    """分析每层对激活值修改的敏感性（每层只推理一次）"""
    torch.backends.quantized.engine = args.backend
    
    # 1. 加载分位数数据
    percentile_file = os.path.join(args.out, "sensitivity_analysis", "layer_percentiles.json")
    if not os.path.exists(percentile_file):
        print("[Info] Percentile file not found, collecting data first...")
        layer_percentiles = collect_activation_percentiles(args)
    else:
        with open(percentile_file, 'r') as f:
            layer_percentiles = json.load(f)
        print(f"[Info] Loaded percentiles from {percentile_file}")
    
    # 2. 构建量化模型
    float_model = VGG16TapQuant(num_classes=10)
    float_model.eval()
    float_model.fuse_model()
    float_model.qconfig = get_default_qat_qconfig(args.backend)
    float_model.train()
    prepare_qat(float_model, inplace=True)
    float_model.load_state_dict(
        torch.load(os.path.join(args.out, "vgg16_qat_preconvert.pth"), map_location="cpu")
    )
    float_model.eval()
    qmodel = convert(float_model, inplace=False)
    
    # 3. 获取基线准确率
    print("\n[Info] Computing baseline accuracy...")
    _, test_loader = get_dataloaders(args.data, args.batch_size, args.workers)
    
    correct_baseline = 0
    total = 0
    for images, targets in test_loader:
        logits = qmodel(images, recorder=None)
        correct_baseline += (logits.argmax(1) == targets).sum().item()
        total += targets.size(0)
    
    baseline_acc = correct_baseline / total * 100.0
    print(f"  Baseline INT8 Accuracy: {baseline_acc:.2f}%")
    
    # 4. 逐层测试敏感性（每层只推理一次）
    conv_layers = [
        "conv1_1", "conv1_2",
        "conv2_1", "conv2_2",
        "conv3_1", "conv3_2", "conv3_3",
        "conv4_1", "conv4_2", "conv4_3",
        "conv5_1", "conv5_2", "conv5_3",
    ]
    
    sensitivity_results = {}
    
    print("\n" + "="*100)
    print("Layer Sensitivity Analysis (applying percentile-based approximation)")
    print("="*100)
    print(f"{'Layer':<15s} | {'p20':<5s} | {'p40':<5s} | {'p60':<5s} | {'p80':<5s} | "
          f"{'Accuracy':<10s} | {'Drop':<10s} | {'Rule':<30s}")
    print("-"*100)
    
    for layer in conv_layers:
        if layer not in layer_percentiles:
            continue
        
        p_vals = layer_percentiles[layer]
        if not all(k in p_vals for k in ['p20', 'p40', 'p60', 'p80']):
            continue
        
        p20 = p_vals['p20']
        p40 = p_vals['p40']
        p60 = p_vals['p60']
        p80 = p_vals['p80']
        
        # 创建该层的修改函数（应用分段规则）
        modifier = create_percentile_modifier(p20, p40, p60, p80)
        edits = {f"block_output.{layer}": modifier}
        
        # 评估准确率（只推理一次）
        correct = 0
        total_samples = 0
        for images, targets in test_loader:
            rec = ActivationRecorder(store_cpu=False, edits=edits)
            logits = qmodel(images, recorder=rec)
            correct += (logits.argmax(1) == targets).sum().item()
            total_samples += targets.size(0)
        
        modified_acc = correct / total_samples * 100.0
        acc_drop = modified_acc - baseline_acc
        
        sensitivity_results[layer] = {
            'percentiles': {'p20': p20, 'p40': p40, 'p60': p60, 'p80': p80},
            'accuracy': modified_acc,
            'drop': acc_drop
        }
        
        rule_desc = f"(0,{p20}]→0, ({p20},{p40}]→{p20}, >{p80}→unchanged"
        print(f"{layer:<15s} | {p20:<5d} | {p40:<5d} | {p60:<5d} | {p80:<5d} | "
              f"{modified_acc:<10.2f} | {acc_drop:>+9.2f} | {rule_desc:<30s}")
    
    # 5. 保存结果
    save_dir = os.path.join(args.out, "sensitivity_analysis")
    with open(os.path.join(save_dir, "layer_sensitivity.json"), 'w') as f:
        json.dump({
            'baseline_accuracy': baseline_acc,
            'approximation_rule': '(0,p20]→0, (p20,p40]→p20, (p40,p60]→p40, (p60,p80]→p60, >p80→unchanged',
            'sensitivity': sensitivity_results
        }, f, indent=2)
    print(f"\n[Saved] Sensitivity results to {save_dir}/layer_sensitivity.json")
    
    # 6. 生成敏感性排名
    print("\n" + "="*80)
    print("Layer Sensitivity Ranking (sorted by accuracy drop)")
    print("="*80)
    
    sorted_layers = sorted(sensitivity_results.items(), key=lambda x: x[1]['drop'])
    
    print(f"{'Rank':<6s} | {'Layer':<15s} | {'Accuracy Drop':<15s} | {'Modified Acc':<15s}")
    print("-"*80)
    for rank, (layer, results) in enumerate(sorted_layers, 1):
        print(f"{rank:<6d} | {layer:<15s} | {results['drop']:>+14.2f} | {results['accuracy']:>14.2f}")
    
    print(f"\n[Info] Most sensitive layer: {sorted_layers[0][0]} (drop: {sorted_layers[0][1]['drop']:+.2f}%)")
    print(f"[Info] Least sensitive layer: {sorted_layers[-1][0]} (drop: {sorted_layers[-1][1]['drop']:+.2f}%)")
    
    return sensitivity_results

class CoverageRecorder:
    """
    专门用于统计覆盖率的 Recorder。
    统计每一层激活值中，有多少比例的值 <= tmax_code。
    """
    def __init__(self, layer_configs: Dict[str, int], ignore_zeros: bool = False):
        """
        Args:
            layer_configs: 字典，key是层名，value是该层的 tmax_code
            ignore_zeros: 是否忽略0值（只统计非零值）
        """
        self.layer_configs = layer_configs
        self.ignore_zeros = ignore_zeros
        self.stats = {}  # {layer_name: {'total': 0, 'covered': 0}}

    def record(self, name: str, x: torch.Tensor) -> None:
        # 这里的 name 可能是 "block_output.conv1_1" 这种格式
        # 我们需要匹配 layer_configs 中的 key
        # layer_configs 的 key 可能是 "conv1_1"
        
        # 尝试从 name 中提取 layer_name
        layer_name = name.replace("block_output.", "").replace("classifier.", "").replace(".out", "")
        
        # 如果直接匹配
        if layer_name in self.layer_configs:
            tmax = self.layer_configs[layer_name]
        # 如果 name 本身就在 config 里
        elif name in self.layer_configs:
            tmax = self.layer_configs[name]
            layer_name = name
        else:
            return  # 不在关注列表中

        with torch.no_grad():
            if hasattr(x, "int_repr"):
                x_int = x.int_repr()
                
                if self.ignore_zeros:
                    # 只统计非零值
                    mask_nonzero = x_int != 0
                    total = mask_nonzero.sum().item()
                    # 在非零值中，小于等于 tmax 的数量
                    covered = ((x_int <= tmax) & mask_nonzero).sum().item()
                else:
                    # 统计所有值
                    total = x_int.numel()
                    covered = (x_int <= tmax).sum().item()
                
                if layer_name not in self.stats:
                    self.stats[layer_name] = {'total': 0, 'covered': 0}
                
                self.stats[layer_name]['total'] += total
                self.stats[layer_name]['covered'] += covered

    def maybe_edit(self, name: str, x: torch.Tensor) -> torch.Tensor:
        """
        为了兼容模型中的调用接口 (rec.maybe_edit)，这里直接返回原值，不做修改。
        """
        # 我们可以在这里顺便调用 record，因为模型里通常是先 maybe_edit 再 record，或者反过来
        # 看 model_vgg16_tap_quant.py 的实现，通常是:
        # x = rec.maybe_edit(name, x)
        # rec.record(name, x)
        # 所以这里不需要调用 record，record 会被单独调用。
        return x

@torch.no_grad()
def analyze_coverage_distribution(args, config_path: str = None):
    """
    统计生成的配置该层被近似值占全部的百分比，并输出到一个excel。
    需要读取 tri_ppo_int_codes/result.json 获取 tmax_code。
    """
    torch.backends.quantized.engine = args.backend
    
    # 1. 读取 PPO 配置
    if config_path is None:
        config_path = os.path.join(args.out, "tri_ppo_int_codes", "result.json")
    if not os.path.exists(config_path):
        print(f"[Error] Config not found: {config_path}")
        return

    with open(config_path, "r") as f:
        cfg = json.load(f)
    
    layer_cfgs = cfg.get("layers", {})
    # 提取 tmax_code
    tmax_map = {}
    for layer_name, params in layer_cfgs.items():
        tmax_map[layer_name] = int(params["tmax_code"])
    
    print(f"[Info] Loaded tmax configs for {len(tmax_map)} layers.")

    # 2. 构建量化模型
    model_name = getattr(args, "model", "vgg16")
    print(f"[Info] Loading model: {model_name}")
    
    if model_name == "vgg19":
        float_model = VGG19TapQuant(num_classes=10)
    elif model_name == "alexnet":
        float_model = AlexNetTapQuant(num_classes=10)
    elif model_name == "lenet5":
        float_model = LeNet5TapQuant(num_classes=10)
    elif model_name == "nin":
        float_model = NiNTapQuant(num_classes=10)
    elif model_name == "mobilenetv1":
        float_model = MobileNetV1TapQuant(num_classes=10)
    else:
        float_model = VGG16TapQuant(num_classes=10)

    float_model.eval()
    float_model.fuse_model()
    float_model.qconfig = get_default_qat_qconfig(args.backend)
    float_model.train()
    prepare_qat(float_model, inplace=True)
    
    ckpt_path = os.path.join(args.out, f"{model_name}_qat_preconvert.pth")
    if not os.path.exists(ckpt_path):
         if model_name == "vgg16" and not os.path.exists(ckpt_path):
             ckpt_path = os.path.join(args.out, "vgg16_qat_preconvert.pth")
    
    print(f"[Info] Loading checkpoint from: {ckpt_path}")
    float_model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    float_model.eval()
    qmodel = convert(float_model, inplace=False)

    # 3. 数据加载
    _, test_loader = get_dataloaders(args.data, args.batch_size, args.workers)

    # 4. 统计覆盖率
    ignore_zeros = getattr(args, "coverage_ignore_zeros", False)
    print(f"[Info] Coverage analysis mode: {'Ignore Zeros' if ignore_zeros else 'Include Zeros'}")
    recorder = CoverageRecorder(tmax_map, ignore_zeros=ignore_zeros)
    
    print("[Info] Running inference to collect coverage stats...")
    # 只需要跑一部分数据即可，比如 10 个 batch，或者全部
    # 为了准确性，建议跑全部测试集，或者至少一部分
    max_batches = 20
    for i, (images, _) in enumerate(test_loader):
        if i >= max_batches:
            break
        qmodel(images, recorder=recorder)
        if (i + 1) % 5 == 0:
            print(f"  Processed {i + 1} batches...")

    # 5. 整理结果并导出 Excel
    results = []
    global_total = 0
    global_covered = 0

    for layer, stats in recorder.stats.items():
        total = stats['total']
        covered = stats['covered']
        
        global_total += total
        global_covered += covered

        ratio = covered / total if total > 0 else 0
        tmax = tmax_map.get(layer, -1)
        
        results.append({
            "Layer": layer,
            "Tmax Code": tmax,
            "Total Pixels": total,
            "Covered Pixels": covered,
            "Coverage Ratio": ratio,
            "Coverage %": f"{ratio*100:.2f}%"
        })
    
    # 排序
    # 尝试按层顺序排序，这里简单按名字排
    results.sort(key=lambda x: x["Layer"])

    # 添加总计行
    global_ratio = global_covered / global_total if global_total > 0 else 0
    results.append({
        "Layer": "OVERALL",
        "Tmax Code": "-",
        "Total Pixels": global_total,
        "Covered Pixels": global_covered,
        "Coverage Ratio": global_ratio,
        "Coverage %": f"{global_ratio*100:.2f}%"
    })
    
    df = pd.DataFrame(results)
    print("\n[Coverage Analysis Result]")
    print(df[["Layer", "Tmax Code", "Coverage %"]])
    
    save_path = os.path.join(args.out, "coverage_analysis.xlsx")
    try:
        df.to_excel(save_path, index=False)
        print(f"\n[Saved] Coverage analysis saved to {save_path}")
    except ImportError:
        print("\n[Error] 'openpyxl' is required to save Excel files. Please install it: pip install openpyxl")
        csv_path = os.path.join(args.out, "coverage_analysis.csv")
        df.to_csv(csv_path, index=False)
        print(f"[Saved] Saved to CSV instead: {csv_path}")
    except Exception as e:
        print(f"\n[Error] Failed to save Excel: {e}")

# inference_int8.py
import os
import torch

from torch.ao.quantization import prepare_qat, get_default_qat_qconfig, convert

from model_vgg16_tap_quant import VGG16TapQuant
from utils import get_dataloaders, ensure_dir

@torch.no_grad()
def export_int8(args):
    # 构建量化模型并 load QAT 权重
    float_model = VGG16TapQuant(num_classes=10)
    float_model.eval()
    float_model.fuse_model()

    torch.backends.quantized.engine = args.backend
    float_model.qconfig = get_default_qat_qconfig(args.backend)
    float_model.train()
    prepare_qat(float_model, inplace=True)

    float_model.load_state_dict(
        torch.load(os.path.join(args.out, "vgg16_qat_preconvert.pth"), map_location="cpu")
    )
    float_model.eval()
    qmodel = convert(float_model, inplace=False)

    ensure_dir(args.out)
    # 存 state_dict
    torch.save(qmodel.state_dict(), os.path.join(args.out, "vgg16_int8_state.pth"))
    print("[Saved] INT8 state_dict.")

    # 存 TorchScript
    example = torch.randn(1, 3, 32, 32)
    ts = torch.jit.trace(qmodel, example)
    ts = torch.jit.freeze(ts)
    torch.jit.save(ts, os.path.join(args.out, "vgg16_int8.ts"))
    print("[Saved] INT8 TorchScript.")

@torch.no_grad()
def eval_int8(args):
    torch.backends.quantized.engine = args.backend
    _, test_loader = get_dataloaders(args.data, args.batch_size, args.workers)

    ts = torch.jit.load(os.path.join(args.out, "vgg16_int8.ts"), map_location="cpu")
    ts.eval()

    correct, total = 0, 0
    for images, targets in test_loader:
        logits = ts(images)
        preds = logits.argmax(1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    acc = correct / total * 100.0
    print(f"[INT8] Accuracy: {acc:.2f}%")
    return acc

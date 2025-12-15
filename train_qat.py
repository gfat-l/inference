# train_qat.py
import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.ao.quantization import prepare_qat, get_default_qat_qconfig

from model_vgg16_tap_quant import VGG16TapQuant
from model_vgg19_tap_quant import VGG19TapQuant
from model_alexnet_tap_quant import AlexNetTapQuant
from model_lenet5_tap_quant import LeNet5TapQuant
from model_nin_tap_quant import NiNTapQuant
from model_mobilenetv1_tap_quant import MobileNetV1TapQuant
from utils import get_dataloaders, ensure_dir, disable_observer, freeze_bn_stats_compat

def train_one_epoch(model, loader, criterion, optimizer, device, epoch, print_freq=100):
    model.train()
    for it, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        if (it + 1) % print_freq == 0:
            acc = (logits.argmax(1) == targets).float().mean().item() * 100
            print(f"Epoch {epoch} | Iter {it+1}/{len(loader)} | Loss {loss.item():.4f} | Acc {acc:.2f}%")

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss_sum += loss.item() * targets.size(0)
        correct += (logits.argmax(1) == targets).sum().item()
        total += targets.size(0)
    return loss_sum / total, correct / total

def train_float(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cudnn.benchmark = True

    train_loader, test_loader = get_dataloaders(args.data, args.batch_size, args.workers)
    
    # 选择模型架构
    model_type = getattr(args, "model", "vgg16")
    if model_type == "vgg19":
        model = VGG19TapQuant(num_classes=10).to(device)
        model_name = "vgg19"
    elif model_type == "alexnet":
        model = AlexNetTapQuant(num_classes=10).to(device)
        model_name = "alexnet"
    elif model_type == "lenet5":
        model = LeNet5TapQuant(num_classes=10).to(device)
        model_name = "lenet5"
    elif model_type == "nin":
        model = NiNTapQuant(num_classes=10).to(device)
        model_name = "nin"
    elif model_type == "mobilenetv1":
        model = MobileNetV1TapQuant(num_classes=10).to(device)
        model_name = "mobilenetv1"
    else:
        model = VGG16TapQuant(num_classes=10).to(device)
        model_name = "vgg16"

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)
    milestones = [int(args.epochs * 0.3), int(args.epochs * 0.6)]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    # 跟踪最佳验证精度
    best_val_acc = 0.0
    best_epoch = 0
    best_model_state = None

    for ep in range(1, args.epochs + 1):
        train_one_epoch(model, train_loader, criterion, optimizer, device, ep)
        vl, va = evaluate(model, test_loader, device)
        print(f"[Float] Epoch {ep} | Val Loss {vl:.4f} | Val Acc {va*100:.2f}%", end="")
        
        # 保存最佳模型
        if va > best_val_acc:
            best_val_acc = va
            best_epoch = ep
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f" <- Best!")
        else:
            print()
        
        scheduler.step()

    # 保存最佳模型
    ensure_dir(args.out)
    save_path = os.path.join(args.out, f"{model_name}_float_unfused.pth")
    torch.save(best_model_state, save_path)
    print(f"\n[Saved] {model_name} float weights (unfused) to {save_path}")
    print(f"[Info] Best validation accuracy: {best_val_acc*100:.2f}% at epoch {best_epoch}")

def train_qat(args):
    # 选择模型架构
    model_type = getattr(args, "model", "vgg16")
    if model_type == "vgg19":
        model_name = "vgg19"
        ModelClass = VGG19TapQuant
    elif model_type == "alexnet":
        model_name = "alexnet"
        ModelClass = AlexNetTapQuant
    elif model_type == "lenet5":
        model_name = "lenet5"
        ModelClass = LeNet5TapQuant
    elif model_type == "nin":
        model_name = "nin"
        ModelClass = NiNTapQuant
    elif model_type == "mobilenetv1":
        model_name = "mobilenetv1"
        ModelClass = MobileNetV1TapQuant
    else:
        model_name = "vgg16"
        ModelClass = VGG16TapQuant
    
    float_pth = os.path.join(args.out, f"{model_name}_float_unfused.pth")
    if not os.path.exists(float_pth):
        print(f"[Info] {model_name} float model not found, pretraining...")
        warmup_args = argparse.Namespace(**vars(args))
        warmup_args.epochs = max(2, args.warmup_float_epochs)
        train_float(warmup_args)

    # 1. 加载 float
    model = ModelClass(num_classes=10)
    model.load_state_dict(torch.load(float_pth, map_location="cpu"))

    # 2. 融合
    model.eval()
    model.fuse_model()
    model.train()

    # 3. QAT 准备
    if args.backend not in ("fbgemm", "qnnpack"):
        raise ValueError("backend must be fbgemm or qnnpack")
    torch.backends.quantized.engine = args.backend
    model.qconfig = get_default_qat_qconfig(args.backend)
    prepare_qat(model, inplace=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    train_loader, test_loader = get_dataloaders(args.data, args.batch_size, args.workers)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.qat_lr, momentum=0.9, weight_decay=5e-4)
    milestones = [int(args.qat_epochs * 0.5), int(args.qat_epochs * 0.75)]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    freeze_after = max(1, int(args.qat_epochs * 0.6))
    
    # 跟踪最佳验证精度（仅用于显示）
    best_val_acc = 0.0
    best_epoch = 0

    for ep in range(1, args.qat_epochs + 1):
        train_one_epoch(model, train_loader, criterion, optimizer, device, ep)
        vl, va = evaluate(model, test_loader, device)
        print(f"[QAT] Epoch {ep} | Val Loss {vl:.4f} | Val Acc {va*100:.2f}%", end="")

        # 跟踪最佳精度
        if va > best_val_acc:
            best_val_acc = va
            best_epoch = ep
            print(f" <- Best!")
        else:
            print()

        if ep == freeze_after:
            disable_observer(model)
            freeze_bn_stats_compat(model)
            print("[QAT] observers disabled & BN frozen")

        scheduler.step()

    # 保存最终模型（不保存中间best，因为QAT量化模块状态复杂）
    ensure_dir(args.out)
    save_path = os.path.join(args.out, f"{model_name}_qat_preconvert.pth")
    torch.save(model.state_dict(), save_path)
    print(f"\n[Saved] {model_name} QAT weights (pre-convert) to {save_path}")
    print(f"[Info] Best validation accuracy: {best_val_acc*100:.2f}% at epoch {best_epoch}")
    print(f"[Info] Final model saved (last epoch)")

def eval_float(args):
    """评估Float模型在测试集上的精度"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 选择模型架构
    model_type = getattr(args, "model", "vgg16")
    if model_type == "vgg19":
        model_name = "vgg19"
        ModelClass = VGG19TapQuant
    elif model_type == "alexnet":
        model_name = "alexnet"
        ModelClass = AlexNetTapQuant
    elif model_type == "lenet5":
        model_name = "lenet5"
        ModelClass = LeNet5TapQuant
    else:
        model_name = "vgg16"
        ModelClass = VGG16TapQuant
    
    float_pth = os.path.join(args.out, f"{model_name}_float_unfused.pth")
    if not os.path.exists(float_pth):
        print(f"[Error] {model_name} float model not found at {float_pth}")
        print(f"[Info] Please train the float model first:")
        print(f"       python main.py --mode train_float --model {model_type} --epochs 30")
        return
    
    # 加载模型
    model = ModelClass(num_classes=10)
    model.load_state_dict(torch.load(float_pth, map_location="cpu"))
    model.to(device)
    model.eval()
    
    # 加载测试集
    _, test_loader = get_dataloaders(args.data, args.batch_size, args.workers)
    
    # 评估
    test_loss, test_acc = evaluate(model, test_loader, device)
    
    print("\n" + "="*60)
    print(f"[Eval Float] Model: {model_name.upper()}")
    print(f"[Eval Float] Test Loss: {test_loss:.4f}")
    print(f"[Eval Float] Test Accuracy: {test_acc*100:.2f}%")
    print("="*60 + "\n")

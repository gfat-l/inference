# utils.py
from pathlib import Path
from typing import Tuple
import os
import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 过滤PyTorch量化相关的警告
warnings.filterwarnings('ignore', message='.*reduce_range will be deprecated.*')
warnings.filterwarnings('ignore', category=UserWarning, module='torch.ao.quantization')

# -------- disable_observer 兼容 --------
# 原文件里是三级 fallback，这里保持
try:
    from torch.ao.quantization import disable_observer  # type: ignore
except Exception:
    try:
        from torch.quantization import disable_observer  # type: ignore
    except Exception:
        # 最兜底：自己遍历 FakeQuantize
        try:
            from torch.ao.quantization.fake_quantize import FakeQuantize as _FQ  # type: ignore
        except Exception:
            try:
                from torch.quantization.fake_quantize import FakeQuantize as _FQ  # type: ignore
            except Exception:
                _FQ = None

        def disable_observer(model: nn.Module) -> None:  # type: ignore[redefinition]
            if _FQ is None:
                return
            for m in model.modules():
                if isinstance(m, _FQ):
                    m.disable_observer()

# -------- 冻结 BN 的兼容实现 --------
def freeze_bn_stats_compat(model: nn.Module) -> None:
    """将模型内所有 BatchNorm 切到 eval，停止更新 running_mean/var。"""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()

# -------- 路径 & 数据 --------
def ensure_dir(d: str) -> None:
    Path(d).mkdir(parents=True, exist_ok=True)

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)

def get_dataloaders(data_dir: str, batch_size: int, workers: int) -> Tuple[DataLoader, DataLoader]:
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_tf)
    test_set  = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_tf)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=workers, pin_memory=True)
    test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                              num_workers=workers, pin_memory=True)
    return train_loader, test_loader

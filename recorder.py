# recorder.py
from typing import Callable, Dict, Optional

import torch

class ActivationRecorder:
    """
    记录每一层的激活，同时支持"如果命中某个名字，就用用户给的函数改掉它"。
    会同时存 int_repr 和 dequantize 后的 float，方便后处理。
    新增：记录激活值的方差分布
    """
    def __init__(self, store_cpu: bool = True,
                 edits: Optional[Dict[str, Callable[[torch.Tensor], torch.Tensor]]] = None,
                 record_variance: bool = False):
        self.store_cpu = store_cpu
        self.storage: Dict[str, torch.Tensor] = {}
        self.storage_float: Dict[str, torch.Tensor] = {}
        self.edits = edits or {}
        
        # 新增：方差统计
        self.record_variance = record_variance
        self.variance_stats: Dict[str, Dict[str, float]] = {}  # {layer_name: {'mean': x, 'var': x, 'std': x}}

    def record(self, name: str, x: torch.Tensor) -> None:
        with torch.no_grad():
            if hasattr(x, "int_repr"):  # 量化张量
                int_part = x.int_repr().detach()
                float_part = x.dequantize().detach()
                if self.store_cpu:
                    int_part = int_part.cpu()
                    float_part = float_part.cpu()
                self.storage[name] = int_part
                self.storage_float[name] = float_part
                
                # 记录方差统计（使用反量化后的浮点值）
                if self.record_variance:
                    self._compute_variance(name, float_part)
            else:
                v = x.detach()
                if self.store_cpu:
                    v = v.cpu()
                self.storage[name] = v
                self.storage_float[name] = v
                
                # 记录方差统计
                if self.record_variance:
                    self._compute_variance(name, v)
    
    def _compute_variance(self, name: str, tensor: torch.Tensor) -> None:
        """计算张量的均值、方差和标准差"""
        self.variance_stats[name] = {
            'mean': tensor.mean().item(),
            'var': tensor.var().item(),
            'std': tensor.std().item(),
            'shape': list(tensor.shape)
        }

    def maybe_edit(self, name: str, x: torch.Tensor) -> torch.Tensor:
        if name in self.edits:
            return self.edits[name](x)
        return x

# ---- 下面是原来 demo 用的两个小编辑函数 ----
def zero_channels(ch_idx):
    def fn(x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        x[:, ch_idx, ...] = 0
        return x
    return fn

def scale_value(alpha: float):
    def fn(x: torch.Tensor) -> torch.Tensor:
        return x * alpha
    return fn


class WeightVarianceRecorder:
    """记录模型中卷积层权重的方差分布"""
    def __init__(self):
        self.variance_stats: Dict[str, Dict[str, float]] = {}
    
    def record_model_weights(self, model: torch.nn.Module) -> None:
        """遍历模型，记录所有Conv2d层的权重方差"""
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                if hasattr(module, 'weight') and module.weight is not None:
                    weight = module.weight.detach()
                    
                    # 整体统计
                    self.variance_stats[name] = {
                        'mean': weight.mean().item(),
                        'var': weight.var().item(),
                        'std': weight.std().item(),
                        'shape': list(weight.shape),
                        'min': weight.min().item(),
                        'max': weight.max().item(),
                    }
                    
                    # 按输出通道统计
                    per_channel_var = weight.var(dim=[1, 2, 3])  # [out_channels]
                    self.variance_stats[name]['per_channel_var_mean'] = per_channel_var.mean().item()
                    self.variance_stats[name]['per_channel_var_std'] = per_channel_var.std().item()
    
    def save_to_file(self, filepath: str) -> None:
        """保存统计结果到文件"""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.variance_stats, f, indent=2)
        print(f"[Saved] Weight variance stats to {filepath}")

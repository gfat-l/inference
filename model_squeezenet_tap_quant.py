"""
SqueezeNet with TAP-based quantization support for CIFAR-10

SqueezeNet核心设计思想：
1. Fire模块：squeeze层(1x1卷积压缩) + expand层(1x1和3x3卷积并行)
2. 大量使用1x1卷积减少参数
3. 延迟降采样保留更多空间信息
4. 参数量仅约1.2M，但精度接近AlexNet

Fire模块结构：
    Input (C_in channels)
       ↓
    Squeeze (1x1 Conv, s1x1 channels) ← 压缩通道数
       ↓
    ReLU
       ↓
    ┌─────────────┬─────────────┐
    ↓             ↓             ↓
 Expand1x1    Expand3x3     (并行)
 (e1x1 ch)    (e3x3 ch)
    ↓             ↓
    └──────concat──────┘
       ↓
    Output (e1x1 + e3x3 channels)

通常设置：s1x1 < e1x1 + e3x3（压缩-扩展策略）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
from recorder import TensorRecorder


class Fire(nn.Module):
    """
    Fire模块：SqueezeNet的核心构建块
    
    参数说明：
        in_channels: 输入通道数
        squeeze_channels: squeeze层输出通道数（压缩后的通道数）
        expand1x1_channels: expand层中1x1卷积的输出通道数
        expand3x3_channels: expand层中3x3卷积的输出通道数
    
    输出通道数 = expand1x1_channels + expand3x3_channels
    
    典型配置：
        Fire(64, 16, 64, 64)  → 输入64，压缩到16，扩展到128(64+64)
    """
    
    def __init__(
        self,
        in_channels: int,
        squeeze_channels: int,
        expand1x1_channels: int,
        expand3x3_channels: int
    ):
        super(Fire, self).__init__()
        
        # Squeeze层：1x1卷积压缩通道数
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        
        # Expand层：1x1和3x3卷积并行
        self.expand1x1 = nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1)
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1)
        
        # Expand后的激活
        self.expand_activation = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Squeeze阶段
        x = self.squeeze(x)
        x = self.squeeze_activation(x)
        
        # Expand阶段：1x1和3x3并行，然后在通道维度拼接
        out1x1 = self.expand1x1(x)
        out3x3 = self.expand3x3(x)
        x = torch.cat([out1x1, out3x3], dim=1)  # 通道维度拼接
        
        x = self.expand_activation(x)
        return x


class SqueezeNetTapQuant(nn.Module):
    """
    SqueezeNet for CIFAR-10 with TAP-based quantization support
    
    网络结构（CIFAR-10适配版本）：
        conv1:  3x3x3x96, stride=1 (CIFAR-10小图不用stride=2)
        maxpool1: 3x3, stride=2
        
        fire2:  Fire(96, 16, 64, 64)   → 128 channels
        fire3:  Fire(128, 16, 64, 64)  → 128 channels
        fire4:  Fire(128, 32, 128, 128) → 256 channels
        maxpool2: 3x3, stride=2
        
        fire5:  Fire(256, 32, 128, 128) → 256 channels
        fire6:  Fire(256, 48, 192, 192) → 384 channels
        fire7:  Fire(384, 48, 192, 192) → 384 channels
        fire8:  Fire(384, 64, 256, 256) → 512 channels
        maxpool3: 3x3, stride=2
        
        fire9:  Fire(512, 64, 256, 256) → 512 channels
        
        conv10: 1x1x512x10 (分类层)
        avgpool: global average pooling
        
    总共10个主要层（1个conv + 8个fire + 1个conv）
    
    量化近似层：conv1 + 8个fire模块 = 9层
    （conv10作为最终分类层，不进行近似）
    """
    
    def __init__(self, num_classes: int = 10, tap_dict: Optional[Dict[str, Any]] = None):
        super(SqueezeNetTapQuant, self).__init__()
        
        self.num_classes = num_classes
        self.tap_dict = tap_dict if tap_dict is not None else {}
        self.recorder = TensorRecorder()
        
        # Stage 1: 初始卷积层
        self.conv1 = nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1)  # 32x32 → 32x32
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)  # 32x32 → 16x16
        
        # Stage 2: Fire模块组 (16x16)
        self.fire2 = Fire(96, 16, 64, 64)      # 96 → 128
        self.fire3 = Fire(128, 16, 64, 64)     # 128 → 128
        self.fire4 = Fire(128, 32, 128, 128)   # 128 → 256
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)  # 16x16 → 8x8
        
        # Stage 3: Fire模块组 (8x8)
        self.fire5 = Fire(256, 32, 128, 128)   # 256 → 256
        self.fire6 = Fire(256, 48, 192, 192)   # 256 → 384
        self.fire7 = Fire(384, 48, 192, 192)   # 384 → 384
        self.fire8 = Fire(384, 64, 256, 256)   # 384 → 512
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)  # 8x8 → 4x4
        
        # Stage 4: 最后的Fire模块 (4x4)
        self.fire9 = Fire(512, 64, 256, 256)   # 512 → 512
        
        # 分类器
        # Dropout防止过拟合
        self.dropout = nn.Dropout(p=0.5)
        # 最终卷积层：1x1卷积将512通道映射到类别数
        self.conv10 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.relu10 = nn.ReLU(inplace=True)
        # 全局平均池化：4x4 → 1x1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def fuse_model(self):
        """
        融合BN层（SqueezeNet原始版本没有BN）
        这里保留接口以兼容框架，但实际不做操作
        """
        pass
    
    def _apply_tap_approx(self, x: torch.Tensor, layer_name: str) -> torch.Tensor:
        """
        应用TAP近似到激活值
        
        TAP近似说明：
        - 将浮点激活值量化为INT8码表中的4个值
        - 使用三角近似：t1, v1, t2, v2
        - 适用于Fire模块的输出激活
        """
        if layer_name not in self.tap_dict:
            return x
        
        codes = self.tap_dict[layer_name]
        if codes is None or len(codes) != 4:
            return x
        
        t1, v1, t2, v2 = codes
        
        # 创建量化后的激活值
        x_q = torch.zeros_like(x)
        
        # 分段线性近似
        mask1 = x <= t1
        mask2 = (x > t1) & (x <= t2)
        mask3 = x > t2
        
        x_q[mask1] = 0.0
        x_q[mask2] = v1 + (x[mask2] - t1) * (v2 - v1) / (t2 - t1)
        x_q[mask3] = v2
        
        return x_q
    
    def _forward_fire_module(self, fire_module: Fire, x: torch.Tensor, layer_name: str) -> torch.Tensor:
        """
        Fire模块的前向传播，支持记录和TAP近似
        
        Fire模块的激活记录在expand后、激活函数后
        """
        # Squeeze阶段
        x = fire_module.squeeze(x)
        x = fire_module.squeeze_activation(x)
        
        # Expand阶段
        out1x1 = fire_module.expand1x1(x)
        out3x3 = fire_module.expand3x3(x)
        x = torch.cat([out1x1, out3x3], dim=1)
        
        # Expand激活
        x = fire_module.expand_activation(x)
        
        # 记录Fire模块的输出激活（激活函数后）
        record_key = f"block_output.{layer_name}"
        self.recorder.record(record_key, x)
        
        # 应用TAP近似
        x = self._apply_tap_approx(x, layer_name)
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        输入: (N, 3, 32, 32) CIFAR-10图像
        输出: (N, 10) 类别logits
        """
        # Stage 1: conv1
        x = self.conv1(x)
        x = self.relu1(x)
        # 记录conv1输出
        self.recorder.record("block_output.conv1", x)
        x = self._apply_tap_approx(x, "conv1")
        x = self.maxpool1(x)  # 32x32 → 16x16
        
        # Stage 2: fire2-4
        x = self._forward_fire_module(self.fire2, x, "fire2")
        x = self._forward_fire_module(self.fire3, x, "fire3")
        x = self._forward_fire_module(self.fire4, x, "fire4")
        x = self.maxpool2(x)  # 16x16 → 8x8
        
        # Stage 3: fire5-8
        x = self._forward_fire_module(self.fire5, x, "fire5")
        x = self._forward_fire_module(self.fire6, x, "fire6")
        x = self._forward_fire_module(self.fire7, x, "fire7")
        x = self._forward_fire_module(self.fire8, x, "fire8")
        x = self.maxpool3(x)  # 8x8 → 4x4
        
        # Stage 4: fire9
        x = self._forward_fire_module(self.fire9, x, "fire9")
        
        # 分类器
        x = self.dropout(x)
        x = self.conv10(x)
        x = self.relu10(x)
        x = self.avgpool(x)  # 4x4 → 1x1
        x = torch.flatten(x, 1)  # (N, 10, 1, 1) → (N, 10)
        
        return x
    
    def get_recorder(self) -> TensorRecorder:
        """获取记录器"""
        return self.recorder
    
    def set_tap_dict(self, tap_dict: Dict[str, Any]):
        """设置TAP码表"""
        self.tap_dict = tap_dict


def get_squeezenet_layer_names():
    """
    返回SqueezeNet中所有需要量化近似的层名称
    
    返回9层：
        - conv1: 初始卷积层
        - fire2-fire9: 8个Fire模块
    
    注意：conv10不进行近似（直接输出logits）
    """
    return [
        "conv1",
        "fire2", "fire3", "fire4",
        "fire5", "fire6", "fire7", "fire8",
        "fire9"
    ]


def get_squeezenet_attention_layers():
    """
    返回推荐的注意力转移层
    
    选择策略：
        - fire4: Stage2末尾（256 channels）
        - fire6: Stage3中部（384 channels）
        - fire8: Stage3末尾（512 channels）
    
    这3层分别代表浅层、中层、深层特征
    """
    return ["fire4", "fire6", "fire8"]


# ========== 网络结构说明 ==========
"""
SqueezeNet vs VGG16 vs AlexNet (CIFAR-10):

参数量对比：
    SqueezeNet:  ~1.2M  (最少)
    AlexNet:     ~3M
    VGG16:       ~15M   (最多)

计算量对比（FLOPs）：
    SqueezeNet:  ~350M  (最少)
    AlexNet:     ~700M
    VGG16:       ~3G    (最多)

精度对比（Float32）：
    SqueezeNet:  ~91%
    AlexNet:     ~89%
    VGG16:       ~93%

SqueezeNet优势：
    1. 参数量极少（1/12 VGG16，1/3 AlexNet）
    2. 大量1x1卷积，适合量化
    3. 压缩-扩展设计，计算高效
    4. 适合边缘设备部署

SqueezeNet劣势：
    1. Fire模块结构复杂（需要特殊处理）
    2. 精度略低于VGG16
    3. 训练调参较敏感

Fire模块设计思想：
    1. Squeeze层：1x1卷积大幅压缩通道数（降低计算量）
    2. Expand层：1x1和3x3并行，增加表达能力
    3. 通道拼接：融合不同感受野的特征
    4. 延迟池化：尽可能保留空间分辨率

示例：Fire(256, 48, 192, 192)
    输入：256通道特征图
    ↓
    Squeeze: 256 → 48 (压缩到18.75%)
    ↓
    Expand1x1: 48 → 192
    Expand3x3: 48 → 192
    ↓
    Concat: 192 + 192 = 384通道输出

计算量分析：
    标准Conv(256→384, 3x3): 256×384×9 = 884,736
    Fire模块:
        - Squeeze(256→48, 1x1): 256×48×1 = 12,288
        - Expand1x1(48→192, 1x1): 48×192×1 = 9,216
        - Expand3x3(48→192, 3x3): 48×192×9 = 82,944
        - 总计: 104,448
    压缩比: 884,736 / 104,448 ≈ 8.5倍

量化挑战：
    1. Fire模块的concat操作需要特殊处理
    2. 1x1和3x3分支的scale可能不同
    3. Squeeze瓶颈层对量化误差敏感
    
建议：
    1. 对Fire模块整体进行TAP近似（在expand后）
    2. 注意力转移选择fire4, fire6, fire8三个关键点
    3. 适当增加calibration数据（Fire模块需要更多样本）
"""

if __name__ == "__main__":
    """
    测试SqueezeNet模型
    """
    print("=" * 60)
    print("SqueezeNet结构测试")
    print("=" * 60)
    
    # 创建模型
    model = SqueezeNetTapQuant(num_classes=10)
    print(f"\n✓ 模型创建成功")
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ 总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"✓ 可训练参数: {trainable_params:,}")
    
    # 测试前向传播
    x = torch.randn(2, 3, 32, 32)
    print(f"\n✓ 输入形状: {x.shape}")
    
    y = model(x)
    print(f"✓ 输出形状: {y.shape}")
    print(f"✓ 输出范围: [{y.min().item():.2f}, {y.max().item():.2f}]")
    
    # 打印层名称
    layer_names = get_squeezenet_layer_names()
    print(f"\n✓ 量化层数量: {len(layer_names)}")
    print(f"✓ 层名称: {layer_names}")
    
    # 打印注意力层
    attention_layers = get_squeezenet_attention_layers()
    print(f"\n✓ 注意力转移层: {attention_layers}")
    
    # 打印记录的激活
    recorder = model.get_recorder()
    print(f"\n✓ 记录的激活数量: {len(recorder.records)}")
    for key in sorted(recorder.records.keys()):
        shape = recorder.records[key].shape
        print(f"  - {key}: {shape}")
    
    # 打印Fire模块详情
    print("\n" + "=" * 60)
    print("Fire模块详细信息")
    print("=" * 60)
    
    fire_modules = [
        ("fire2", model.fire2, 96),
        ("fire3", model.fire3, 128),
        ("fire4", model.fire4, 128),
        ("fire5", model.fire5, 256),
        ("fire6", model.fire6, 256),
        ("fire7", model.fire7, 384),
        ("fire8", model.fire8, 384),
        ("fire9", model.fire9, 512),
    ]
    
    for name, module, in_ch in fire_modules:
        s_ch = module.squeeze.out_channels
        e1_ch = module.expand1x1.out_channels
        e3_ch = module.expand3x3.out_channels
        out_ch = e1_ch + e3_ch
        print(f"{name}: {in_ch}→[squeeze:{s_ch}]→[expand:{e1_ch}+{e3_ch}]→{out_ch}")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)

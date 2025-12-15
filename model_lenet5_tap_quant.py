"""
LeNet-5 with TAP-based quantization support for CIFAR-10

LeNet-5是最经典的CNN网络之一（1998年Yann LeCun提出）
原始设计用于手写数字识别（MNIST 28x28灰度图）
这里适配到CIFAR-10（32x32彩色图）

网络结构：
    Conv1 (5x5) → Pool1 → Conv2 (5x5) → Pool2 → FC1 → FC2 → FC3
    
特点：
    - 最简单的CNN结构，纯顺序执行
    - 无并行分支、无concat、无残差连接
    - 2个卷积层 + 3个全连接层 = 5层
    - 参数量极少（~60K）
    - 训练速度极快（适合快速验证）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
from recorder import ActivationRecorder


class LeNet5TapQuant(nn.Module):
    """
    LeNet-5 for CIFAR-10 with TAP-based quantization support
    
    网络结构（CIFAR-10适配版本）：
        Input: (N, 3, 32, 32)
        
        Conv1: 3x3x5x5 → 6 channels, 32x32 → 28x28
        ReLU
        MaxPool1: 2x2, stride=2 → 28x28 → 14x14
        
        Conv2: 6x5x5 → 16 channels, 14x14 → 10x10
        ReLU
        MaxPool2: 2x2, stride=2 → 10x10 → 5x5
        
        Flatten: 16x5x5 = 400
        
        FC1: 400 → 120
        ReLU
        
        FC2: 120 → 84
        ReLU
        
        FC3: 84 → 10 (分类层)
    
    量化近似层：conv1, conv2, fc1, fc2 = 4层
    （fc3作为最终分类层，不进行近似）
    """
    
    def __init__(self, num_classes: int = 10, tap_dict: Optional[Dict[str, Any]] = None):
        super(LeNet5TapQuant, self).__init__()
        
        self.num_classes = num_classes
        self.tap_dict = tap_dict if tap_dict is not None else {}
        
        # 卷积层
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0)   # 32x32 → 28x28
        self.relu1 = nn.ReLU(inplace=False)  # 需要记录激活，不能inplace
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)                 # 28x28 → 14x14
        
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)  # 14x14 → 10x10
        self.relu2 = nn.ReLU(inplace=False)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)                 # 10x10 → 5x5
        
        # 全连接层
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 400 → 120
        self.relu3 = nn.ReLU(inplace=False)
        
        self.fc2 = nn.Linear(120, 84)          # 120 → 84
        self.relu4 = nn.ReLU(inplace=False)
        
        self.fc3 = nn.Linear(84, num_classes)  # 84 → 10（分类层，不量化）
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def fuse_model(self):
        """
        融合BN层（LeNet-5原始版本没有BN）
        保留接口以兼容框架
        """
        pass
    
    def _apply_tap_approx(self, x: torch.Tensor, layer_name: str) -> torch.Tensor:
        """
        应用TAP近似到激活值
        
        TAP近似：将浮点激活值量化为INT8码表中的4个值
        使用三角近似：t1, v1, t2, v2
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
    
    def forward(self, x: torch.Tensor, recorder: Optional[ActivationRecorder] = None) -> torch.Tensor:
        """
        前向传播
        
        输入: (N, 3, 32, 32) CIFAR-10图像
        输出: (N, 10) 类别logits
        """
        # Conv1 层
        x = self.conv1(x)           # (N, 3, 32, 32) → (N, 6, 28, 28)
        x = self.relu1(x)
        # 记录conv1输出（ReLU后）
        if recorder is not None:
            recorder.record("block_output.conv1", x)
            x = recorder.maybe_edit("block_output.conv1", x)
        x = self._apply_tap_approx(x, "conv1")
        x = self.pool1(x)           # (N, 6, 28, 28) → (N, 6, 14, 14)
        
        # Conv2 层
        x = self.conv2(x)           # (N, 6, 14, 14) → (N, 16, 10, 10)
        x = self.relu2(x)
        # 记录conv2输出（ReLU后）
        if recorder is not None:
            recorder.record("block_output.conv2", x)
            x = recorder.maybe_edit("block_output.conv2", x)
        x = self._apply_tap_approx(x, "conv2")
        x = self.pool2(x)           # (N, 16, 10, 10) → (N, 16, 5, 5)
        
        # Flatten
        x = torch.flatten(x, 1)     # (N, 16, 5, 5) → (N, 400)
        
        # FC1 层
        x = self.fc1(x)             # (N, 400) → (N, 120)
        x = self.relu3(x)
        # 记录fc1输出（ReLU后）
        # 使用与AlexNet一致的命名格式：classifier.fc_name.out
        if recorder is not None:
            recorder.record("classifier.fc1.out", x)
            x = recorder.maybe_edit("classifier.fc1.out", x)
        x = self._apply_tap_approx(x, "fc1")
        
        # FC2 层
        x = self.fc2(x)             # (N, 120) → (N, 84)
        x = self.relu4(x)
        # 记录fc2输出（ReLU后）
        if recorder is not None:
            recorder.record("classifier.fc2.out", x)
            x = recorder.maybe_edit("classifier.fc2.out", x)
        x = self._apply_tap_approx(x, "fc2")
        
        # FC3 层（分类层，不量化）
        x = self.fc3(x)             # (N, 84) → (N, 10)
        
        return x
    
    def set_tap_dict(self, tap_dict: Dict[str, Any]):
        """设置TAP码表"""
        self.tap_dict = tap_dict


def get_lenet5_layer_names():
    """
    返回LeNet-5中所有需要量化近似的层名称
    
    返回4层：
        - conv1, conv2: 2个卷积层
        - fc1, fc2: 2个全连接层
    
    注意：fc3不进行近似（直接输出logits）
    """
    return ["conv1", "conv2", "fc1", "fc2"]


def get_lenet5_attention_layers():
    """
    返回推荐的注意力转移层
    
    LeNet-5只有4个量化层，较浅
    选择策略：
        - conv2: 卷积层的输出特征
        - fc1: 全连接层的第一层特征
    
    由于网络太浅，只选2层进行注意力对齐
    """
    return ["conv2", "fc1"]


# ========== 网络结构说明 ==========
"""
LeNet-5详细说明：

1. 历史地位：
   - 1998年Yann LeCun提出
   - 最早成功应用的CNN之一
   - 用于手写数字识别（MNIST）
   - 奠定了现代CNN的基础架构

2. 原始设计 vs CIFAR-10适配：
   
   原始LeNet-5（MNIST 28x28灰度）：
   - Input: 28x28x1
   - Conv1: 1→6 (5x5) → 28x28x6
   - Pool: 28x28 → 14x14
   - Conv2: 6→16 (5x5) → 10x10x16
   - Pool: 10x10 → 5x5
   - FC1: 400→120
   - FC2: 120→84
   - FC3: 84→10
   
   CIFAR-10适配版（32x32彩色）：
   - Input: 32x32x3（彩色3通道）
   - Conv1: 3→6 (5x5) → 28x28x6（输入通道改为3）
   - 其余结构保持不变
   - 利用32x32比28x28大的优势

3. 层连接关系：
   
   严格的顺序连接：
   Input → Conv1 → ReLU → Pool1 
         → Conv2 → ReLU → Pool2 
         → Flatten 
         → FC1 → ReLU 
         → FC2 → ReLU 
         → FC3 → Output
   
   特点：
   - 无跳跃连接
   - 无并行分支
   - 无concat操作
   - 最简单的前馈结构

4. 参数量分析：
   
   Conv1: 3×6×5×5 = 450
   Conv2: 6×16×5×5 = 2,400
   FC1: 400×120 = 48,000
   FC2: 120×84 = 10,080
   FC3: 84×10 = 840
   
   总参数量 ≈ 62K（极少！）
   
   对比：
   - AlexNet: ~3M（50倍）
   - VGG16: ~15M（240倍）

5. 量化特点：
   
   优势：
   ✓ 结构极简单，易于理解和调试
   ✓ 参数量少，训练/推理速度快
   ✓ 纯顺序执行，量化实现简单
   ✓ 包含Conv和FC两种层，覆盖全面
   ✓ 适合快速验证量化算法
   
   劣势：
   ✗ 网络太浅（只有2个Conv层）
   ✗ 精度较低（CIFAR-10约75-80%）
   ✗ 特征提取能力有限
   ✗ 不适合复杂任务

6. 适用场景：
   
   推荐用于：
   - 量化算法快速验证
   - 教学和演示
   - 基准测试
   - 快速原型开发
   
   不推荐用于：
   - 追求高精度的应用
   - 复杂视觉任务
   - 生产环境部署

7. 与其他网络对比：

   | 网络 | 卷积层 | FC层 | 参数量 | 精度 | 训练时间 |
   |------|-------|------|--------|------|---------|
   | LeNet-5 | 2 | 3 | 60K | ~75% | 10min |
   | AlexNet | 5 | 2 | 3M | ~89% | 1h |
   | VGG16 | 13 | 0 | 15M | ~93% | 3h |
   | VGG19 | 16 | 0 | 20M | ~93% | 4h |

8. 记录点格式：
   
   卷积层：block_output.{layer_name}
   - block_output.conv1
   - block_output.conv2
   
   全连接层：classifier.{fc_name}.out
   - classifier.fc1.out
   - classifier.fc2.out
   
   （与AlexNet的FC层格式保持一致）

9. 注意力转移：
   
   由于网络较浅，只选2层：
   - conv2: 卷积特征的最后一层
   - fc1: FC特征的第一层
   
   这2层代表了卷积特征和全连接特征的关键位置

10. 训练建议：
    
    - Epochs: 30-50（快速收敛）
    - Learning rate: 0.01
    - Batch size: 128
    - Optimizer: SGD with momentum
    - 预期精度: 75-80%（Float32）
    - 量化后精度: 70-75%（INT8）
"""

if __name__ == "__main__":
    """
    测试LeNet-5模型
    """
    print("=" * 60)
    print("LeNet-5结构测试")
    print("=" * 60)
    
    # 创建模型
    model = LeNet5TapQuant(num_classes=10)
    print(f"\n✓ 模型创建成功")
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ 总参数量: {total_params:,} ({total_params/1e3:.2f}K)")
    print(f"✓ 可训练参数: {trainable_params:,}")
    
    # 各层参数量
    print(f"\n✓ 各层参数量：")
    print(f"  - Conv1 (3→6, 5×5): {3*6*5*5:,}")
    print(f"  - Conv2 (6→16, 5×5): {6*16*5*5:,}")
    print(f"  - FC1 (400→120): {400*120:,}")
    print(f"  - FC2 (120→84): {120*84:,}")
    print(f"  - FC3 (84→10): {84*10:,}")
    
    # 测试前向传播
    x = torch.randn(2, 3, 32, 32)
    print(f"\n✓ 输入形状: {x.shape}")
    
    # 创建recorder
    rec = ActivationRecorder()
    y = model(x, recorder=rec)
    print(f"✓ 输出形状: {y.shape}")
    print(f"✓ 输出范围: [{y.min().item():.2f}, {y.max().item():.2f}]")
    
    # 打印层名称
    layer_names = get_lenet5_layer_names()
    print(f"\n✓ 量化层数量: {len(layer_names)}")
    print(f"✓ 层名称: {layer_names}")
    
    # 打印注意力层
    attention_layers = get_lenet5_attention_layers()
    print(f"\n✓ 注意力转移层: {attention_layers}")
    
    # 打印记录的激活
    print(f"\n✓ 记录的激活数量: {len(rec.storage)}")
    for key in sorted(rec.storage.keys()):
        shape = rec.storage[key].shape
        print(f"  - {key}: {shape}")
    
    # 打印网络结构
    print("\n" + "=" * 60)
    print("完整网络结构")
    print("=" * 60)
    print(model)
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)

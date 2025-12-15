from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
from torch.ao.quantization import QuantStub, DeQuantStub, fuse_modules

from recorder import ActivationRecorder


def conv_seq(in_ch, out_ch, k=3, s=1, p=1):
    """
    卷积序列：conv + bn + relu
    用于构建AlexNet的卷积层
    """
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)),
        ("bn",   nn.BatchNorm2d(out_ch)),
        ("relu", nn.ReLU(inplace=True)),
    ]))


class AlexNetTapQuant(nn.Module):
    """
    AlexNet for CIFAR-10 (32x32 input), 可量化
    
    网络结构：
    - 5个卷积层 (conv1-conv5)
    - 3个全连接层 (fc1-fc3)
    - 带BN层
    
    关键点：
      - 融合后的路径：block_output.{name}
      - 融合前的路径：features.{name}.conv_out / bn_out / relu_out
      - 每块入口：block_input.{name}
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        # 特征提取层
        # conv1: 3->64, kernel=5, stride=2, padding=2 (匹配参考模型，stride=2降采样)
        self.conv1 = conv_seq(3, 64, k=5, s=2, p=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 16->8 (因为conv1已降采样到16x16)

        # conv2: 64->192, kernel=5, stride=1, padding=2
        self.conv2 = conv_seq(64, 192, k=5, s=1, p=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 8->4

        # conv3: 192->384, kernel=3, stride=1, padding=1
        self.conv3 = conv_seq(192, 384, k=3, s=1, p=1)

        # conv4: 384->256, kernel=3, stride=1, padding=1
        self.conv4 = conv_seq(384, 256, k=3, s=1, p=1)

        # conv5: 256->256, kernel=3, stride=1, padding=1
        self.conv5 = conv_seq(256, 256, k=3, s=1, p=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 4->2 (重命名为pool3以匹配参考模型)

        # 分类器 - 匹配参考模型，无avgpool
        # fc1: 256*2*2 -> 512
        self.fc1 = nn.Sequential(OrderedDict([
            ("linear", nn.Linear(256 * 2 * 2, 512)),
            ("relu", nn.ReLU(inplace=True)),
            ("dropout", nn.Dropout(0.5)),
        ]))
        
        # fc2: 512 -> 256
        self.fc2 = nn.Sequential(OrderedDict([
            ("linear", nn.Linear(512, 256)),
            ("relu", nn.ReLU(inplace=True)),
            ("dropout", nn.Dropout(0.5)),
        ]))
        
        # fc3: 256 -> num_classes
        self.fc3 = nn.Linear(256, num_classes)

    def fuse_model(self):
        """
        融合操作：
        1. 卷积层的 conv+bn+relu
        2. 全连接层的 linear+relu (Dropout在量化时自动跳过)
        """
        # 融合卷积层
        for name, m in self.named_modules():
            if isinstance(m, nn.Sequential) and set(m._modules.keys()) == {"conv", "bn", "relu"}:
                fuse_modules(m, ["conv", "bn", "relu"], inplace=True)
        
        # 融合全连接层的 linear+relu
        # 注意：现在FC层结构是 linear->relu->dropout，只融合linear+relu
        if "linear" in self.fc1._modules and "relu" in self.fc1._modules:
            fuse_modules(self.fc1, ["linear", "relu"], inplace=True)
        if "linear" in self.fc2._modules and "relu" in self.fc2._modules:
            fuse_modules(self.fc2, ["linear", "relu"], inplace=True)

    def _forward_convseq(
        self,
        x: torch.Tensor,
        seq: nn.Sequential,
        name: str,
        rec: Optional[ActivationRecorder],
    ) -> torch.Tensor:
        """
        处理卷积序列的前向传播，支持记录和编辑
        
        Args:
            x: 输入张量
            seq: 卷积序列模块
            name: 层名称 (如 'conv1', 'conv2', ...)
            rec: 激活记录器
        
        Returns:
            输出张量
        """
        # 记录块入口
        if rec is not None:
            rec.record(f"block_input.{name}", x)

        # 判断是否为未融合结构
        unfused = (
            isinstance(seq, nn.Sequential)
            and "bn" in seq._modules
            and not isinstance(seq._modules["bn"], nn.Identity)
        )

        if unfused:
            # 未融合：分别处理 conv, bn, relu
            conv = seq._modules["conv"]
            bn = seq._modules["bn"]
            relu = seq._modules["relu"]

            # conv
            x = conv(x)
            if rec is not None:
                rec.record(f"features.{name}.conv_out", x)
                x = rec.maybe_edit(f"features.{name}.conv_out", x)

            # bn
            x = bn(x)
            if rec is not None:
                rec.record(f"features.{name}.bn_out", x)
                x = rec.maybe_edit(f"features.{name}.bn_out", x)

            # relu
            x = relu(x)
            if rec is not None:
                rec.record(f"features.{name}.relu_out", x)
                x = rec.maybe_edit(f"features.{name}.relu_out", x)

        else:
            # 融合后：一次forward
            x = seq(x)
            if rec is not None:
                rec.record(f"block_output.{name}", x)
                x = rec.maybe_edit(f"block_output.{name}", x)

        return x

    def _forward_fc_seq(
        self,
        x: torch.Tensor,
        seq: nn.Sequential,
        name: str,
        rec: Optional[ActivationRecorder],
    ) -> torch.Tensor:
        """
        处理全连接序列的前向传播
        
        Args:
            x: 输入张量
            seq: 全连接序列模块
            name: 层名称 (如 'fc1', 'fc2')
            rec: 激活记录器
        
        Returns:
            输出张量
        """
        # 记录块入口
        if rec is not None:
            rec.record(f"block_input.classifier.{name}", x)

        # 判断是否为未融合结构
        unfused = (
            isinstance(seq, nn.Sequential)
            and "bn" in seq._modules
            and not isinstance(seq._modules["bn"], nn.Identity)
        )

        if unfused:
            # 未融合：分别处理 linear, bn, relu
            linear = seq._modules["linear"]
            bn = seq._modules["bn"]
            relu = seq._modules["relu"]

            # linear
            x = linear(x)
            if rec is not None:
                rec.record(f"classifier.{name}.linear_out", x)
                x = rec.maybe_edit(f"classifier.{name}.linear_out", x)

            # bn
            x = bn(x)
            if rec is not None:
                rec.record(f"classifier.{name}.bn_out", x)
                x = rec.maybe_edit(f"classifier.{name}.bn_out", x)

            # relu
            x = relu(x)
            if rec is not None:
                rec.record(f"classifier.{name}.relu_out", x)
                x = rec.maybe_edit(f"classifier.{name}.relu_out", x)

        else:
            # 融合后：一次forward
            x = seq(x)
            if rec is not None:
                rec.record(f"classifier.{name}.out", x)
                x = rec.maybe_edit(f"classifier.{name}.out", x)

        return x

    def forward(self, x: torch.Tensor, recorder: Optional[ActivationRecorder] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入图像 [B, 3, 32, 32]
            recorder: 激活记录器（可选）
        
        Returns:
            输出logits [B, num_classes]
        """
        x = self.quant(x)

        # 特征提取
        # conv1 + pool1
        x = self._forward_convseq(x, self.conv1, "conv1", recorder)
        x = self.pool1(x)

        # conv2 + pool2
        x = self._forward_convseq(x, self.conv2, "conv2", recorder)
        x = self.pool2(x)

        # conv3
        x = self._forward_convseq(x, self.conv3, "conv3", recorder)

        # conv4
        x = self._forward_convseq(x, self.conv4, "conv4", recorder)

        # conv5 + pool3
        x = self._forward_convseq(x, self.conv5, "conv5", recorder)
        x = self.pool3(x)

        # 匹配参考模型 - 直接flatten，无avgpool
        x = torch.flatten(x, 1)

        # 分类器
        # fc1
        x = self._forward_fc_seq(x, self.fc1, "fc1", recorder)

        # fc2
        x = self._forward_fc_seq(x, self.fc2, "fc2", recorder)

        # fc3 (最后一层，不带BN和ReLU)
        if recorder is not None:
            recorder.record("block_input.classifier.fc3", x)
        x = self.fc3(x)
        if recorder is not None:
            recorder.record("classifier.fc3.out", x)

        x = self.dequant(x)
        return x


# ========== 辅助函数：获取层名列表 ==========

def get_alexnet_layer_names():
    """
    返回AlexNet中所有需要近似探索的层名称
    用于PPO训练和评估
    注意：匹配参考模型后，fc维度改为 1024→512→256→10
    """
    # 卷积层
    conv_layers = [f"conv{i}" for i in range(1, 6)]  # conv1 到 conv5
    
    # 全连接层（不包括最后的fc3，因为它直接输出logits）
    fc_layers = ["fc1", "fc2"]
    
    return conv_layers + fc_layers


def get_alexnet_tap_points():
    """
    返回所有tap点（用于记录和编辑的关键位置）
    
    Returns:
        dict: 层名到tap点名称的映射
    """
    tap_points = {}
    
    # 卷积层的tap点
    for i in range(1, 6):
        name = f"conv{i}"
        tap_points[name] = f"block_output.{name}"
    
    # 全连接层的tap点
    for fc_name in ["fc1", "fc2"]:
        tap_points[fc_name] = f"classifier.{fc_name}.out"
    
    # 最后一层
    tap_points["fc3"] = "classifier.fc3.out"
    
    return tap_points


if __name__ == "__main__":
    # 简单测试
    model = AlexNetTapQuant(num_classes=10)
    print("AlexNet模型结构：")
    print(model)
    
    print("\n可探索的层名称：")
    print(get_alexnet_layer_names())
    
    print("\nTap点映射：")
    print(get_alexnet_tap_points())
    
    # 测试前向传播
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print(f"\n输入shape: {x.shape}")
    print(f"输出shape: {y.shape}")

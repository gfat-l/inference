from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
from torch.ao.quantization import QuantStub, DeQuantStub, fuse_modules

from recorder import ActivationRecorder


def conv_seq(in_ch, out_ch, k=3, s=1, p=1):
    # conv + bn + relu
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)),
        ("bn",   nn.BatchNorm2d(out_ch)),
        ("relu", nn.ReLU(inplace=True)),
    ]))


class VGG19TapQuant(nn.Module):
    """
    VGG19，可量化；16个卷积层（比VGG16多3层）
    
    网络结构：
    - block1: 2层 (conv1_1, conv1_2)
    - block2: 2层 (conv2_1, conv2_2)
    - block3: 4层 (conv3_1, conv3_2, conv3_3, conv3_4)
    - block4: 4层 (conv4_1, conv4_2, conv4_3, conv4_4)
    - block5: 4层 (conv5_1, conv5_2, conv5_3, conv5_4)
    
    关键点：
      - 融合后的路径：block_output.{name}
      - 融合前的路径：features.{name}.conv_out / bn_out / relu_out
      - 每块入口：block_input.{name}
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        # block1 - 64 channels
        self.conv1_1 = conv_seq(3, 64)
        self.conv1_2 = conv_seq(64, 64)
        self.pool1 = nn.MaxPool2d(2, 2)

        # block2 - 128 channels
        self.conv2_1 = conv_seq(64, 128)
        self.conv2_2 = conv_seq(128, 128)
        self.pool2 = nn.MaxPool2d(2, 2)

        # block3 - 256 channels (4 layers)
        self.conv3_1 = conv_seq(128, 256)
        self.conv3_2 = conv_seq(256, 256)
        self.conv3_3 = conv_seq(256, 256)
        self.conv3_4 = conv_seq(256, 256)  # VGG19多的第1层
        self.pool3 = nn.MaxPool2d(2, 2)

        # block4 - 512 channels (4 layers)
        self.conv4_1 = conv_seq(256, 512)
        self.conv4_2 = conv_seq(512, 512)
        self.conv4_3 = conv_seq(512, 512)
        self.conv4_4 = conv_seq(512, 512)  # VGG19多的第2层
        self.pool4 = nn.MaxPool2d(2, 2)

        # block5 - 512 channels (4 layers)
        self.conv5_1 = conv_seq(512, 512)
        self.conv5_2 = conv_seq(512, 512)
        self.conv5_3 = conv_seq(512, 512)
        self.conv5_4 = conv_seq(512, 512)  # VGG19多的第3层
        self.pool5 = nn.MaxPool2d(2, 2)

        # FC层 - 匹配参考模型，只有一个fc层
        self.fc = nn.Linear(512, num_classes)

    def fuse_model(self):
        # fuse conv+bn+relu
        for _, m in self.named_modules():
            if isinstance(m, nn.Sequential) and set(m._modules.keys()) == {"conv", "bn", "relu"}:
                fuse_modules(m, ["conv", "bn", "relu"], inplace=True)
        # 参考模型中fc层不需要融合

    def _forward_convseq(
        self,
        x: torch.Tensor,
        seq: nn.Sequential,
        name: str,
        rec: Optional[ActivationRecorder],
    ) -> torch.Tensor:
        """
        name: 'conv1_1' / 'conv2_2' ...
        融合前：记录 features.{name}.xxx_out
        融合后：记录 block_output.{name}
        """
        # 先把入口记一下
        if rec is not None:
            rec.record(f"block_input.{name}", x)

        # 判断是否还是"conv+bn+relu"这种未融合结构
        unfused = (
            isinstance(seq, nn.Sequential)
            and "bn" in seq._modules
            and not isinstance(seq._modules["bn"], nn.Identity)
        )

        if unfused:
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
            # 融合后的情况：一次 forward，名字叫 block_output.{name}
            x = seq(x)
            if rec is not None:
                rec.record(f"block_output.{name}", x)
                x = rec.maybe_edit(f"block_output.{name}", x)

        return x

    def forward(self, x: torch.Tensor, recorder: Optional[ActivationRecorder] = None) -> torch.Tensor:
        x = self.quant(x)

        # block1
        x = self._forward_convseq(x, self.conv1_1, "conv1_1", recorder)
        x = self._forward_convseq(x, self.conv1_2, "conv1_2", recorder)
        x = self.pool1(x)

        # block2
        x = self._forward_convseq(x, self.conv2_1, "conv2_1", recorder)
        x = self._forward_convseq(x, self.conv2_2, "conv2_2", recorder)
        x = self.pool2(x)

        # block3
        x = self._forward_convseq(x, self.conv3_1, "conv3_1", recorder)
        x = self._forward_convseq(x, self.conv3_2, "conv3_2", recorder)
        x = self._forward_convseq(x, self.conv3_3, "conv3_3", recorder)
        x = self._forward_convseq(x, self.conv3_4, "conv3_4", recorder)
        x = self.pool3(x)

        # block4
        x = self._forward_convseq(x, self.conv4_1, "conv4_1", recorder)
        x = self._forward_convseq(x, self.conv4_2, "conv4_2", recorder)
        x = self._forward_convseq(x, self.conv4_3, "conv4_3", recorder)
        x = self._forward_convseq(x, self.conv4_4, "conv4_4", recorder)
        x = self.pool4(x)

        # block5
        x = self._forward_convseq(x, self.conv5_1, "conv5_1", recorder)
        x = self._forward_convseq(x, self.conv5_2, "conv5_2", recorder)
        x = self._forward_convseq(x, self.conv5_3, "conv5_3", recorder)
        x = self._forward_convseq(x, self.conv5_4, "conv5_4", recorder)
        x = self.pool5(x)

        # 匹配参考模型 - 直接flatten，无avgpool
        x = x.view(x.size(0), -1)

        # fc - 只有一个fc层
        if recorder is not None:
            recorder.record("block_input.fc", x)
        x = self.fc(x)
        if recorder is not None:
            recorder.record("fc.out", x)

        x = self.dequant(x)
        return x


# ========== 辅助函数 ==========

def get_vgg19_layer_names():
    """
    返回VGG19中所有需要近似探索的层名称
    注意：匹配参考模型后，只有一个fc层（无fc1/fc2）
    由于fc是最后一层（输出logits），不进行近似探索，以免破坏预测结果。
    """
    return [
        "conv1_1", "conv1_2",
        "conv2_1", "conv2_2",
        "conv3_1", "conv3_2", "conv3_3", "conv3_4",
        "conv4_1", "conv4_2", "conv4_3", "conv4_4",
        "conv5_1", "conv5_2", "conv5_3", "conv5_4",
        # "fc", # Exclude final layer
    ]


if __name__ == "__main__":
    # 简单测试
    model = VGG19TapQuant(num_classes=10)
    print("VGG19模型结构：")
    print(model)
    
    print("\n可探索的层名称：")
    print(get_vgg19_layer_names())
    print(f"总共 {len(get_vgg19_layer_names())} 层")
    
    # 测试前向传播
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print(f"\n输入shape: {x.shape}")
    print(f"输出shape: {y.shape}")

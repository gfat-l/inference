from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
from torch.ao.quantization import QuantStub, DeQuantStub, fuse_modules

from recorder import ActivationRecorder


def conv_seq(in_ch, out_ch, k=3, s=1, p=1):
    # 和你原文件一样：conv + bn + relu
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)),
        ("bn",   nn.BatchNorm2d(out_ch)),
        ("relu", nn.ReLU(inplace=True)),
    ]))


class VGG16TapQuant(nn.Module):
    """
    VGG16，可量化；**关键点：卷积块的记录/可编辑名字要和原文件对上**：
      - 融合后的路径：block_output.{name}
      - 融合前的路径：features.{name}.conv_out / bn_out / relu_out
      - 每块入口：block_input.{name}
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        # block1
        self.conv1_1 = conv_seq(3, 64)
        self.conv1_2 = conv_seq(64, 64)
        self.pool1 = nn.MaxPool2d(2, 2)

        # block2
        self.conv2_1 = conv_seq(64, 128)
        self.conv2_2 = conv_seq(128, 128)
        self.pool2 = nn.MaxPool2d(2, 2)

        # block3
        self.conv3_1 = conv_seq(128, 256)
        self.conv3_2 = conv_seq(256, 256)
        self.conv3_3 = conv_seq(256, 256)
        self.pool3 = nn.MaxPool2d(2, 2)

        # block4
        self.conv4_1 = conv_seq(256, 512)
        self.conv4_2 = conv_seq(512, 512)
        self.conv4_3 = conv_seq(512, 512)
        self.pool4 = nn.MaxPool2d(2, 2)

        # block5
        self.conv5_1 = conv_seq(512, 512)
        self.conv5_2 = conv_seq(512, 512)
        self.conv5_3 = conv_seq(512, 512)
        self.pool5 = nn.MaxPool2d(2, 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Sequential(OrderedDict([
            ("linear", nn.Linear(512, 512)),
            ("relu", nn.ReLU(inplace=True)),
        ]))
        self.fc2 = nn.Linear(512, num_classes)

    # -------- 融合，与原文件一致 --------
    def fuse_model(self):
        # fuse conv+bn+relu
        for _, m in self.named_modules():
            if isinstance(m, nn.Sequential) and set(m._modules.keys()) == {"conv", "bn", "relu"}:
                fuse_modules(m, ["conv", "bn", "relu"], inplace=True)
        # fuse fc1
        fuse_modules(self.fc1, ["linear", "relu"], inplace=True)

    # -------- 核心修复点：记录名字要和原来一样 --------
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

        # 判断是否还是“conv+bn+relu”这种未融合结构
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
                # ！！！这一句就是你在 eval_int8_modified 里要命中的 key
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
        x = self.pool3(x)

        # block4
        x = self._forward_convseq(x, self.conv4_1, "conv4_1", recorder)
        x = self._forward_convseq(x, self.conv4_2, "conv4_2", recorder)
        x = self._forward_convseq(x, self.conv4_3, "conv4_3", recorder)
        x = self.pool4(x)

        # block5
        x = self._forward_convseq(x, self.conv5_1, "conv5_1", recorder)
        x = self._forward_convseq(x, self.conv5_2, "conv5_2", recorder)
        x = self._forward_convseq(x, self.conv5_3, "conv5_3", recorder)
        x = self.pool5(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # fc1
        if recorder is not None:
            recorder.record("block_input.classifier.fc1", x)
        x = self.fc1._modules["linear"](x)
        if recorder is not None:
            recorder.record("classifier.fc1.linear_out", x)
            x = recorder.maybe_edit("classifier.fc1.linear_out", x)
        x = self.fc1._modules["relu"](x)
        if recorder is not None:
            recorder.record("classifier.fc1.relu_out", x)
            x = recorder.maybe_edit("classifier.fc1.relu_out", x)

        x = self.fc2(x)
        if recorder is not None:
            recorder.record("classifier.fc2.out", x)

        x = self.dequant(x)
        return x

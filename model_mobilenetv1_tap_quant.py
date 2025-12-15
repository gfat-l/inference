from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
from torch.ao.quantization import QuantStub, DeQuantStub, fuse_modules

from recorder import ActivationRecorder


def conv_bn_relu(in_ch, out_ch, stride):
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)),
        ("bn", nn.BatchNorm2d(out_ch)),
        ("relu", nn.ReLU(inplace=True))
    ]))

def conv_dw(in_ch, stride):
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False)),
        ("bn", nn.BatchNorm2d(in_ch)),
        ("relu", nn.ReLU(inplace=True))
    ]))

def conv_pw(in_ch, out_ch):
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)),
        ("bn", nn.BatchNorm2d(out_ch)),
        ("relu", nn.ReLU(inplace=True))
    ]))


class MobileNetV1TapQuant(nn.Module):
    """
    MobileNetV1 for CIFAR-10.
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        # Standard Conv
        self.conv1 = conv_bn_relu(3, 32, 1) # Stride 1 for CIFAR

        # Depthwise Separable Convolutions
        self.dw1 = conv_dw(32, 1); self.pw1 = conv_pw(32, 64)
        self.dw2 = conv_dw(64, 2); self.pw2 = conv_pw(64, 128)
        self.dw3 = conv_dw(128, 1); self.pw3 = conv_pw(128, 128)
        self.dw4 = conv_dw(128, 2); self.pw4 = conv_pw(128, 256)
        self.dw5 = conv_dw(256, 1); self.pw5 = conv_pw(256, 256)
        self.dw6 = conv_dw(256, 2); self.pw6 = conv_pw(256, 512)

        # 5x 512 blocks
        self.dw7 = conv_dw(512, 1); self.pw7 = conv_pw(512, 512)
        self.dw8 = conv_dw(512, 1); self.pw8 = conv_pw(512, 512)
        self.dw9 = conv_dw(512, 1); self.pw9 = conv_pw(512, 512)
        self.dw10 = conv_dw(512, 1); self.pw10 = conv_pw(512, 512)
        self.dw11 = conv_dw(512, 1); self.pw11 = conv_pw(512, 512)

        self.dw12 = conv_dw(512, 2); self.pw12 = conv_pw(512, 1024)
        self.dw13 = conv_dw(1024, 1); self.pw13 = conv_pw(1024, 1024)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, num_classes)

    def fuse_model(self):
        for _, m in self.named_modules():
            if isinstance(m, nn.Sequential) and set(m._modules.keys()) == {"conv", "bn", "relu"}:
                fuse_modules(m, ["conv", "bn", "relu"], inplace=True)

    def _forward_convseq(
        self,
        x: torch.Tensor,
        seq: nn.Sequential,
        name: str,
        rec: Optional[ActivationRecorder],
    ) -> torch.Tensor:
        if rec is not None:
            rec.record(f"block_input.{name}", x)

        unfused = (
            isinstance(seq, nn.Sequential)
            and "bn" in seq._modules
            and not isinstance(seq._modules["bn"], nn.Identity)
        )

        if unfused and rec is not None:
            x = seq.conv(x)
            rec.record(f"features.{name}.conv_out", x)
            x = seq.bn(x)
            rec.record(f"features.{name}.bn_out", x)
            x = seq.relu(x)
            rec.record(f"features.{name}.relu_out", x)
            out = x
        else:
            out = seq(x)

        if rec is not None:
            rec.record(f"block_output.{name}", out)
            out = rec.maybe_edit(f"block_output.{name}", out)
        
        return out

    def forward(self, x, recorder: Optional[ActivationRecorder] = None):
        x = self.quant(x)

        x = self._forward_convseq(x, self.conv1, "conv1", recorder)

        x = self._forward_convseq(x, self.dw1, "dw1", recorder)
        x = self._forward_convseq(x, self.pw1, "pw1", recorder)

        x = self._forward_convseq(x, self.dw2, "dw2", recorder)
        x = self._forward_convseq(x, self.pw2, "pw2", recorder)

        x = self._forward_convseq(x, self.dw3, "dw3", recorder)
        x = self._forward_convseq(x, self.pw3, "pw3", recorder)

        x = self._forward_convseq(x, self.dw4, "dw4", recorder)
        x = self._forward_convseq(x, self.pw4, "pw4", recorder)

        x = self._forward_convseq(x, self.dw5, "dw5", recorder)
        x = self._forward_convseq(x, self.pw5, "pw5", recorder)

        x = self._forward_convseq(x, self.dw6, "dw6", recorder)
        x = self._forward_convseq(x, self.pw6, "pw6", recorder)

        x = self._forward_convseq(x, self.dw7, "dw7", recorder)
        x = self._forward_convseq(x, self.pw7, "pw7", recorder)
        x = self._forward_convseq(x, self.dw8, "dw8", recorder)
        x = self._forward_convseq(x, self.pw8, "pw8", recorder)
        x = self._forward_convseq(x, self.dw9, "dw9", recorder)
        x = self._forward_convseq(x, self.pw9, "pw9", recorder)
        x = self._forward_convseq(x, self.dw10, "dw10", recorder)
        x = self._forward_convseq(x, self.pw10, "pw10", recorder)
        x = self._forward_convseq(x, self.dw11, "dw11", recorder)
        x = self._forward_convseq(x, self.pw11, "pw11", recorder)

        x = self._forward_convseq(x, self.dw12, "dw12", recorder)
        x = self._forward_convseq(x, self.pw12, "pw12", recorder)

        x = self._forward_convseq(x, self.dw13, "dw13", recorder)
        x = self._forward_convseq(x, self.pw13, "pw13", recorder)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.dequant(x)
        return x

def get_mobilenetv1_layer_names():
    layers = ["conv1"]
    for i in range(1, 14):
        layers.append(f"dw{i}")
        layers.append(f"pw{i}")
    return layers

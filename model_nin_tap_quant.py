from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
from torch.ao.quantization import QuantStub, DeQuantStub, fuse_modules

from recorder import ActivationRecorder


def conv_seq(in_ch, out_ch, k=3, s=1, p=1):
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)),
        ("bn",   nn.BatchNorm2d(out_ch)),
        ("relu", nn.ReLU(inplace=True)),
    ]))


class NiNTapQuant(nn.Module):
    """
    Network in Network (NiN) for CIFAR-10.
    Structure:
    - Block 1: Conv5x5 -> Conv1x1 -> Conv1x1 -> MaxPool
    - Block 2: Conv5x5 -> Conv1x1 -> Conv1x1 -> MaxPool
    - Block 3: Conv3x3 -> Conv1x1 -> Conv1x1 -> AvgPool
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        # Block 1
        self.conv1 = conv_seq(3, 192, k=5, p=2)
        self.cccp1 = conv_seq(192, 160, k=1, p=0)
        self.cccp2 = conv_seq(160, 96, k=1, p=0)
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)

        # Block 2
        self.conv2 = conv_seq(96, 192, k=5, p=2)
        self.cccp3 = conv_seq(192, 192, k=1, p=0)
        self.cccp4 = conv_seq(192, 192, k=1, p=0)
        self.pool2 = nn.MaxPool2d(3, stride=2, padding=1)

        # Block 3
        self.conv3 = conv_seq(192, 192, k=3, p=1)
        self.cccp5 = conv_seq(192, 192, k=1, p=0)
        self.cccp6 = conv_seq(192, num_classes, k=1, p=0)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

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
            # Step 1: Conv
            x = seq.conv(x)
            rec.record(f"features.{name}.conv_out", x)
            # Step 2: BN
            x = seq.bn(x)
            rec.record(f"features.{name}.bn_out", x)
            # Step 3: ReLU
            x = seq.relu(x)
            rec.record(f"features.{name}.relu_out", x)
            # Final output
            out = x
        else:
            # Fused execution
            out = seq(x)

        if rec is not None:
            rec.record(f"block_output.{name}", out)
            out = rec.maybe_edit(f"block_output.{name}", out)
        
        return out

    def forward(self, x, recorder: Optional[ActivationRecorder] = None):
        x = self.quant(x)

        x = self._forward_convseq(x, self.conv1, "conv1", recorder)
        x = self._forward_convseq(x, self.cccp1, "cccp1", recorder)
        x = self._forward_convseq(x, self.cccp2, "cccp2", recorder)
        x = self.pool1(x)

        x = self._forward_convseq(x, self.conv2, "conv2", recorder)
        x = self._forward_convseq(x, self.cccp3, "cccp3", recorder)
        x = self._forward_convseq(x, self.cccp4, "cccp4", recorder)
        x = self.pool2(x)

        x = self._forward_convseq(x, self.conv3, "conv3", recorder)
        x = self._forward_convseq(x, self.cccp5, "cccp5", recorder)
        x = self._forward_convseq(x, self.cccp6, "cccp6", recorder)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dequant(x)
        return x

def get_nin_layer_names():
    return [
        "conv1", "cccp1", "cccp2",
        "conv2", "cccp3", "cccp4",
        "conv3", "cccp5", "cccp6"
    ]

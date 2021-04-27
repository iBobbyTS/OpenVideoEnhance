import math
import torch
import torch.nn as nn

from ove.utils.io import empty_cache
from ove.utils.modeling import Sequential


def conv3x3(in_planes, out_planes, dilation=1, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=(3, 3), stride=(stride, stride),
        padding=(dilation, dilation), dilation=(dilation, dilation), bias=False
    )


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, dilation=1, stride=1):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.block = Sequential(
            conv3x3(inplanes, planes, dilation, stride),
            self.relu,
            conv3x3(planes, planes)
        )

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        empty_cache()
        return out


class S2DF(nn.Module):
    def __init__(self, num_blocks):
        super().__init__()
        self.num_block = num_blocks
        self.block1 = Sequential(
            nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False),
            nn.ReLU(inplace=True)
        )
        self.block2 = BasicBlock(64, 64, dilation=4)
        self.block3 = BasicBlock(64, 64, dilation=8)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        y = [x]
        x = self.block1(x)
        y.append(x)
        x = self.block2(x)
        y.append(x)
        x = self.block3(x)
        y.append(x)
        return torch.cat(y, dim=1)


def S2DF_3dense():
    return S2DF(3)

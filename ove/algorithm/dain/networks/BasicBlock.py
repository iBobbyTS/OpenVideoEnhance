import math

import torch.nn as nn

from ove.utils.modeling import Sequential


def conv3x3():
    return nn.Conv2d(
        128, 128, kernel_size=(3, 3), stride=(1, 1),
        padding=(1, 1), dilation=(1, 1), bias=False
    )


class BasicBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.block = Sequential(
            conv3x3(),
            self.relu,
            conv3x3()
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        return out


class MultipleBasicBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_block = 4
        self.block = Sequential(
            nn.Conv2d(
                437, 128,
                kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=True
            ),
            nn.ReLU(inplace=True),
            BasicBlock(),
            BasicBlock(),
            BasicBlock(),
            nn.Conv2d(128, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.block(x)
        return x


def MultipleBasicBlock_4():
    return MultipleBasicBlock()

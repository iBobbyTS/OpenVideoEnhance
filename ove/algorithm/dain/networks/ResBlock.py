import math
import torch.nn as nn
from ove.utils.models import Sequential
from .GeneralModels import BasicBlock


class MultipleBasicBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = Sequential(
            nn.Conv2d(437, 128,
                      kernel_size=(7, 7), stride=(1, 1), padding=3, bias=True),
            nn.ReLU(inplace=True)
        )
        self.block2 = BasicBlock(128, 128, dilation=1, init_modules=True)
        self.block3 = BasicBlock(128, 128, dilation=1, init_modules=True)
        self.block4 = BasicBlock(128, 128, dilation=1, init_modules=True)
        self.block5 = Sequential(nn.Conv2d(128, 3, (3, 3), (1, 1), (1, 1)))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x

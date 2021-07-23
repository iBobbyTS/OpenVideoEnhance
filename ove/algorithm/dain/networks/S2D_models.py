import math
import torch
import torch.nn as nn
from .GeneralModels import BasicBlock


class S2DF(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(1, 1), padding=3, bias=False),
            nn.ReLU(inplace=True)
        )
        self.block2 = BasicBlock(64, 64, dilation=4, init_modules=False)
        self.block3 = BasicBlock(64, 64, dilation=8, init_modules=False)

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
        del x
        return torch.cat(y, dim=1)

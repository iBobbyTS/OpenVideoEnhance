import math
import torch.nn as nn


def conv3x3(in_planes, out_planes, dilation=1, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=(3, 3), stride=(stride, stride),
        padding=dilation, dilation=(dilation, dilation), bias=False
    )


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, dilation=1, stride=1, downsample=None, init_modules=False):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, dilation, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride
        if init_modules:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

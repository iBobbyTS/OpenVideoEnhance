import torch
from torch import nn as nn
from torch.nn import functional as F

from vrt.utils.arch import default_init_weights


def make_layer(basic_block, num_basic_block, **kwarg):
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return layers


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            num_feat, num_grow_ch, 3, 1, 1
        )
        self.conv2 = nn.Conv2d(
            num_feat + num_grow_ch, num_grow_ch, 3, 1, 1
        )
        self.conv3 = nn.Conv2d(
            num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1
        )
        self.conv4 = nn.Conv2d(
            num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1
        )
        self.conv5 = nn.Conv2d(
            num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1
        )

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        default_init_weights(
            [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5],
            0.1
        )

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(
            self,
            num_feat, num_grow_ch=32
    ):
        super().__init__()
        self.block = nn.Sequential(
            ResidualDenseBlock(num_feat, num_grow_ch),
            ResidualDenseBlock(num_feat, num_grow_ch),
            ResidualDenseBlock(num_feat, num_grow_ch)
        )

    def forward(self, x):
        return self.block(x) * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(
            self,
            num_in_ch, num_out_ch,
            num_feat=64, num_block=23, num_grow_ch=32,
            interpolate_opt=None
    ):
        super().__init__()
        self.interpolate_opt = interpolate_opt
        lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # Networks
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.block1 = nn.Sequential(
            *make_layer(
                RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch
            ),
            nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            lrelu
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            lrelu
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            lrelu,
            nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        )

    def forward(self, x):
        feat = self.conv_first(x)
        feat = feat + self.block1(feat)
        # Up sample
        feat = self.block2(F.interpolate(
            feat, scale_factor=2, **self.interpolate_opt
        ))
        feat = self.block3(F.interpolate(
            feat, scale_factor=2, **self.interpolate_opt
        ))
        out = self.block4(feat)
        return out

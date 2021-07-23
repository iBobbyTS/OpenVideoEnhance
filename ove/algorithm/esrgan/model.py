import torch
from torch import nn as nn

from ove.utils.arch import default_init_weights, make_layer
from ove.utils.modeling import Sequential


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1 = Sequential(
            nn.Conv2d(num_feat, num_grow_ch, (3, 3), (1, 1), (1, 1)),
            relu
        )
        self.conv2 = Sequential(
            nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, (3, 3), (1, 1), (1, 1)),
            relu
        )
        self.conv3 = Sequential(
            nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, (3, 3), (1, 1), (1, 1)),
            relu
        )
        self.conv4 = Sequential(
            nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, (3, 3), (1, 1), (1, 1)),
            relu
        )
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, (3, 3), (1, 1), (1, 1))
        # Initialization
        default_init_weights(
            [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5],
            0.1
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(
        self,
        num_feat, num_grow_ch=32
    ):
        super().__init__()
        self.block = Sequential(
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
        interpolate_opt=None, scale_factors=(2, 2)
    ):
        super().__init__()
        relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # Networks
        self.block1 = nn.Conv2d(num_in_ch, num_feat, (3, 3), (1, 1), (1, 1))
        self.block2 = Sequential(
            *[RRDB(num_feat=num_feat, num_grow_ch=num_grow_ch) for _ in range(num_block)],
            nn.Conv2d(num_feat, num_feat, (3, 3), (1, 1), (1, 1))
        )
        self.block3 = Sequential(
            nn.Upsample(scale_factor=scale_factors[0], **interpolate_opt),
            nn.Conv2d(num_feat, num_feat, (3, 3), (1, 1), (1, 1)),
            relu,
            nn.Upsample(scale_factor=scale_factors[1], **interpolate_opt),
            nn.Conv2d(num_feat, num_feat, (3, 3), (1, 1), (1, 1)),
            relu,
            nn.Conv2d(num_feat, num_feat, (3, 3), (1, 1), (1, 1)),
            relu,
            nn.Conv2d(num_feat, num_out_ch, (3, 3), (1, 1), (1, 1))
        )
        # nn.Upsample(
        #     scale_factor=scale_factors[1], **interpolate_opt,
        #     **({} if isinstance(scale_factors[1], int) else {'recompute_scale_factor': True})
        # )

    def forward(self, x, target, count):
        feat = self.block1(x)
        feat = feat + self.block2(feat)
        feat = self.block3(feat)
        target[[count]] = feat

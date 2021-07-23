import torch
import torch.nn as nn
from networks import SPyNet, ResidualBlocksWithInputConv, PixelShufflePack, warp
from ove.utils.modeling import Sequential


class BasicVSRNet(nn.Module):
    def __init__(self, mid_channels=64, num_blocks=30, spynet_pretrained=None):
        super().__init__()
        self.mid_channels = mid_channels
        self.__dict__['spynet'] = SPyNet(spynet_pretrained)
        self.backward_resblocks = ResidualBlocksWithInputConv(
            mid_channels + 3, mid_channels, num_blocks
        )
        self.forward_resblocks = ResidualBlocksWithInputConv(
            mid_channels + 3, mid_channels, num_blocks
        )
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.img_upsample = nn.Upsample(scale_factor=4, mode='bicubic', align_corners=False)
        self.upsample = Sequential(
            nn.Conv2d(mid_channels * 2, mid_channels, (1, 1), (1, 1), 0, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            PixelShufflePack(mid_channels, mid_channels, 2, upsample_kernel=3),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            PixelShufflePack(mid_channels, 64, 2, upsample_kernel=3),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(64, 64, (3, 3), (1, 1), 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(64, 3, (3, 3), (1, 1), 1)
        )


    def check_if_mirror_extended(self, lrs):
        self.is_mirror_extended = False
        if lrs.size(1) % 2 == 0:
            lrs_1, lrs_2 = torch.chunk(lrs, 2, dim=1)
            if torch.norm(lrs_1 - lrs_2.flip(1)) == 0:
                self.is_mirror_extended = True

    def compute_flow(self, lrs):
        b, t, c, h, w = lrs.size()
        t -= 1  # Real t
        lrs_1 = lrs[:, :-1, :, :, :].view(b * t, c, h, w)
        lrs_2 = lrs[:, 1:, :, :, :].view(b * t, c, h, w)
        flows_backward = self.spynet(lrs_1, lrs_2).view(b, t, 2, h, w)
        if self.is_mirror_extended:
            flows_forward = None
        else:
            flows_forward = self.spynet(lrs_2, lrs_1).view(b, t, 2, h, w)
        return flows_forward, flows_backward

    def forward(self, lrs):
        n, t, c, h, w = lrs.size()
        assert h >= 64 and w >= 64, f'The height and width of inputs should be at least 64, but got {h} and {w}.'
        self.check_if_mirror_extended(lrs)
        flows_forward, flows_backward = self.compute_flow(lrs)
        outputs = []
        feat_prop = lrs.new_zeros(n, self.mid_channels, h, w)
        for i in range(t - 1, -1, -1):
            if i < t - 1:
                flow = flows_backward[:, i, :, :, :]
                feat_prop = warp(feat_prop, flow.permute(0, 2, 3, 1))
            feat_prop = torch.cat([lrs[:, i, :, :, :], feat_prop], dim=1)
            feat_prop = self.backward_resblocks(feat_prop)
            outputs.append(feat_prop)
        outputs = outputs[::-1]
        feat_prop *= 0
        for i in range(0, t):
            lr_curr = lrs[:, i, :, :, :]
            if i > 0:
                if flows_forward is not None:
                    flow = flows_forward[:, i - 1, :, :, :]
                else:
                    flow = flows_backward[:, -i, :, :, :]
                feat_prop = warp(feat_prop, flow.permute(0, 2, 3, 1))
            feat_prop = torch.cat([lr_curr, feat_prop], dim=1)
            feat_prop = self.forward_resblocks(feat_prop)
            out = torch.cat([outputs[i], feat_prop], dim=1)
            out = self.upsample(out)
            base = self.img_upsample(lr_curr)
            out += base
            print(out.shape)
            outputs[i] = out
        return torch.stack(outputs, dim=1)

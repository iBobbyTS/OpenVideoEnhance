import torch
import torch.nn as nn
import torch.nn.functional as F
from ove.utils.modeling import Sequential


def conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, inplace=False):
    return Sequential(
        nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(kernel_size, kernel_size), stride=(stride, stride),
            padding=(padding, padding), dilation=(dilation, dilation), bias=True),
        nn.LeakyReLU(0.1, inplace)
    )


def warp(
        x, flo,
        use_mask=True, return_mask=False,
        interpolation='bilinear', padding_mode='zeros'):
    batch_size, channels, height, width = x.shape
    print(height, width)
    vgrid = torch.autograd.Variable(torch.stack((
        torch.arange(0, width, dtype=torch.int16, device=flo.device, requires_grad=False).view(1, width).expand(height, width),
        torch.arange(0, height, dtype=torch.int16, device=flo.device, requires_grad=False).view(height, 1).expand(height, width)
    )).view(1, 2, height, width).expand(batch_size, 2, height, width).type_as(x))
    print(vgrid.shape, flo.shape)
    vgrid += flo
    vgrid *= 2.0
    vgrid[:, 0, :, :] /= width - 1
    vgrid[:, 1, :, :] /= height - 1
    vgrid -= 1.0
    vgrid = vgrid.permute(0, 2, 3, 1)
    output = F.grid_sample(
        x, vgrid, mode=interpolation, padding_mode=padding_mode, align_corners=True
    )
    if use_mask:
        mask = torch.autograd.Variable(torch.ones(x.size(), device=x.device))
        mask = F.grid_sample(mask, vgrid, mode=interpolation, padding_mode=padding_mode, align_corners=True)
        mask = mask.masked_fill_(mask < 0.999, 0)
        mask = mask.masked_fill_(mask > 0, 1)
        if return_mask:
            return output * mask, mask
        else:
            return output * mask
    else:
        return output



class UndefinedConv(nn.Module):
    def __init__(
        self,
        width, height, batch, device,
        md, inplace, corr
    ):
        super().__init__()
        self.device = device
        self.leakyRELU = nn.LeakyReLU(0.1, inplace)
        self.conv1 = Sequential(
            conv(3, 16, kernel_size=3, stride=1, inplace=inplace),
            conv(16, 16, kernel_size=3, stride=1, inplace=inplace),
            conv(16, 16, kernel_size=3, stride=1, inplace=inplace)
        )
        self.conv2 = Sequential(
            conv(16, 32, kernel_size=3, stride=2, inplace=inplace),
            conv(32, 32, kernel_size=3, stride=1, inplace=inplace),
            conv(32, 32, kernel_size=3, stride=1, inplace=inplace)
        )
        self.conv3 = Sequential(
            conv(32, 64, kernel_size=3, stride=2, inplace=inplace),
            conv(64, 64, kernel_size=3, stride=1, inplace=inplace),
            conv(64, 64, kernel_size=3, stride=1, inplace=inplace)
        )
        self.conv4 = Sequential(
            conv(64, 96, kernel_size=3, stride=2, inplace=inplace),
            conv(96, 96, kernel_size=3, stride=1, inplace=inplace),
            conv(96, 96, kernel_size=3, stride=1, inplace=inplace)
        )
        self.conv5 = Sequential(
            conv(96, 128, kernel_size=3, stride=2, inplace=inplace),
            conv(128, 128, kernel_size=3, stride=1, inplace=inplace),
            conv(128, 128, kernel_size=3, stride=1, inplace=inplace)
        )
        self.conv6 = Sequential(
            conv(128, 196, kernel_size=3, stride=2, inplace=inplace),
            conv(196, 196, kernel_size=3, stride=1, inplace=inplace),
            conv(196, 196, kernel_size=3, stride=1, inplace=inplace)
        )
        self.corr = corr()

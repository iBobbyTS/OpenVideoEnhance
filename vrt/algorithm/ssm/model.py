import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class down(nn.Module):
    def __init__(self, inChannels, outChannels, filterSize):
        super().__init__()
        self.block = nn.Sequential(
            torch.nn.AvgPool2d(2),
            nn.Conv2d(inChannels, outChannels, filterSize, stride=1, padding=int((filterSize - 1) / 2)),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(outChannels, outChannels, filterSize, stride=1, padding=int((filterSize - 1) / 2)),
            nn.LeakyReLU(negative_slope=0.1)
        )

    def forward(self, x):
        return self.block(x)


class up(nn.Module):
    def __init__(self, inChannels, outChannels):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(inChannels, outChannels, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(2 * outChannels, outChannels, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1)
        )

    def forward(self, x, skpCn):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.block1(x)
        x = torch.cat((x, skpCn), 1)
        x = self.block2(x)
        return x


class UNet(nn.Module):
    def __init__(self, inChannels, outChannels):
        super().__init__()
        # Initialize neural network blocks.
        self.block1 = nn.Sequential(
            nn.Conv2d(inChannels, 32, 7, stride=1, padding=3),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(32, 32, 7, stride=1, padding=3),
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.down1 = down(32, 64, 5)
        self.down2 = down(64, 128, 3)
        self.down3 = down(128, 256, 3)
        self.down4 = down(256, 512, 3)
        self.down5 = down(512, 512, 3)
        self.up1 = up(512, 512)
        self.up2 = up(512, 256)
        self.up3 = up(256, 128)
        self.up4 = up(128, 64)
        self.up5 = up(64, 32)
        self.block2 = nn.Sequential(
            nn.Conv2d(32, outChannels, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1)
        )

    def forward(self, x):
        s1 = self.block1(x)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        s5 = self.down4(s4)
        x = self.down5(s5)
        x = self.up1(x, s5)
        x = self.up2(x, s4)
        x = self.up3(x, s3)
        x = self.up4(x, s2)
        x = self.up5(x, s1)
        x = self.block2(x)
        return x


class backWarp(nn.Module):
    def __init__(self, W, H, device):
        super().__init__()
        self.W = W
        self.H = H
        self.gridY, self.gridX = torch.meshgrid(
            torch.arange(H, requires_grad=False, device=device),
            torch.arange(W, requires_grad=False, device=device)
        )

    def forward(self, img, flow):
        u = flow[:, 0, :, :]
        v = flow[:, 1, :, :]
        x = self.gridX.unsqueeze(0).expand_as(u).float() + u
        y = self.gridY.unsqueeze(0).expand_as(v).float() + v
        x = 2 * (x / self.W - 0.5)
        y = 2 * (y / self.H - 0.5)
        grid = torch.stack((x, y), dim=3)
        imgOut = torch.nn.functional.grid_sample(
            img, grid, align_corners=True
        )
        return imgOut


t = np.linspace(0.125, 0.875, 7)


def getFlowCoeff(indices, device):
    ind = indices.detach().numpy()
    C11 = C00 = - (1 - (t[ind])) * (t[ind])
    C01 = (t[ind]) * (t[ind])
    C10 = (1 - (t[ind])) * (1 - (t[ind]))
    return (torch.Tensor(C00)[None, None, None, :].permute(3, 0, 1, 2).to(device),
        torch.Tensor(C01)[None, None, None,:].permute(3, 0, 1, 2).to(device),
        torch.Tensor(C10)[None, None, None, :].permute(3, 0, 1, 2).to(device),
        torch.Tensor(C11)[None, None,None, :].permute(3, 0, 1, 2).to(device))


def getWarpCoeff(indices, device):
    ind = indices.detach().numpy()
    C0 = 1 - t[ind]
    C1 = t[ind]
    return (torch.Tensor(C0)[None, None, None, :].permute(3, 0, 1, 2).to(device),
        torch.Tensor(C1)[None, None, None,:].permute(3, 0, 1, 2).to(device)
    )

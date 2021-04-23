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
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(inChannels, outChannels, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(2 * outChannels, outChannels, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1)
        )

    def forward(self, x, skpCn):
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


class SSM(nn.Module):
    def __init__(self, height, width, device, sf):
        super().__init__()
        self.sf = sf
        self.flowComp = UNet(6, 4)
        self.ArbTimeFlowIntrp = UNet(20, 5)
        self.flowBackWarp = backWarp(width, height, device)

    def forward(self, I0, I1, target, count):
        flowOut = self.flowComp(torch.cat((I0, I1), dim=1))
        F_0_1 = flowOut[:, :2, :, :]
        F_1_0 = flowOut[:, 2:, :, :]
        target[[count]] = I0
        count += 1
        for intermediateIndex in range(1, self.sf):
            t = intermediateIndex / self.sf
            temp = -t * (1 - t)
            fCoeff = [temp, t * t, (1 - t) * (1 - t), temp]
            F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
            F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0
            g_I0_F_t_0 = self.flowBackWarp(I0, F_t_0)
            g_I1_F_t_1 = self.flowBackWarp(I1, F_t_1)
            intrpOut = self.ArbTimeFlowIntrp(torch.cat((
                I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0
            ), dim=1))
            F_t_0_f = intrpOut[:, :2, :, :] + F_t_0
            F_t_1_f = intrpOut[:, 2:4, :, :] + F_t_1
            V_t_0 = torch.sigmoid(intrpOut[:, 4:5, :, :])
            V_t_1 = 1 - V_t_0
            g_I0_F_t_0_f = self.flowBackWarp(I0, F_t_0_f)
            g_I1_F_t_1_f = self.flowBackWarp(I1, F_t_1_f)
            wCoeff = [1 - t, t]
            Ft_p = (
                wCoeff[0] * V_t_0 * g_I0_F_t_0_f + wCoeff[1] * V_t_1 * g_I1_F_t_1_f
            ) / (
                wCoeff[0] * V_t_0 + wCoeff[1] * V_t_1
            )
            target[[count]] = Ft_p.detach()
            count += 1
        return count

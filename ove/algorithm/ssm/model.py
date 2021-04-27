import torch
import torch.nn as nn
import torch.nn.functional as F

from ove.utils.modeling import Sequential


class Down(nn.Module):
    def __init__(self, inChannels, outChannels, filterSize):
        super().__init__()
        filterSize -= 1
        filterSize /= 2
        padding = int(filterSize)
        padding = (padding, padding)
        self.block = Sequential(
            torch.nn.AvgPool2d(2),
            nn.Conv2d(inChannels, outChannels, filterSize, stride=(1, 1), padding=padding),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(outChannels, outChannels, filterSize, stride=(1, 1), padding=padding),
            nn.LeakyReLU(negative_slope=0.1)
        )

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    def __init__(self, inChannels, outChannels):
        super().__init__()
        self.block1 = Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(inChannels, outChannels, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.block2 = Sequential(
            nn.Conv2d(2 * outChannels, outChannels, (3, 3), stride=(1, 1), padding=(1, 1)),
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
        self.block1 = Sequential(
            nn.Conv2d(inChannels, 32, (7, 7), stride=(1, 1), padding=(3, 3)),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(32, 32, (7, 7), stride=(1, 1), padding=(3, 3)),
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.down1 = Down(32, 64, 5)
        self.down2 = Down(64, 128, 3)
        self.down3 = Down(128, 256, 3)
        self.down4 = Down(256, 512, 3)
        self.down5 = Down(512, 512, 3)
        self.up1 = Up(512, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.up5 = Up(64, 32)
        self.block2 = Sequential(
            nn.Conv2d(32, outChannels, (3, 3), stride=(1, 1), padding=(1, 1)),
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


class BackWarp(nn.Module):
    def __init__(self, W, H, device):
        super().__init__()
        self.W = W
        self.H = H
        self.gridY, self.gridX = torch.meshgrid(
            torch.arange(H, requires_grad=False, device=device, dtype=torch.float32),
            torch.arange(W, requires_grad=False, device=device, dtype=torch.float32)
        )
        self.gridY = self.gridY.unsqueeze(0)
        self.gridX = self.gridX.unsqueeze(0)

    def forward(self, img, flow):
        f = flow.clone()
        f[:, 0] += self.gridX
        f[:, 0] *= 2
        f[:, 0] /= self.W
        f[:, 0] -= 1
        f[:, 1] += self.gridY
        f[:, 1] *= 2
        f[:, 1] /= self.H
        f[:, 1] -= 1
        f = f.permute(0, 2, 3, 1)
        imgOut = F.grid_sample(
            img, f, mode='bilinear', align_corners=True
        )
        return imgOut


class SSM(nn.Module):
    def __init__(self, height, width, device, sf):
        super().__init__()
        self.sf = sf
        self.flowComp = UNet(6, 4)
        self.ArbTimeFlowIntrp = UNet(20, 5)
        self.flowBackWarp = BackWarp(width, height, device)

    def forward(self, I0, I1, target, count):
        flowOut = self.flowComp(torch.cat((I0, I1), dim=1))
        F_0_1 = flowOut[:, :2, :, :]
        F_1_0 = flowOut[:, 2:, :, :]
        target[[count]] = I0
        count += 1
        for intermediateIndex in range(1, self.sf):
            t = intermediateIndex / self.sf
            t_inv = 1.0 - t
            temp = -t * t_inv
            F_t_0 = temp * F_0_1 + t * t * F_1_0
            F_t_1 = t_inv * t_inv * F_0_1 + temp * F_1_0
            g_I0_F_t_0 = self.flowBackWarp(I0, F_t_0)
            g_I1_F_t_1 = self.flowBackWarp(I1, F_t_1)
            intrpOut = self.ArbTimeFlowIntrp(torch.cat((
                I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0
            ), dim=1))
            F_t_0 += intrpOut[:, 0:2, :, :]
            F_t_1 += intrpOut[:, 2:4, :, :]
            V_t_0 = torch.sigmoid(intrpOut[:, 4:5, :, :])
            V_t_1 = 1 - V_t_0
            g_I0_F_t_0_f = self.flowBackWarp(I0, F_t_0)
            g_I1_F_t_1_f = self.flowBackWarp(I1, F_t_1)
            Ft_p = (
                t_inv * V_t_0 * g_I0_F_t_0_f + t * V_t_1 * g_I1_F_t_1_f
            ) / (
                t_inv * V_t_0 + t * V_t_1
            )
            target[[count]] = Ft_p.detach()
            count += 1
        return count

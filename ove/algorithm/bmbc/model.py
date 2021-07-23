from math import ceil
import torch
import torch.nn as nn
import torch.nn.functional as F

from ove.utils.modeling import Sequential
from ove.utils.models import warp
from .networks import DynFilter, DFNet, BMNet


class BMBC(nn.Module):
    def __init__(self, height, width, sf, device):
        super().__init__()
        # Args
        self.device = device
        self.Hi = height
        self.H = float(height)
        self.Wi = width
        self.W = float(width)
        self.H_i = ceil(height / 32) * 32
        self.H_ = float(self.H_i)
        self.W_i = ceil(width / 32) * 32
        self.W_ = float(self.W_i)
        self.time_step = [kk / sf for kk in range(1, sf)]
        self.time_step_inverse = [1.0 - t for t in self.time_step]
        # Layers
        self.context_layer = Sequential(
            nn.Conv2d(3, 64, (7, 7), stride=(1, 1), padding=(3, 3), bias=False),
            nn.ReLU()
        )
        self.BMNet = BMNet(width, height, batch=1, device=device)
        self.DFNet = DFNet(32, 4, 16, 6)
        self.filtering = DynFilter(device=device)

    def forward(self, I0, I1, target, count):
        target[[count]] = I0
        count += 1
        F_0_1 = F.interpolate(
            self.BMNet(
                F.interpolate(
                    torch.cat((I0, I1), dim=1), (int(self.H_), int(self.W_)), mode='bilinear', align_corners=True
                ), time=0) * 2.0,
            (self.Hi, self.Wi), mode='bilinear', align_corners=True
        )
        F_0_1[:, 0, :, :] *= self.W / self.W_
        F_0_1[:, 1, :, :] *= self.H / self.H_
        F_1_0 = F.interpolate(
            self.BMNet(
                F.interpolate(
                    torch.cat((I0, I1), dim=1), (int(self.H_), int(self.W_)), mode='bilinear', align_corners=True
                ), time=1) * (-2.0),
            (self.Hi, self.Wi), mode='bilinear', align_corners=True
        )
        F_1_0[:, 0, :, :] *= self.W / self.W_
        F_1_0[:, 1, :, :] *= self.H / self.H_
        for t, t_inv in zip(self.time_step, self.time_step_inverse):
            BM = F.interpolate(
                self.BMNet(
                    F.interpolate(
                        torch.cat((I0, I1), dim=1), (int(self.H_), int(self.W_)), mode='bilinear', align_corners=True
                    ), time=t),
                (self.Hi, self.Wi), mode='bilinear', align_corners=True
            )
            BM[:, 0, :, :] *= self.W / self.W_
            BM[:, 1, :, :] *= self.H / self.H_
            CI0 = self.context_layer(I0)
            CI1 = self.context_layer(I1)
            C1 = warp(torch.cat((I0, CI0), dim=1), (-t) * F_0_1)
            C2 = warp(torch.cat((I1, CI1), dim=1), t_inv * F_0_1)
            C3 = warp(torch.cat((I0, CI0), dim=1), t * F_1_0)
            C4 = warp(torch.cat((I1, CI1), dim=1), (t - 1) * F_1_0)
            C5 = warp(torch.cat((I0, CI0), dim=1), BM * (-2 * t))
            C6 = warp(torch.cat((I1, CI1), dim=1), BM * 2 * t_inv)
            inp = torch.cat((I0, C1, C2, C3, C4, C5, C6, I1), dim=1)
            DF = F.softmax(self.DFNet(inp), dim=1)
            candidates = inp[:, 3:-3, :, :]
            R = self.filtering(candidates[:, 0::67, :, :], DF)
            G = self.filtering(candidates[:, 1::67, :, :], DF)
            B = self.filtering(candidates[:, 2::67, :, :], DF)
            target[[count], 0:1] = R
            target[[count], 1:2] = G
            target[[count], 2:3] = B
            count += 1
        return count

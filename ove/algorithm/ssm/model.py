import torch
import torch.nn as nn
from .networks import UNet, BackWarp


class SSM(nn.Module):
    def __init__(self, height, width, device, sf):
        super().__init__()
        self.time_offsets = [kk / sf for kk in range(1, sf)]
        self.flowComp = UNet(6, 4)
        self.ArbTimeFlowIntrp = UNet(20, 5)
        self.flowBackWarp = BackWarp(width, height, device)

    def forward(self, I0, I1, target, count):
        flowOut = self.flowComp(torch.cat((I0, I1), dim=1))
        F_0_1 = flowOut[:, :2, :, :]
        F_1_0 = flowOut[:, 2:, :, :]
        target[[count]] = I0
        count += 1
        for t in self.time_offsets:
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

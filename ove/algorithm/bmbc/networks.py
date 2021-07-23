import torch
import torch.nn as nn
import torch.nn.functional as F
from ove.utils.models import warp
from ove.utils.modeling import Sequential


# BilateralCostVolume
class BilateralCostVolume(nn.Module):
    def __init__(self, md=4, device=torch.device('cpu')):
        super().__init__()
        self.device = device
        self.md = md  # displacement (default = 4pixels)
        self.grid = torch.ones(1, device=device)

        self.range = (md * 2 + 1) ** 2  # (default = 9*9 = 81)
        d_u = torch.linspace(-self.md, self.md, 2 * self.md + 1, device=device).view(1, -1).repeat((2 * self.md + 1, 1)).view(
            self.range, 1)  # (25,1)
        d_v = torch.linspace(-self.md, self.md, 2 * self.md + 1, device=device).view(-1, 1).repeat((1, 2 * self.md + 1)).view(
            self.range, 1)  # (25,1)

        self.d = torch.cat((d_u, d_v), dim=1)

    def L2normalize(self, x, d=1):
        norm = x ** 2
        norm = norm.sum(dim=d, keepdim=True) + 1e-6
        norm = torch.sqrt(norm)
        return x / norm

    def Make_UniformGrid(self, Input):
        '''
        Make uniform grid
        :param Input: tensor(N,C,H,W)
        :return grid: (N,2,H,W)
        '''
        # torchHorizontal = torch.linspace(-1.0, 1.0, W).view(1, 1, 1, W).expand(N, 1, H, W)
        # torchVertical = torch.linspace(-1.0, 1.0, H).view(1, 1, H, 1).expand(N, 1, H, W)
        # grid = torch.cat([torchHorizontal, torchVertical], 1).cuda()

        B, _, H, W = Input.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, 1, 1, W).expand(self.range, 1, H, W)
        yy = torch.arange(0, H).view(1, 1, H, 1).expand(self.range, 1, H, W)

        grid = torch.cat((xx, yy), 1).float()

        if Input.is_cuda:
            grid = grid.cuda()

        return grid

    def warp(self, x, BM_d):
        vgrid = self.grid + BM_d
        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(x.size(3) - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(x.size(2) - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = F.grid_sample(x, vgrid, mode='bilinear', padding_mode='border', align_corners=False)
        mask = torch.autograd.Variable(torch.ones(x.size(), device=self.device))
        mask = F.grid_sample(mask, vgrid, align_corners=False)

        mask = mask.masked_fill_(mask < 0.999, 0)
        mask = mask.masked_fill_(mask > 0, 1)

        return output * mask

    def forward(self, feature1, feature2, BM, time=0.5):
        feature1 = self.L2normalize(feature1)
        feature2 = self.L2normalize(feature2)

        if torch.equal(self.grid, torch.ones(1, device=self.device)):
            # Initialize first uniform grid
            self.grid = torch.autograd.Variable(self.Make_UniformGrid(BM))

        if BM.size(2) != self.grid.size(2) or BM.size(3) != self.grid.size(3):
            # Update uniform grid to fit on input tensor shape
            self.grid = torch.autograd.Variable(self.Make_UniformGrid(BM))

        # Displacement volume (N,(2d+1)^2,2,H,W) d = (i,j) , i in [-md,md] & j in [-md,md]
        D_vol = self.d.view(1, self.range, 2, 1, 1).expand(BM.size(0), -1, -1, BM.size(2), BM.size(
            3))  # (N,(2d+1)^2,2,H,W) \\ ex. D_vol(0,0,:,0,0) == (-2,-2), D_vol(0,13,:,0,0) == (1,0)

        # BM_d(N,(2*d+1)^2,2,H,W) : BM + d (displacement d from [-2,-2] to[2,2])
        BM_d = BM.view(BM.size(0), 1, BM.size(1), BM.size(2), BM.size(3)).expand(-1, self.range, -1, -1, -1) + D_vol

        # Bilateral Cost Volume(list)
        BC_list = []

        for i in range(BM.size(0)):
            if time == 0:
                bw_feature = feature1[i].view((1,) + feature1[i].size()).expand(self.range, -1, -1, -1)  # (D**2,C,H,W)
                fw_feature = self.warp(feature2[i].view((1,) + feature2[i].size()).expand(self.range, -1, -1, -1),
                                       2 * (1 - time) * BM_d[i])  # (D**2,C,H,W)

            elif time > 0 and time < 1:
                bw_feature = self.warp(feature1[i].view((1,) + feature1[i].size()).expand(self.range, -1, -1, -1),
                                       2 * (-time) * BM_d[i])  # (D**2,C,H,W)
                fw_feature = self.warp(feature2[i].view((1,) + feature2[i].size()).expand(self.range, -1, -1, -1),
                                       2 * (1 - time) * BM_d[i])  # (D**2,C,H,W)

            elif time == 1:
                bw_feature = self.warp(feature1[i].view((1,) + feature1[i].size()).expand(self.range, -1, -1, -1),
                                       2 * (-time) * BM_d[i])  # (D**2,C,H,W)
                fw_feature = feature2[i].view((1,) + feature2[i].size()).expand(self.range, -1, -1, -1)  # (D**2,C,H,W)

            BC_list.append(
                torch.sum(torch.mul(fw_feature, bw_feature), dim=1).view(1, self.range, BM.size(2), BM.size(3)))

        # BC (N,(2d+1)^2,H,W)
        return torch.cat(BC_list)


# DFNet
class DynFilter(nn.Module):
    def __init__(self, kernel_size=(5, 5), padding=2, device=torch.device('cpu')):
        super().__init__()
        self.padding = padding
        self.filter_local_expand = torch.eye(
            (ksp := kernel_size[0] * kernel_size[1]),
            device=device, dtype=torch.float32
        ).reshape(
            ksp, 1, *kernel_size
        )

    def forward(self, x, filter):
        return torch.sum(torch.mul(torch.cat([
            F.conv2d(
                x[:, c:c + 1, :, :], self.filter_local_expand, padding=self.padding
            ) for c in range(x.size(1))], dim=1), filter), dim=1).unsqueeze(1)


class MakeDense(nn.Module):
    def __init__(self, intInput, intOutput):
        super().__init__()
        self.dense_layer = Sequential(
            torch.nn.Conv2d(
                in_channels=intInput, out_channels=intOutput,
                kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
            ),
            torch.nn.ReLU()
        )

    def forward(self, x):
        return torch.cat((x,  self.dense_layer(x)), dim=1)


class RDB(nn.Module):
    def __init__(self, fusion_Feat, Conv_Num, Growth_rate):
        super().__init__()
        self.layer = Sequential(
            *[MakeDense(
                fusion_Feat + i * Growth_rate, Growth_rate
            ) for i in range(Conv_Num)],
            torch.nn.Conv2d(
                in_channels=fusion_Feat + Conv_Num * Growth_rate, out_channels=fusion_Feat,
                kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)
            )
        )

    def forward(self, x):
        Feature_d_LF = self.layer(x)
        return Feature_d_LF + x


class DFNet(nn.Module):
    def __init__(self, fusion_Feat, RDB_Num, Growth_rate, Conv_Num):
        super().__init__()
        SFE = lambda intInput, intOutput: nn.Conv2d(
            in_channels=intInput, out_channels=intOutput,
            kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.fusion_Feat = fusion_Feat
        self.RDB_Num = RDB_Num
        self.Growth_rate = Growth_rate
        self.SFE_First = SFE(408, fusion_Feat)
        self.SFE_Second = SFE(fusion_Feat, fusion_Feat)
        self.GRL = Sequential(*[RDB(
            fusion_Feat=self.fusion_Feat, Conv_Num=Conv_Num, Growth_rate=self.Growth_rate
        ) for i in range(self.RDB_Num)])
        self.Global_feature_fusion = Sequential(
            nn.Conv2d(
                in_channels=self.RDB_Num * self.fusion_Feat, out_channels=self.fusion_Feat, kernel_size=(1, 1),
                stride=(1, 1), padding=(0, 0)),
            nn.Conv2d(
                in_channels=self.fusion_Feat, out_channels=self.fusion_Feat, kernel_size=(3, 3),
                stride=(1, 1), padding=(1, 1))
        )
        self.Conv_Output = nn.Conv2d(
            in_channels=self.fusion_Feat, out_channels=5 ** 2 * 6,
            kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )

    def forward(self, x):
        F_B1 = self.SFE_First(x)
        F_d = self.SFE_Second(F_B1)
        for i in range(self.RDB_Num):
            F_d = self.GRL[i](F_d)
            if i == 0:
                F_RDB_group = F_d
            else:
                F_RDB_group = torch.cat((F_RDB_group, F_d), dim=1)
        F_GF = self.Global_feature_fusion(F_RDB_group)
        F_DF = F_GF + F_B1
        return self.Conv_Output(F_DF)


# BMNet
def conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
    return Sequential(
        nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(kernel_size, kernel_size), stride=(stride, stride),
            padding=(padding, padding), dilation=(dilation, dilation), bias=True),
        nn.LeakyReLU(0.1)
    )


def predict_flow(in_channels):
    return nn.Conv2d(in_channels, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)


def deconv(in_channels, out_channels, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(
        in_channels, out_channels,
        (kernel_size, kernel_size), (stride, stride), (padding, padding), bias=True
    )


class BMNet(nn.Module):
    def __init__(self, width, height, batch, device=torch.device('cpu')):
        super().__init__()
        self.device = device
        self.conv1 = Sequential(
            conv(3, 16, kernel_size=3, stride=1),
            conv(16, 16, kernel_size=3, stride=1),
            conv(16, 16, kernel_size=3, stride=1)
        )
        self.conv2 = Sequential(
            conv(16, 32, kernel_size=3, stride=2),
            conv(32, 32, kernel_size=3, stride=1),
            conv(32, 32, kernel_size=3, stride=1)
        )
        self.conv3 = Sequential(
            conv(32, 64, kernel_size=3, stride=2),
            conv(64, 64, kernel_size=3, stride=1),
            conv(64, 64, kernel_size=3, stride=1)
        )
        self.conv4 = Sequential(
            conv(64, 96, kernel_size=3, stride=2),
            conv(96, 96, kernel_size=3, stride=1),
            conv(96, 96, kernel_size=3, stride=1)
        )
        self.conv5 = Sequential(
            conv(96, 128, kernel_size=3, stride=2),
            conv(128, 128, kernel_size=3, stride=1),
            conv(128, 128, kernel_size=3, stride=1)
        )
        self.conv6 = Sequential(
            conv(128, 196, kernel_size=3, stride=2),
            conv(196, 196, kernel_size=3, stride=1),
            conv(196, 196, kernel_size=3, stride=1)
        )

        self.leakyRELU = nn.LeakyReLU(0.1)

        dd = [128, 256, 352, 416, 448]  # numpy.cumsum([128, 128, 96, 64, 32])

        od = (2 * 6 + 1) ** 2
        self.bilateral_corr6 = BilateralCostVolume(md=6, device=device)
        self.conv6_0 = conv(od, 128, kernel_size=3, stride=1)
        self.conv6_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv6_2 = conv(od + dd[1], 96, kernel_size=3, stride=1)
        self.conv6_3 = conv(od + dd[2], 64, kernel_size=3, stride=1)
        self.conv6_4 = conv(od + dd[3], 32, kernel_size=3, stride=1)
        self.deconv6 = Sequential(
            predict_flow(od + dd[4]),
            deconv(2, 2, kernel_size=4, stride=2, padding=1)
        )
        self.upfeat6 = deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)

        od = (2 * 4 + 1) ** 2 + 128 * 2 + 4
        self.bilateral_corr5 = BilateralCostVolume(md=4, device=device)
        self.conv5_0 = conv(od, 128, kernel_size=3, stride=1)
        self.conv5_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv5_2 = conv(od + dd[1], 96, kernel_size=3, stride=1)
        self.conv5_3 = conv(od + dd[2], 64, kernel_size=3, stride=1)
        self.conv5_4 = conv(od + dd[3], 32, kernel_size=3, stride=1)
        self.deconv5 = Sequential(
            predict_flow(od + dd[4]),
            deconv(2, 2, kernel_size=4, stride=2, padding=1)
        )
        self.upfeat5 = deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)

        od = (2 * 4 + 1) ** 2 + 96 * 2 + 4
        self.bilateral_corr4 = BilateralCostVolume(md=4, device=device)
        self.conv4_0 = conv(od, 128, kernel_size=3, stride=1)
        self.conv4_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv4_2 = conv(od + dd[1], 96, kernel_size=3, stride=1)
        self.conv4_3 = conv(od + dd[2], 64, kernel_size=3, stride=1)
        self.conv4_4 = conv(od + dd[3], 32, kernel_size=3, stride=1)
        self.deconv4 = Sequential(
            predict_flow(od + dd[4]),
            deconv(2, 2, kernel_size=4, stride=2, padding=1)
        )
        self.upfeat4 = deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)

        od = (2 * 2 + 1) ** 2 + 64 * 2 + 4
        self.bilateral_corr3 = BilateralCostVolume(md=2, device=device)
        self.conv3_0 = conv(od, 128, kernel_size=3, stride=1)
        self.conv3_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv3_2 = conv(od + dd[1], 96, kernel_size=3, stride=1)
        self.conv3_3 = conv(od + dd[2], 64, kernel_size=3, stride=1)
        self.conv3_4 = conv(od + dd[3], 32, kernel_size=3, stride=1)
        self.deconv3 = Sequential(
            predict_flow(od + dd[4]),
            deconv(2, 2, kernel_size=4, stride=2, padding=1)
        )
        self.upfeat3 = deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)

        od = (2 * 2 + 1) ** 2 + 32 * 2 + 4
        self.bilateral_corr2 = BilateralCostVolume(md=2, device=device)
        self.conv2_0 = conv(od, 128, kernel_size=3, stride=1)
        self.conv2_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv2_2 = conv(od + dd[1], 96, kernel_size=3, stride=1)
        self.conv2_3 = conv(od + dd[2], 64, kernel_size=3, stride=1)
        self.conv2_4 = conv(od + dd[3], 32, kernel_size=3, stride=1)
        self.predict_flow2 = predict_flow(od + dd[4])

        self.dc_conv = Sequential(
            conv(od + dd[4], 128, kernel_size=3, stride=1, padding=1, dilation=1),
            conv(128, 128, kernel_size=3, stride=1, padding=2, dilation=2),
            conv(128, 128, kernel_size=3, stride=1, padding=4, dilation=4),
            conv(128, 96, kernel_size=3, stride=1, padding=8, dilation=8),
            conv(96, 64, kernel_size=3, stride=1, padding=16, dilation=16),
            conv(64, 32, kernel_size=3, stride=1, padding=1, dilation=1),
            predict_flow(32)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, time=0.5):
        im1 = x[:, :3, :, :]
        im2 = x[:, 3:, :, :]
        c11 = self.conv1(im1)
        c12 = self.conv2(c11)
        c13 = self.conv3(c12)
        c14 = self.conv4(c13)
        c15 = self.conv5(c14)
        c16 = self.conv6(c15)

        c21 = self.conv1(im2)
        c22 = self.conv2(c21)
        c23 = self.conv3(c22)
        c24 = self.conv4(c23)
        c25 = self.conv5(c24)
        c26 = self.conv6(c25)
        bicorr6 = self.leakyRELU(self.bilateral_corr6(c16, c26, torch.zeros(c16.size(0), 2, c16.size(2), c16.size(3), device=self.device), time))

        x = torch.cat((self.conv6_0(bicorr6), bicorr6), 1)
        x = torch.cat((self.conv6_1(x), x), 1)
        x = torch.cat((self.conv6_2(x), x), 1)
        x = torch.cat((self.conv6_3(x), x), 1)
        x = torch.cat((self.conv6_4(x), x), 1)
        up_flow6 = self.deconv6(x)
        up_feat6 = self.upfeat6(x)
        warp1_5 = warp(c15, up_flow6 * (-1.25) * time * 2)
        warp2_5 = warp(c25, up_flow6 * 1.25 * (1 - time) * 2)
        bicorr5 = self.leakyRELU(self.bilateral_corr5(c15, c25, up_flow6 * 1.25, time))
        x = torch.cat((bicorr5, warp1_5, warp2_5, up_flow6, up_feat6), 1)
        x = torch.cat((self.conv5_0(x), x), 1)
        x = torch.cat((self.conv5_1(x), x), 1)
        x = torch.cat((self.conv5_2(x), x), 1)
        x = torch.cat((self.conv5_3(x), x), 1)
        x = torch.cat((self.conv5_4(x), x), 1)
        up_flow5 = self.deconv5(x)
        up_feat5 = self.upfeat5(x)

        warp1_4 = warp(c14, up_flow5 * (-2.5) * time * 2)
        warp2_4 = warp(c24, up_flow5 * 2.5 * (1 - time) * 2)
        bicorr4 = self.leakyRELU(self.bilateral_corr4(c14, c24, up_flow5 * 2.5, time))
        x = torch.cat((bicorr4, warp1_4, warp2_4, up_flow5, up_feat5), 1)
        x = torch.cat((self.conv4_0(x), x), 1)
        x = torch.cat((self.conv4_1(x), x), 1)
        x = torch.cat((self.conv4_2(x), x), 1)
        x = torch.cat((self.conv4_3(x), x), 1)
        x = torch.cat((self.conv4_4(x), x), 1)
        up_flow4 = self.deconv4(x)
        up_feat4 = self.upfeat4(x)

        warp1_3 = warp(c13, up_flow4 * (-5.0) * time * 2)
        warp2_3 = warp(c23, up_flow4 * 5.0 * (1 - time) * 2)
        bicorr3 = self.leakyRELU(self.bilateral_corr3(c13, c23, up_flow4 * 5.0, time))
        x = torch.cat((bicorr3, warp1_3, warp2_3, up_flow4, up_feat4), 1)
        x = torch.cat((self.conv3_0(x), x), 1)
        x = torch.cat((self.conv3_1(x), x), 1)
        x = torch.cat((self.conv3_2(x), x), 1)
        x = torch.cat((self.conv3_3(x), x), 1)
        x = torch.cat((self.conv3_4(x), x), 1)
        up_flow3 = self.deconv3(x)
        up_feat3 = self.upfeat3(x)

        warp1_2 = warp(c12, up_flow3 * (-10.0) * time * 2)
        warp2_2 = warp(c22, up_flow3 * 10.0 * (1 - time) * 2)
        bicorr2 = self.leakyRELU(self.bilateral_corr2(c12, c22, up_flow3 * 10.0, time))
        x = torch.cat((bicorr2, warp1_2, warp2_2, up_flow3, up_feat3), 1)
        x = torch.cat((self.conv2_0(x), x), 1)
        x = torch.cat((self.conv2_1(x), x), 1)
        x = torch.cat((self.conv2_2(x), x), 1)
        x = torch.cat((self.conv2_3(x), x), 1)
        x = torch.cat((self.conv2_4(x), x), 1)
        flow2 = self.predict_flow2(x) + self.dc_conv(x)
        flow = F.interpolate(flow2, (im1.size(2), im1.size(3)), mode='bilinear', align_corners=True) * 20.0
        return flow

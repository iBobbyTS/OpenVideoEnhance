import os
os.environ['PYTHON_EGG_CACHE'] = 'tmp/'

import torch
import torch.nn as nn
from torch.autograd import Variable, Function
from torch.nn.modules.module import Module
import torch.nn.functional as F
import numpy as np
import ove_ext.dain.correlation as correlation

from ove.utils.modeling import Sequential


class CorrelationFunction(Function):
    @staticmethod
    def forward(
        ctx, input1, input2, pad_size=3, kernel_size=3, max_displacement=20, stride1=1, stride2=2,
        corr_multiply=1
    ):
        ctx.save_for_backward(input1, input2)

        ctx.pad_size = pad_size
        ctx.kernel_size = kernel_size
        ctx.max_displacement = max_displacement
        ctx.stride1 = stride1
        ctx.stride2 = stride2
        ctx.corr_multiply = corr_multiply

        with torch.cuda.device_of(input1):
            rbot1 = input1.new()
            rbot2 = input2.new()
            output = input1.new()
            correlation.forward(
                input1, input2, rbot1, rbot2, output,
                ctx.pad_size, ctx.kernel_size, ctx.max_displacement, ctx.stride1, ctx.stride2, ctx.corr_multiply
            )

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input1, input2 = ctx.saved_tensors

        with torch.cuda.device_of(input1):
            rbot1 = input1.new()
            rbot2 = input2.new()

            grad_input1 = input1.new()
            grad_input2 = input2.new()

            correlation.backward(
                input1, input2, rbot1, rbot2, grad_output, grad_input1, grad_input2,
                ctx.pad_size, ctx.kernel_size, ctx.max_displacement, ctx.stride1, ctx.stride2,
                ctx.corr_multiply
            )

        return grad_input1, grad_input2, None, None, None, None, None, None


class Correlation(Module):
    def __init__(self, pad_size=0, kernel_size=0, max_displacement=0, stride1=1, stride2=2, corr_multiply=1):
        super(Correlation, self).__init__()
        self.pad_size = pad_size
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride1 = stride1
        self.stride2 = stride2
        self.corr_multiply = corr_multiply

    def forward(self, input1, input2):
        return CorrelationFunction.apply(
            input1, input2, self.pad_size, self.kernel_size, self.max_displacement,
            self.stride1, self.stride2, self.corr_multiply)


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=(kernel_size, kernel_size), stride=(stride,),
                  padding=(padding,), dilation=(dilation,), bias=True),
        nn.LeakyReLU(0.1, True)
    )


def predict_flow(in_planes):
    return nn.Conv2d(in_planes, 2, kernel_size=(3, 3), stride=(1,), padding=(1,), bias=True)


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_planes, out_planes, (kernel_size, kernel_size), (stride,), (padding,), bias=True)


class PWCDCNet(nn.Module):
    def __init__(self, md, width, height, batch):
        super().__init__()
        self.device = torch.device('cuda')
        self.leakyRELU = nn.LeakyReLU(0.1, True)
        self.conv1 = Sequential(
            conv(3, 16, kernel_size=3, stride=2),
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
        self.corr = Correlation(pad_size=md, kernel_size=1, max_displacement=md, stride1=1, stride2=1, corr_multiply=1)
        nd = (2 * md + 1) ** 2
        dd = list(np.cumsum([128, 128, 96, 64, 32], dtype=np.int16))
        od = nd
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

        od = nd + 132
        self.conv5_0 = conv(od, 128, kernel_size=3, stride=1)
        self.conv5_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv5_2 = conv(od + dd[1], 96, kernel_size=3, stride=1)
        self.conv5_3 = conv(od + dd[2], 64, kernel_size=3, stride=1)
        self.conv5_4 = conv(od + dd[3], 32, kernel_size=3, stride=1)
        self.predict_flow5 = predict_flow(od + dd[4])
        self.deconv5 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat5 = deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)

        od = nd + 100
        self.conv4_0 = conv(od, 128, kernel_size=3, stride=1)
        self.conv4_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv4_2 = conv(od + dd[1], 96, kernel_size=3, stride=1)
        self.conv4_3 = conv(od + dd[2], 64, kernel_size=3, stride=1)
        self.conv4_4 = conv(od + dd[3], 32, kernel_size=3, stride=1)
        self.predict_flow4 = predict_flow(od + dd[4])
        self.deconv4 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat4 = deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)
        od = nd + 68
        self.conv3_0 = conv(od, 128, kernel_size=3, stride=1)
        self.conv3_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv3_2 = conv(od + dd[1], 96, kernel_size=3, stride=1)
        self.conv3_3 = conv(od + dd[2], 64, kernel_size=3, stride=1)
        self.conv3_4 = conv(od + dd[3], 32, kernel_size=3, stride=1)
        self.predict_flow3 = predict_flow(od + dd[4])
        self.deconv3 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat3 = deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)
        od = nd + 36
        self.conv2_0 = conv(od, 128, kernel_size=3, stride=1)
        self.conv2_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv2_2 = conv(od + dd[1], 96, kernel_size=3, stride=1)
        self.conv2_3 = conv(od + dd[2], 64, kernel_size=3, stride=1)
        self.conv2_4 = conv(od + dd[3], 32, kernel_size=3, stride=1)
        self.predict_flow2 = predict_flow(od + dd[4])
        self.deconv2 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
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
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
        W_MAX = width
        H_MAX = height
        B_MAX = batch
        self.grid = Variable(
            torch.cat((
                torch.arange(0, W_MAX).repeat(H_MAX, 1).view(1, 1, H_MAX, W_MAX).repeat(B_MAX, 1, 1, 1),
                torch.arange(0, H_MAX).repeat(1, W_MAX).view(1, 1, H_MAX, W_MAX).repeat(B_MAX, 1, 1, 1)
            ), 1), requires_grad=False
        )

    def warp(self, x, flo):
        B, C, H, W = x.size()
        vgrid = self.grid.clone()
        vgrid += flo
        vgrid *= 2.0
        vgrid[:, 0, :, :] /= max(W - 1, 1)
        vgrid[:, 1, :, :] /= max(H - 1, 1)
        vgrid -= 1.0
        vgrid = vgrid.permute(0, 2, 3, 1)
        output = F.grid_sample(x, vgrid, align_corners=True)
        mask = Variable(x.resize_(x.size()).zero_() + 1, requires_grad=False, device=self.device)
        mask = F.grid_sample(mask, vgrid, align_corners=True)
        mask = (mask >= torch.Tensor((0.9999,))).float()
        return output * mask

    def forward(self, im1, im2):
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
        corr6 = self.leakyRELU(self.corr(c16, c26))
        x = torch.cat((self.conv6_0(corr6), corr6), 1)
        x = torch.cat((self.conv6_1(x), x), 1)
        x = torch.cat((self.conv6_2(x), x), 1)
        x = torch.cat((self.conv6_3(x), x), 1)
        x = torch.cat((self.conv6_4(x), x), 1)
        up_flow6 = self.deconv6(x)
        up_feat6 = self.upfeat6(x)
        corr5 = self.leakyRELU(self.corr(c15, self.warp(c25, up_flow6 * 0.625)))
        x = torch.cat((corr5, c15, up_flow6, up_feat6), 1)
        x = torch.cat((self.conv5_0(x), x), 1)
        x = torch.cat((self.conv5_1(x), x), 1)
        x = torch.cat((self.conv5_2(x), x), 1)
        x = torch.cat((self.conv5_3(x), x), 1)
        x = torch.cat((self.conv5_4(x), x), 1)
        up_flow5 = self.deconv5(self.predict_flow5(x))
        up_feat5 = self.upfeat5(x)
        corr4 = self.leakyRELU(self.corr(c14, self.warp(c24, up_flow5 * 1.25)))
        x = torch.cat((corr4, c14, up_flow5, up_feat5), 1)
        x = torch.cat((self.conv4_0(x), x), 1)
        x = torch.cat((self.conv4_1(x), x), 1)
        x = torch.cat((self.conv4_2(x), x), 1)
        x = torch.cat((self.conv4_3(x), x), 1)
        x = torch.cat((self.conv4_4(x), x), 1)
        up_flow4 = self.deconv4(self.predict_flow4(x))
        up_feat4 = self.upfeat4(x)
        corr3 = self.leakyRELU(self.corr(c13, self.warp(c23, up_flow4 * 2.5)))
        x = torch.cat((corr3, c13, up_flow4, up_feat4), 1)
        x = torch.cat((self.conv3_0(x), x), 1)
        x = torch.cat((self.conv3_1(x), x), 1)
        x = torch.cat((self.conv3_2(x), x), 1)
        x = torch.cat((self.conv3_3(x), x), 1)
        x = torch.cat((self.conv3_4(x), x), 1)
        up_flow3 = self.deconv3(self.predict_flow3(x))
        up_feat3 = self.upfeat3(x)
        corr2 = self.leakyRELU(self.corr(c12, self.warp(c22, up_flow3 * 5.0)))
        x = torch.cat((corr2, c12, up_flow3, up_feat3), 1)
        x = torch.cat((self.conv2_0(x), x), 1)
        x = torch.cat((self.conv2_1(x), x), 1)
        x = torch.cat((self.conv2_2(x), x), 1)
        x = torch.cat((self.conv2_3(x), x), 1)
        x = torch.cat((self.conv2_4(x), x), 1)
        flow2 = self.predict_flow2(x)
        flow2 += self.dc_conv(x)
        return flow2


def pwc_dc_net(size, batch=1):
    model = PWCDCNet(4, size[0], size[1], batch)
    return model

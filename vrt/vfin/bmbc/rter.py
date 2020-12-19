from math import ceil

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from .utils import warp
from .model import DynFilter, DFNet, BMNet


class rter:
    torch.set_grad_enabled(False)

    def __init__(self, sf, height, width, model_directory='model_weights/BMBC/Official.pth', *args, **kwargs):
        self.model = dict()
        state_dict = torch.load(model_directory)
        self.model['context_layer'] = nn.Conv2d(3, 64, (7, 7), stride=(1, 1), padding=(3, 3), bias=False)
        self.model['context_layer'].load_state_dict(state_dict['context_layer'])
        self.model['BMNet'] = BMNet()
        self.model['BMNet'].load_state_dict(state_dict['BMNet_weights'])
        self.model['DFNet'] = DFNet(32, 4, 16, 6)
        self.model['DFNet'].load_state_dict(state_dict['DFNet_weights'])
        self.model['filtering'] = DynFilter()
        self.ReLU = torch.nn.ReLU()

        for param in self.model['context_layer'].parameters():
            param.requires_grad = False
        for param in self.model['BMNet'].parameters():
            param.requires_grad = False
        for param in self.model['DFNet'].parameters():
            param.requires_grad = False

        if torch.cuda.is_available():
            self.model['BMNet'].cuda()
            self.model['DF_Net'].cuda()
            self.model['context_layer'].cuda()
            self.model['filtering'].cuda()
            self.ReLU.cuda()

        divisor = 32.
        self.H = height
        self.H_ = int(ceil(self.H / divisor) * divisor)
        self._H = self.H / float(self.H_)
        self.W = width
        self.W_ = int(ceil(self.W / divisor) * divisor)
        self._W = self.W / float(self.W_)
        self.size = (self.H, self.W)
        self.size_ = (self.H_, self.W_)
        self.time_step = [_ / sf for _ in range(1, sf)]

    def init_batch(self, buffer):
        self.inited = False

    def store_ndarray_in_tensor(self, frames):  # 内部调用
        I0, I1 = map(TF.to_tensor, *frames)

        I0 = I0.unsqueeze(0).cuda()
        I1 = I1.unsqueeze(0).cuda()
        return I0, I1

    def rt(self, frames, *args, **kwargs):
        out = []
        for time_step in self.time_step:
            I0, I1 = self.store_ndarray_in_tensor(frames)
            F_0_1 = self.model['BMNet'](F.interpolate(torch.cat((I0, I1), dim=1), self.size_, mode='bilinear'), time=0) * 2.0
            F_1_0 = self.model['BMNet'](F.interpolate(torch.cat((I0, I1), dim=1), self.size_, mode='bilinear'), time=1) * (-2.0)
            BM = self.model['BMNet'](F.interpolate(torch.cat((I0, I1), dim=1), self.size_, mode='bilinear'), time=time_step)  # V_t_1

            F_0_1 = F.interpolate(F_0_1, self.size, mode='bilinear')
            F_1_0 = F.interpolate(F_1_0, self.size, mode='bilinear')
            BM = F.interpolate(BM, self.size, mode='bilinear')

            F_0_1[:, 0, :, :] *= self._W
            F_0_1[:, 1, :, :] *= self._H
            F_1_0[:, 0, :, :] *= self._W
            F_1_0[:, 1, :, :] *= self._H
            BM[:, 0, :, :] *= self._W
            BM[:, 1, :, :] *= self._H

            C1 = warp(torch.cat((I0, self.ReLU(self.model['context_layer'](I0))), dim=1), (-time_step) * F_0_1)  # F_t_0
            C2 = warp(torch.cat((I1, self.ReLU(self.model['context_layer'](I1))), dim=1), (1 - time_step) * F_0_1)  # F_t_1
            C3 = warp(torch.cat((I0, self.ReLU(self.model['context_layer'](I0))), dim=1), (time_step) * F_1_0)  # F_t_0
            C4 = warp(torch.cat((I1, self.ReLU(self.model['context_layer'](I1))), dim=1), (time_step - 1) * F_1_0)  # F_t_1
            C5 = warp(torch.cat((I0, self.ReLU(self.model['context_layer'](I0))), dim=1), BM * (-2 * time_step))
            C6 = warp(torch.cat((I1, self.ReLU(self.model['context_layer'](I1))), dim=1), BM * 2 * (1 - time_step))

            input_ = torch.cat((I0, C1, C2, C3, C4, C5, C6, I1), dim=1)
            DF = F.softmax(self.model['DF_Net'](input_), dim=1)

            candidates = input_[:, 3:-3, :, :]

            R = self.model['filtering'](candidates[:, 0::67, :, :], DF)
            G = self.model['filtering'](candidates[:, 1::67, :, :], DF)
            B = self.model['filtering'](candidates[:, 2::67, :, :], DF)

            I2 = torch.cat((R, G, B), dim=1)
            out.append(I2)
        return out

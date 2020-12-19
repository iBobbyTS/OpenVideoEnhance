import warnings
import math
import os

import numpy
import torch

from . import model


class rter:
    warnings.filterwarnings("ignore")
    torch.set_grad_enabled(False)

    def __init__(self, height: int, width: int, batch_size=1, model_directory='model_weights/SSM/Official.pth', *args, **kwargs):
        # sf
        self.sf = kwargs['coef']
        self.batch_size = batch_size

        # Check if need to expand image
        self.h_w = [int(math.ceil(height / 32) * 32 - height) if height % 32 else 0,
                    int(math.ceil(width / 32) * 32) - width if width % 32 else 0]
        self.dim = [height + self.h_w[0], width + self.h_w[1]]

        self.cuda_availability = torch.cuda.is_available()
        device = torch.device("cuda:0" if self.cuda_availability else "cpu")

        # Initialize model
        self.flowComp = model.UNet(6, 4).to(device)
        for param in self.flowComp.parameters():
            param.requires_grad = False
        self.ArbTimeFlowIntrp = model.UNet(20, 5).to(device)
        for param in self.ArbTimeFlowIntrp.parameters():
            param.requires_grad = False
        self.flowBackWarp = model.backWarp(self.dim[1], self.dim[0], device).to(device)
        dict1 = torch.load(model_directory, map_location='cpu')
        self.ArbTimeFlowIntrp.load_state_dict(dict1['state_dictAT'])
        self.flowComp.load_state_dict(dict1['state_dictFC'])
        # Initialize batch
        self.batch = torch.cuda.FloatTensor(self.batch_size + 1, 3, self.dim[0], self.dim[1]) if self.cuda_availability \
            else torch.FloatTensor(self.batch_size + 1, 3, self.dim[0], self.dim[1])

    def init_batch(self, buffer):
        self.inited = False

    def store_ndarray_in_tensor(self, frame: numpy.ndarray, index: int):  # 内部调用
        self.batch[index, :, self.h_w[0]:, self.h_w[1]:] = \
            torch.cuda.ByteTensor(frame[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255 \
            if self.cuda_availability else \
            torch.FloatTensor(numpy.transpose(frame, (2, 0, 1))[::-1].astype('float32') / 255)

    def tensor2ndarray(self, frames: torch.tensor):
        if self.cuda_availability:
            return torch.round(frames.detach() * 255).byte().clamp(0, 255)[:, :, self.h_w[0]:, self.h_w[1]:] \
                       .permute(0, 2, 3, 1).detach().cpu().numpy()[:, :, :, ::-1]
        else:
            return numpy.transpose((numpy.array(frames.detach()) * 255) \
                       .astype(numpy.uint8)[:, ::-1, self.h_w[0]:, self.h_w[1]:], (0, 2, 3, 1))

    def rt(self, frames: list, *args, **kwargs):
        if not self.inited:
            self.store_ndarray_in_tensor(frames[0], 0)
            self.inited = True
            return [frames[0]]
        for i, f in enumerate(frames, 1):
            self.store_ndarray_in_tensor(f, i)
        I0 = self.batch[:-1]
        I1 = self.batch[1:]
        flowOut = self.flowComp(torch.cat((I0, I1), dim=1))
        F_0_1 = flowOut[:, :2, :, :]
        F_1_0 = flowOut[:, 2:, :, :]
        intermediate_frames = list(range(self.sf - 1))  # Each item contains intermediate frames
        for intermediateIndex in range(1, self.sf):
            t = intermediateIndex / self.sf
            temp = -t * (1 - t)
            fCoeff = [temp, t * t, (1 - t) * (1 - t), temp]

            F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
            F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0

            g_I0_F_t_0 = self.flowBackWarp(I0, F_t_0)
            g_I1_F_t_1 = self.flowBackWarp(I1, F_t_1)

            intrpOut = self.ArbTimeFlowIntrp(
                torch.cat((I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0), dim=1))

            F_t_0_f = intrpOut[:, :2, :, :] + F_t_0
            F_t_1_f = intrpOut[:, 2:4, :, :] + F_t_1
            V_t_0 = torch.sigmoid(intrpOut[:, 4:5, :, :])
            V_t_1 = 1 - V_t_0

            g_I0_F_t_0_f = self.flowBackWarp(I0, F_t_0_f)
            g_I1_F_t_1_f = self.flowBackWarp(I1, F_t_1_f)

            wCoeff = [1 - t, t]

            Ft_p = (wCoeff[0] * V_t_0 * g_I0_F_t_0_f + wCoeff[1] * V_t_1 * g_I1_F_t_1_f) / (
                    wCoeff[0] * V_t_0 + wCoeff[1] * V_t_1)

            # Save intermediate frame
            # Ft_p contains batches of one intermediate frame
            intermediate_frames[intermediateIndex - 1] = self.tensor2ndarray(Ft_p)
        self.batch[0] = self.batch[-1]
        if kwargs['duplicate']:
            return [intermediate_frames[0][0], frames[0], frames[0]]
        else:
            return [intermediate_frames[0][0], frames[0]]

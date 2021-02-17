import os

import numpy
import torch
import cv2

from vrt import dictionaries, utils
from . import model
from vrt.utils import modeling


class RTer:
    def __init__(
            self,
            height, width,
            model_path=None, default_model_dir=None,
            *args, **kwargs
    ):
        torch.set_grad_enabled(False)
        # Save parameters
        self.sf = kwargs['sf']
        # Initialize pader
        self.pader = modeling.Pader(
            width, height, 32, extend_func='replication'
        )
        self.dim = self.pader.paded_size[::-1]
        self.pading_result = self.pader.pading_result
        # Check GPU
        self.cuda_availability = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.cuda_availability else "cpu")
        self.flowComp = model.UNet(6, 4).to(self.device)
        for param in self.flowComp.parameters():
            param.requires_grad = False
        self.ArbTimeFlowIntrp = model.UNet(20, 5).to(self.device)
        for param in self.ArbTimeFlowIntrp.parameters():
            param.requires_grad = False
        self.flowBackWarp = model.backWarp(*self.dim[::-1], self.device)  # .to(device)
        # Solve for model path
        if model_path is None:
            model_path = os.path.abspath(os.path.join(
                default_model_dir, dictionaries.model_paths['ssm']
            ))
        utils.folder.check_model(model_path)
        state_dict = torch.load(model_path, **({} if self.cuda_availability else {'map_location': self.device}))
        self.ArbTimeFlowIntrp.load_state_dict(state_dict['state_dictAT'])
        self.flowComp.load_state_dict(state_dict['state_dictFC'])
        # Initialize batch
        self.need_to_init = True

    def get_output_effect(self):
        return {
            'height': 1,
            'width': 1,
            'fps': self.sf
        }

    def ndarray2tensor(self, frame: list):
        if self.cuda_availability:
            frame = torch.cuda.ByteTensor(frame)
            frame = frame.permute(0, 3, 1, 2)
            frame = frame[:, [2, 1, 0]]
            frame = frame.float()
            frame /= 255
        else:
            frame = numpy.transpose(frame, (0, 3, 1, 2))
            frame = frame[:, ::-1]
            frame = frame.astype('float32')
            frame /= 255
            frame = torch.FloatTensor(frame)
        frame = self.pader.pad(frame)
        return frame

    def tensor2ndarray(self, tensor: list):
        if self.cuda_availability:
            tensor = torch.cuda.FloatTensor(tensor)
            tensor = tensor.clamp(0.0, 1.0)
            tensor *= 255
            tensor = tensor.byte()
            tensor = tensor[
                     :, [2, 1, 0],
                     self.pading_result[2]:(-_ if (_ := self.pading_result[3]) else None),
                     self.pading_result[0]:(-_ if (_ := self.pading_result[1]) else None)
                     ]
            tensor = tensor.permute(0, 2, 3, 1)
            tensor = tensor.detach().cpu().numpy()
        else:
            tensor = [_.numpy() for _ in tensor]
            tensor = numpy.array(tensor, dtype=numpy.float32)
            tensor = tensor.clip(0.0, 1.0)
            tensor *= 255
            tensor = tensor.astype(numpy.uint8)
            tensor = numpy.squeeze(tensor, 1)
            tensor = tensor[
                     :, ::-1,
                     self.pading_result[2]:(-_ if (_ := self.pading_result[3]) else None),
                     self.pading_result[0]:(-_ if (_ := self.pading_result[1]) else None)
                     ]
            tensor = numpy.transpose(tensor, (0, 2, 3, 1))
        tensor = list(tensor)
        return tensor

    def rt(self, frame: list, *args, **kwargs):
        if self.need_to_init:
            self.need_to_init = False
            self.tensor_1 = self.ndarray2tensor(frame)
            self.ndarray_1 = frame
            return []
        self.tensor_0, self.tensor_1 = self.tensor_1, self.ndarray2tensor(frame)
        self.ndarray_0, self.ndarray_1 = self.ndarray_1, frame
        I0 = self.tensor_0
        I1 = self.tensor_1
        flowOut = self.flowComp(torch.cat((I0, I1), dim=1))
        F_0_1 = flowOut[:, :2, :, :]
        F_1_0 = flowOut[:, 2:, :, :]
        intermediate_frames = []
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

            intermediate_frames.append(Ft_p)
        # print(intermediate_frames)
        intermediate_frames = self.tensor2ndarray(intermediate_frames)
        return_ = [self.ndarray_0[0], *intermediate_frames]
        if kwargs['duplicate']:
            return_.append(frame[0])
        return return_

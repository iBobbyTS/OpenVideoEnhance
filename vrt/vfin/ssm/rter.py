import os

import numpy
import torch

from vrt import dictionaries, utils
from . import model


class RTer:
    def __init__(
            self,
            height, width,
            model_path=None, default_model_dir=None,
            sf=2, resize_hotfix=False,
            *args, **kwargs
    ):
        torch.set_grad_enabled(False)
        # Save parameters
        self.sf = round(float(sf))
        self.resize_hotfix = resize_hotfix
        # Initialize pader
        self.pader = utils.modeling.Pader(
            width, height, 32, extend_func='replication'
        )
        self.dim = self.pader.paded_size[::-1]
        self.pading_result = self.pader.pading_result
        # Solve for model path
        model_path = utils.folder.check_model(default_model_dir, model_path, dictionaries.model_paths['ssm'])
        # Check GPU
        self.cuda_availability = torch.cuda.is_available()
        if self.cuda_availability:
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
        # self.cuda_availability = False
        self.device = torch.device("cuda:0" if self.cuda_availability else "cpu")
        # Initialize model
        self.flowComp = model.UNet(6, 4).to(self.device)
        self.ArbTimeFlowIntrp = model.UNet(20, 5).to(self.device)
        for param in [*self.flowComp.parameters(), *self.ArbTimeFlowIntrp.parameters()]:
            param.requires_grad = False
        self.flowBackWarp = model.backWarp(*self.dim[::-1], self.device)  # .to(device)
        # Load state dict
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
            frame = torch.stack([torch.from_numpy(_.copy()).cuda() for _ in frame])
            frame = frame.permute(0, 3, 1, 2)
            frame = frame[:, [2, 1, 0]]
            frame = frame.float()
            frame /= 255.0
        else:
            frame = numpy.transpose(frame, (0, 3, 1, 2))
            frame = frame[:, ::-1]
            frame = frame.astype('float32')
            frame /= 255.0
            frame = torch.FloatTensor(frame)
        frame = self.pader.pad(frame)
        return frame

    def tensor2ndarray(self, tensor: list):
        if self.cuda_availability:
            tensor = torch.stack(tensor)
            if self.resize_hotfix:
                tensor = utils.modeling.resize_hotfix(tensor)
            tensor = tensor.clamp(0.0, 1.0)
            tensor *= 255.0
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
            tensor = numpy.stack(tensor)
            tensor = tensor.clip(0.0, 1.0)
            tensor *= 255.0
            tensor = tensor.astype(numpy.uint8)
            tensor = tensor[
                :, ::-1,
                self.pading_result[2]:(-_ if (_ := self.pading_result[3]) else None),
                self.pading_result[0]:(-_ if (_ := self.pading_result[1]) else None)
            ]
            tensor = numpy.transpose(tensor, (0, 2, 3, 1))
            tensor = utils.modeling.resize_hotfix_numpy(tensor)
        return tensor

    def rt(self, frames: list, *args, **kwargs):
        numpy_frames = frames
        return_ = []
        if frames:
            frames = self.ndarray2tensor(frames)
        else:
            return frames
        for i, numpy_frame, frame in zip(range(1, len(numpy_frames)+1), numpy_frames, frames):
            frame = frame.unsqueeze(0)
            if self.need_to_init:
                self.need_to_init = False
                self.tensor_1 = frame
                self.ndarray_1 = numpy_frame
                if len(frames) > 1:
                    continue
                else:
                    return []
            self.tensor_0, self.tensor_1 = self.tensor_1, frame
            self.ndarray_0, self.ndarray_1 = self.ndarray_1, numpy_frame
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

                intermediate_frames.append(Ft_p[0])
            intermediate_frames = self.tensor2ndarray(intermediate_frames)
            return_.extend([self.ndarray_0, *intermediate_frames])
            if kwargs['duplicate'] and i == len(frames):
                return_.extend([numpy_frame]*self.sf)
        return return_

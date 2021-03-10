import numpy
import torch

from vrt import utils
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
        self.dim = self.pader.padded_size[::-1]
        self.pading_result = self.pader.padding_result
        # Solve for model path
        model_path = utils.folder.check_model(default_model_dir, model_path, utils.dictionaries.model_paths['ssm'])
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

    def ndarray2tensor(self, frame):
        # if isinstance(frame, (list, tuple)):
        #     frame = utils.tensor.stack(frame)
        frame.convert(
            place='torch', dtype='float32',
            shape_order='fchw', channel_order='rgb', range_=(0.0, 1.0)
        )
        frame.tensor = self.pader.pad(frame.tensor)
        frame.unsqueeze(1)
        return frame

    def tensor2ndarray(self, tensor: utils.tensor.Tensor):
        tensor.tensor = tensor.tensor[
            :, :,
            self.pading_result[2]:(-_ if (_ := self.pading_result[3]) else None),
            self.pading_result[0]:(-_ if (_ := self.pading_result[1]) else None)
        ]
        return tensor

    def rt(self, frames: utils.tensor.Tensor, last, *args, **kwargs):
        if not frames:
            return frames
        frames = self.ndarray2tensor(frames)
        returning_tensor = utils.tensor.Tensor(
            tensor=torch.empty(
                (len(frames) * self.sf - (1 if self.need_to_init else 0) * self.sf + (self.sf if last else 0), 3, *self.dim),
                dtype=torch.float32,
                device=self.device
            ),
            shape_order='fchw', channel_order='rgb',
            range_=(0.0, 1.0), clamp=False
        )
        count = 0
        for i, frame in enumerate(frames, 1):
            if self.need_to_init:
                self.need_to_init = False
                self.tensor_1 = frame
                if len(frames) > 1:
                    continue
                else:
                    return []
            self.tensor_0, self.tensor_1 = self.tensor_1, frame
            I0 = self.tensor_0.tensor
            I1 = self.tensor_1.tensor
            flowOut = self.flowComp(torch.cat((I0, I1), dim=1))
            F_0_1 = flowOut[:, :2, :, :]
            F_1_0 = flowOut[:, 2:, :, :]
            intermediate_frames = []
            returning_tensor[count:count + 1] = self.tensor_0.tensor
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
                intermediate_frames.append(Ft_p[0])
                returning_tensor[count:count + 1] = Ft_p.detach()
                count += 1
            if last and i == len(frames):
                for _ in range(self.sf):
                    returning_tensor[count:count + 1] = frame.tensor
                    count += 1
        returning_tensor = self.tensor2ndarray(returning_tensor)
        return returning_tensor

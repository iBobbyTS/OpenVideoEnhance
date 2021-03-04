import os

import numpy
import torch

from vrt import dictionaries, utils
from .rrdbnet import RRDBNet as ESRGAN


class RTer:
    def __init__(
            self,
            model_path=None, default_model_dir=None,
            model_name='r',
            mode='bilinear', align_corners=False,
            *args, **kwargs
    ):
        torch.set_grad_enabled(False)
        # Make sure model exists
        assert model_name in dictionaries.model_paths['esrgan'].keys(), \
            f"Choose nets between {list(dictionaries.model_paths['esrgan'].keys())}"
        # Solve for model path
        model_path = utils.folder.check_model(default_model_dir, model_path, dictionaries.model_paths['esrgan'][model_name])
        # Check GPU
        self.cuda_availability = torch.cuda.is_available()
        if self.cuda_availability:
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
        # self.cuda_availability = False
        # Initialize model
        interpolate_opt = {
            'mode': mode,
            **({} if mode in ('nearest', 'area') else {'align_corners': align_corners})
        }
        self.model = ESRGAN(
            num_in_ch=3, num_out_ch=3,
            interpolate_opt=interpolate_opt
        )
        self.model.load_state_dict((torch.load(model_path)['params']), strict=True)
        self.model.eval()
        if self.cuda_availability:
            self.model = self.model.cuda()
        # Load state dict

    @staticmethod
    def get_output_effect():
        return {
            'height': 4,
            'width': 4,
            'fps': 1
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
        return frame

    def tensor2ndarray(self, tensor: list):
        if self.cuda_availability:
            tensor = torch.stack(tensor)
            tensor = tensor.clamp(0.0, 1.0)
            tensor *= 255.0
            tensor = tensor.byte()
            tensor = tensor[:, [2, 1, 0]]
            tensor = tensor.permute(0, 2, 3, 1)
            tensor = tensor.detach().cpu().numpy()
        else:
            tensor = [_.detach().numpy() for _ in tensor]
            tensor = numpy.stack(tensor)
            tensor = tensor.clip(0.0, 1.0)
            tensor *= 255.0
            tensor = tensor.astype(numpy.uint8)
            tensor = tensor[:, [2, 1, 0]]
            tensor = numpy.transpose(tensor, (0, 2, 3, 1))
        return list(tensor)

    def rt(self, frames: list, *args, **kwargs):
        if not frames:
            return frames
        lr_frames = self.ndarray2tensor(frames)
        sr_frames = []
        for lr_frame in lr_frames:
            lr_frame = self.model(lr_frame.unsqueeze(0))
            sr_frames.append(lr_frame[0])
        sr_frames = self.tensor2ndarray(sr_frames)
        return sr_frames

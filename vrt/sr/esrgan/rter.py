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
        if model_path is None:
            model_path = os.path.abspath(os.path.join(
                default_model_dir, dictionaries.model_paths['esrgan'][model_name]
            ))
        utils.folder.check_model(model_path)
        # Check GPU
        self.cuda_availability = torch.cuda.is_available()
        # Initialize model
        interpolate_opt = {
            'mode': mode,
            **({} if mode in ('nearest', 'area') else {'align_corners': align_corners})
        }
        self.model = ESRGAN(
            num_in_ch=3, num_out_ch=3,
            interpolate_opt=interpolate_opt
        )
        if self.cuda_availability:
            self.model = self.model.cuda()
        # Load state dict
        self.model.load_state_dict((torch.load(model_path)['params']), strict=True)
        self.model.eval()

    @staticmethod
    def get_output_effect():
        return {
            'height': 4,
            'width': 4,
            'fps': 1
        }

    def ndarray2tensor(self, frame: list):
        if self.cuda_availability:
            frame = torch.from_numpy(frame[0]).cuda()
            frame = frame.permute(2, 0, 1)
            frame = frame[[2, 1, 0]]
            frame = frame.unsqueeze(0)
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
        return tensor

    def rt(self, frames: list, *args, **kwargs):
        if not frames:
            return frames
        lr_frames = self.ndarray2tensor(frames)
        sr_frames = []
        for lr_frame in lr_frames:
            lr_frame = self.model(lr_frame.unsqueeze(0))
            lr_frame = self.tensor2ndarray(lr_frame)
            sr_frames.append(lr_frame[0])
        return sr_frames

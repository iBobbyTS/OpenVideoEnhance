import numpy
import torch

from vrt import utils
from .rrdbnet import RRDBNet as ESRGAN


class RTer:
    def __init__(
            self,
            height, width,
            model_path=None, default_model_dir=None,
            model_name='r',
            mode='bilinear', align_corners=False,
            *args, **kwargs
    ):
        torch.set_grad_enabled(False)
        # Make sure model exists
        assert model_name in utils.dictionaries.model_paths['esrgan'].keys(), \
            f"Choose nets between {list(utils.dictionaries.model_paths['esrgan'].keys())}"
        # Store arg
        self.height = height * 4
        self.width = width * 4
        # Solve for model path
        model_path = utils.folder.check_model(default_model_dir, model_path, utils.dictionaries.model_paths['esrgan'][model_name])
        # Check GPU
        self.cuda_availability = torch.cuda.is_available()
        if self.cuda_availability:
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
        self.device = torch.device('cuda' if self.cuda_availability else 'cpu')
        # Initialize model
        interpolate_opt = {
            'mode': mode,
            **({} if mode in ('nearest', 'area') else {'align_corners': align_corners})
        }
        self.model = ESRGAN(
            num_in_ch=3, num_out_ch=3,
            interpolate_opt=interpolate_opt
        )
        # Load state dict
        self.model.load_state_dict((torch.load(model_path)['params']), strict=True)
        self.model.eval()
        self.model.to(self.device)

    @staticmethod
    def get_output_effect():
        return {
            'height': 4,
            'width': 4,
            'fps': 1
        }

    @staticmethod
    def ndarray2tensor(frame: list):
        if isinstance(frame, (list, tuple)):
            frame = utils.tensor.stack(frame)
        frame.convert(
            place='torch', dtype='float32',
            shape_order='fchw', channel_order='rgb', range_=(0.0, 1.0)
        )
        return frame

    @staticmethod
    def tensor2ndarray(tensor: utils.tensor.Tensor):
        tensor.detach()
        return tensor

    def rt(self, frames: list, *args, **kwargs):
        if not frames:
            return frames
        returning_tensor = utils.tensor.Tensor(
            tensor=torch.empty(
                len(frames), 3, self.height, self.width,
                dtype=torch.float32,
                device=self.device
            ),
            shape_order='fchw', channel_order='rgb',
            range_=(0.0, 1.0), clamp=False
        )
        frames = self.ndarray2tensor(frames)
        for i in range(len(frames)):
            returning_tensor[i:i+1] = self.model(frames[i:i+1].tensor)
        returning_tensor = self.tensor2ndarray(returning_tensor)
        return returning_tensor

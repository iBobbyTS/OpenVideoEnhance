from math import ceil
import torch

from ove import utils
from .model import RRDBNet


class RTer:
    def __init__(
        self,
        height, width, device=torch.device('cpu'),
        model_path=None, default_model_dir=None,
        model_name='r',
        mode='bilinear', align_corners=False,
        scale_factors=(2, 2),
        *args, **kwargs
    ):
        # Make sure model exists
        assert model_name in utils.dictionaries.model_paths['esrgan'].keys(), \
            f"Choose nets between {list(utils.dictionaries.model_paths['esrgan'].keys())}"
        # Store arg
        self.ori_height = height
        self.ori_width = width
        self.height = int(int(height * scale_factors[0]) * scale_factors[1])
        self.width = int(int(width * scale_factors[0]) * scale_factors[1])
        # Solve for model path
        model_path = utils.folder.check_model(default_model_dir, model_path, utils.dictionaries.model_paths['esrgan'][model_name])
        # Check GPU
        self.device = device
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        # Initialize model
        interpolate_opt = {
            'mode': mode,
            **({} if mode in ('nearest', 'area') else {'align_corners': align_corners})
        }
        self.model = RRDBNet(
            num_in_ch=3, num_out_ch=3,
            interpolate_opt=interpolate_opt, scale_factors=scale_factors
        )
        self.model.to(self.dtype).to(self.device)
        # Load state dict
        self.model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
        self.model.eval()

    def get_output_effect(self):
        return {
            'height': self.height / self.ori_height,
            'width': self.width / self.ori_width,
            'fps': 1
        }

    def encode(self, frame):
        frame.convert(
            place='torch', dtype=str(self.dtype).split('.')[1],
            shape_order='fchw', channel_order='rgb', range_=(0.0, 1.0)
        )
        return frame

    @staticmethod
    def decode(tensor: utils.tensor.Tensor):
        tensor.detach()
        return tensor

    def rt(self, frames: list, *args, **kwargs):
        if not frames:
            return False
        returning_tensor = utils.tensor.Tensor(
            tensor=torch.empty(
                len(frames), 3, self.height, self.width,
                dtype=self.dtype,
                device=self.device
            ),
            shape_order='fchw', channel_order='rgb',
            range_=(0.0, 1.0), clamp=False
        )
        frames = self.encode(frames)
        for i in range(len(frames)):
            self.model(frames[[i]].tensor, returning_tensor, i)
        returning_tensor = self.decode(returning_tensor)
        return returning_tensor

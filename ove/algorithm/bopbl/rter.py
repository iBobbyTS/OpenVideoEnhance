import torch
import numpy

from ove import utils
from . import model


class RTer:
    def __init__(
            self,
            height, width,
            model_path=None, default_model_dir=None,
            with_scratch=False,
            *args, **kwargs
    ):
        torch.set_grad_enabled(False)
        # Make sure model exists
        self.height = height
        self.width = width
        # Check GPU
        self.cuda_availability = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.cuda_availability else 'cpu')
        # Initialize model
        self.overall_restore = model.OverallRestore(
            model_path=model_path, default_model_dir=default_model_dir, device=self.device,
            with_scratch=with_scratch,
            height=height, width=width
        )

    @staticmethod
    def get_output_effect():
        return {
            'height': 1,
            'width': 1,
            'fps': 1
        }

    @staticmethod
    def encode(frame: list):
        if isinstance(frame, (list, tuple)):
            frame = utils.tensor.stack(frame)
        frame.convert(
            place='torch', dtype='float32',
            shape_order='fchw', channel_order='rgb', range_=(-1.0, 1.0)
        )
        return frame

    @staticmethod
    def decode(tensor: utils.tensor.Tensor):
        tensor.detach()
        return tensor

    def rt(self, frames: list, *args, **kwargs):
        if not frames:
            return frames
        returning_tensor = utils.tensor.Tensor(
            tensor=numpy.empty(
                (len(frames), self.height, self.width, 3),
                dtype=numpy.float32
            ),
            shape_order='fhwc', channel_order='rgb',
            range_=(0.0, 255.0), clamp=False
        )
        frames = self.encode(frames)
        for i in range(len(frames)):
            returning_tensor[i] = self.overall_restore.rt(frames[i:i+1].tensor)
        returning_tensor = self.decode(returning_tensor)
        return returning_tensor

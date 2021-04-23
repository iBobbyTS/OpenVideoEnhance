import copy
import numpy
import torch
from PIL import Image

from ove import utils
from .visualize import MyNet
from . import device


class RTer:
    def __init__(
            self,
            height, width,
            temp_path,
            model_path=None, default_model_dir=None,
            model_name='v', render_factor=35,
            *args, **kwargs
    ):
        self.height = height
        self.width = width
        self.render_size = render_factor * 16
        # Make sure model exists
        assert model_name in utils.dictionaries.model_paths['deoldify'].keys(), \
            f"Choose nets between {list(utils.dictionaries.model_paths['deoldify'].keys())}"
        # Solve for model path
        model_path = utils.folder.check_model(
            default_model_dir, model_path, utils.dictionaries.model_paths['deoldify'][model_name]
        )
        # Check GPU
        self.cuda_availability = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.cuda_availability else 'cpu')
        device.set(self.cuda_availability)
        # Initialize model_name
        self.model = MyNet(
            state_dict=torch.load(model_path, map_location=self.device),
            model=model_name,
            temp_path=temp_path
        )

    @staticmethod
    def get_output_effect():
        return {
            'height': 1,
            'width': 1,
            'fps': 1
        }

    def encode(self, frame: list):
        if isinstance(frame, (list, tuple)):
            frame = utils.tensor.stack(frame)
        frame.convert(
            place='torch', dtype='float32',
            shape_order='fchw', channel_order='rgb', range_=(0.0, 1.0)
        )
        frame.update(torch.stack([
            torch.mean(frame.tensor, frame.shape_order['c'])
        ]*3, frame.shape_order['c']))
        resized_frame = numpy.empty(
            (len(frame), self.render_size, self.render_size, 3),
            dtype=numpy.uint8
        )
        for i, f in enumerate(copy.deepcopy(frame).convert(
            place='numpy', dtype='uint8',
            shape_order='fhwc', channel_order='rgb', range_=(0.0, 255.0)
        )):
            resized_frame[i] = numpy.asarray(Image.fromarray(f).resize(
                (self.render_size, self.render_size), resample=Image.BICUBIC
            ))
        resized_frame = torch.from_numpy(resized_frame).to(self.device).permute(0, 3, 1, 2).float()/255.0
        return len(frame), (frame, resized_frame)

    @staticmethod
    def decode(tensor: utils.tensor.Tensor):
        tensor.detach()
        return tensor

    def rt(self, frames: list, *args, **kwargs):
        if not frames:
            return frames
        returning_tensor = utils.tensor.Tensor(
            tensor=(torch.empty(
                len(frames), 3, self.height, self.width,
                dtype=torch.float32,
                device=self.device
            ) if self.cuda_availability else numpy.empty(
                (len(frames), 3, self.height, self.width),
                dtype=numpy.float32
            )),
            shape_order='fchw', channel_order='rgb',
            range_=(0.0, 1.0), clamp=False
        )
        frame_count, frames = self.encode(frames)
        for i, frame, resized in zip(range(frame_count), *frames):
            self.model(frame.tensor, resized, returning_tensor, i)
        returning_tensor = self.decode(returning_tensor)
        return returning_tensor

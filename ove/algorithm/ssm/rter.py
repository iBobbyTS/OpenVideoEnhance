import torch

from ove import utils
from .model import SSM


class RTer:
    def __init__(
            self,
            height, width,
            model_path=None, default_model_dir=None,
            sf=2, resize_hotfix=False, device=torch.device('cpu'),
            *args, **kwargs
    ):
        # Save parameters
        self.sf = sf
        self.resize_hotfix = resize_hotfix
        self.device = device
        # Initialize pader
        self.pader = utils.modeling.Pader(
            width, height, 32, extend_func='replication'
        ).to(self.device)
        self.padded_width, self.padded_height = self.pader.padded_size
        self.padding_result = self.pader.padding_result
        # Solve for model path
        model_path = utils.folder.check_model(default_model_dir, model_path, utils.dictionaries.model_paths['ssm'])
        # Initialize model
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = SSM(self.padded_height, self.padded_width, self.device, self.sf)
        self.model.to(self.dtype).to(self.device)
        # Load state dict
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        # Initialize batch
        self.need_to_init = True

    def get_output_effect(self):
        return {
            'height': 1,
            'width': 1,
            'fps': self.sf
        }

    def encode(self, frame):
        frame.convert(
            place='torch', dtype=str(self.dtype).split('.')[1],
            shape_order='fchw', channel_order='rgb', range_=(0.0, 1.0)
        )
        frame.tensor = self.pader(frame.tensor)
        frame.unsqueeze(1)
        return frame

    def decode(self, tensor: utils.tensor.Tensor):
        tensor.tensor = tensor.tensor[
            :, :,
            self.padding_result[2]:(-_ if (_ := self.padding_result[3]) else None),
            self.padding_result[0]:(-_ if (_ := self.padding_result[1]) else None)
        ]
        return tensor

    def rt(self, frames: utils.tensor.Tensor, last, *args, **kwargs):
        if not frames:
            return False
        frames = self.encode(frames)
        returning_tensor = utils.tensor.Tensor(
            tensor=torch.empty(
                (len(frames) * self.sf - (1 if self.need_to_init else 0) * self.sf + (self.sf if last else 0),
                 3, self.padded_height, self.padded_width),
                dtype=self.dtype,
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
                    return False
            self.tensor_0, self.tensor_1 = self.tensor_1, frame
            I0 = self.tensor_0.tensor
            I1 = self.tensor_1.tensor
            count = self.model(I0, I1, returning_tensor, count)
            if last and i == len(frames):
                for _ in range(1, self.sf + 1):
                    returning_tensor[[-_]] = frame.tensor
        returning_tensor = self.decode(returning_tensor)
        return returning_tensor

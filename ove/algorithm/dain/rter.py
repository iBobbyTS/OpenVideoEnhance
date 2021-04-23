import torch

from ove import utils
from . import models


class RTer:
    def __init__(
            self,
            height, width,
            model_path=None, default_model_dir=None,
            sf=2, resize_hotfix=False,
            net_name='DAIN_slowmotion', rectify=False, animation=False,
            *args, **kwargs
    ):
        # Save parameters
        self.sf = sf
        self.resize_hotfix = resize_hotfix
        self.network = net_name
        self.dim = (height, width)
        # Initialize pader
        self.pader = utils.modeling.Pader(
            width, height, 128, extend_func='replication'
        )
        # Solve for model path
        model_path = utils.folder.check_model(
            default_model_dir, model_path, utils.dictionaries.model_paths['dain']
        )
        # Initialize model
        self.model = models.__dict__[self.network](
            padding=self.pader.slice,
            channel=3, filter_size=4,
            timestep=1/self.sf,
            rectify=rectify,
            useAnimationMethod=animation,
        ).cuda()
        self.device = torch.device('cuda')
        # Load state dict
        model_dict = self.model.state_dict()
        pretrained_dict = {k: v for k, v in torch.load(model_path).items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)
        self.model.eval()
        # Initialize batch
        self.need_to_init = True

    def get_output_effect(self):
        return {
            'height': 1,
            'width': 1,
            'fps': self.sf
        }

    def encode(self, frame: utils.tensor.Tensor):
        frame.convert(
            place='torch', dtype='float32',
            shape_order='fchw', channel_order='rgb', range_=(0.0, 1.0)
        )
        frame.tensor = self.pader.pad(frame.tensor)
        frame.unsqueeze(1)
        return frame

    def decode(self, tensor: utils.tensor.Tensor):
        if self.resize_hotfix:
            tensor.tensor = utils.modeling.resize_hotfix(tensor.tensor)
        return tensor

    def rt(self, frames, last, *args, **kwargs):
        if not frames:
            return False
        frames = self.encode(frames)
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
                    return False
            self.tensor_0, self.tensor_1 = self.tensor_1, frame
            I0 = self.tensor_0.tensor
            I1 = self.tensor_1.tensor
            count = self.model(I0, I1, returning_tensor, count)
            if last and i == len(frames):
                for _ in range(1, self.sf + 1):
                    returning_tensor[[-_]] = frame.tensor[
                            :, :, self.pader.slice[2]: self.pader.slice[3], self.pader.slice[0]: self.pader.slice[1]
                        ]
        returning_tensor = self.decode(returning_tensor)
        return returning_tensor

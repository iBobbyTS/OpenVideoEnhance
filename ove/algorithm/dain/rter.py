import torch
import cv2

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
        self.width = width
        self.height = height
        # Initialize pader
        self.dtype = torch.float32
        self.pader = utils.modeling.Pader(
            width, height, 128, extend_func='replication'
        )
        # Solve for model path
        base_model_path = utils.folder.check_model(
            default_model_dir, model_path, utils.dictionaries.model_paths['dain'].replace('.pth', '-base.pth')
        )
        rectify_model_path = utils.folder.check_model(
            default_model_dir, model_path, utils.dictionaries.model_paths['dain'].replace('.pth', '-rectify.pth')
        ) if rectify else None
        # Initialize model
        self.model = models.__dict__[self.network](
            size=(width, height),
            padding=self.pader.slice,
            batch_size=1,
            sf=self.sf,
            rectify=rectify,
            useAnimationMethod=animation,
        ).cuda()
        self.device = torch.device('cuda')
        # Load state dict
        self.model.load_state_dict(dict(
            **torch.load(base_model_path),
            **(torch.load(rectify_model_path) if rectify else {})
        ))
        self.model.eval()
        self.model.to(self.dtype)
        # Initialize batch
        self.need_to_init = True
        self.count = 0

    def get_output_effect(self):
        return {
            'height': 1,
            'width': 1,
            'fps': self.sf
        }

    def encode(self, frame: utils.tensor.Tensor):
        frame.convert(
            place='torch', dtype=str(self.dtype).split('.')[1],
            shape_order='fchw', channel_order='rgb', range_=(0.0, 1.0)
        )
        frame.tensor = self.pader(frame.tensor)
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
                (len(frames) * self.sf - (1 if self.need_to_init else 0) * self.sf + (self.sf if last else 0),
                 3, self.height, self.width),
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
            self.count += 1
            cv2.imwrite(f'/content/img/{self.count}-I0.jpg', (I0[0]*255.0).permute(1, 2, 0).round().byte().cpu().numpy()[:, :, ::-1])
            cv2.imwrite(f'/content/img/{self.count}-I1.jpg', (I1[0]*255.0).permute(1, 2, 0).round().byte().cpu().numpy()[:, :, ::-1])
            count = self.model(I0, I1, returning_tensor, count)
            if last and i == len(frames):
                for _ in range(1, self.sf + 1):
                    returning_tensor[[-_]] = frame.tensor[
                            :, :, self.pader.slice[2]: self.pader.slice[3], self.pader.slice[0]: self.pader.slice[1]
                        ]
        returning_tensor = self.decode(returning_tensor)
        return returning_tensor

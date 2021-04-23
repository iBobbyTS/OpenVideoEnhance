from subprocess import getoutput

import torch
from .model import EDVR

from ove import utils

vram = round(int(getoutput(
    'nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits'
)) / 1000)


class RTer:
    def __init__(
            self,
            width, height, 
            model_path=None, default_model_dir=None,
            model_name='l4v',
            *args, **kwargs
    ):
        # Find model preference
        model_pref = utils.dictionaries.model_configs['edvr']
        # Make sure model exists
        assert model_name in model_pref.keys(), \
            f"Choose nets between {list(utils.dictionaries.model_paths['edvr'].keys())}"
        model_pref = utils.dictionaries.model_configs['edvr'][model_name]
        # Solve for model path
        model_path = utils.folder.check_model(
            default_model_dir, model_path, utils.dictionaries.model_paths['edvr'][model_name]
        )
        self.num_frame = model_pref[1]
        self.enlarge = model_pref[2]
        max_res = model_pref[4][vram]
        self.need_to_init = model_pref[1] // 2 - 1
        # Initialize pader
        self.pader = utils.modeling.Pader(
            width, height, model_pref[3], extend_func='replication', aline_to_edge=True
        )
        self.paded_dim = self.pader.padded_size
        self.input_dim = (width, height)
        self.output_dim = (width * self.enlarge, height * self.enlarge)
        # Calculate splition
        self.splition = utils.modeling.cal_split((self.paded_dim[0], self.paded_dim[1]), max_res)
        # Initialize model
        self.model = EDVR(**model_pref[0]).cuda()
        self.model.load_state_dict(torch.load(model_path)['params'], strict=True)
        self.model.eval()
        # Initialize batch
        self.batch = torch.cuda.FloatTensor(1, self.num_frame, 3, self.paded_dim[1], self.paded_dim[0])

    def get_output_effect(self):
        return {
            'height': self.enlarge,
            'width': self.enlarge,
            'fps': 1
        }

    def encode(self, frame: utils.tensor.Tensor):
        frame.convert(
            place='torch', dtype='float32',
            shape_order='fchw', channel_order='rgb', range_=(0.0, 1.0)
        )
        frame.update(self.pader.pad(frame.tensor))
        frame.unsqueeze(1, name='t')
        return frame

    def decode(self, frames) -> utils.tensor.Tensor:
        tensor = torch.cuda.FloatTensor(len(frames), 3, self.output_dim[1], self.output_dim[0])
        for i, parts in enumerate(frames):
            for slice_, part in zip(self.splition, parts):
                tensor[i, :, slice_[2] * self.enlarge:slice_[3] * self.enlarge, slice_[0] * self.enlarge:slice_[1] * self.enlarge] = part[0, :, :(slice_[3]-slice_[2])*self.enlarge, :(slice_[1]-slice_[0])*self.enlarge]
        tensor = utils.tensor.Tensor(
            tensor=tensor,
            shape_order='fchw', channel_order='rgb',
            range_=(0.0, 1.0)
        )
        return tensor

    def rt(self, frames: utils.tensor.Tensor, last, *args, **kwargs):
        if not frames:
            return frames
        frames = self.encode(frames)
        return_ = []
        for i, frame in enumerate(frames, 1):
            if self.need_to_init:
                if self.need_to_init == self.num_frame // 2:
                    for i in range(self.num_frame // 2 + 1):
                        self.batch[0, i:i+1] = frame.tensor  # no 0:4
                self.batch[
                    0, self.num_frame-self.need_to_init-1:self.num_frame-self.need_to_init
                ] = frame.tensor  # 5:6
                self.need_to_init -= 1  # 0
                if len(frames) > 1:
                    continue
                else:
                    return []
            else:
                self.batch[0, :-1] = self.batch.clone()[0, 1:]
                self.batch[0, -2:-1] = frame.tensor
            parts = []
            for j in self.splition:
                torch.cuda.empty_cache()
                hr = self.model(self.batch[:, :, :, j[2]:j[3], j[0]:j[1]])
                parts.append(hr)
            return_.append(parts)
        if last:
            for i in range(self.num_frame // 2 - 1):
                self.batch[0, :-1] = self.batch.clone()[0, 1:]
                parts = []
                for j in self.splition:
                    torch.cuda.empty_cache()
                    hr = self.model(self.batch[:, :, :, j[2]:j[3], j[0]:j[1]])
                    parts.append(hr)
                return_.append(parts)
        return_ = self.decode(return_)
        return return_

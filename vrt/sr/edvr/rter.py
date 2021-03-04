
import cv2
from subprocess import getoutput

import numpy
import torch
from basicsr.models.archs.edvr_arch import EDVR

from vrt import dictionaries, utils

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
        torch.set_grad_enabled(False)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        # Find model preference
        model_pref = dictionaries.model_paths['edvr']
        # Make sure model exists
        assert model_name in model_pref.keys(), \
            f"Choose nets between {list(dictionaries.model_paths['edvr'].keys())}"
        model_pref = dictionaries.model_paths['edvr'][model_name]
        # Solve for model path
        model_path = utils.folder.check_model(default_model_dir, model_path, dictionaries.model_paths['edvr'][model_name][5])
        self.num_frame = model_pref[1]
        self.enlarge = model_pref[2]
        max_res = model_pref[4][vram]
        self.need_to_init = model_pref[1] // 2
        # Initialize pader
        self.pader = utils.modeling.Pader(
            width, height, model_pref[3], extend_func='replication', aline_to_edge=True
        )
        self.dim = self.pader.paded_size
        self.ori_dim = (width, height)
        # Calculate splition
        self.splition = utils.modeling.cal_split((self.dim[0], self.dim[1]), max_res)
        # Initialize model
        self.model = EDVR(**model_pref[0]).cuda()
        self.model.load_state_dict(torch.load(model_path)['params'], strict=True)
        self.model.eval()
        # Initialize batch
        self.batch = torch.cuda.FloatTensor(1, self.num_frame, 3, self.dim[1], self.dim[0])

    def get_output_effect(self):
        return {
            'height': self.enlarge,
            'width': self.enlarge,
            'fps': 1
        }

    def ndarray2tensor(self, frame: list):
        frame = [torch.from_numpy(_.copy()).cuda() for _ in frame]
        frame = torch.stack(frame)
        frame = frame.permute(0, 3, 1, 2)
        frame = frame[:, [2, 1, 0]]
        frame = frame.float()
        frame /= 255.0
        frame = self.pader.pad(frame)
        frame = frame.unsqueeze(1)
        return frame

    def tensor2ndarray(self, frames):
        tensor = torch.cuda.FloatTensor(len(frames), 3, self.ori_dim[1] * self.enlarge, self.ori_dim[0] * self.enlarge)
        for i, parts in enumerate(frames):
            for slice_, part in zip(self.splition, parts):
                tensor[i, :, slice_[2] * self.enlarge:slice_[3] * self.enlarge, slice_[0] * self.enlarge:slice_[1] * self.enlarge] = part[0, :, :(slice_[3]-slice_[2])*self.enlarge, :(slice_[1]-slice_[0])*self.enlarge]
        tensor = tensor.clamp(0.0, 1.0)
        tensor *= 255.0
        tensor = tensor.byte()
        tensor = tensor[:, [2, 1, 0]]
        tensor = tensor.permute(0, 2, 3, 1)
        tensor = tensor.cpu().numpy()
        return list(tensor)

    def rt(self, frames: list, *args, **kwargs):
        if not frames:
            return frames
        numpy_frames = frames
        return_ = []
        torch_frames = self.ndarray2tensor(frames)
        for i, numpy_frame, torch_frame in zip(range(1, len(numpy_frames)+1), numpy_frames, torch_frames):
            if self.need_to_init:
                if self.need_to_init == self.num_frame // 2:
                    for i in range(self.num_frame // 2):
                        self.batch[0, i] = torch_frame
                self.batch[0, self.num_frame - self.need_to_init] = torch_frame
                self.need_to_init -= 1
                if len(frames) > 1:
                    continue
                else:
                    return []
            else:
                self.batch[0, :-1] = self.batch.clone()[0, 1:]
                self.batch[0, -1] = torch_frame
            parts = []
            for j in self.splition:
                torch.cuda.empty_cache()
                hr = self.model(self.batch[:, :, :, j[2]:j[3], j[0]:j[1]])
                parts.append(hr)
            return_.append(parts)
        if kwargs['duplicate']:
            for i in range(self.num_frame // 2):
                self.batch[0, :-1] = self.batch.clone()[0, 1:]
                parts = []
                for j in self.splition:
                    torch.cuda.empty_cache()
                    hr = self.model(self.batch[:, :, :, j[2]:j[3], j[0]:j[1]])
                    parts.append(hr)
                return_.append(parts)
        return_ = self.tensor2ndarray(return_)
        return return_





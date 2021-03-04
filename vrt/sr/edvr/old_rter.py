import math
from subprocess import getoutput

import numpy
import torch

from basicsr.models.archs.edvr_arch import EDVR

vram = round(int(getoutput(
    'nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits'
)) / 1000)

models = {  # 'model_name': [[model_args], num_of_frame, enlarge_factor:ef, multiple, {vram:max_res}]
    'ld': [{'num_feat': 128, 'num_reconstruct_block': 40, 'hr_in': True, 'with_predeblur': True}, 5, 1, 16, {16: 994}],
    'ldc': [{'num_feat': 128, 'num_reconstruct_block': 40, 'hr_in': True, 'with_predeblur': True}, 5, 1, 16, {16: 994}],
    'l4r': [{'num_feat': 128, 'num_reconstruct_block': 40}, 5, 4, 4, {16: 288}],
    'l4v': [{'num_feat': 128, 'num_reconstruct_block': 40, 'num_frame': 7, 'center_frame_idx': 3}, 7, 4, 4, {16: 256}],
    'l4br': [{'num_feat': 128, 'num_reconstruct_block': 40, 'with_predeblur': True}, 5, 4, 4, {16: 256}],
    'm4r': [{'with_tsa': False}, 5, 4, 4, {16: 416}],
    'mt4r': [{}, 5, 4, 4, {16: 416}]
}


def cal_split(h_w, max_side_length):
    splition = []
    for i in range(math.ceil(h_w[0] / max_side_length)):
        for j in range(math.ceil(h_w[1] / max_side_length)):
            ws = i * max_side_length
            we = (i + 1) * max_side_length if (i + 1) * max_side_length <= h_w[0] else h_w[0]
            hs = j * max_side_length
            he = (j + 1) * max_side_length if (j + 1) * max_side_length <= h_w[1] else h_w[1]
            splition.append((ws, we, hs, he))
    return splition


class rter:
    def __init__(self, model_name, model_path, height, width):
        model_info = models[model_name]
        self.model = EDVR(**model_info[0]).cuda()
        self.num_frame = model_info[1]
        self.enlarge = model_info[2]
        ef = model_info[3]
        max_res = model_info[4][vram]
        self.height, self.width = height, width

        self.h_w = [int(math.ceil(height / ef) * ef - height) if height % ef else 0,
                    int(math.ceil(width / ef) * ef) - width if width % ef else 0]
        self.dim = [height + self.h_w[0], width + self.h_w[1]]
        self.splition = cal_split((self.dim[0], self.dim[1]), max_res)
        self.pader = torch.nn.ReplicationPad2d([0, self.h_w[1], 0, self.h_w[0]])

        self.model.load_state_dict(torch.load(model_path)['params'], strict=True)
        self.model.eval()

    def init_batch(self, video):
        self.batch = torch.cuda.FloatTensor(1, self.num_frame, 3, self.dim[0], self.dim[1])
        frame = self.ndarray2tensor(video.read()[1])
        for i in range(self.num_frame // 2 + 1):
            self.batch[0, i] = frame
        for i in range(self.num_frame // 2 + 1, self.num_frame - 1):
            self.batch[0, i] = self.ndarray2tensor(video.read()[1])

    def ndarray2tensor(self, frame: numpy.ndarray):
        out_frame = torch.cuda.ByteTensor(frame)[:, :, [2, 1, 0]].permute(2, 0, 1).unsqueeze(0)
        out_frame = self.pader(out_frame.float()) / 255.0
        return out_frame

    def rt(self, f):
        e = self.enlarge
        if f[0]:
            self.batch[0, -1] = self.ndarray2tensor(f[1])
        outputs = []
        for i in self.splition:
            torch.cuda.empty_cache()
            outputs.append(
                (self.model(self.batch[:, :, :, i[0]:i[1], i[2]:i[3]]) * 255.0).clamp(0, 255).byte().squeeze()[
                    [2, 1, 0]].permute(1, 2, 0).cpu().numpy())
        self.batch[0, :-1] = self.batch.clone()[0, 1:]
        output = numpy.zeros((self.height * e, self.width * e, 3), dtype=numpy.uint8)
        for slice_, tensor in zip(self.splition, outputs):
            shape = output[slice_[0] * e:slice_[1] * e, slice_[2] * e:slice_[3] * e].shape
            output[slice_[0] * e:slice_[1] * e, slice_[2] * e:slice_[3] * e] = tensor[:shape[0], :shape[1]]
        return output

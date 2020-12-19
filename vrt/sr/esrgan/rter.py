import math

import numpy
import torch

from basicsr.models.archs.rrdbnet_arch import RRDBNet as ESRGAN


model_paths = {
    'pd': 'PSNR_SRx4_DF2K.pth',
    'dk': 'SRx4_DF2KOST.pth',
    'r': 'ESRGAN_x4.pth',
    'ro': 'ESRGAN_x4_old_arch.pth',
    'p': 'PSNR_x4.pth',
    'po': 'PSNR_x4_old_arch.pth'
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
    def __init__(self, net_name, model_dir, height, width, **esrgan):
        assert net_name in model_paths.keys(), f'Choose nets between {list(model_paths.keys())}'
        model_path = model_paths[net_name]
        self.model = ESRGAN(num_in_ch=3, num_out_ch=3)
        self.cuda_compatibility = torch.cuda.is_available()
        if self.cuda_compatibility:
            self.model = self.model.cuda()
        if not model_dir:
            model_dir = 'model_weights/ESRGAN/' + model_path
        self.model.load_state_dict((torch.load(model_dir)['params']), strict=True)
        self.model.eval()

    def init_batch(self, video):
        return

    def ndarray2tensor(self, frame: numpy.ndarray):
        return torch.cuda.ByteTensor(frame)[:, :, [2, 1, 0]].permute(2, 0, 1).unsqueeze(0).float() / 255.0 \
            if self.cuda_compatibility else \
            torch.FloatTensor(numpy.transpose(frame, (2, 0, 1))[::-1].astype('float32') / 255).unsqueeze(0)

    def tensor2ndarray(self, frame: torch.tensor):
        return (frame * 255.).clamp(0, 255).byte().squeeze()[[2, 1, 0]].permute(1, 2, 0).cpu().numpy()

    def rt(self, frames, *args, **kwargs):
        output = []
        for frame in frames:
            output.append(self.tensor2ndarray(self.model(self.ndarray2tensor(frame))))
        return output

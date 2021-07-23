import torch
import torch.nn.functional as F
from fastai import vision, basic_data

from .generators import gen_inference_deep, gen_inference_wide


class MyNet:
    def __init__(self, temp_path, state_dict, model='v'):
        self.learn = (
            gen_inference_deep if model == 'a' else gen_inference_wide
        )(state_dict=state_dict, temp_path=temp_path)
        self.learn.model.to(torch.float16 if torch.cuda.is_available() else torch.float32)
        self.norm, self.denorm = vision.normalize_funcs(*vision.data.imagenet_stats)

    def __call__(self, frame, resized, target, count):
        x, y = self.norm((resized, resized), do_x=True)
        color = self.learn.pred_batch(
            ds_type=basic_data.DatasetType.Valid, batch=(x[None], y[None]), reconstruct=True
        )[0]
        color = self.denorm(color.px, do_x=False)
        # Postprocess
        color = F.interpolate(color.unsqueeze(0), frame.shape[1:3], mode='bicubic', align_corners=True).squeeze(0)
        # Original RGB
        R, G, B = color
        # Original YUV
        U = -0.148 * R - 0.291 * G + 0.439 * B
        V = 0.439 * R - 0.368 * G - 0.071 * B
        # Color Y
        Y = 0.257 * frame[0] + 0.504 * frame[1] + 0.098 * frame[2]
        # Final RGB
        R = 1.164 * Y + 1.596 * V
        G = 1.164 * Y - 0.392 * U - 0.813 * V
        B = 1.164 * Y + 2.017 * U
        target[count, 0] = R
        target[count, 1] = G
        target[count, 2] = B

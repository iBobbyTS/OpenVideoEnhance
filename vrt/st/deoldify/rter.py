import warnings
from pathlib import Path
import os

import PIL

from . import visualize

model_paths = {
    'video': 'ColorizeVideo_gen',
    'artistic': 'ColorizeArtistic_gen',
    'stable': 'ColorizeStable_gen',
}


class rter:
    warnings.filterwarnings("ignore", category=UserWarning, message=".*?Your .*? set is empty.*?")

    def __init__(self, net_name, model_dir, height, width, **deoldify):
        if model_dir:
            model_dir = os.path.splitext(model_dir)
        else:
            model_dir = 'model_weights/DeOldify/' + model_paths[net_name]
        os.makedirs(os.path.join(deoldify['temp_dir'], 'models'), exist_ok=True)
        learn = visualize.gen_inference_wide(
            root_folder=Path(deoldify['temp_dir']),
            weights_name=model_dir)
        filtr = visualize.MasterFilter(
            [visualize.ColorizerFilter(learn=learn)],
            render_factor=deoldify['coef'])
        colorizer = visualize.ModelImageVisualizer(filtr)
        self.colorizer = visualize.VideoColorizer(colorizer)

    def init_batch(self, video):
        return

    def rt(self, frames, *args, **kwargs):
        out = []
        for frame in frames:
            self.colorizer.vis._clean_mem()
            frame = PIL.Image.fromarray(frame[:, :, ::-1])
            out.append(self.colorizer.vis.filter.filter(frame, frame, 35))
        return out

import os
import sys

vrt_root = '/'.join(os.path.abspath(__file__).split('/')[:-2])
if os.getcwd() != vrt_root:
    os.chdir(vrt_root)
if vrt_root not in [os.path.abspath(path) for path in sys.path]:
    sys.path.append(vrt_root)

from enhancer import enhance

# Args
input_opt = {
    'path': '/Users/ibobby/Dataset/resolution_test/120p.mp4'
}
temp_opt = {
    'path': '../tmp',
    'remove': False
}
preprocess_opt = {
    'lib': 'ffmpeg',
    'frame_range': (0, 0),
    'resize': (64, 48),
    # if FFmpeg
    'decoder': None,
}
model_opt = {
    'empty_cache': True,
    'default_model_dir': '/Users/ibobby/Dataset/model_weights',
    'to_do': ['esrgan'],
    'model_path': [None],
    'args': [[]],
    'kwargs': [{'render_factor': 20}]
}
postprocess_opt = {
    # Share
    'type': 'img',
    'lib': 'ffmpeg',
    # Video
    # CV2
    'fourcc': 'hvc1',
    # FFmpeg
    'encoder': 'libx264',
    'pix_fmt': 'yuv420p',
    'resize': None,
    'in_fps': None,
    # Set out_fps if you want the final fps to be 60 in order not to waste space.
    # Keep it None unless you know the difference of -r option in FFmpeg for input and output streams.
    'out_fps': None,
    'crf': 28,
    'ffmpeg-params': '',
    # Image & Internal Data Format
    'ext': 'jpg',  # If img
    # Internal Data Format
    'dtype': 'uint8'
}
output_opt = {
    'path': '/Users/ibobby/Dataset/out/o'
}

if __name__ == '__main__':
    enhance(
        input_opt,
        temp_opt,
        preprocess_opt, model_opt, postprocess_opt,
        output_opt
    )

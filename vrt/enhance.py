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
    'path': '/content/frame_test.mov'
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
    'default_model_dir': '/content/model_weights',
    'to_do': ['dain', 'edvr', 'ssm', 'esrgan'],
    'model_path': [None]*7,
    'args': [[]]*7,
    'kwargs': [{'model_name': 'ldc'}, {'sf': 3}, {}, {'sf': 4}, {}, {}, {}, {}]
}
postprocess_opt = {
    # Share
    'type': 'vid',
    'lib': 'ffmpeg',
    # Video
    # CV2
    'fourcc': 'hvc1',
    # FFmpeg
    'encoder': 'libx265',
    'pix_fmt': 'yuv420p',
    'resize': None,
    'in_fps': None,
    # Set out_fps if you want the final fps to be 60 in order not to waste space.
    # Keep it None unless you know the difference of -r option in FFmpeg for input and output streams.
    'out_fps': None,
    'crf': 18,
    'ffmpeg-params': '',
    # Image & Internal Data Format
    'ext': 'jpg',  # If img
    # Internal Data Format
    'dtype': 'uint8'
}
output_opt = {
    'path': '/content/out/o'
}

enhance(input_opt, temp_opt, preprocess_opt, model_opt, postprocess_opt, output_opt)

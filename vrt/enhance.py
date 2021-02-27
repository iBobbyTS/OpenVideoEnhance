import os
import sys
sys.path.insert(0, '/content/OpenVideoEnhance')
vrt_root = os.path.split(os.path.abspath(__file__))[0]
if os.getcwd() != vrt_root:
    os.chdir(vrt_root)
if vrt_root not in [os.path.abspath(path) for path in sys.path]:
    sys.path.append(vrt_root)

from enhancer import enhance

# Args
input_opt = {
    'path': '/Users/ibobby/Dataset/frame_test.mov'
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
    'default_model_dir': '../model_weights',
    'to_do': ['ssm', 'esrgan', 'ssm', 'esrgan'],
    'model_path': [None, None, None, None],
    'args': [[], [], [], []],
    'kwargs': [{}, {'mode': 'nearest'}, {}, {'mode': 'nearest'}]
}
postprocess_opt = {
    # Share
    'type': 'img',
    'lib': 'cv2',
    # Video
    # CV2
    'fourcc': 'hvc1',
    # FFmpeg
    'encoder': 'libx265',
    'pix_fmt': 'yuv420p',
    'resize': None,
    'in_fps': 1,
    # Set out_fps if you want the final fps to be 60 in order not to waste space.
    # Keep it None unless you know the difference of -r option in FFmpeg for input and output streams.
    'out_fps': None,
    'crf': 24,
    'ffmpeg-params': '',
    # Image & Internal Data Format
    'ext': 'tiff',  # If img
    # Internal Data Format
    'dtype': 'uint8'
}
output_opt = {
    'path': '/Users/ibobby/Dataset/out/o'
}

enhance(input_opt, temp_opt, preprocess_opt, model_opt, postprocess_opt, output_opt)

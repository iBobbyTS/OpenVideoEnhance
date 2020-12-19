import os
import sys

vrt_root = os.path.split(os.path.abspath(__file__))[0]
if os.getcwd() != vrt_root:
    os.chdir(vrt_root)
if vrt_root not in [os.path.abspath(path) for path in sys.path]:
    sys.path.append(vrt_root)

from enhancer import enhance

# Args
external_opt = {
    'ffmpeg_dir': ''
}
input_opt = {
    'path': '/Users/ibobby/Dataset/resolution_test/gray'
}
temp_opt = {
    'path': 'tmp',
    'remove': False
}
preprocess_opt = {
    'reader': 'ffmpeg',
    'hardware_decoder': 'apple',
    'out_fmt': 'tiff',
    'frame_range': (0, 0),
    'resize': False,
    'crop': False
}
model_opt = {
    'to_do': ['DeOldify'],
    'model_name': ['video'],
    'model_path': [False],
    'sf': [35],
    'batch_size': [1],
    'empty_cache': True
}
postprocess_opt = {
    # Share
    'type': 'img',
    'ext': 'png',
    'codec': 'hevc',  # dtype if type is idf
    'resize': False,
    # Video
    'writer': 'cv2',
    'mac_compatibility': True,
    'fps': 30,
    'hardware_encoder': 'apple',
    'extra_video_meta': True,
    'br': False,
    'crf': 20,
    # Image
    # Internal Data Format
    # None
}
output_opt = {
    'path': '/Users/ibobby/Dataset/resolution_test/out/120p'
}

enhance(external_opt, input_opt, temp_opt, preprocess_opt, model_opt, postprocess_opt, output_opt)

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
    'path': '/content/drive/Shareddrives/iBobbyTS/Colab/Test/Video/Test/resolution_test/720p'
}
temp_opt = {
    'path': '/tmp/OpenVideoEnhance',
    'remove': False
}
preprocess_opt = {
    'reader': 'ffmpeg',
    'hardware_decoder': 'nvidia',
    'out_fmt': 'tiff',
    'frame_range': (0, 0),
    'resize': False,
    'crop': False,
    'buffer_size': 2
}
model_opt = {
    'default_model_path': '/content/model_weights',
    'empty_cache': False,
    'to_do': ['BMBC'],
    'model_name': [False],
    'model_path': [False],
    'batch_size': [1],
    'extra_args': [{}]
}
postprocess_opt = {
    # Share
    'type': 'vid',
    'ext': 'mov',
    'resize': False,
    # Video
    'codec': 'hevc',
    'writer': 'cv2',
    'mac_compatibility': True,
    'fps': 30,
    'hardware_encoder': 'nvidia',
    'extra_video_meta': True,
    'br': False,
    'crf': 20,
    # Image
    # None
    # Internal Data Format
    'dtype': 'uint8'
}
output_opt = {
    'path': '/content/drive/Shareddrives/iBobbyTS/Colab/Project/OpenVideoEnhance/GitHub/OpenVideoEnhance/out/720p.mov'
}

enhance(external_opt, input_opt, temp_opt, preprocess_opt, model_opt, postprocess_opt, output_opt)

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
    'to_do': ['BMBC'],
    'model_name': ['DAIN_slowmotion'],
    'model_path': [False],
    'coef': [3],
    'batch_size': [1],
    'empty_cache': False,
    'extra_args': [{'rectify': False}]
}
postprocess_opt = {
    # Share
    'type': 'vid',
    'ext': 'mov',
    'codec': 'hevc',  # vcodec if vid, dtype if type is idf
    'resize': False,
    # Video
    'writer': 'cv2',
    'mac_compatibility': True,
    'fps': 30,
    'hardware_encoder': 'nvidia',
    'extra_video_meta': True,
    'br': False,
    'crf': 20,
    # Image
    # Internal Data Format
    # None
}
output_opt = {
    'path': '/content/drive/Shareddrives/iBobbyTS/Colab/Project/OpenVideoEnhance/GitHub/OpenVideoEnhance/out/720p.mov'
}

enhance(external_opt, input_opt, temp_opt, preprocess_opt, model_opt, postprocess_opt, output_opt)

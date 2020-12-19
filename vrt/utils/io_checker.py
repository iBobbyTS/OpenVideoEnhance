# Model checking
import os
import torch

__all__ = ('listdir', 'check_model', 'empty_cache', 'path2list', 'detect_input_type', 'check_dir_availability', 'solve_start_end_frame')


def path2list(path):
    out = list(os.path.split(path))
    out.extend(os.path.splitext(out[1]))
    out.pop(1)
    return out


# Empty cache
if 'CUDA_EMPTY_CACHE' in os.environ and int(os.environ['CUDA_EMPTY_CACHE']):
    empty_cache = torch.cuda.empty_cache
else:
    empty_cache = lambda: None


def listdir(folder):  # 输入文件夹路径，输出文件夹内的文件，排序并移除可能的无关文件
    disallow = ['.DS_Store', '.ipynb_checkpoints', '$RECYCLE.BIN', 'Thumbs.db', 'desktop.ini']
    files = []
    for file in os.listdir(folder):
        if file not in disallow and file[:2] != '._':
            files.append(file)
    files.sort()
    return files


def check_model(paths):
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model {path} doesn't exist, exiting")


def detect_input_type(input_dir):  # 检测输入类型
    if os.path.isfile(input_dir):
        if os.path.splitext(input_dir)[1].lower() == '.json':
            input_type_ = 'continue'
        else:
            input_type_ = 'vid'
    else:
        extension = os.path.splitext(listdir(input_dir)[0])[1].replace('.', '')
        if extension == 'npz':
            input_type_ = 'npz'
        elif extension == 'npy':
            input_type_ = 'npy'
        elif extension == 'pmg':
            input_type_ = 'pmg'
        elif extension in ('dpx', 'jpg', 'jpeg', 'exr', 'psd', 'png', 'tif', 'tiff'):
            input_type_ = 'img'
        else:
            input_type_ = 'mix'
    return input_type_


def check_dir_availability(dire, ext=''):
    if not os.path.exists(os.path.split(dire)[0]):  # If mother directory doesn't exist
        os.makedirs(os.path.split(dire)[0])  # Create one
    if os.path.exists(dire + ext):  # If target file/folder exists
        count = 2
        while os.path.exists(f'{dire}_{count}{ext}'):
            count += 1
        dire = f'{dire}_{count}{ext}'
    else:
        dire = f'{dire}{ext}'
    if not ext:  # Output as folder
        os.mkdir(dire)
    return dire


def solve_start_end_frame(frame_range, frame_count):
    start_frame, end_frame = frame_range
    if end_frame == 0 or end_frame >= frame_count:
        copy = True
        end_frame = frame_count
    else:
        copy = False
    if start_frame == 0 or start_frame >= frame_count:
        start_frame = 0
    return start_frame, end_frame, copy

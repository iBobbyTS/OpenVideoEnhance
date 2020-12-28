# Model checking
import os
import torch

from .io_utils import listdir


def path2list(path):
    out = list(os.path.split(path))
    out.extend(os.path.splitext(out[1]))
    out.pop(1)
    return out


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


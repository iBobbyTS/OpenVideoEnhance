import os

import torch


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


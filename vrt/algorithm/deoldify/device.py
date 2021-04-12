import os

import torch


def set(cuda_availability):
    os.environ['CUDA_VISIBLE_DEVICES'] = {False: '', True: '1'}[cuda_availability]
    if cuda_availability:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    os.environ['OMP_NUM_THREADS'] = '1'

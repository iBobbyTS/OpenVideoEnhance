import os
from time import time

import torch

from . import str_fmt, folder, dictionaries

__all__ = (
    'empty_cache',
    'solve_input',
    'solve_start_end_frame',
    'Timer'
)


def empty_cache():
    if int(os.environ.get('CUDA_EMPTY_CACHE', 0)):
        torch.cuda.empty_cache()


def detect_input_type(input_dir):
    """
        Detect input type.

        Parameters
        ----------
        input_dir : str
            Input path

        Returns
        -------
        type : str
            continue, mix, vid, npy, npz, pmg, img or mix

        Examples
        --------
        >>> detect_input_type('/content/videos/test.mov')
        vid
    """
    if os.path.isfile(input_dir):
        if os.path.splitext(input_dir)[1].lower() == '.json':
            input_type_ = 'continue'
        else:
            input_type_ = 'vid'
    else:
        if '%' in input_dir.split('/')[-1]:
            extension = os.path.splitext(input_dir)[1].replace('.', '').lower()
        else:
            extension = os.path.splitext(folder.listdir(input_dir)[0])[1].replace('.', '').lower()
        if extension == 'npz':
            input_type_ = 'npz'
        elif extension == 'npy':
            input_type_ = 'npy'
        elif extension == 'pmg':
            input_type_ = 'pmg'
        elif extension in ('jpg', 'jpeg', 'png', 'tif', 'tiff'):
            input_type_ = 'img'
        else:
            input_type_ = 'mix'
    return input_type_


def solve_input(inputs):
    """
        Put inputs in to a list of tuples with input type and input directory

        Parameters
        ----------
        inputs : str or list
            Input path

        Returns
        -------
        inputs : tuple
            Including type and directory

        Examples
        --------
        >>> solve_input('/content/videos/test.mov')
        ('vid', ['/content/videos', 'test', '.mov'])
    """
    return detect_input_type(inputs), folder.path2list(inputs)


def solve_start_end_frame(frame_range, frame_count, before, after):
    start_frame, end_frame = frame_range
    if end_frame == 0 or end_frame >= frame_count:
        end_frame = frame_count
        after = 0
    if start_frame == 0 or start_frame >= frame_count:
        start_frame = 0
        before = 0
    return start_frame, end_frame, before, after


def solve_before_after_frame(model_opt):
    extra = list(map(dictionaries.model_extra_frames.get, model_opt['to_do']))
    for i, j in zip(*map(model_opt.get, ('to_do', 'kwargs'))):
        if 'edvr' in i:
            _ = 2
            if 'num_frames' in (model_config := dictionaries.model_configs['edvr'][j['model_name']][0]).keys():
                _ = model_config['num_frames']
            extra.append((_, _))
    before = max([_[0] for _ in extra])
    after = min([_[1] for _ in extra])
    return before, after


class Timer:
    def __init__(self, total_count):
        self.total_count = total_count
        self.timer = 0
        self.count = 0
        self.frames_left = total_count
        self.start_time = time()
        self.init = True

    def print(self):
        time_spent = time() - self.start_time
        self.start_time = time()
        self.frames_left -= 1
        if self.init:
            self.initialize_time = time_spent
            print(
                f'Initialized and processed frame 1/{self.total_count} | '
                f'{self.total_count - self.count - 1} frames left | '
                f'Time spent: {round(self.initialize_time, 2)}s',
                end=''
            )
            self.init = False
        else:
            self.timer += time_spent
            frames_processed = self.count + 1
            if self.frames_left != 0:
                print(
                    f'\rProcessed batch {frames_processed}/{self.total_count} | '
                    f"{self.frames_left} {'batches' if self.frames_left > 1 else 'batch'} left | "
                    f'Time spent: {round(time_spent, 2)}s | '
                    f'Time left: {str_fmt.second2time(self.frames_left * self.timer / (frames_processed - 1))} | '
                    f'Total time spend: {str_fmt.second2time(self.timer + self.initialize_time)}',
                    end='', flush=True
                )
            else:
                print(
                    f'\rFinished processing in {str_fmt.second2time(self.timer + self.initialize_time)}',
                    flush=True
                )
        self.count += 1

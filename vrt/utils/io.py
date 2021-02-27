import os
from time import time

import torch

from . import str_fmt, folder

__all__ = (
    'empty_cache',
    'solve_input',
    'solve_start_end_frame',
    'Timer'
)

def empty_cache():
    if int(os.environ.get('CUDA_EMPTY_CACHE', 0)):
        torch.cuda.empty_cache()

# def empty_cache(lvl):
#     if lvl:
#         def fn(l):
#             if l < lvl:
#                 torch.cuda.empty_cache()
#         return fn
#     else:
#         return lambda: None


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
        extension = os.path.splitext(folder.listdir(input_dir)[0])[1].replace('.', '').lower()
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
        [('vid', ['/content/videos', 'test', '.mov'])]
    """
    inputs = inputs if isinstance(inputs, (list, tuple, set)) else [inputs]
    return_inputs = []
    for i in inputs:
        return_inputs.append((detect_input_type(i), folder.path2list(i)))
    return return_inputs


def solve_start_end_frame(frame_range, frame_count):
    """
        Use frame range and frame count to solve for start_frame, end_frame and copy

        Parameters
        ----------
        frame_range : tuple
            Frame range want to process.
        frame_count : int
            Frame count of the input video

        Returns
        -------
        start_frame : int
        end_frame : int
        copy : bool

        Examples
        --------
        >>> solve_start_end_frame((2, 200), 300)
        (2, 200, False)
    """
    start_frame, end_frame = frame_range
    if end_frame == 0 or end_frame >= frame_count:
        copy = True
        end_frame = frame_count
    else:
        copy = False
    if start_frame == 0 or start_frame >= frame_count:
        start_frame = 0
    return start_frame, end_frame, copy


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
                end='')
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
                    end='', flush=True)
            else:
                print(
                    f'\rFinished processing in {str_fmt.second2time(self.timer + self.initialize_time)}',
                    flush=True)
        self.count += 1

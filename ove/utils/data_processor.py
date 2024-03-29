import os
import subprocess
import threading

import numpy
import cv2

from ove import utils


class DataLoader:
    def __init__(self, video_input, opt, channel_order, global_opt, frames_before):
        """
        Parameters
        ----------
        video_input: path to video in a format of [path, filename, ext]
        opt: preprocess_opt
        """
        ffmpeg_bin_path = global_opt['ffmpeg_bin_path']
        self.count = max((0 if ((_ := opt['frame_range']) is None or _[0] == 0) else _[0]) - frames_before, 0)
        self.input_type = video_input[0]
        self.lib = opt['lib']
        real_path = f'{video_input[1][0]}/{video_input[1][1]}{video_input[1][2]}'
        if self.input_type == 'vid':
            if self.lib == 'cv2':
                self.video = cv2.VideoCapture(real_path)
                width = int(self.video.get(3))
                height = int(self.video.get(4))
                fps = self.video.get(5)
                frame_count = int(self.video.get(7))
                self.video.set(1, self.count)
                self.read_func = lambda x: self.video.read()[1]
            elif self.lib == 'ffmpeg':
                # FFprobe
                ffprobe_out = eval(subprocess.getoutput(
                    f"{os.path.join(ffmpeg_bin_path, 'ffprobe')} -hide_banner -v quiet -select_streams v -show_entries "
                    'stream=height,width,r_frame_rate,nb_frames,codec_tag_string:stream_tags=duration '
                    f"-print_format json '{real_path}'"
                ))['streams'][0]
                width, height = map(ffprobe_out.get, ('width', 'height'))
                fps = eval(ffprobe_out['r_frame_rate'])
                if ffprobe_out.get('nb_frames', False):
                    frame_count = int(ffprobe_out['nb_frames'])
                else:
                    duration = ffprobe_out['tags']['DURATION']
                    duration = duration.split(':')
                    duration = int(duration[0]) * 3600 + int(duration[1]) * 60 + float(duration[2])
                    frame_count = round(duration * fps)
                # Open pipe
                command = [os.path.join(ffmpeg_bin_path, 'ffmpeg'), '-loglevel', 'panic']
                if opt['decoder'] is not None:
                    command.extend(['-c:v', opt['decoder']])
                command.extend([
                    '-ss', str(self.count / fps),
                    '-i', real_path,
                    *([] if (_ := opt['resize']) is None else ['-s', '%dx%d' % tuple(_)]),
                    '-f', 'rawvideo', '-pix_fmt', f'{channel_order}48',
                    '-'
                ])
                self.pipe = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=width * height * 3)
                # Define functions
                self.read_func = lambda: numpy.frombuffer(
                    self.pipe.stdout.read(width * height * 3 * 2),
                    dtype='uint16'
                ).reshape((height, width, 3)).astype(numpy.int32)
        elif self.input_type == 'img':
            self.count += 1
            self.files = utils.folder.listdir(os.path.split(real_path)[0] if '%' in os.path.split(real_path)[-1] else real_path)
            print(real_path)
            frame_count = len(self.files)
            height, width = cv2.imread(os.path.join(
                os.path.split(real_path)[0] if '%' in os.path.split(real_path)[-1] else real_path, self.files[0]
            )).shape[0:2]
            fps = 1
            if self.lib == 'cv2':
                self.read_func = lambda x: cv2.imread(os.path.join(
                    real_path, self.files[self.count - 1]
                ))
            elif self.lib == 'ffmpeg':
                if (_ := opt['resize']) is not None:
                    width, height = _
                command = [os.path.join(ffmpeg_bin_path, 'ffmpeg'), '-loglevel', 'panic']
                if opt['decoder'] is not None:
                    command.extend(['-c:v', opt['decoder']])
                command.extend([
                    '-ss', str((self.count - 1) / 25),
                    '-i', real_path,
                    *([] if (_ := opt['resize']) is None else ['-s', '%dx%d' % tuple(_)]),
                    '-f', 'rawvideo', '-pix_fmt', f'{channel_order}48',
                    '-'
                ])
                self.pipe = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=100000000)
                self.read_func = lambda: numpy.frombuffer(
                    self.pipe.stdout.read(width * height * 3 * 2),
                    dtype=numpy.uint16
                ).reshape((height, width, 3)).astype(numpy.int32)

        # Public
        if self.lib == 'ffmpeg':
            print(' '.join(command))
        if self.lib == 'cv2' and (_ := opt['resize']) is not None:
            self.read_func = utils.modeling.sequential([
                self.read_func,
                lambda x: cv2.resize(x, tuple(_), interpolation=cv2.INTER_CUBIC)
            ])
        dtype = numpy.int32 if self.lib == 'ffmpeg' else numpy.uint8
        self.range_ = (0.0, (65535.0 if self.lib == 'ffmpeg' else 255.0))
        if (_ := opt['resize']) is not None:
            width, height = _
        self.info = [
            width, height, fps, frame_count,
            numpy.ndarray, dtype
        ]
        self.channel_order = channel_order if self.lib == 'ffmpeg' else 'bgr'

    def get(self, idx):
        """
        Parameters
        ----------
        idx : int
        from 0 to 5
        0: Height
        1: Width
        2: Framerate if available
        3: Total frame count
        4: Returning object type of read()
        5: Returning data type

        Returns
        -------
        Value
        """
        return self.info[idx]

    def read(self, frame_num=None):
        """

        Parameters
        ----------
        frame_num
        Apply frame_num if not reading the following frame.

        Returns
        -------
        requested frame
        """
        if frame_num is not None:
            if frame_num != self.info_func()[1]:
                if self.input_type == 'vid' and self.lib == 'cv2':
                    self.video.set(1, frame_num)
                if self.input_type == 'img' and self.lib == 'cv2':
                    self.count = frame_num
        img = self.read_func()
        self.count += 1
        return img

    def close(self):
        if self.input_type == 'vid' and self.lib == 'cv2':
            self.video.release()
        if self.lib == 'ffmpeg':
            self.pipe.terminate()


class DataWriter:
    default = {
        'extensions': {
            'vtag': {
                'avc1': 'mp4',
                'hvc1': 'mov',
                'av01': 'mp4',
                'vp09': 'webm'
            },
            'encoder': {
                'h264': 'mp4',
                'libx264': 'mp4',
                'libx264rgb': 'mp4',
                'h264_nvenc': 'mp4',
                'hevc': 'mov',
                'libx265': 'mov',
                'hevc_nvenc': 'mov',
                'av1': 'mp4',
                'vp9': 'webm'
            }
        }
    }

    def __init__(
            self,
            input_dir, output_path, opt, res, fps, channel_order, ffmpeg_bin_path
    ):
        """
        Parameters
        ----------
        output_path
        output path

        opt : dict
            preprocess_opt
        """
        self.type = opt['type']
        self.lib = opt['lib']
        self.count = 0
        if output_path is None:
            output_path = f'{os.path.splitext(input_dir)[0]}_Enhanced'
        if self.type == 'vid':
            output_path, ext = os.path.splitext(output_path)
            if not ext:
                ext = self.default['extensions']['vtag'][opt['fourcc']] if self.lib == 'cv2' else \
                    self.default['extensions']['encoder'][opt['encoder']]
            self.output_path = utils.folder.check_dir_availability(output_path, ext=ext)
            # Solve for output file name
            if self.lib == 'cv2':
                self.video = cv2.VideoWriter(
                    self.output_path,
                    cv2.VideoWriter_fourcc(*opt['fourcc']),
                    fps, res
                )
                self.write_func = lambda frame: self.video.write(frame)
            elif self.lib == 'ffmpeg':
                command = [
                    os.path.join(ffmpeg_bin_path, 'ffmpeg'),
                    '-loglevel', 'error',
                    '-f', 'rawvideo',
                    '-pix_fmt', f'{channel_order}48',
                    '-s', '%dx%d' % res, '-r', str(fps),
                    '-i', '-',
                    *([] if (_ := opt['resize']) is None else ['-s', '%dx%d' % _]),
                    *([] if (_ := opt['out_fps']) is None else ['-r', str(_)]),
                    '-c:v', opt['encoder'],
                    *([] if (_ := opt['pix_fmt']) is None else ['-pix_fmt', str(_)]),
                    *(['-tag:v', 'hvc1'] if opt['encoder'] in ('hevc', 'hevc_nvenc', 'libx265') else []),
                    *([] if (_ := opt['crf']) is None else ['-crf', str(_)]),
                    *([] if (_ := opt['ffmpeg-params']) is None else _.split(' ')),
                    self.output_path
                ]
                self.pipe = subprocess.Popen(command, stdin=subprocess.PIPE)
                self.write_func = lambda frame: (self.pipe.stdin.write(frame.tobytes()), self.pipe.stdin.flush())
        elif self.type == 'img':
            self.output_path = utils.folder.check_dir_availability(output_path)
            self.ext = opt['ext']
            if self.lib == 'cv2':
                self.write_func = lambda frame: cv2.imwrite(f"{self.output_path}/{self.count}.{self.ext}", frame)
            elif self.lib == 'ffmpeg':
                command = [
                    os.path.join(ffmpeg_bin_path, 'ffmpeg'),
                    '-loglevel', 'error',
                    '-f', 'rawvideo', '-pix_fmt', f'{channel_order}48',
                    '-s', '%dx%d' % res,
                    '-i', '-',
                    *([] if (_ := opt['resize']) is None else ['-s', '%dx%d' % _]),
                    *([] if (_ := opt['ffmpeg-params']) is None else _.split(' ')),
                    f'{self.output_path}/%d.{self.ext}'
                ]
                self.pipe = subprocess.Popen(command, stdin=subprocess.PIPE)
                self.write_func = lambda frame: (self.pipe.stdin.write(frame.tobytes()), self.pipe.stdin.flush())
        if self.lib == 'ffmpeg':
            print(' '.join(command))
        self.dtype = 'uint16' if self.lib == 'ffmpeg' else 'uint8'
        self.range_ = (0.0, (65535.0 if self.lib == 'ffmpeg' else 255.0))
        self.channel_order = channel_order if self.lib == 'ffmpeg' else 'bgr'
        self.thread = threading.Thread()
        self.thread.start()

    def write(self, obj):
        self.thread.join()
        del self.thread
        self.count += 1
        self.thread = threading.Thread(target=self.write_func, args=(obj,))
        self.thread.start()

    def close(self):
        if self.type == 'vid' and self.lib == 'cv2':
            self.video.release()
        if self.lib == 'ffmpeg':
            self.thread.join()
            self.pipe.communicate()
            self.pipe.terminate()


class DataBuffer:
    def __init__(self, video: DataLoader):
        self.video = video
        self.buff = utils.tensor.Tensor(
            tensor=numpy.empty(
                (2, *map(self.video.get, (1, 0)), 3),
                dtype=self.video.get(5)),
            shape_order='fhwc', channel_order=self.video.channel_order,
            range_=video.range_)
        self.count = 0
        # Thread
        self.get_thread = lambda: threading.Thread(target=self.get_frame_ready)
        self.thread = self.get_thread()
        self.thread.start()

    def get_frame(self, last):
        self.thread.join()
        del self.thread
        if last:
            i = 1 if self.count else 0
            return self.buff[[i]]
        returning = self.buff[[self.count]]
        self.thread = self.get_thread()
        self.thread.start()
        return returning

    def get_frame_ready(self):
        self.count += -1 if self.count else 1
        self.buff[self.count] = self.video.read()

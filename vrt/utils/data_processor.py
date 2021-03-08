import os
import subprocess
import threading

import numpy
import cv2

from vrt import utils


class DataLoader:
    def __init__(self, video_input, opt):
        """
        Parameters
        ----------
        video_input: path to video in a format of [path, filename, ext]
        opt: preprocess_opt
        """
        self.count = 0
        self.input_type = video_input[0]
        self.lib = opt['lib']
        real_path = f'{video_input[1][0]}/{video_input[1][1]}{video_input[1][2]}'
        if self.input_type == 'vid':
            if self.lib == 'cv2':
                self.video = cv2.VideoCapture(real_path)
                width = int(self.video.get(3))
                height = int(self.video.get(4))
                fps = self.video.get(5)
                fourcc = self.video.get(6)
                frame_count = int(self.video.get(7))
                self.info_func = lambda: (self.video.get(0), int(self.video.get(1)), None)
                self.read_func = lambda: self.video.read()[1]
            elif self.lib == 'ffmpeg':
                command = ['ffmpeg', '-loglevel', 'error']
                if opt['decoder'] is not None:
                    command.extend(['-c:v', opt['decoder']])
                command.extend([
                    '-i', real_path,
                    *([] if (_ := opt['resize']) is None else ['-s', '%dx%d' % _]),
                    '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                    '-'
                ])
                self.pipe = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=10 ** 8)
                ffprobe_out = eval(subprocess.getoutput(
                    f"ffprobe -hide_banner -v quiet -select_streams v -show_entries "
                    'stream=height,width,r_frame_rate,nb_frames,codec_tag_string '
                    f"-print_format json '{real_path}'"))['streams'][0]
                if (_ := opt['resize']) is None:
                    width, height = map(ffprobe_out.get, ('width', 'height'))
                else:
                    width, height = _
                frame_count = int(ffprobe_out['nb_frames'])
                fps = eval(ffprobe_out['r_frame_rate'])
                fourcc = cv2.VideoWriter_fourcc(*ffprobe_out['codec_tag_string'])
                self.info_func = lambda: (
                    (1000 / fps) * (self.count - 1 if self.count > 1 else 0),
                    self.count, self.count == frame_count)
                self.read_func = lambda: numpy.frombuffer(
                    self.pipe.stdout.read(width * height * 3),
                    dtype='uint8'
                ).reshape((height, width, 3))
        elif self.input_type == 'img':
            self.files = utils.folder.listdir(real_path)
            frame_count = len(self.files)
            height, width = cv2.imread(os.path.join(
                real_path, self.files[0]
            )).shape[0:2]
            fps = 1
            fourcc = cv2.VideoWriter_fourcc(*'img2')
            if self.lib == 'cv2':
                self.info_func = lambda: (
                    (1000 / fps) * (self.count - 1 if self.count > 1 else 0),
                    self.count, self.count == frame_count)
                self.read_func = lambda: cv2.imread(os.path.join(
                    real_path, self.files[self.count - 1]
                ))
            elif self.lib == 'ffmpeg':
                if (_ := opt['resize']) is not None:
                    width, height = _
                command = ['ffmpeg', '-loglevel', 'error']
                if opt['decoder'] is not None:
                    command.extend(['-c:v', opt['decoder']])
                command.extend([
                    '-pattern_type', 'glob',
                    '-i', f'{real_path}/*{os.path.splitext(self.files[0])[1]}',
                    *([] if (_ := opt['resize']) is None else ['-s', '%dx%d' % _]),
                    '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                    '-'
                ])
                self.pipe = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=100000000)
                self.info_func = lambda: (
                    (1000 / fps) * (self.count - 1 if self.count > 1 else 0),
                    self.count, self.count == frame_count)
                self.read_func = lambda: numpy.frombuffer(
                    self.pipe.stdout.read(width * height * 3),
                    dtype='uint8'
                ).reshape((height, width, 3))

        # Public
        self.info = [
            *self.info_func(),  # Time in milliseconds, frame count, start/end
            width, height, fps, fourcc, frame_count,
            numpy.ndarray, numpy.uint8
        ]

    def get(self, idx):
        """
        Parameters
        ----------
        idx : int
        from 0 to 8
        0: Current time in milliseconds if framerate is available
        1: Current frame number
        2: Relative time: 0 if at the beginning, 1 at the end
        3: Height
        4: Width
        5: Framerate if available
        6: Fourcc code
        7: Total frame count
        8: Returning object type of read()
        9: Returning data type

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
            },
            'encoder': {
                'h264': 'mp4',
                'libx264': 'mp4',
                'libx264rgb': 'mp4',
                'hevc': 'mov',
                'libx265': 'mov',
            }
        }
    }

    def __init__(self, input_dir, output_path, opt, res, fps):
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
                    'ffmpeg',
                    # '-flush_packets', '1',
                    '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                    '-s', '%dx%d' % res, '-r', str(fps),
                    '-i', '-',
                    *([] if (_ := opt['resize']) is None else ['-s', '%dx%d' % _]),
                    *([] if (_ := opt['out_fps']) is None else ['-r', str(_)]),
                    '-c:v', opt['encoder'],
                    *([] if (_ := opt['pix_fmt']) is None else ['-pix_fmt', str(_)]),
                    *(['-tag:v', 'hvc1'] if opt['encoder'] in ('hevc', 'libx265') else []),
                    *([] if (_ := opt['crf']) is None else ['-crf', str(_)]),
                    *([] if (_ := [opt['ffmpeg-params'].split(' ')]) else _),
                    self.output_path
                ]
                self.pipe = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
                self.write_func = lambda frame: (self.pipe.stdin.write(frame.tobytes()), self.pipe.stdin.flush())
        elif self.type == 'img':
            self.output_path = utils.folder.check_dir_availability(output_path)
            self.ext = opt['ext']
            if self.lib == 'cv2':
                self.write_func = lambda frame: cv2.imwrite(f"{self.output_path}/{self.count}.{self.ext}", frame)
            elif self.lib == 'ffmpeg':
                command = [
                    'ffmpeg',
                    # '-flush_packets', '1',
                    '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                    '-s', '%dx%d' % res,
                    '-i', '-',
                    *([] if (_ := opt['resize']) is None else ['-s', '%dx%d' % _]),
                    *([] if (_ := [opt['ffmpeg-params'].split(' ')]) else _),
                    f'{self.output_path}/%d.{self.ext}'
                ]
                self.pipe = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
                self.write_func = lambda frame: (self.pipe.stdin.write(frame.tobytes()), self.pipe.stdin.flush())
        self.thread = threading.Thread()
        self.thread.start()

    def write(self, obj):
        self.thread.join()
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
        self.buff = numpy.empty(
            (2, *map(self.video.get, (4, 3)), 3),
            dtype=self.video.get(9)
        )
        self.buff = utils.tensor.Tensor(
            tensor=numpy.empty(
                (2, *map(self.video.get, (4, 3)), 3),
                dtype=self.video.get(9)),
            shape_order='fhwc', channel_order='bgr',
            range_=(0, 255)
        )
        self.count = 0
        # Thread
        self.get_thread = lambda: threading.Thread(target=self.get_frame_ready)
        self.thread = self.get_thread()
        self.thread.start()

    def get_frame(self, last):
        self.thread.join()
        if last:
            i = 1 if self.count else 0
            return self.buff[i: i+1]
        returning = self.buff[self.count:self.count+1]
        self.thread = self.get_thread()
        self.thread.start()
        return returning

    def get_frame_ready(self):
        self.count += -1 if self.count else 1
        self.buff[self.count] = self.video.read()

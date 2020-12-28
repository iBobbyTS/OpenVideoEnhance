from os import system, mkdir
from os.path import join, splitext
from subprocess import getoutput

from numpy import load, save, savez_compressed
from cv2 import imread, imwrite, resize, VideoCapture, VideoWriter, VideoWriter_fourcc
import torch

from utils.io_checker import check_dir_availability
from utils.io_utils import listdir


class DataLoader:
    sequence_read_funcs = {'img': imread,
                           'npz': lambda path: load(path)['arr_0'],
                           'npy': load,
                           'pmg': torch.load
                           }
    hwaccel_name = {'nvidia': 'cuvid',
                    'amd': 'amf',
                    'intel': 'qsv'
                    }

    def __init__(self, input_dir, input_type, reader,
                 resize, crop, frame_range,
                 ffmpeg_dir, hardware_decoder, out_fmt, temp_dir):
        assert not (crop and resize), "Crop and resize can't be on at the same time"
        if input_type == 'vid':
            if reader == 'cv2':
                # Internal usage
                self.cap = VideoCapture(input_dir)
                self.resize = resize
                self.crop = crop
                self.cap.set(0, frame_range[0])
                self.read_func = None
                # External usage
                self.fps = self.cap.get(5)
                self.frame_count = int(self.cap.get(7))
                if resize:
                    self.width, self.height = resize
                elif crop:
                    self.width, self.height = crop[1] - crop[0], crop[3] - crop[2]
                else:
                    self.width = int(self.cap.get(3))
                    self.height = int(self.cap.get(4))
            else:  # reader == 'ffmpeg':
                # Local usage
                mkdir(join(temp_dir, 'input_img'))
                cmd = [join(ffmpeg_dir, 'ffmpeg'), '-hide_banner', '-loglevel', 'error']
                ffprobe_out = eval(getoutput(
                    f"{join(ffmpeg_dir, 'ffprobe')} -hide_banner -v quiet -select_streams v -show_entries "
                    'stream=codec_name,height,width,r_frame_rate,nb_frames '
                    f"-print_format json '{input_dir}'"))['streams'][0]
                self.codec_name = ffprobe_out['codec_name']
                if hardware_decoder and hardware_decoder in self.hwaccel_name.keys():
                    hardware_decoder = f'{self.hwaccel_name[hardware_decoder]}' if hardware_decoder else ''
                    cmd.extend(['-c:v', f'{self.codec_name}_{hardware_decoder}'])
                cmd.extend(['-i', f"'{input_dir}'"])
                video_filter = []
                if frame_range[0] or frame_range[1]:
                    video_filter.append(f"select=between(n\\,{frame_range[0]}\\,{frame_range[1]})")
                if crop:
                    video_filter.append(f"crop={crop[1]-crop[0]}:{crop[3]-crop[2]}:{crop[0]}:{crop[2]}")
                if video_filter:
                    cmd.extend(['-filter_complex', f"'{';'.join(video_filter)}'"])
                if resize:
                    cmd.extend(['-s', f'{resize[0]}x{resize[1]}'])
                if out_fmt == 'tiff':
                    cmd.extend(['-pix_fmt', 'rgb24'])
                cmd.append(f"{join(temp_dir, 'input_img')}/%0{len(ffprobe_out['nb_frames'])}d.{out_fmt}")
                print(' '.join(cmd))
                system(' '.join(cmd))
                # Internal usage
                self.read_func = self.sequence_read_funcs['img']
                self.img_read_dir = join(temp_dir, 'input_img')
                self.files = listdir(join(temp_dir, 'input_img'))
                self.resize = False
                self.crop = False
                # External usage
                if resize:
                    self.width, self.height = resize
                elif crop:
                    self.width, self.height = crop[1] - crop[0], crop[3] - crop[2]
                else:
                    self.height = int(ffprobe_out['height'])
                    self.width = int(ffprobe_out['width'])
                self.fps = eval(ffprobe_out['r_frame_rate'])
                self.frame_count = int(ffprobe_out['nb_frames'])
        else:  # Input sequence
            # Internal usage
            self.img_read_dir = input_dir
            self.files = listdir(input_dir)
            self.read_func = self.sequence_read_funcs[input_type]
            # External usage
            self.fps = None
            self.frame_count = len(self.files)
            self.crop = crop
            self.resize = resize
            self.height, self.width = (resize[1], resize[0]) if resize else self.read_func(join(self.img_read_dir, self.files[0])).shape[0:2]
        self.count = 0

    def read(self, count=False):
        if self.read_func is None:
            if count:
                self.cap.set(1, count)
            img = self.cap.read()
            if not img[0]:
                return False
            else:
                img = img[1]
        else:
            if not count:
                count = self.count
            img = self.read_func(join(self.img_read_dir, self.files[count]))
            self.count = count + 1

        if self.crop:
            img = img[slice(*self.crop[2:4]), slice(*self.crop[0:2])]
        if self.resize:
            img = resize(img, self.resize)
        return img

    def close(self):
        if self.read_func is None:
            self.cap.release()


class DataWriter:
    write_funcs = {'tiff': lambda path, img: imwrite(path + '.tiff', img),
                   'png': lambda path, img: imwrite(path + '.png', img),
                   'jpg': lambda path, img: imwrite(path + '.jpg', img),
                   'npz': savez_compressed,
                   'npy': save,
                   'pmg': lambda path, img: torch.save(img, path + '.pmg')
                   }
    codec_vtags = {'hevc': 'hvc1',
                   'h264': 'avc1',
                   'vp9': 'vp90'
                   }
    hwaccel_name = {'nvidia': 'nvenc',
                    'apple': 'videotoolbox',
                    'amd': 'amf',
                    'intel': 'qsv'
                    }

    def __init__(self, input_dir, output_dir,
                 temp_dir, ext,
                 output_type, writer, fps, size, resize, codec,
                 ffmpeg_dir, hardware_encoder, mac_compatibility, extra_video_meta, crf,
                 complete):
        if output_type != 'vid' and not ext:
            if not splitext(output_dir)[1]:
                ext = splitext(input_dir)[1][1:]
            else:
                ext = splitext(output_dir)[1][1:]
        self.resize = resize
        output_dir = splitext(output_dir)[0]
        if output_type == 'vid':
            output_dir = check_dir_availability(output_dir, ext)
        else:
            output_dir = check_dir_availability(output_dir)
        self.write_with_cv2 = True if output_type == 'vid' and writer == 'cv2' else False
        self.run_cmd = True if output_type == 'vid' and writer == 'ffmpeg' else False
        self.count = 0
        if output_type == 'vid':
            vtag = self.codec_vtags[codec]
            if writer == 'cv2':
                self.writer = VideoWriter(output_dir, VideoWriter_fourcc(*vtag), fps, size)
                self.write_func = self.writer.write
            else:  # writer == 'ffmpeg'
                hardware_encoder = f'_{self.hwaccel_name[hardware_encoder]}' if hardware_encoder else ''
                cmd = [join(ffmpeg_dir, 'ffmpeg'), '-loglevel', 'error', '-vsync', '0',
                       '-r', fps, '-i', f"'{join(temp_dir, 'out_img')}/%d.{ext}'"]
                has_audio = eval(getoutput(f"'{join(ffmpeg_dir, 'ffprobe')}' -v quiet -show_streams "
                                           f"-select_streams a -print_format json "
                                           f"'{input_dir}'"))['streams']
                if has_audio and complete:
                    cmd.extend(['-vn', '-i', input_dir])
                cmd.extend(['-c:v', f"{codec}{hardware_encoder}", '-c:a', 'copy'])
                if mac_compatibility:
                    if 'hevc' in codec:
                        cmd.extend(['-tag:v', vtag])
                    cmd.extend(['-pix_fmt', 'yuv420p'])
                if extra_video_meta:
                    cmd.extend(['-color_primaries', '1', '-color_trc', '1', '-colorspace', '1'])
                if crf and isinstance(crf, int):
                    cmd.extend(['-crf', crf])
                cmd.append(f"'{output_dir}'")
                self.cmd = ' '.join(cmd)
                self.write_func = self.write_funcs[ext]
                self.write_path = join(temp_dir, 'input_img')
        else:
            self.write_func = self.write_funcs[ext]
            self.write_path = output_dir

    def write(self, img):
        if self.resize:
            img = resize(img, self.resize)
        if self.write_with_cv2:
            self.write_func(img)
        else:
            self.write_func(f"{self.write_path}/{self.count}", img)
            self.count += 1

    def close(self):
        if self.write_with_cv2:
            self.writer.release()
        if self.run_cmd:
            system(self.cmd)


class DataBuffer:
    def __init__(self, video: DataLoader, buffer_size=2):
        self.video = video
        self.buff = []
        self.buff_frame_index = {}
        self.buffer_size = buffer_size

    def get_frame(self, frame_index):
        if frame_index not in self.buff_frame_index.keys():
            if len(self.buff) > self.buffer_size:
                self.buff.pop(0)
                del self.buff_frame_index[tuple(self.buff_frame_index.keys())[0]]
            frame = self.video.read(frame_index)
            self.buff_frame_index[frame_index] = len(self.buff)
            self.buff.append(frame)
        return self.buff[self.buff_frame_index[frame_index]]

    def __del__(self):
        pass

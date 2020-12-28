from time import time

everything_start_time = time()

import os

from utils.io_checker import path2list, detect_input_type, check_dir_availability, check_model
from utils.io_rw_utils import DataLoader, DataWriter, DataBuffer
from utils.other import second2time, solve_start_end_frame
from utils import modules


def enhance(external_opt, input_opt, temp_opt, preprocess_opt, model_opt, postprocess_opt, output_opt):
    """
    external_opt
        ffmpeg_dir = None
    input_opt
        path = /Users/ibobby/Dataset/resolution_test/120p.mp4

    temp_opt
        path = tmp
        remove = False

    preprocess_opt
        reader= cv2  # choices: cv2, ffmpeg
        if reader==ffmpeg:
            hardware_decoder: None  # choices: cuvid, None
            out_fmt = [tiff, png, jpg]
        frame_range = (None, None): All, (0, 10): 0 to 9
        resize = None  # ex. (2688, 1512)
        crop = None  # ex. ((32, 1025), (0, 1081)) (width_slice, height_slice)


    model_opt
        to_do = [DAIN, EDVR]
        model_name = [DAIN_slowmotion, l4v]
        model_path = [None, None]
        sf = [2, None]
        batch_size = [5, 1]
        empty_cache = True  # choices: True, False

    postprocess_opt
        type = vid  # choices: vid, img, idf
        if type == vid:
            writer = cv2  # choice: cv2, ffmpeg
            codec = hevc  # choice: hevc, h264, vp9
            if codec != vp9:
                mac_compatibility = True  # or False
            if writer == ffmpeg:
                hardware_encoder = nvenc  # choice: None, videotoolbox
                extra_video_meta = True  # or False (-color_primaries 1 -color_trc 1 -colorspace 1)
                br = None  # ex. 10M
                crf = 20  # between 2 and 35
            ext = mov  # choice: mp4, mov, webm
        if type == img:
            ext = tiff  # choices: tiff, png, jpg
        if type == idf:
            ext = npz  # choices: npz, npy, pmg
            codec = uint8  # choices: uint8, float32
        resize = None  # ex. (3840, 2160)
        fps = 60  # or None

    output_opt
        path = None  # or /Users/ibobby/Dataset/resolution_test/120p_2.mp4
    """
    input_type = detect_input_type(input_opt['path'])
    if input_type != 'continue':
        input_file_name_list = path2list(input_opt['path'])
        temp_file_path = check_dir_availability(os.path.join(temp_opt['path'], input_file_name_list[1]))
        video = DataLoader(
            input_opt['path'], input_type, preprocess_opt['reader'],
            preprocess_opt['resize'], preprocess_opt['crop'], preprocess_opt['frame_range'],
            # FFmpeg ↓
            external_opt['ffmpeg_dir'], preprocess_opt['hardware_decoder'], preprocess_opt['out_fmt'],
            temp_file_path
        )
        frame_count = video.frame_count
        # fps and sf
        original_fps = video.fps if input_type == 'video' else postprocess_opt['fps']
        target_fps = original_fps
        for sf_, algorithm in zip(model_opt['coef'], model_opt['to_do']):
            if modules.belong_to(algorithm) == 'vfin':
                target_fps *= sf_ if sf_ and sf_ is not None else 1
        # codec
        if not postprocess_opt['codec'] and input_type == 'vid':
            postprocess_opt['codec'] = video.codec_name
        # Start/End frame
        start_frame, end_frame, copy = solve_start_end_frame(preprocess_opt['frame_range'], frame_count)
        saver = DataWriter(
            input_opt['path'], output_opt['path'],
            temp_file_path, postprocess_opt['ext'],
            postprocess_opt['type'], postprocess_opt['writer'], postprocess_opt['fps'],
            (video.width, video.height), postprocess_opt['resize'], postprocess_opt['codec'],
            external_opt['ffmpeg_dir'], postprocess_opt['hardware_encoder'],
            postprocess_opt['mac_compatibility'], postprocess_opt['extra_video_meta'],
            postprocess_opt['crf'], preprocess_opt['frame_range'] == [0, 0]
        )
        buffer = DataBuffer(video, buffer_size=preprocess_opt['buffer_size'])
        # Log
        cag = {}
        # Empty cache
        if model_opt['empty_cache']:
            os.environ['CUDA_EMPTY_CACHE'] = str(int(model_opt['empty_cache']))
        # Model checking
        check_model(model_opt['model_path'])
        # Setup models
        restorers = []
        for algorithm, model_path, coef, model_name, extra_args in zip(
                model_opt['to_do'], model_opt['model_path'], model_opt['coef'], model_opt['model_name'], model_opt['extra_args']):
            model = modules.get(algorithm)(
                height=video.height, width=video.width, model_dir=model_path, net_name=model_name,  # General
                coef=coef,  # VFIN
                rectify=extra_args['rectify'],  # DAIN
                temp_dir=temp_file_path  # DeOldify
            )
            model.init_batch(buffer)
            restorers.append(model)
        # Start
        timer = 0
        start_time = time()
        batch_count = end_frame - start_frame
        for i in range(start_frame, end_frame):
            frames = [buffer.get_frame(i)]
            # Inference
            for model in restorers:
                frames = model.rt(frames, duplicate=i+1 == end_frame)
            for frame in frames:
                saver.write(frame)
            # Time
            time_spent = time() - start_time
            start_time = time()
            if i == start_frame:
                initialize_time = time_spent
                print(f'Initialized and processed frame 1/{batch_count} | '
                      f'{batch_count - i - 1} frames left | '
                      f'Time spent: {round(initialize_time, 2)}s',
                      end='')
            else:
                timer += time_spent
                frames_processes = i + 1
                frames_left = batch_count - frames_processes
                print(f'\rProcessed batch {frames_processes}/{batch_count} | '
                      f"{frames_left} {'batches' if frames_left > 1 else 'batch'} left | "
                      f'Time spent: {round(time_spent, 2)}s | '
                      f'Time left: {second2time(frames_left * timer / i)} | '
                      f'Total time spend: {second2time(timer + initialize_time)}', end='', flush=True)
        video.close()
        saver.close()
        del buffer

from time import time

everything_start_time = time()

import os

from vrt import utils


def enhance(input_opt, temp_opt, preprocess_opt, model_opt, postprocess_opt, output_opt):
    """
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
        empty_cache = True  # choices: True, False

    postprocess_opt
        type = vid  # choices: vid, img, idf
        if type == vid:
            writer = cv2  # choice: cv2, ffmpeg
            codec = hevc  # choice: hevc, h264, vp9
            mac_compatibility = True  # or False
            if writer == ffmpeg:
                hardware_encoder = nvenc  # choice: None, videotoolbox
                extra_video_meta = True  # or False (-color_primaries 1 -color_trc 1 -colorspace 1)
                crf = 20  # between 2 and 35
            ext = mov  # choice: mp4, mov, webm
        if type == img:
            ext = tiff  # choices: tiff, png, jpg
        if type == idf:
            ext = npz  # choices: npz, npy, pmg
            codec = uint8  # choices: uint8, float32
        fps = 60  # or None

    output_opt
        path = None  # or /Users/ibobby/Dataset/resolution_test/120p_2.mp4
    """
    inputs = utils.io.solve_input(input_opt['path'])
    for solved_input in inputs:
        # Create temporary folder
        temp_path = os.path.join((temp_opt['path']), solved_input[1][1])
        utils.folder.check_dir_availability(temp_path)
        # Load video
        video = utils.data_processor.DataLoader(
            video_input=solved_input, opt=preprocess_opt,
            channel_order=utils.dictionaries.model_channel_order[model_opt['to_do'][0]]
        )
        # Initialize buffer
        buffer = utils.data_processor.DataBuffer(video)

        # Solve for start/end frame
        start, end, copy = utils.io.solve_start_end_frame(
            preprocess_opt['frame_range'], video.get(7)
        )

        # Empty Cache
        os.environ['CUDA_EMPTY_CACHE'] = '1'

        # Initialize restorers
        restorers = []
        width, height, fps = map(video.get, (3, 4, 5))
        for i, (to_do, model_path, args, kwargs) in enumerate(zip(*map(
            model_opt.get, ('to_do', 'model_path', 'args', 'kwargs')
        ))):
            rter = utils.algorithm.get(to_do)(
                height=height, width=width,
                model_path=model_path, default_model_dir=model_opt['default_model_dir'],
                *args, **kwargs
            )
            output_effect = rter.get_output_effect()
            height *= output_effect['height']
            width *= output_effect['width']
            fps *= output_effect['fps']
            restorers.append(rter)
        # Solve for fps
        if (fps_ := postprocess_opt['in_fps']) is not None:
            fps = fps_
        # Initialize saver
        saver = utils.data_processor.DataWriter(
            input_dir=input_opt['path'], output_path=output_opt['path'],
            opt=postprocess_opt, fps=fps, res=(width, height),
            channel_order=utils.dictionaries.model_channel_order[model_opt['to_do'][-1]]
        )
        channel_order = utils.dictionaries.model_channel_order[model_opt['to_do'][-1]]
        # Start processing
        timer = utils.io.Timer(end - start)
        x = []
        for i in range(start, end):
            frames = buffer.get_frame(last=(i+1 == end))
            # Inference
            for model in restorers:
                frames = model.rt(frames, last=(i+1 == end))
                if frames:
                    frames.detach()
            # Save
            x.append(len(frames))
            if frames:
                for frame in frames.convert(
                    place='numpy', dtype='uint8', shape_order='fhwc', channel_order=channel_order, range_=(0.0, 255.0)
                ):
                    saver.write(frame)
            # Show progress
            timer.print()
        video.close()
        del buffer
        saver.close()

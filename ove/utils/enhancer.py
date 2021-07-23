from time import time
everything_start_time = time()
import os
from ove import utils


def enhance(
    global_opt, input_opt, temp_opt, preprocess_opt, model_opt, postprocess_opt, output_opt
):
    solved_input = utils.io.solve_input(input_opt['path'])
    # Create temporary folder
    temp_path = os.path.abspath(os.path.join(
        (temp_opt['path']), solved_input[1][1]
    ))
    utils.folder.check_dir_availability(temp_path)
    # Get frame need to process before and after
    before, after = utils.io.solve_before_after_frame(model_opt)
    # Load video
    video = utils.data_processor.DataLoader(
        video_input=solved_input, opt=preprocess_opt,
        channel_order=utils.dictionaries.model_channel_order[model_opt['to_do'][0]],
        global_opt=global_opt, frames_before=before
    )
    # Initialize buffer
    buffer = utils.data_processor.DataBuffer(video)

    # Solve for start/end frame
    start, end, before, after = utils.io.solve_start_end_frame(
        preprocess_opt['frame_range'], video.get(3), before, after
    )

    # Empty Cache
    os.environ['CUDA_EMPTY_CACHE'] = '1'

    # Set cuDNN
    device = utils.modeling.set_gpu_status(model_opt)
    # Initialize restorers
    restorers = []
    width, height, fps = map(video.get, (0, 1, 2))
    for i, (to_do, model_path, kwargs) in enumerate(zip(*map(
        model_opt.get, ('to_do', 'model_path', 'kwargs')
    ))):
        rter = utils.algorithm.get(to_do)(
            height=height, width=width, device=device,
            model_path=model_path, default_model_dir=model_opt['default_model_dir'],
            temp_path=temp_path,
            **kwargs
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
        channel_order=utils.dictionaries.model_channel_order[model_opt['to_do'][-1]],
        ffmpeg_bin_path=global_opt['ffmpeg_bin_path']
    )
    channel_order = utils.dictionaries.model_channel_order[model_opt['to_do'][-1]] if postprocess_opt['lib'] == 'ffmpeg' else 'bgr'
    # Start processing
    timer = utils.io.Timer(end + after - start + before)
    for i in range(start-before, end+after):
        frames = buffer.get_frame(last=(i+1 == end+after))
        # Inference
        for model in restorers:
            frames = model.rt(frames, last=(i+1 == end+after))
            if frames:
                frames.clamp()
        # Save
        if frames and i <= end:
            for frame in frames.convert(
                place='numpy', dtype=saver.dtype, shape_order='fhwc', channel_order=channel_order, range_=saver.range_
            ):
                saver.write(frame)
        del frames
        # Show progress
        timer.print()
    video.close()
    del buffer
    saver.close()

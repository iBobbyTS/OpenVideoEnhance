import os
import math

from ove import utils


def run(
    global_opt, input_opt, temp_opt, preprocess_opt, model_opt, postprocess_opt, output_opt
):
    file = open(input_opt['path'], 'r+')
    process_bins = eval(file.read())
    for process_bin in process_bins:
        if process_bin['state'] != 0:
            """
            0: Not started yet
            1: Processing
            2: Done
            """
            continue
        else:
            enhance(global_opt, input_opt, temp_opt, preprocess_opt, model_opt, postprocess_opt, output_opt)


def enhance(
    global_opt, input_opt, temp_opt, preprocess_opt, model_opt, postprocess_opt, output_opt
):
    pass


def allocate(
    global_opt, input_opt, temp_opt, preprocess_opt, model_opt, postprocess_opt, output_opt
):
    solved_input = utils.io.solve_input(input_opt['path'])
    if solved_input[0] == 'continue':
        run(global_opt, input_opt, temp_opt, preprocess_opt, model_opt, postprocess_opt, output_opt)
    before, after = utils.io.solve_before_after_frame(model_opt)
    video = utils.data_processor.DataLoader(
        video_input=solved_input, opt=preprocess_opt,
        channel_order=utils.dictionaries.model_channel_order[model_opt['to_do'][0]],
        global_opt=global_opt, frames_before=before
    )
    start, end = utils.io.solve_start_end_frame(
        preprocess_opt['frame_range'], video.get(3)
    )
    start = max(0, start - before)
    end = min(end + after, video.get(3))
    process_batch_count = math.ceil((end - start) / global_opt['process_batch_size'])

    run(global_opt, input_opt, temp_opt, preprocess_opt, model_opt, postprocess_opt, output_opt)

def second2time(second):
    m, s = divmod(second, 60)
    h, m = divmod(m, 60)
    t = '%d:%02d:%05.2f' % (h, m, s)
    return t


def solve_start_end_frame(frame_range, frame_count):
    start_frame, end_frame = frame_range
    if end_frame == 0 or end_frame >= frame_count:
        copy = True
        end_frame = frame_count
    else:
        copy = False
    if start_frame == 0 or start_frame >= frame_count:
        start_frame = 0
    return start_frame, end_frame, copy
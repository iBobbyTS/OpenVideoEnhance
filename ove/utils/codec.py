def get_extension(codec):
    if ('avc' or '264') in codec:
        ext = 'mp4'
    elif ('hvc' or 'hev' or '265') in codec:
        ext = 'mov'
    elif 'prores' in codec:
        ext = 'mov'
    elif ('av1' or 'av01') in codec:
        ext = 'mp4'
    elif 'vp' in codec:
        ext = 'webm'
    else:
        ext = 'mp4'
    return ext

model_paths = {
    'ssm': 'SSM/official.pth',
    'dain': 'DAIN/official.pth',
    'bmbc': 'BMBC/official.pth',
    'esrgan': {
        'pd': 'ESRGAN/PSNR_SRx4_DF2K.pth',
        'dk': 'ESRGAN/SRx4_DF2KOST.pth',
        'r': 'ESRGAN/ESRGAN_x4.pth',
        'p': 'ESRGAN/PSNR_x4.pth',
    },
    'edvr': {
        'ld': 'EDVR/official_L_deblur_REDS.pth',
        'ldc': 'EDVR/official_L_deblurcomp_REDS.pth',
        'l4r': 'EDVR/official_L_x4_SR_REDS.pth',
        'l4v': 'EDVR/official_L_x4_SR_Vimeo90K.pth',
        'l4br': 'EDVR/official_L_x4_SRblur_REDS.pth',
        'm4r': 'EDVR/official_M_woTSA_x4_SR_REDS.pth',
        'mt4r': 'EDVR/official_M_x4_SR_REDS.pth'
    },
    'bopbl': {
        'common': 'BOPBL/official_common.pth',
        'with_scratch': 'BOPBL/official_with_scratch.pth',
        'no_scratch': 'BOPBL/official_no_scratch.pth',
        'face_detect': 'BOPBL/official_face_detect.dat'
    },
    'deoldify': {
        'a': 'DeOldify/official_artistic.pth',
        's': 'DeOldify/official_stable.pth',
        'v': 'DeOldify/official_video.pth',
    }
}

model_configs = {
    'edvr': {  # 'model_name': [[model_args], num_of_frame, enlarge_factor:ef, multiple, {vram:max_res}]
        'ld': [{'num_feat': 128, 'num_reconstruct_block': 40, 'hr_in': True, 'with_predeblur': True}, 5, 1, 16, {16: 992}],
        'ldc': [{'num_feat': 128, 'num_reconstruct_block': 40, 'hr_in': True, 'with_predeblur': True}, 5, 1, 16, {16: 992}],
        'l4r': [{'num_feat': 128, 'num_reconstruct_block': 40}, 5, 4, 4, {16: 288}],
        'l4v': [{'num_feat': 128, 'num_reconstruct_block': 40, 'num_frame': 7}, 7, 4, 4, {16: 388}],
        'l4br': [{'num_feat': 128, 'num_reconstruct_block': 40, 'with_predeblur': True}, 5, 4, 4, {16: 256}],
        'm4r': [{'with_tsa': False}, 5, 4, 4, {16: 416}],
        'mt4r': [{}, 5, 4, 4, {16: 416}]
    }
}

model_channel_order = {
    'ssm': 'rgb',
    'dain': 'rgb',
    'bmbc': 'rgb',
    'esrgan': 'rgb',
    'edvr': 'rgb',
    'bopbl': 'rgb',
    'deoldify': 'rgb'
}

model_extra_frames = {
    'ssm': (0, 2),
    'dain': (0, 2),
    'bmbc': (0, 2),
    'esrgan': (0, 0),
    'edvr': (3, 3),
    'bopbl': (0, 0),
    'deoldify': (0, 0)
}

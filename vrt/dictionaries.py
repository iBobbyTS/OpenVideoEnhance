import vfin

__all__ = (
    'model_paths'
)

model_paths = {
    'ssm': 'SSM/official.pth',
    'dain': 'DAIN/official.pth',
    'esrgan': {
        'pd': 'ESRGAN/PSNR_SRx4_DF2K.pth',
        'dk': 'ESRGAN/SRx4_DF2KOST.pth',
        'r': 'ESRGAN/ESRGAN_x4.pth',
        'ro': 'ESRGAN/ESRGAN_x4_old_arch.pth',
        'p': 'ESRGAN/PSNR_x4.pth',
        'po': 'ESRGAN/PSNR_x4_old_arch.pth'
    },
    'edvr': {  # 'model_name': [[model_args], num_of_frame, enlarge_factor:ef, multiple, {vram:max_res}, model_dir]
        'ld': [{'num_feat': 128, 'num_reconstruct_block': 40, 'hr_in': True, 'with_predeblur': True}, 5, 1, 16, {16: 994}, 'EDVR/official_L_deblur_REDS.pth'],
        'ldc': [{'num_feat': 128, 'num_reconstruct_block': 40, 'hr_in': True, 'with_predeblur': True}, 5, 1, 16, {16: 994}, 'EDVR/official_L_deblurcomp_REDS.pth'],
        'l4r': [{'num_feat': 128, 'num_reconstruct_block': 40}, 5, 4, 4, {16: 288}, 'EDVR/official_L_x4_SR_REDS.pth'],
        'l4v': [{'num_feat': 128, 'num_reconstruct_block': 40, 'num_frame': 7}, 7, 4, 4, {16: 256}, 'EDVR/official_L_x4_SR_Vimeo90K.pth'],
        'l4br': [{'num_feat': 128, 'num_reconstruct_block': 40, 'with_predeblur': True}, 5, 4, 4, {16: 256}, 'EDVR/official_L_x4_SRblur_REDS.pth'],
        'm4r': [{'with_tsa': False}, 5, 4, 4, {16: 416}, 'EDVR/official_M_woTSA_x4_SR_REDS.pth'],
        'mt4r': [{}, 5, 4, 4, {16: 416}, 'EDVR/official_M_x4_SR_REDS.pth']
    }
}
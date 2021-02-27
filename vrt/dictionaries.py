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
    }
}

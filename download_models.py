import os
from ove.utils.download import download_file_from_google_drive


class Downloader:
    def __init__(
        self,
        download_model=True, model_path='../model_weights'
    ):
        self.opt = {
            'download_model': download_model,
            'model_path': os.path.abspath(model_path)
        }
        os.makedirs(self.opt['model_path'], exist_ok=True)

    def resolve_options(self, opt):
        return self.opt | opt

    # Algorithms
    def BasicSR(self, **kwargs):
        opt = self.resolve_options(kwargs)
        links = {
            'EDVR': [
                ('1LGhWdzAIu818_IDptIUBGCBJoE11jQLk', 'official_L_deblur_REDS.pth'),
                ('1eEWNZCCL17cf-G4yKF65rjV8Yy4eXnwM', 'official_L_deblurcomp_REDS.pth'),
                ('1C6tFY8CjjLaGqpPddWRrgqRThNVd9DZD', 'official_L_x4_SR_REDS.pth'),
                ('1ehwhFsVG8WCJ5tTfJRCpYzexPrB-ru5e', 'official_L_x4_SR_Vimeo90K.pth'),
                ('1WUwcPvp6rHrgxgfUtByfoosVUZ7w0i-N', 'official_L_x4_SRblur_REDS.pth'),
                ('1ddnMOCu87T_WbUFNvY0yihs44cViHvoY', 'official_M_woTSA_x4_SR_REDS.pth'),
                ('1scZpjI0iMRXdNSklR5j5Ei3mXbzIES9r', 'official_M_x4_SR_REDS.pth')
            ],
            'esrgan': [
                ('1ZZUHpIHdK2WijNiiV_QyFooJV9SuEgk1', 'official_ESRGAN_x4_old_arch.pth'),
                ('1AIyRcdAHj4l-pwTfUHaSoOgy2L2uuphN', 'official_ESRGAN_x4.pth'),
                ('1r9CEwpWaBQvFjuEJk7J8rDP9cnuOwhgK', 'official_PSNR_SRx4_DF2K.pth'),
                ('1l48p8GCErCrg_p3zFBNCjJ7Jc21eP-vb', 'official_PSNR_x4_old_arch.pth'),
                ('1SWZDffT4iZJ3ufsPBSbIRcPCcTGbM3vw', 'official_PSNR_x4.pth'),
                ('1qSSyzbxnnRgH11DGEXpcrSma2fCLRXfK', 'official_SR_x4_DF2KOST.pth')
            ]
        }
        # Resolve download_model
        opt['download_model'] = links.keys() if opt['download_model'] == True else []
        for a in opt['download_model']:
            if a in links.keys():
                os.makedirs(f'{opt["model_path"]}/{a}', exist_ok=True)
                for link in links[a]:
                    download_file_from_google_drive(link[0], f'{opt["model_path"]}/{a}/{link[1]}')

    def SSM(self, **kwargs):
        opt = self.resolve_options(kwargs)
        os.makedirs(f'{opt["model_path"]}/SSM', exist_ok=True)
        download_file_from_google_drive('10cOGtYTheDg2rF3geLtOUYvYyMjfCUct', f'{opt["model_path"]}/SSM/official.pth')

    def DAIN(self, cc=None, **kwargs):
        opt = self.resolve_options(kwargs)
        os.makedirs(f'{opt["model_path"]}/DAIN', exist_ok=True)
        download_file_from_google_drive('1r-gVVu6oxCSZyBij4d4tPtssifGZlG5X', f'{opt["model_path"]}/DAIN/dain_app_experimental.pth')
        download_file_from_google_drive('1vxRb52qyJt3J_AJzzA1LiEdfPEyf9bXf', f'{opt["model_path"]}/DAIN/official.pth')

    def DeOldify(self, **kwargs):
        opt = self.resolve_options(kwargs)
        download_file_from_google_drive()

    def BOPBL(self, **kwargs):
        opt = self.resolve_options(kwargs)
        download_file_from_google_drive()

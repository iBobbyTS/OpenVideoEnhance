import os
import sys
import json
import shutil


class Build:
    def __init__(
            self,
            build_type='develop', rebuild=False,
            download_model=True, model_path='../model_weights',
            develop=False,
            gitee=False
    ):
        """
        build_type: what goes after python setup.py
        download_model: weather you wantt o downloadt he model
        model_path: where you want to store the downloaded models
        develop: weather you want to install packages for jobs other than inferencing
        """
        self.python_executable = sys.executable
        self.cwd = os.path.split(__file__)[0]
        os.chdir(self.cwd)
        self.opt = {
            'build_type': build_type,
            'develop': develop,
            'download_model': download_model,
            'model_path': os.path.abspath(model_path),
            'gitee': gitee,
            'rebuild': rebuild
        }
        # Make sure building tools are there
        self.pip_install(['pip', 'setuptools', 'wheel'])
        # Packages that all algorithms will use
        self.pip_install(['numpy', 'opencv-python', 'Pillow', 'torch', 'torchvision'])
        # Create folder for model
        if download_model:
            os.makedirs(self.opt['model_path'], exist_ok=True)

    @staticmethod  # Internal
    def terms_to_delete(path: str):
        return ['dist', 'build', [fil for fil in os.listdir(path) if fil[-9:] == '.egg-info'][0]]

    # Internal
    def pip_install(self, libs: list):
        if isinstance(libs, str):
            libs = [libs]
        if libs:
            os.system(f"{self.python_executable} -m pip install -U {' '.join(libs)}")

    def download_from_google_drive(self, link, save_dir):
        try:
            from utils.download import download_file_from_google_drive
        except ModuleNotFoundError:
            self.pip_install(['requests', 'tqdm'])
            from utils.download import download_file_from_google_drive
        download_file_from_google_drive(link, save_dir)

    # Algorithms
    def BasicSR(self, cuda_extensions=None, **kwargs):
        # Resolve options
        opt = self.opt
        opt.update(kwargs)
        if cuda_extensions is None:
            import torch
            cuda_extensions = True if torch.cuda.is_available() else False
        # Install packages
        self.pip_install(['addict', 'future', 'lmdb', 'pyyaml', 'requests', 'scikit-image', 'scipy', 'tb-nightly', 'tqdm', 'yapf'])
        if opt['develop']:
            pass
        os.chdir('../third_party')
        if os.path.exists('BasicSR') and opt['rebuild']:
            shutil.rmtree('BasicSR')
        os.system(f"git clone https://{'gitee' if opt['gitee'] else 'github'}.com/xinntao/BasicSR.git")
        os.chdir('BasicSR')
        os.system(f"{self.python_executable} setup.py {opt['build_type']}{'' if cuda_extensions else ' --no_cuda_ext'}")
        # Download models
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
                    self.download_from_google_drive(link[0], f'{opt["model_path"]}/{a}/{link[1]}')
        os.chdir(self.cwd)

    def SSM(self, **kwargs):
        # Resolve options
        opt = self.opt
        opt.update(kwargs)
        if opt['develop']:
            self.pip_install(['click', 'tensorboardX'])
        if opt['download_model']:
            os.makedirs(f'{opt["model_path"]}/SSM', exist_ok=True)
            self.download_from_google_drive('10cOGtYTheDg2rF3geLtOUYvYyMjfCUct', f'{opt["model_path"]}/SSM/official.pth')

    def DAIN_all_in_one(self, cc=None, **kwargs):
        # Resolve options
        opt = self.opt
        opt.update(kwargs)
        if cc is None:
            import torch
            cc = ['%d%d' % torch.cuda.get_device_capability()]
        if opt['develop']:
            self.pip_install(['bisect'])
        os.chdir('vfin/dain')
        # Write compiler args
        nvcc_args = []
        for cc_ in cc:
            nvcc_args.append('-gencode')
            nvcc_args.append(f'arch=compute_{cc_},code=sm_{cc_}')
        nvcc_args.append('-w')
        with open('compiler_args.json', 'w') as f:
            json.dump({
                'develop': opt['develop'],
                'extra_compile_args': {'nvcc': nvcc_args, 'cxx': ['-std=c++14', '-w']}
            }, f)
        print(f'Compiling for compute compatibility {cc}')
        # Compile
        # DAIN's package
        os.system(f'{self.python_executable} setup.py {opt["build_type"]}')
        """
        os.chdir('my_package')
        packages = ['DepthFlowProjection', 'FilterInterpolation', 'FlowProjection']
        if opt['develop']:
            packages.extend(['InterpolationCh', 'SeparableConv', 'SeparableConvFlow', 'MinDepthFlowProjection', 'Interpolation'])
        for folder in packages:
            os.chdir(f"{'' if folder == packages[0] else '../'}{folder}")
            os.system(f'{self.python_executable} setup.py {opt["build_type"]}')
            if opt['build_type'] == 'install':
                for file_to_delete in self.terms_to_delete('.'):
                    shutil.rmtree(file_to_delete)
        # PWCNet
        os.chdir('../../PWCNet/correlation_package_pytorch1_0')
        os.system(f'{self.python_executable} setup.py {opt["build_type"]}')
        if opt['build_type'] == 'install':
            for file_to_delete in self.terms_to_delete('.'):
                shutil.rmtree(file_to_delete)
        os.chdir('../..')
        """
        if opt['build_type'] == 'install':
            for file_to_delete in self.terms_to_delete('.'):
                shutil.rmtree(file_to_delete)
        os.remove('compiler_args.json')
        os.chdir(self.cwd)
        # Download model
        if opt['download_model']:
            os.makedirs(f'{opt["model_path"]}/DAIN', exist_ok=True)
            self.download_from_google_drive('1r-gVVu6oxCSZyBij4d4tPtssifGZlG5X', f'{opt["model_path"]}/DAIN/dain_app_experimental.pth')
            self.download_from_google_drive('1vxRb52qyJt3J_AJzzA1LiEdfPEyf9bXf', f'{opt["model_path"]}/DAIN/official.pth')

    def DAIN(self, cc=None, **kwargs):
        # Resolve options
        opt = self.opt
        opt.update(kwargs)
        if cc is None:
            import torch
            cc = ['%d%d' % torch.cuda.get_device_capability()]
        if opt['develop']:
            self.pip_install(['bisect'])
        os.chdir('vfin/dain')
        # Write compiler args
        nvcc_args = []
        for cc_ in cc:
            nvcc_args.append('-gencode')
            nvcc_args.append(f'arch=compute_{cc_},code=sm_{cc_}')
        nvcc_args.append('-w')
        with open('compiler_args.json', 'w') as f:
            json.dump({'nvcc': nvcc_args, 'cxx': ['-std=c++14', '-w']}, f)
        print(f'Compiling for compute compatibility {cc}')
        # Compile
        # DAIN's package
        os.chdir('my_package')
        packages = ['DepthFlowProjection', 'FilterInterpolation', 'FlowProjection']
        if opt['develop']:
            packages.extend(['InterpolationCh', 'SeparableConv', 'SeparableConvFlow', 'MinDepthFlowProjection', 'Interpolation'])
        for folder in packages:
            os.chdir(f"{'' if folder == packages[0] else '../'}{folder}")
            os.system(f'{self.python_executable} setup.py {opt["build_type"]}')
            if opt['build_type'] == 'install':
                for file_to_delete in self.terms_to_delete('.'):
                    shutil.rmtree(file_to_delete)
        # PWCNet
        os.chdir('../../PWCNet/correlation_package_pytorch1_0')
        os.system(f'{self.python_executable} setup.py {opt["build_type"]}')
        if opt['build_type'] == 'install':
            for file_to_delete in self.terms_to_delete('.'):
                shutil.rmtree(file_to_delete)
        os.chdir('../..')
        os.remove('compiler_args.json')
        os.chdir(self.cwd)
        # Download model
        if opt['download_model']:
            os.makedirs(f'{opt["model_path"]}/DAIN', exist_ok=True)
            self.download_from_google_drive('1r-gVVu6oxCSZyBij4d4tPtssifGZlG5X', f'{opt["model_path"]}/DAIN/dain_app_experimental.pth')
            self.download_from_google_drive('1vxRb52qyJt3J_AJzzA1LiEdfPEyf9bXf', f'{opt["model_path"]}/DAIN/official.pth')

    def DeOldify(self, download_model=True):
        os.chdir('plugins')
        os.system(f'{self.python_executable} -m pip install ')
        if download_model:
            pass

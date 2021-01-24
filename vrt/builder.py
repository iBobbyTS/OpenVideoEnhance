import os
import sys
import json
import shutil

from utils.download_util import download_file_from_google_drive


class Build:
    def __init__(self, build_type='develop', model_path='../model_weights', develop=False):
        self.python_executable = sys.executable
        self.cwd = os.getcwd()
        self.build_type = build_type
        self.model_path = os.path.abspath(model_path)
        self.develop = develop
        # Make sure building tools are there
        self.pip_install(['-U', 'pip', 'setuptools', 'wheel'])
        # Packages that all algorithms will use
        self.pip_install(['numpy', 'opencv-python', 'Pillow', 'torch', 'torchvision'])
        # Create folder for model
        os.makedirs(self.model_path, exist_ok=True)

    @staticmethod  # Internal
    def terms_to_delete(path: str):
        return ['dist', 'build', [fil for fil in os.listdir(path) if fil[-9:] == '.egg-info'][0]]

    # Internal
    def pip_install(self, libs: list):
        if isinstance(libs, str):
            libs = [libs]
        if libs:
            os.system(f"{self.python_executable} -m pip install {' '.join(libs)}")

    # Algorithms
    def BasicSR(self, build=True, download_model=None, cuda_extensions=True, build_type=None, develop=None):
        if build_type is None:
            build_type = self.build_type
        if download_model is None:
            if not download_model:
                download_model = []
            else:
                download_model = links.keys()
        if develop is None:
            develop = self.develop
        if develop:
            pass
        os.chdir('../third_party')
        if build:
            os.system('git clone https://github.com/xinntao/BasicSR.git')
            os.chdir('BasicSR')
            self.pip_install(['addict', 'future', 'lmdb', 'pyyaml', 'requests', 'scikit-image', 'scipy', 'tb-nightly', 'tqdm', 'yapf'])
            os.system(f'{self.python_executable} setup.py {self.build_type}' + '' if cuda_extensions else ' --no_cuda_ext')
        else:
            os.chdir('BasicSR')
        # Download models
        links = {
            'EDVR': [
                ('1-OeHaUQKJbNRC-U0GZb98Y-UhioGWonO', 'official_L_deblur_REDS.pth'),
                ('1-PTtKIkWOpKNKV1D2I_OYjTlBByD2E3i', 'official_L_deblurcomp_REDS.pth'),
                ('1-7yBQj3U5oV9-oT9IhuDXrIbA3ABY6hr', 'official_L_x4_SR_REDS.pth'),
                ('1-Em9TYAu-v6U-eeELTAQOXNs6E0gduy8', 'official_L_x4_SR_Vimeo90K.pth'),
                ('1-OKoqv1e_YQJOEV3AYeotraS7SsFRijD', 'official_L_x4_SRblur_REDS.pth'),
                ('1-FDVYUDQkAqkQz0LyFhlOciQn-o6zJdG', 'official_M_woTSA_x4_SR_REDS.pth'),
                ('1-KDkMMB-4i4AS9kVCDF0ppFS3EuDKHOM', 'official_M_x4_SR_REDS.pth')
            ],
            'ESRGAN': [
                ('1ZZUHpIHdK2WijNiiV_QyFooJV9SuEgk1', 'official_ESRGAN_x4_old_arch.pth'),
                ('1AIyRcdAHj4l-pwTfUHaSoOgy2L2uuphN', 'official_ESRGAN_x4.pth'),
                ('1r9CEwpWaBQvFjuEJk7J8rDP9cnuOwhgK', 'official_PSNR_SRx4_DF2K.pth'),
                ('1l48p8GCErCrg_p3zFBNCjJ7Jc21eP-vb', 'official_PSNR_x4_old_arch.pth'),
                ('1SWZDffT4iZJ3ufsPBSbIRcPCcTGbM3vw', 'official_PSNR_x4.pth'),
                ('1qSSyzbxnnRgH11DGEXpcrSma2fCLRXfK', 'official_SR_x4_DF2KOST.pth')
            ]
        }
        for a in download_model:
            if a in links.keys():
                os.makedirs(f'{self.model_path}/{a}')
                for link in links[a]:
                    download_file_from_google_drive(link[0], f'{self.model_path}/{a}/{link[1]}')
        os.chdir(self.cwd)

    @staticmethod
    def SSM(download_model=True, develop=False):
        if develop is None:
            develop = self.develop
        if develop:
            pip_install(['click', 'tensorboardX'])
        if download_model:
            os.makedirs(f'{self.model_path}/SSM', exist_ok=True)
            download_file_from_google_drive('1WTb1C6IAICq5DBlmTSwJ5zQ8AcOGj_vR', f'{self.model_path}/SSM/official.pth')

    def DAIN(self, cc=None, download_model=True, build_all=False, build_type=None, develop=None):
        if cc is None:
            import torch
            cc = ['%d%d' % torch.cuda.get_device_capability()]
        if build_type is None:
            build_type = self.build_type
        if develop is None:
            develop = self.develop
        if develop:
            pip_install(['bisect'])
        os.chdir('vfin/dain')
        # Write compiler args
        nvcc_args = []
        for cc_ in cc:
            nvcc_args.append('-gencode')
            nvcc_args.append(f'arch=compute_{cc_},code=sm_{cc_}')
        nvcc_args.append('-w')
        with open('compiler_args.json', 'w') as f:
            json.dump({'nvcc': nvcc_args, 'cxx': ['-std=c++14', '-w']}, f)
        print(f'Compiling for compute compatilility {cc}')
        # Compile
        # DAIN's package
        os.chdir('my_package')
        packages = ['DepthFlowProjection', 'FilterInterpolation', 'FlowProjection']
        if build_all:
            packages.extend(['InterpolationCh', 'SeparableConv', 'SeparableConvFlow', 'MinDepthFlowProjection', 'Interpolation'])
        for folder in packages:
            os.chdir(f"{'' if folder == packages[0] else '../'}{folder}")
            os.system(f'{self.python_executable} setup.py {build_type}')
            if build_type == 'install':
                for file_to_delete in self.terms_to_delete('.'):
                    shutil.rmtree(file_to_delete)
        # PWCNet
        os.chdir('../../PWCNet/correlation_package_pytorch1_0')
        os.system(f'{self.python_executable} setup.py {build_type}')
        if build_type == 'install':
            for file_to_delete in self.terms_to_delete('.'):
                shutil.rmtree(file_to_delete)
        os.chdir('../..')
        os.remove('compiler_args.json')
        os.chdir(self.cwd)
        if download_model:
            os.makedirs(f'{self.model_path}/DAIN', exist_ok=True)
            download_file_from_google_drive('18_0UwKWnXT3fHw81eZZO0V1arXCAIG3X', '{self.model_path}/DAIN/experimental.pth')
            download_file_from_google_drive('10X1XS6E0eUK5hf-5P6qHgWI-U-heVxCk', '{self.model_path}/DAIN/best.pth')

    def DeOldify(self, download_model=True):
        os.chdir('plugins')
        os.system(f'{self.python_executable} -m pip install ')
        if download_model:
            pass

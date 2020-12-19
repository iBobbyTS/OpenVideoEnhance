import os
import sys
import json
import shutil

from utils.download_util import download_file_from_google_drive


class Build:
    def __init__(self, check_pytorch_version=True, build_type='develop'):
        self.python_executable = sys.executable
        self.cwd = os.getcwd()
        self.build_type = build_type

        if check_pytorch_version:
            self.check_pytorch_version_func(self.python_executable)

    @staticmethod  # Internal
    def terms_to_delete(path: str):
        return ['dist', 'build', [fil for fil in os.listdir(path) if fil[-9:] == '.egg-info'][0]]

    @staticmethod  # Internal
    def check_pytorch_version_func(python_executable: str):
        import torch
        torch_version = torch.__version__
        torch_version_split = torch_version.split('.')
        prefix = 'You need torch>=1.3.0, <=1.4.0, you have torch=='
        if torch_version_split[0] == '0' or (torch_version_split[0] == '1' and torch_version_split[1] <= '3'):
            raise RuntimeError(prefix + torch_version + ' < 1.3.0')
        elif int(torch_version_split[0]) > 1 or int(torch_version_split[1]) > 4:
            raise RuntimeError(prefix + torch_version + ' > 1.4.0')
        print(f'Building CUDAExtension for PyTorch in {python_executable}')

    def BasicSR(self, build_=True, download_model='', cuda_extensions=True):
        os.chdir('plugins')
        if build_:
            os.system('git clone https://github.com/xinntao/BasicSR.git')
            os.chdir('BasicSR')
            os.system(f'{self.python_executable} -m pip install -r requirements.txt')
            os.system(f'{self.python_executable} setup.py {self.build_type}' + '' if cuda_extensions else ' --no_cuda_ext')
        else:
            os.chdir('BasicSR')
        if download_model:
            os.system(f'{self.python_executable} scripts/download_pretrained_models {download_model}')
        os.chdir(self.cwd)

    @staticmethod
    def SSM(download_model=True):
        if download_model:
            download_file_from_google_drive('1WTb1C6IAICq5DBlmTSwJ5zQ8AcOGj_vR', 'vfin/ssm/SuperSloMo.ckpt')

    def DAIN(self, compute_compatibility=None, download_model=True, build_all=False):
        if compute_compatibility is None:
            compute_compatibility = [37, 61, 60, 70, 75]
        os.chdir('vfin/DAIN')
        # Write compiler args
        nvcc_args = []
        for cc in compute_compatibility:
            nvcc_args.append('-gencode')
            nvcc_args.append(f'arch=compute_{cc},code=sm_{cc}')
        nvcc_args.append('-w')
        with open('compiler_args.json', 'w') as f:
            json.dump({'nvcc': nvcc_args, 'cxx': ['-std=c++11', '-w']}, f)
        print(f'Compiling for compute compatilility {compute_compatibility}')
        # Compile
        os.chdir('my_package')
        folders = ['DepthFlowProjection', 'FilterInterpolation', 'FlowProjection']
        if build_all:
            folders.extend(['InterpolationCh', 'SeparableConv', 'SeparableConvFlow', 'MinDepthFlowProjection', 'Interpolation'])
        for folder in folders:
            os.chdir(f"{'' if folder == folders[0] else '../'}{folder}")
            os.system(f'{self.python_executable} setup.py {self.build_type}')
            for file_to_delete in self.terms_to_delete(folder):
                shutil.rmtree(f'{folder}/{file_to_delete}')
        os.chdir('../../PWCNet/correlation_package_pytorch1_0')
        os.system(f'{self.python_executable} setup.py {self.build_type}')
        for file_to_delete in self.terms_to_delete('.'):
            shutil.rmtree(file_to_delete)
        os.chdir('../..')
        os.remove('compiler_args.json')
        os.makedirs('model_weights')
        if download_model:
            download_file_from_google_drive('18_0UwKWnXT3fHw81eZZO0V1arXCAIG3X', 'model_weights/experimental.pth')
            download_file_from_google_drive('10X1XS6E0eUK5hf-5P6qHgWI-U-heVxCk', 'model_weights/best.pth')
        os.chdir(self.cwd)

    def DeOldify(self, download_model=True):
        os.chdir('plugins')
        os.system(f'{self.python_executable} -m pip install ')
        if download_model:
            pass

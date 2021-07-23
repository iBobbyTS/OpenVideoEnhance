import os
import sys
from setuptools import setup

import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


root = '/'.join(os.path.abspath(__file__).split('/')[:-1])
if os.getcwd() != root:
    os.chdir(root)

with open('version.txt', 'r') as f:
    version = f.read()


def make_extension(
        belong: str, name: str,
        extra_compile_args: dict,
        extra_file: list = None, define_macros: list = None
):
    return CUDAExtension(
        name=f'ove_ext.{belong}.{name}',
        sources=[
            f'extension/{belong}/{name}/{f}'
            for f in (
                'cuda_kernel.cu', 'cuda.cc',
                *([] if extra_file is None else extra_file)
            )
        ],
        **({} if define_macros is None else {'define_macros': define_macros}),
        extra_compile_args=extra_compile_args
    )


if torch.cuda.is_available():
    if len(sys.argv) > 2:
        cc = sys.argv[2].split(',')
        del sys.argv[2:]
    else:
        cc = ['%d%d' % torch.cuda.get_device_capability()]
    print(f"Building for CUDA Compatibility {','.join(cc)}")
else:
    print('No available CUDA device.')
    exit(1)

dain_extra_compile_args = {
    'cxx': ['-std=c++17', '-w'],
    'nvcc': ['-w']
}
for i in cc:
    dain_extra_compile_args['nvcc'].extend(['-gencode', f'arch=compute_{i},code=sm_{i}'])

basicsr_extra_compile_args = {
    'cxx': [],
    'nvcc': [
        '-D__CUDA_NO_HALF_OPERATORS__',
        '-D__CUDA_NO_HALF_CONVERSIONS__',
        '-D__CUDA_NO_HALF2_OPERATORS__'
    ]
}

ext_modules = [
    *[make_extension(
        belong='dain',
        name=name,
        extra_compile_args=dain_extra_compile_args
    ) for name in (
        'correlation', 'depth_flow_projection', 'filter_interpolation', 'flow_projection'
    )],
    make_extension(
        belong='basicsr',
        name='dcn',
        extra_compile_args=basicsr_extra_compile_args,
        define_macros=[('WITH_CUDA', None)],
        extra_file=['ext.cc']
    )
]

setup(
    name='ove_ext',
    version=version,
    packages=['extension'],
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False
)

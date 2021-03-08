from json import load
from setuptools import find_packages, setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def make_my_cuda_ext(names):
    if not isinstance(names, list):
        names = [names]
    return_ = []
    for name in names:
        lower_name = name.lower()
        return_.append(CUDAExtension(
            name=f'dain.{name}',
            sources=[
                f'my_package/{name}/{lower_name}_cuda.cc',
                f'my_package/{name}/{lower_name}_cuda_kernel.cu',
            ],
            extra_compile_args=extra_compile_args
        ))
    return return_


if __name__ == '__main__':
    with open('compiler_args.json') as f:
        json = load(f)
        develop = json['develop']
        extra_compile_args = json['extra_compile_args']
    packages = ['DepthFlowProjection', 'FilterInterpolation', 'FlowProjection']
    if develop:
        packages.extend(['Interpolation', 'InterpolationCh', 'MinDepthFlowProjection', 'SeparableConv', 'SeparableConvFlow'])
    ext_modules = make_my_cuda_ext(packages)
    # PWCNet
    ext_modules.append(CUDAExtension(
        name='dain.Correlation',
        sources=[
            'PWCNet/correlation_package_pytorch1_0/correlation_cuda.cc',
            'PWCNet/correlation_package_pytorch1_0/correlation_cuda_kernel.cu'
        ],
        extra_compile_args=extra_compile_args
    ))
    setup(
        name='dain',
        # packages=[f'my_package/{_}' for _ in packages] + ['PWCNet/correlation_package_pytorch1_0'],
        packages=find_packages(),
        include_package_data=True,
        zip_safe=False,
        # ext_modules=ext_modules,
        cmdclass={'build_ext': BuildExtension}
    )

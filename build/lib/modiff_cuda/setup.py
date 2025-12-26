from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

setup(
    name='modiff_cuda',
    packages=find_packages(),
    # package_dir={'': '.'}, # This is default
    ext_modules=[
        CUDAExtension(
            name='modiff_cuda_backend',
            sources=[
                'interface.cpp',
                'csrc/conv_w8a8_cuda.cu',
                'csrc/conv_simple.cu',
                'csrc/conv_tiled.cu',
                'csrc/conv_fast.cu',
                'csrc/conv_optimized.cu',
                'csrc/test_kernels.cu',
            ],
            include_dirs=[os.path.abspath('csrc')],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '-gencode=arch=compute_80,code=sm_80', # Ampere
                    '-gencode=arch=compute_90,code=sm_90', # Hopper
                    '--use_fast_math',
                    '-U__CUDA_NO_HALF_OPERATORS__',
                    '-U__CUDA_NO_HALF_CONVERSIONS__',
                    '-U__CUDA_NO_HALF2_OPERATORS__'
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

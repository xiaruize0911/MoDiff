from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

setup(
    name='modiff_cuda_cublas',
    ext_modules=[
        CUDAExtension(
            name='modiff_cuda_cublas',
            sources=[
                'csrc/conv2d_cublas_interface.cpp',
                'csrc/conv2d_cublas.cu',
                'csrc/im2col_cuda.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-gencode=arch=compute_80,code=sm_80',  # A100
                    '-lcublas',
                ]
            },
            libraries=['cublas'],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

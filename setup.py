from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Assumes CUTLASS is cloned at /workspace/cutlass or similar, or users provide include dir
CUTLASS_PATH = os.environ.get("CUTLASS_PATH", "/workspace/cutlass")

setup(
    name='modiff',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name='modiff_cutlass',
            sources=[
                'csrc/pybind.cpp',
                'csrc/cuda_kernels.cu',
            ],
            include_dirs=[
                os.path.join(CUTLASS_PATH, 'include'),
                os.path.join(CUTLASS_PATH, 'tools/util/include'),
            ],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': [
                    '-O3', '-std=c++17', 
                    '-U__CUDA_NO_HALF_OPERATORS__', 
                    '-U__CUDA_NO_HALF_CONVERSIONS__',
                    '-U__CUDA_NO_HALF2_OPERATORS__'
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    install_requires=[
        'torch',
    ],
)
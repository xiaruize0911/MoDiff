from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Bypass CUDA version mismatch check (handles CUDA 12.4 nvcc vs torch built with CUDA 13.x)
import torch.utils.cpp_extension as _cpp_ext
_cpp_ext._check_cuda_version = lambda *a, **kw: None

# Assumes CUTLASS is cloned at /workspace/cutlass or similar, or users provide include dir
CUTLASS_PATH = os.environ.get("CUTLASS_PATH", "/workspace/cutlass")

# Build RPATH so the .so can find torch and CUDA libs without LD_LIBRARY_PATH at runtime
TORCH_LIB_DIR = os.path.join(os.path.dirname(torch.__file__), 'lib')
CUDA_LIB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.environ.get("CUDA_HOME", "/usr/local/cuda"))), "cuda", "lib64")
RPATH_FLAGS = [
    f'-Wl,-rpath,{TORCH_LIB_DIR}',
    f'-Wl,-rpath,/usr/local/cuda/lib64',
]

setup(
    name='modiff',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name='modiff_cutlass',
            sources=[
                'csrc/pybind.cpp',
                'csrc/kernels/quantize.cu',
                'csrc/kernels/modiff_delta_quantize.cu',
                'csrc/kernels/conv_epilogue.cu',
                'csrc/kernels/conv2d_int8.cu',
                'csrc/kernels/conv2d_int4.cu',
                'csrc/kernels/layout_transform.cu',
                'csrc/kernels/group_norm_silu.cu',
                'csrc/kernels/fused_gn_qkv.cu',
                'csrc/kernels/flash_attn_int8.cu',
                'csrc/kernels/quantize_qkv.cu',
                'csrc/kernels/gemm_wxax.cu',
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
            },
            extra_link_args=RPATH_FLAGS,
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    install_requires=[
        'torch',
    ],
)
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
                # quantize: activation quantize + MoDiff temporal-delta quantize
                # family 1 of the csrc split (2026-08-12): quantize/ now lives in the two
                # per-datapath trees. See csrc/README.md for the boundary rule.
                'csrc/baseline/quantize/quantize.cu',
                'csrc/modiff/quantize/delta_quantize.cu',
                # conv: W8A8/W4A4 int conv (CUTLASS implicit GEMM) + shared epilogue
                'csrc/kernels/conv/conv_epilogue.cu',
                'csrc/kernels/conv/conv2d_int8.cu',
                'csrc/kernels/conv/conv2d_int4.cu',
                'csrc/kernels/conv/conv2d_evt.cu',
                # norm: GroupNorm(+SiLU)(+quantize) and fused GroupNorm->qkv
                'csrc/kernels/norm/group_norm_silu.cu',
                'csrc/kernels/norm/fused_gn_qkv.cu',
                # linear: W8A8/W4A4 Linear GEMM (own AWQ-tiling port + vendored AWQ)
                # family 2 of the csrc split (2026-08-12)
                'csrc/baseline/linear/gemm_wxax.cu',
                'csrc/modiff/linear/gemm_wxax.cu',
                # attention: W8A8/W4A4 materialized attention + fused int8/int4 flash
                'csrc/kernels/attention/attn_quant_gemm.cu',
                'csrc/kernels/attention/flash_attn_int8.cu',
                # util: NCHW<->NHWC / packing layout transforms
                # family 6 of the csrc split (2026-08-12)
                'csrc/baseline/util/layout_transform.cu',
                'csrc/modiff/util/layout_transform.cu',
            ],
            include_dirs=[
                os.path.join(CUTLASS_PATH, 'include'),
                os.path.join(CUTLASS_PATH, 'tools/util/include'),
                # Absolute paths: ninja runs compile commands from the build-temp dir, so
                # relative include dirs (e.g. 'csrc') fail to resolve after the csrc/ reorg
                # moved the .cu sources into subdirectories.
                os.path.abspath('csrc'),                 # common.cuh, modiff_kernels_api.h
                os.path.abspath('csrc/kernels/common'),  # mma_int8.cuh (shared by linear + attention)
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
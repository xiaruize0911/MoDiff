"""
Build script for INT8 Convolution kernel using CUTLASS
"""
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch
import os
import subprocess

# Get CUDA compute capability
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability()
    arch = f"sm_{major}{minor}"
else:
    arch = "sm_89"  # Default to L4

# Find CUTLASS include path
def get_cutlass_include():
    try:
        result = subprocess.run(
            ['python', '-c', 'import cutlass_library; import os; print(os.path.dirname(cutlass_library.__file__))'],
            capture_output=True, text=True
        )
        cutlass_base = result.stdout.strip()
        return os.path.join(cutlass_base, 'source', 'include')
    except:
        return None

cutlass_include = get_cutlass_include()
if cutlass_include is None or not os.path.exists(cutlass_include):
    raise RuntimeError("CUTLASS not found. Install with: pip install nvidia-cutlass")

print(f"Using CUTLASS from: {cutlass_include}")
print(f"Building for architecture: {arch}")

setup(
    name='modiff_int8',
    version='1.0.0',
    ext_modules=[
        CUDAExtension(
            name='modiff_int8',
            sources=[
                'csrc/conv_int8_cutlass.cu',
                'csrc/conv_int8_cutlass_interface.cpp',
            ],
            include_dirs=[
                os.path.abspath('csrc'),
                cutlass_include,
            ],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': [
                    '-O3',
                    f'-arch={arch}',
                    '--use_fast_math',
                    '-std=c++17',
                    f'-I{cutlass_include}',
                    '-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1',
                    '--expt-relaxed-constexpr',
                    '-lineinfo',  # For debugging
                ],
            },
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

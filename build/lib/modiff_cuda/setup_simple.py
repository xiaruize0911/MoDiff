from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='conv_simple',
    ext_modules=[
        CUDAExtension(
            name='conv_simple',
            sources=[
                'csrc/conv_simple_interface.cpp',
                'csrc/conv_simple.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '--use_fast_math', '-gencode=arch=compute_80,code=sm_80']
            }
        ),
    ],
    cmdclass={'build_ext': BuildExtension}
)

set -euo pipefail
cd /workspace/MoDiff
NVCC=/usr/local/cuda-12.4/bin/nvcc
TMP=build/temp.linux-x86_64-cpython-311
TORCH_LIB=$(python -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"lib"))')
INC="-I/workspace/cutlass/include -I/workspace/cutlass/tools/util/include -I/workspace/MoDiff/csrc"
for d in $(python -c 'from torch.utils.cpp_extension import include_paths;print(" ".join(include_paths()))'); do INC="$INC -I$d"; done
INC="$INC -I/usr/include/python3.11"
$NVCC $INC -c csrc/modiff/conv/conv2d_int8_blockk.cu \
  -o $TMP/csrc/modiff/conv/conv2d_int8_blockk.o \
  -O3 -std=c++17 -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__ \
  -U__CUDA_NO_HALF2_OPERATORS__ -DTORCH_API_INCLUDE_EXTENSION_H \
  -DTORCH_EXTENSION_NAME=modiff_cutlass \
  -D_GLIBCXX_USE_CXX11_ABI=$(python -c 'import torch;print(int(torch._C._GLIBCXX_USE_CXX11_ABI))') \
  '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1011"' \
  --expt-relaxed-constexpr --compiler-options "'-fPIC'" -gencode=arch=compute_86,code=sm_86
c++ -shared $(find "$TMP" -name '*.o') -o modiff_cutlass.cpython-311-x86_64-linux-gnu.so \
  -L"$TORCH_LIB" -L/usr/local/cuda-12.4/lib64 \
  -lc10 -lc10_cuda -ltorch -ltorch_cpu -ltorch_cuda -ltorch_python -lcudart -Wl,-rpath,"$TORCH_LIB"
echo BUILD_OK

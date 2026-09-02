#!/bin/bash
# Incremental rebuild of only the a_hat-touching translation units + relink.
# A full `python setup.py build_ext --inplace` recompiles every CUTLASS conv (~32 min);
# these three .cu files are the only ones that include ahat_cache.cuh's hot path.
#
#   source /workspace/MoDiff/setup_cuda_env.sh
#   bash docs/ahat_blockwise_2026-09-01/scripts/rebuild_ahat.sh [extra .cu paths...]
set -euo pipefail

cd /workspace/MoDiff
NVCC=/usr/local/cuda-12.4/bin/nvcc
TMP=build/temp.linux-x86_64-cpython-311
TORCH_DIR=$(python -c 'import torch,os;print(os.path.dirname(torch.__file__))')
TORCH_LIB=$TORCH_DIR/lib

INC="-I/workspace/cutlass/include -I/workspace/cutlass/tools/util/include -I/workspace/MoDiff/csrc"
for d in $(python -c 'from torch.utils.cpp_extension import include_paths;print(" ".join(include_paths()))'); do
    INC="$INC -I$d"
done
INC="$INC -I/usr/include/python3.11"

# The PYBIND11_* quotes are load-bearing: without them the pybind ABI tag mismatches
# pybind.o and the module fails to import with a duplicate-registration error.
FLAGS=(-O3 -std=c++17
    -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__ -U__CUDA_NO_HALF2_OPERATORS__
    -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=modiff_cutlass
    -D_GLIBCXX_USE_CXX11_ABI=$(python -c 'import torch;print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')
    '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1011"'
    --expt-relaxed-constexpr --compiler-options "'-fPIC'"
    -gencode=arch=compute_86,code=sm_86)

SRCS=(csrc/modiff/norm/group_norm_silu.cu
      csrc/modiff/quantize/delta_quantize.cu
      csrc/baseline/quantize/quantize.cu
      csrc/pybind.cpp
      "$@")

for src in "${SRCS[@]}"; do
    obj="$TMP/${src%.*}.o"
    mkdir -p "$(dirname "$obj")"
    if [[ $src == *.cpp ]]; then
        echo "c++  $src"
        c++ $INC -c "$src" -o "$obj" -O3 -std=c++17 -fPIC \
            -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=modiff_cutlass \
            "-D_GLIBCXX_USE_CXX11_ABI=$(python -c 'import torch;print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')" \
            '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' \
            '-DPYBIND11_BUILD_ABI="_cxxabi1011"'
    else
        echo "nvcc $src"
        $NVCC $INC -c "$src" -o "$obj" "${FLAGS[@]}"
    fi
done

echo "link"
c++ -shared $(find "$TMP" -name '*.o') \
    -o modiff_cutlass.cpython-311-x86_64-linux-gnu.so \
    -L"$TORCH_LIB" -L/usr/local/cuda-12.4/lib64 \
    -lc10 -lc10_cuda -ltorch -ltorch_cpu -ltorch_cuda -ltorch_python -lcudart \
    -Wl,-rpath,"$TORCH_LIB"
echo "ok"

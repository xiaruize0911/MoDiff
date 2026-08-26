import os, torch
from torch.utils.cpp_extension import load
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")
m = load(name="ahat_probe", sources=["probe.cu"], extra_cuda_cflags=["-O3", "--use_fast_math"],
         verbose=False, build_directory=os.path.abspath("build"))
print("built:", m)

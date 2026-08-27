import os
from torch.utils.cpp_extension import load

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")
m = load(name="ahat_skip2_probe", sources=["probe.cu"], extra_cuda_cflags=["-O3"],
         verbose=True, build_directory=os.path.abspath("build"))
print("built:", m)

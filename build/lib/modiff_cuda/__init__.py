import torch
import os

# Load the C++ extension
try:
    import modiff_cuda_backend
except ImportError:
    # If not installed, try to load from build directory or just use the name if it's in path
    import modiff_cuda as modiff_cuda_backend

from .nn import W8A8MoDiffConv2dCUDA

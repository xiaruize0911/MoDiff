#!/bin/bash
# Setup CUDA environment for MoDiff

export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
# nvjitlink 12.8 (from pip) MUST come before cuda-12.4 system path to avoid
# "undefined symbol: __nvJitLinkCreate_12_8" when loading PyTorch (built w/ CUDA 12.8)
export LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/nvidia/nvjitlink/lib:/usr/local/lib/python3.11/dist-packages/torch/lib:/usr/local/lib/python3.11/dist-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.11/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.11/dist-packages/nvidia/cusparse/lib:$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

echo "CUDA environment configured:"
echo "CUDA_HOME=$CUDA_HOME"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

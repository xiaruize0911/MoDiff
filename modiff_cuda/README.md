# MoDiff Fused CUDA Kernels

High-performance fused CUDA kernels for accelerating diffusion models, specifically optimized for U-Net architectures used in LDM (Latent Diffusion Models) and DDIM.

## Overview

This module provides fused CUDA kernels that combine multiple operations into single kernel launches, reducing memory bandwidth requirements and improving throughput. The primary optimization is **Fused GroupNorm + SiLU**, which is a common pattern in diffusion U-Nets.

## Key Optimizations

### 1. Fused GroupNorm + SiLU Kernel

**Problem**: In standard implementations, GroupNorm and SiLU activation are separate operations:
```python
# Sequential (2 kernel launches, 4 memory operations)
x = group_norm(x)  # Read → Compute → Write
x = silu(x)        # Read → Compute → Write
```

**Solution**: Single fused kernel that computes both in one pass:
```python
# Fused (1 kernel launch, 2 memory operations)
x = fused_groupnorm_silu(x, weight, bias, num_groups, eps)
```

**Memory Bandwidth Reduction**: 50% (4 mem ops → 2 mem ops)

### 2. Kernel Variants

| Kernel | Description | Use Case |
|--------|-------------|----------|
| `fused_groupnorm_silu` | FP32 single-pass implementation | Default, maximum accuracy |
| `fused_groupnorm_silu_fp16` | FP16 with automatic casting | Memory-constrained scenarios |
| `fused_groupnorm_silu_two_pass` | Two-pass for numerical stability | Large group sizes |

## Performance Results

### Isolated Kernel Benchmark (GroupNorm + SiLU)

| Input Shape | Sequential (ms) | Fused (ms) | Speedup |
|-------------|-----------------|------------|---------|
| (4, 320, 32×32) | 0.069 | 0.030 | **2.32x** |
| (4, 640, 16×16) | 0.069 | 0.016 | **4.21x** |
| (4, 1280, 8×8) | 0.068 | 0.011 | **6.23x** |

### Full Model Benchmark (DDIM CIFAR-10, 100 steps)

| Batch Size | Baseline (ms/step) | Fused (ms/step) | Speedup |
|------------|-------------------|-----------------|---------|
| 1 | 13.35 | 12.06 | **1.11x** |
| 4 | 13.39 | 12.05 | **1.11x** |
| 8 | 13.05 | 11.50 | **1.13x** |

**Overall: 11-13% end-to-end speedup on full diffusion model inference**

## Installation

### Prerequisites
- CUDA 11.0+
- PyTorch 1.9+
- Ninja build system

### Build from Source

```bash
cd /workspace/MoDiff
pip install -e .
```

Or build just the CUDA extension:

```bash
cd /workspace/MoDiff/modiff_cuda
python setup.py build_ext --inplace
```

## Usage

### Basic Usage

```python
import torch
from modiff_cuda import fused_groupnorm_silu

# Input tensor (N, C, H, W)
x = torch.randn(4, 320, 32, 32, device='cuda')
weight = torch.ones(320, device='cuda')
bias = torch.zeros(320, device='cuda')

# Fused operation
out = fused_groupnorm_silu(x, weight, bias, num_groups=32, eps=1e-6)
```

### Drop-in Replacement Class

```python
from integration.fused_layers import FusedGroupNormSiLU

# Create fused layer (drop-in replacement for nn.GroupNorm + SiLU)
fused_layer = FusedGroupNormSiLU(num_groups=32, num_channels=320)

# Use like normal
out = fused_layer(x)
```

### Apply to Existing U-Net

```python
from integration.fused_layers import apply_fused_kernels_to_unet

# Load your model
model = load_unet_model()

# Automatically replace GroupNorm+SiLU patterns
model = apply_fused_kernels_to_unet(model)

# Run inference as normal
output = model(x, t)
```

## File Structure

```
modiff_cuda/
├── csrc/
│   ├── fused_conv_norm_act.cu          # CUDA kernel implementations
│   ├── fused_conv_norm_act_interface.cpp  # PyTorch C++ bindings
│   ├── int8_gemm.cu                    # INT8 quantized GEMM
│   ├── int8_gemm_interface.cpp         # INT8 bindings
│   └── bindings.cpp                    # Module entry point
├── __init__.py                         # Python module init
└── README.md                           # This file

integration/
├── fused_layers.py                     # High-level Python API
├── benchmark_fused_kernels.py          # Microbenchmarks
├── benchmark_ddim_fused.py             # Full model benchmark (CIFAR-10)
└── benchmark_lsun_church_fid.py        # LDM benchmark with FID
```

## Technical Details

### Kernel Implementation

The fused kernel uses the following optimization strategies:

1. **Shared Memory**: Statistics (mean, variance) computed in shared memory
2. **Warp Reduction**: Efficient parallel reduction using warp shuffle instructions
3. **Memory Coalescing**: Contiguous memory access patterns for maximum bandwidth
4. **Register Blocking**: Intermediate values kept in registers

### Thread/Block Configuration

```cpp
// Optimal configuration for GroupNorm+SiLU
dim3 blocks(batch_size, num_groups);
dim3 threads(min(256, group_size));
```

### Numerical Stability

The kernel uses Welford's online algorithm for computing mean and variance in a single pass, ensuring numerical stability even for large tensors.

## Benchmarking

### Run Microbenchmarks

```bash
cd /workspace/MoDiff
python integration/benchmark_fused_kernels.py
```

### Run Full Model Benchmark

```bash
# DDIM on CIFAR-10
python integration/benchmark_ddim_fused.py \
    --config configs/cifar10.yml \
    --model_path models/ema_diffusion_cifar10_model/model-790000.ckpt \
    --num_samples 100 \
    --ddim_steps 100

# LDM on LSUN Churches (with FID)
python integration/benchmark_lsun_church_fid.py \
    --config configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml \
    --model_path models/ldm/lsun_churches256/model.ckpt \
    --num_samples 2000 \
    --ddim_steps 200
```

## Compatibility

| Model | Tested | Speedup |
|-------|--------|---------|
| DDIM (CIFAR-10) | ✅ | 11-13% |
| LDM (LSUN Churches) | ✅ | ~10-15% |
| Stable Diffusion | 🔄 | Expected similar |

## Future Optimizations

- [ ] Fused Conv2D + GroupNorm + SiLU (single kernel)
- [ ] INT8 quantized fused kernels
- [ ] Flash Attention integration
- [ ] CUDA Graphs for full U-Net
- [ ] Multi-GPU support

## Citation

If you use these optimizations in your work, please cite:

```bibtex
@misc{modiff2024,
  title={MoDiff: Optimized Diffusion Model Inference with Fused CUDA Kernels},
  year={2024},
  url={https://github.com/your-repo/modiff}
}
```

## License

Apache 2.0

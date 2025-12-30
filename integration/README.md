# MoDiff Integration Benchmarks

Benchmark scripts for evaluating diffusion model inference optimizations.

## Quick Start

```bash
cd /workspace/MoDiff
python integration/benchmark_ldm.py --steps 20 --batch_size 4
```

## Benchmark Results (LDM LSUN-Churches, 20 steps, batch=4, NVIDIA L4)

| Mode | Time (s) | ms/step | Speedup |
|------|----------|---------|---------|
| FP32 | 0.966 | 48.32 | baseline |
| **TF32** | **0.701** | **35.05** | **1.38x** |
| FP16 | 0.845 | 42.24 | 1.14x |
| BF16 | 0.851 | 42.55 | 1.14x |
| INT8 MoDiff | 1.067 | 53.34 | 0.91x |

**Key Finding**: TF32 (TensorFloat-32) provides the best speedup with zero code changes. FP16/BF16 autocast also provides good speedups. The INT8 MoDiff implementation has overhead from error-compensated modulation caching.

## Benchmark Scripts

### 1. `benchmark_fused_kernels.py` - Isolated Kernel Microbenchmarks

Tests individual fused kernel performance vs sequential PyTorch operations.

```bash
python benchmark_fused_kernels.py
```

**Output**: Comparison of fused vs sequential GroupNorm+SiLU across different tensor shapes.

### 2. `benchmark_ddim_fused.py` - DDIM CIFAR-10 Full Model

End-to-end benchmark on DDIM with CIFAR-10.

```bash
python benchmark_ddim_fused.py \
    --config ../configs/cifar10.yml \
    --model_path ../models/ema_diffusion_cifar10_model/model-790000.ckpt \
    --num_samples 100 \
    --ddim_steps 100 \
    --batch_size 8
```

**Options**:
- `--num_samples`: Number of images to generate (default: 100)
- `--ddim_steps`: DDIM sampling steps (default: 100)
- `--batch_size`: Batch size for generation (default: 8)
- `--output_dir`: Directory for generated images

### 3. `benchmark_lsun_church_fid.py` - LDM LSUN-Churches with FID

Full benchmark with FID quality evaluation on LDM.

```bash
python benchmark_lsun_church_fid.py \
    --config ../configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml \
    --model ../models/ldm/lsun_churches256/model.ckpt \
    --num_samples 2000 \
    --ddim_steps 200 \
    --batch_size 8
```

**Options**:
- `--num_samples`: Number of samples for FID (recommend 2000+)
- `--ddim_steps`: DDIM steps (default: 200)
- `--ddim_eta`: DDIM eta parameter (default: 0.0)
- `--output_dir`: Directory for outputs

**Output**: 
- Generated images for baseline and fused variants
- FID scores computed against LSUN Church dataset
- Timing comparison and speedup metrics

### 4. `benchmark_ldm_fused.py` - Generic LDM Benchmark

General-purpose LDM benchmark without FID.

```bash
python benchmark_ldm_fused.py \
    --config <config_path> \
    --model_path <model_path>
```

### 5. `benchmark_ldm_cuda_graphs.py` - CUDA Graphs Optimization

Benchmark with CUDA Graphs for reduced kernel launch overhead.

```bash
python benchmark_ldm_cuda_graphs.py \
    --config <config_path> \
    --model_path <model_path>
```

## Results Summary

### LDM LSUN-Churches Performance (NVIDIA L4)

| Mode | Time | Speedup | Notes |
|------|------|---------|-------|
| TF32 | 35.05 ms/step | **1.38x** | Best performance, default on Ampere+ |
| FP16 | 42.24 ms/step | 1.14x | torch.cuda.amp.autocast |
| BF16 | 42.55 ms/step | 1.14x | torch.cuda.amp.autocast |
| FP32 | 48.32 ms/step | 1.00x | Baseline |
| INT8 | 53.34 ms/step | 0.91x | MoDiff overhead |

### Fused GroupNorm+SiLU (Isolated Kernel)

| Shape (N,C,H,W) | Sequential | Fused | Speedup |
|-----------------|------------|-------|---------|
| (4, 320, 32, 32) | 0.069 ms | 0.030 ms | 2.32x |
| (4, 640, 16, 16) | 0.069 ms | 0.016 ms | 4.21x |
| (4, 1280, 8, 8) | 0.068 ms | 0.011 ms | 6.23x |

## Helper Modules

### `fused_layers.py` - High-Level API

Provides `FusedGroupNormSiLU` class and `apply_fused_kernels_to_unet()` function for easy integration.

```python
from fused_layers import apply_fused_kernels_to_unet

model = apply_fused_kernels_to_unet(your_model)
```

### `cuda_graph_wrapper.py` - CUDA Graphs Utilities

Wrapper for capturing and replaying CUDA graphs for reduced kernel launch overhead.

## Requirements

- CUDA 11.0+
- PyTorch 1.9+
- `clean-fid` (for FID computation)
- MoDiff CUDA extensions built (`pip install -e ..`)

## Model Checkpoints

Download required checkpoints:

```bash
# CIFAR-10 DDIM
# Place at: models/ema_diffusion_cifar10_model/model-790000.ckpt

# LSUN Churches LDM
# Place at: models/ldm/lsun_churches256/model.ckpt
```

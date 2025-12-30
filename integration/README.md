# MoDiff Integration

Optimized INT8 MoDiff implementation with static quantization and NHWC pipeline.

## Files

| File | Description |
|------|-------------|
| `int8_optimized.py` | Main INT8 implementation with static quantization |
| `modiff_layers.py` | Legacy INT8 implementation (for compatibility) |
| `benchmark_ldm.py` | Unified LDM benchmark for all precision modes |

## Quick Start

```bash
cd /workspace/MoDiff

# Run all modes (FP32, FP16, BF16, INT8)
python integration/benchmark_ldm.py --mode all --steps 50 --num_samples 16

# Run only INT8 with FID evaluation
python integration/benchmark_ldm.py --mode int8 --num_samples 100 --eval_fid
```

## Benchmark Results (LDM LSUN-Churches, 50 steps, batch=4, NVIDIA L4)

| Mode | Time/Sample | Speedup | Notes |
|------|-------------|---------|-------|
| FP32 | 0.625s | baseline | TF32 disabled |
| FP16 | 0.542s | 1.15x | torch.amp.autocast |
| BF16 | 0.544s | 1.15x | torch.amp.autocast |
| **INT8 Optimized** | **0.509s** | **1.23x** | Static quantization |

## Key Features

### Static Quantization
- Pre-calibrates activation scales during warmup
- Eliminates per-layer `find_max` overhead (~15% faster)

### Adaptive Precision
- Automatically uses INT8 for large convolutions (768+ channels)
- Falls back to FP16 for small convolutions

### MoDiff Temporal Caching
- Reuses previous timestep outputs
- Computes only residual convolutions for subsequent steps

## Usage

```python
from integration.int8_optimized import (
    convert_model_to_optimized_int8,
    enable_modiff_mode,
    reset_modiff_state,
    set_calibrating,
    get_calibration_config,
    reset_calibration,
)

# Convert model
convert_model_to_optimized_int8(model.model.diffusion_model)
enable_modiff_mode(model.model.diffusion_model, True)

# Calibrate (10 runs, 5 steps each)
reset_calibration()
set_calibrating(model.model.diffusion_model, True)
for _ in range(10):
    sampler.sample(S=5, batch_size=2, shape=(4,32,32), eta=0.0, verbose=False)
get_calibration_config().finalize()
set_calibrating(model.model.diffusion_model, False)

# Inference (reset state per sample)
reset_modiff_state(model.model.diffusion_model)
samples, _ = sampler.sample(S=50, batch_size=4, shape=(4,32,32), eta=0.0)
```

## Dependencies

- PyTorch 2.0+
- CUTLASS INT8 kernels (`modiff_cuda/`)
- OmegaConf
- pytorch-fid (optional, for FID evaluation)

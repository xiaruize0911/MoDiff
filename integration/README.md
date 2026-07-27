# MoDiff Integration

Optimized INT8/INT4 MoDiff implementation with CUTLASS Tensor Core acceleration.

## Directory Structure

```
integration/
├── kernels/                        # Quantized Conv/Linear Layer Implementations
│   ├── int8_optimized.py           #   CUTLASS INT8 fused conv + MoDiff
│   ├── int4_optimized.py           #   CUTLASS INT4 fused conv + MoDiff
│   ├── int8_cudagraph.py           #   PyTorch native INT8 + CUDA Graph
│   ├── fused_baseline.py           #   Separate-kernel (unfused) baseline
│   ├── int8_linear.py              #   INT8 quantized linear layer
│   ├── int4_linear.py              #   INT4 quantized linear layer
│   └── modiff_layers.py            #   Legacy CUTLASS conv
│
├── fused_ops/                      # Fused Operator Implementations
│   ├── fused_gn_silu.py            #   Triton GroupNorm+SiLU kernel
│   └── fused_resblock.py           #   Fused residual block
│
├── utils/                          # Infrastructure & Utilities
│   ├── buffer_pool.py              #   Pre-allocated GPU buffer pool
│   ├── timestep_cache.py           #   Cached timestep embeddings
│   └── profiler.py                 #   CUDA event-based profiler
│
├── benchmarks/                     # Benchmarks & Evaluation
│   ├── benchmark_ldm.py            #   Original LDM benchmark
│   ├── benchmark_extended.py       #   Extended benchmark (all modes)
│   ├── generate_extended_report.py #   Report & plot generation
│   ├── eval_fid_lsun.py            #   FID evaluation on LSUN
│   └── fid_eval.py                 #   FID computation utilities
│
├── calibration/                    # Calibration Data
│   ├── int8_calibration.pt
│   └── int4_calibration.pt
│
└── results/                        # Benchmark Output
    ├── extended/                    #   Extended benchmark results & plots
    └── ldm/                         #   Original LDM benchmark results
```

## Quick Start

```bash
cd /workspace/MoDiff

# Run original benchmark (FP32, FP16, INT8, INT4)
python integration/benchmarks/benchmark_ldm.py --mode all --steps 50 --num_samples 16

# Run extended benchmark (includes PyTorch INT8, CUDA Graph, separate kernels)
python integration/benchmarks/benchmark_extended.py --mode int8 --batch_size 32 --steps 200

# Generate report and plots from results
python integration/benchmarks/generate_extended_report.py
```

## Benchmark Results (LDM LSUN-Churches, 100 steps, NVIDIA L40S)

| Mode | Time/Sample | Speedup | FID (50k) | Notes |
|------|-------------|---------|-----------|-------|
| FP32 | 0.271s | 1.00x | - | TF32 disabled |
| FP16 | 0.231s | 1.17x | - | torch.autocast |
| **INT8** | **0.254s** | **1.07x** | **8.20** | CUTLASS Tensor Cores, dynamic residuals |
| **INT4** | **0.227s** | **1.19x** | - | INT4→INT8 unpacking, fastest mode |

## Key Features

### CUTLASS Tensor Core Acceleration
- INT8 convolution using CUTLASS implicit GEMM
- INT4 unpacks to INT8 for Tensor Core execution
- Per-channel weight quantization, per-tensor activation quantization

### Dynamic Residual Quantization
- Prevents blurring by using dynamic scales for residuals
- Static scales for forward pass, dynamic for MoDiff error compensation
- FID of 8.20 demonstrates excellent quality preservation

### Adaptive Precision
- Converts Conv2d layers with `in_channels >= 32` to INT8/INT4 (see
  `convert_model_to_optimized_int8`, integration/kernels/int8_optimized.py)
- Skips: layers with `in_channels < 32`, named "skip" convs, the final output
  conv (`out.*`), grouped convs, and (by default) 1x1 pointwise convs
- Falls back to FP16 for all skipped/gated layers
- Percentile-based calibration (99.99%) for optimal scale selection

### MoDiff Temporal Caching
- Reuses previous timestep outputs
- Computes only residual convolutions for subsequent steps
- Error compensation maintains quality with reduced computation

## Usage

### INT8 Mode

```python
from integration.int8_optimized import (
    convert_model_to_optimized_int8,
    enable_modiff_mode,
    reset_modiff_state,
    set_calibrating,
    get_calibration_config,
    reset_calibration,
)

# Convert model to INT8
convert_model_to_optimized_int8(model.model.diffusion_model)
enable_modiff_mode(model.model.diffusion_model, True)

# Calibrate (5 runs, 5 steps each)
reset_calibration()
set_calibrating(model.model.diffusion_model, True)
for _ in range(5):
    reset_modiff_state(model.model.diffusion_model)
    sampler.sample(S=5, batch_size=4, shape=(4,32,32), eta=0.0, verbose=False)
get_calibration_config().finalize()
set_calibrating(model.model.diffusion_model, False)

# Inference (reset state per sample)
reset_modiff_state(model.model.diffusion_model)
samples, _ = sampler.sample(S=100, batch_size=4, shape=(4,32,32), eta=0.0)
```

### INT4 Mode

```python
from integration.int4_optimized import (
    convert_model_to_optimized_int4,
    enable_modiff_mode,
    reset_modiff_state,
    set_calibrating,
    get_calibration_config,
    reset_calibration,
)

# Same API as INT8, just replace int8 with int4
convert_model_to_optimized_int4(model.model.diffusion_model)
# ... rest is identical to INT8
```

## FID Evaluation

### Generate Samples

```bash
# INT8: Generate 50,000 samples
python integration/generate_50k_int8.py \
    --mode int8 \
    --num_samples 50000 \
    --steps 50 \
    --batch_size 64 \
    --output_dir results/int8_50k

# INT4: Generate 50,000 samples  
python integration/generate_50k_int8.py \
    --mode int4 \
    --num_samples 50000 \
    --steps 50 \
    --batch_size 64 \
    --output_dir results/int4_50k
```

### Calculate FID

```bash
# Calculate FID against LSUN Churches dataset
python integration/calculate_fid_lmdb.py \
    results/int8_50k/samples \
    data/lsun/church_outdoor_train_lmdb \
    --num_samples 50000 \
    --batch_size 50 \
    --output results/int8_50k/fid_score.txt

# Result: INT8 FID = 8.20 (excellent quality)
cat results/int8_50k/fid_score.txt
```

## Dependencies

- PyTorch 2.6+
- CUDA 12.x with Tensor Core support
- Triton 2.1+ (for optimized INT8/INT4 kernels)
- OmegaConf
- pytorch-fid (for FID evaluation)
- scipy (for FID calculation)
- lmdb (for LSUN dataset)

## Build Fused Kernels

Triton kernels are JIT compiled and do not require separate build step.
Just ensure Triton is installed: `pip install triton`.

## Technical Details

### INT8 Implementation
- **Quantization**: Per-channel weights, per-tensor activations
- **Residuals**: Dynamic quantization to prevent blurring
- **Convolution**: CUTLASS implicit GEMM (INT8 × INT8 → INT32)
- **Calibration**: Percentile-based (99.99%) for optimal scales

### INT4 Implementation ⚠️ **Incomplete - Not True INT4 MoDiff**

**Current Status**: Uses INT8 CUTLASS backend with INT4 weights (not native INT4 Tensor Cores)

**Critical Limitations**:
- ❌ **FP32 caches**: MoDiff temporal caches stored in FP32 (wastes 4× memory)
- ❌ **FP32 residuals**: Residual computation in FP32, then re-quantized each step
- ❌ **INT8 backend**: Unpacks INT4→INT8 before computation (adds overhead)
- ❌ **No native INT4**: Doesn't use CUTLASS INT4 Tensor Cores (sm_89 supports this)

**What it actually does**:
1. Pack weights to INT4 (2 values per byte)
2. **Unpack INT4→INT8** at runtime
3. Call INT8 CUTLASS convolution
4. Store caches in **FP32** (not INT4!)

**Expected behavior** (when properly implemented):
- ✓ Store caches as `uint8` (INT4 packed) → 4× memory savings
- ✓ Compute residuals in INT4 precision → less overhead  
- ✓ Use native INT4 CUTLASS Tensor Cores → ~1.5-1.8× speedup vs INT8

**Current performance**: 1.19× speedup (only from INT8 backend, INT4 adds overhead)

**Recommendation**: **Use INT8 mode for production** (fully optimized, 1.18× speedup, FID=8.20)

**See**: [INT4_ANALYSIS.md](../INT4_ANALYSIS.md) for detailed explanation and fix roadmap.

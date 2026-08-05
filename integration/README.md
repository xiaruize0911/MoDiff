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

> ⚠ **The FID column cannot be reproduced in this checkout, and the INT4 note is wrong.**
> `models/ldm/lsun_churches256/model.ckpt` here is an 856-byte stub whose `state_dict` has 0 entries,
> loaded `strict=False`, so every weight is random; on top of that `UNetModel.out[-1]` is a
> `zero_module`, which makes the UNet's epsilon prediction identically zero. No image-quality number
> can come from this tree — see `docs/gn_qkv_fusion_2026-08-03/FINDINGS.md` §5. Treat 8.20 as
> inherited from elsewhere (provenance unknown) until it is regenerated against a real checkpoint.
> The timings are on an L40S and predate the current kernels; the measured, reproducible numbers are
> in `docs/MEASUREMENT_REPORT_2026-08-01.md` (A40, batch 128).

| Mode | Time/Sample | Speedup | FID (50k) | Notes |
|------|-------------|---------|-----------|-------|
| FP32 | 0.271s | 1.00x | - | TF32 disabled |
| FP16 | 0.231s | 1.17x | - | torch.autocast |
| **INT8** | **0.254s** | **1.07x** | 8.20 ⚠ unreproducible here | CUTLASS Tensor Cores |
| **INT4** | **0.227s** | **1.19x** | - | native CUTLASS s4 (not "INT4→INT8 unpacking") |

## Key Features

### CUTLASS Tensor Core Acceleration
- INT8 convolution using CUTLASS implicit GEMM
- INT4 uses native CUTLASS s4 Tensor Cores (`cutlass::int4b_t`, csrc/kernels/conv/conv2d_int4.cu:52);
  it does NOT unpack to INT8 -- that description was stale
- Per-channel weight quantization, per-tensor activation quantization

### Dynamic Residual Quantization
- Prevents blurring by using dynamic scales for residuals
- Static scales for forward pass, dynamic for MoDiff error compensation
- (An FID of 8.20 was previously claimed here. It cannot be produced in this checkout --
  stub checkpoint, zero UNet output -- so the claim is withdrawn pending a real ckpt.)

### Adaptive Precision
- Converts Conv2d layers with `in_channels >= 32` to INT8/INT4 (see
  `convert_model_to_optimized_int8`, integration/kernels/int8_optimized.py)
- Skips: layers with `in_channels < 32`, named "skip" convs, the final output
  conv (`out.*`), grouped convs, and (by default) 1x1 pointwise convs
- Falls back to FP16 for all skipped/gated layers
- Percentile-based calibration (99.99%) for optimal scale selection

### MoDiff Temporal Caching
- Reuses previous timestep outputs: `ô_t = A(Q(a_t − â_{t+1})) + ô_{t+1}` (paper Eqs 13–14)
- **Same FLOPs, not fewer.** "Computes only residual convolutions" used to appear here and is wrong:
  every convolution still runs at full size every step. What changes is *what is quantized* — the
  temporal delta instead of the activation — which by Theorem 4.3 (`‖x − Q(x)‖² ≤ s²d`) shrinks the
  quantization error because the delta's range is smaller. Measured on this tree: a 12.5× median
  reduction in quantizer step, i.e. ~155× in squared error
  (`docs/modiff_correctness_2026-08-03/FINDINGS.md`).
- Costs extra HBM traffic, which is intrinsic: the `â` read-modify-write plus the `ô` accumulate mean
  ~2.3× the bytes of the non-MoDiff path per large conv. The honest framing is quality-at-low-bits,
  not wall-clock speedup.
- Error compensation (`e_t = a_t − â_t` fed forward) is what stops that error accumulating across
  steps; verified by `test_kernel_correctness.py::modiff_invariants`.

## Script prerequisites

Checked 2026-08-03 by importing each:

| script | status |
|---|---|
| `scripts/sample_diffusion_ldm.py` | **cannot run** — imports `qdiff` (Q-Diffusion), which is neither vendored in this tree nor in `requirements.txt`. Vendor it or drop the script. |
| `scripts/sample_diffusion_ddim.py` | fine — needs `lmdb`, which *is* pinned (`requirements.txt:27`) |
| `scripts/txt2img.py` | fine — needs `opencv-python`, which *is* pinned (`requirements.txt:30`) |

If an import fails on `lmdb`/`cv2`/`omegaconf`/`einops`, the container has simply lost its packages:
run `pip install -r requirements.txt`. That file pins `torchmetrics==0.6.0` deliberately —
`pytorch-lightning==1.4.2` imports `torchmetrics.utilities.data.get_num_classes`, which newer
torchmetrics removed, so installing packages one at a time will miss the constraint and break.

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

# NOTE: needs a REAL checkpoint. With the 856-byte stub in this tree the UNet output is
# identically zero, so every mode produces the same latents and FID is meaningless.
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

### INT4 Implementation

All four limitations previously listed here were **out of date and are removed** — each was checked
against the code on 2026-08-03:

- *"Uses an INT8 backend / unpacks INT4→INT8"* — no. `csrc/kernels/conv/conv2d_int4.cu:52-53`
  instantiates CUTLASS with `cutlass::int4b_t` for both operands; the weight stays packed
  (`weight_packed.data_ptr()` cast to `int4b_t*`, `:167-168`). It is a native s4 convolution.
- *"No native INT4 Tensor Cores (sm_89 supports this)"* — native s4 IMMA is available from sm_75, and
  this repo's target is **sm_86** (A40), where it works. The claim also misidentifies the requirement.
- *"FP32 caches"* — no. `_cache_dtype()` (`integration/kernels/int4_optimized.py:175`) returns fp16
  once calibrated. fp32 is the uncalibrated window only.
- *"FP32 residuals, re-quantized each step"* — the temporal delta is formed and quantized inside one
  kernel (`group_norm_silu_delta_quantize_pack_nhwc` / `step1_static_quantize_pack_int4_fprop`), with
  the `o_hat` accumulate folded into the conv's CUTLASS EVT epilogue (`conv2d_int4_evt_o_hat`). There
  is no fp32 round trip on the calibrated path.

**Real remaining limitation**: the `a_hat`/`o_hat` caches are fp16, not packed int4, so MoDiff's
memory overhead is 2 bytes per activation element rather than 0.5. Storing `a_hat` as int codes is
tracked as a possible future change; it is a bandwidth question, not a correctness one.

**Current performance**: 1.19× speedup (only from INT8 backend, INT4 adds overhead)

**Recommendation**: **Use INT8 mode for production** (fully optimized, 1.18× speedup).
The FID=8.20 that used to appear here is withdrawn -- unreproducible in this checkout, see
the note under Benchmark Results.

**See**: [INT4_ANALYSIS.md](../INT4_ANALYSIS.md) for detailed explanation and fix roadmap.

# Nsight Systems Memory Analysis: why INT8/INT4 use more memory than FP16

Date: 2026-06-22

GPU: NVIDIA A40

Profiler: NVIDIA Nsight Systems 2024.1.1

Primary artifacts:

- Redo report: `integration/results/nsys_memory_redo/NSYS_MEMORY_REDO_REPORT.md`
- Parsed summary: `integration/results/nsys_memory_redo/nsys_memory_summary.json`
- Nsight exports: `integration/results/nsys_memory_redo/profiles/*_s50_b168.{nsys-rep,sqlite}`
- Optimized FP16-cache redo report: `integration/results/nsys_memory_fp16_cache/NSYS_MEMORY_REDO_REPORT.md`
- Optimized parsed summary: `integration/results/nsys_memory_fp16_cache/nsys_memory_summary.json`
- Optimized Nsight exports: `integration/results/nsys_memory_fp16_cache/profiles/*_s50_b168.{nsys-rep,sqlite}`
- Benchmark JSON: `integration/results/nsys_memory_redo/benchmarks/{fp16,int8,int4}/results.json`

## Executive Summary

INT8/INT4 memory is larger than FP16 because the current low-bit implementation does not make the UNet low-bit resident. It adds quantized convolution/linear execution on top of a full-precision activation and cache pipeline.

The full Nsight redo confirms the root cause:

- FP16 produced only 111.7 MiB of Device-to-Device CUDA copies in the profiled 50-step run.
- INT8 produced 15,268.2 MiB of Device-to-Device copies, about 137x FP16.
- INT4 produced 7,901.1 MiB of Device-to-Device copies, about 71x FP16.
- H2D/D2H traffic is effectively unchanged across modes, so the problem is inside the GPU execution path, not image output or dataset movement.

The source-level memory accounting after warmup shows the resident culprit:

- INT8 tracks 10,911.2 MiB of quantized-module resident state at batch 168; 10,377.9 MiB is cache/residual state, while only 532.0 MiB is weights.
- INT4 tracks 6,988.8 MiB; 6,694.4 MiB is cache/residual state, while only 293.1 MiB is weights.

So the weights are smaller, but the implementation keeps large FP32 `a_hat_cache`, FP32 `o_hat_cache`, and, for INT8 Conv, FP32 residual buffers. Those tensors scale with batch size and feature-map size, and they dominate any low-bit weight savings.

## What Changed In This Pass

I made `nsys` usable from the existing Nsight installation:

```bash
/usr/local/bin/nsys -> /opt/nvidia/nsight-compute/2024.1.1/host/target-linux-x64/nsys
nsys --version
# NVIDIA Nsight Systems version 2024.1.1.0
```

I also added repeatable analysis tooling:

- `integration/utils/quant_memory.py`: buckets resident tensors held by INT8/INT4 modules after warmup.
- `integration/benchmarks/benchmark_extended.py`: now records `quant_memory_after_warmup`.
- `integration/benchmarks/benchmark_ldm.py`: now records `quant_memory_after_warmup`.
- `integration/benchmarks/run_nsys_memory_redo.sh`: profiles FP16, INT8, and INT4 and exports SQLite.
- `integration/benchmarks/analyze_nsys_memory.py`: parses Nsight SQLite plus benchmark JSON into JSON/Markdown.

## Full Nsight Redo

Command:

```bash
STEPS=50 BATCH_SIZE=168 NUM_SAMPLES=168 LINEAR_BACKEND=int_gemm \
  integration/benchmarks/run_nsys_memory_redo.sh
```

### Benchmark Timing

| Mode | Samples | Time/sample | Time/step |
|---|---:|---:|---:|
| fp16 | 168 | 0.0777s | 1.55 ms |
| int8 | 168 | 0.0823s | 1.65 ms |
| int4 | 168 | 0.0825s | 1.65 ms |

### Nsight CUDA Memcpy

| Mode | H2D MiB | D2H MiB | D2D MiB | D2D count |
|---|---:|---:|---:|---:|
| fp16 | 2,570.0 | 126.0 | 111.7 | 210 |
| int8 | 2,570.6 | 126.0 | 15,268.2 | 1,264 |
| int4 | 2,570.6 | 126.0 | 7,901.1 | 1,124 |

The D2D copy explosion matches the low-bit implementation shape: repeated channels-last normalization, quantized activation materialization, cache update copies, FP32 residual computation, and FP32 outputs between quantized Conv islands.

### Nsight Host OS Runtime I/O

Nsight OSRT records host API counts and duration for this capture. This export does not include byte counts for `read`/`write`, so the reliable host-I/O signal is time and call count rather than volume.

| Mode | `read` count | `read` ms | `write` count | `write` ms | `open64` count | `open64` ms |
|---|---:|---:|---:|---:|---:|---:|
| fp16 | 15,494 | 2,492.2 | 743 | 1,385.1 | 7,249 | 506.9 |
| int8 | 15,552 | 3,385.6 | 732 | 1,534.2 | 7,239 | 402.4 |
| int4 | 16,860 | 3,450.1 | 765 | 911.8 | 7,368 | 388.8 |

Host I/O is not the reason INT8/INT4 memory is larger:

- `read`/`write` counts are broadly similar across FP16, INT8, and INT4.
- INT8/INT4 do not show a host-side I/O explosion comparable to the CUDA Device-to-Device copy explosion.
- CUDA H2D and D2H byte totals are also effectively unchanged across modes.
- The huge mode-specific delta is internal GPU D2D traffic: 111.7 MiB for FP16, 15,268.2 MiB for INT8, and 7,901.1 MiB for INT4.

So, from Nsight's I/O view, the quantized modes are not spending memory because they read/write more files or transfer more data between CPU and GPU. They spend memory and copy bandwidth inside the GPU while materializing/layout-normalizing quantized activations and maintaining full-precision caches.

### Warmup Resident Quant Memory

| Mode | Tracked quant MiB | Cache/residual MiB | Weight MiB | Scale/bias MiB |
|---|---:|---:|---:|---:|
| fp16 | 0.0 | 0.0 | 0.0 | 0.0 |
| int8 | 10,911.2 | 10,377.9 | 532.0 | 1.3 |
| int4 | 6,988.8 | 6,694.4 | 293.1 | 1.3 |

Top buckets:

| Mode | Bucket | MiB |
|---|---|---:|
| int8 | `conv_a_hat_cache_mib` | 3,683.5 |
| int8 | `conv_residual_buf_mib` | 3,683.5 |
| int8 | `conv_o_hat_cache_mib` | 2,968.9 |
| int8 | `conv_quant_weights_mib` | 450.6 |
| int4 | `conv_a_hat_cache_mib` | 3,683.5 |
| int4 | `conv_o_hat_cache_mib` | 2,968.9 |
| int4 | `conv_quant_weights_mib` | 225.3 |

## Smaller PyTorch Peak-Memory Run

I also ran `benchmark_extended.py` at `batch_size=42` so PyTorch allocated/peak memory could be compared directly:

```bash
python integration/benchmarks/benchmark_extended.py \
  --mode {fp16,int8,int4} \
  --steps 50 \
  --batch_size 42 \
  --num_samples 42 \
  --output_dir integration/results/nsys_memory_redo/extended_b42_s50 \
  --skip_plots
```

| Mode | Time/sample | Setup allocated MB | Peak MB | Peak - setup MB | Tracked quant MiB | Cache/residual MiB | Weight MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| fp16 | 0.089s | 2,596 | 8,406 | 5,810 | 0.0 | 0.0 | 0.0 |
| int8 | 0.091s | 3,976 | 8,400 | 4,424 | 3,035.8 | 2,584.0 | 450.6 |
| int4 | 0.085s | 3,736 | 7,228 | 3,492 | 1,889.6 | 1,663.1 | 225.3 |

At this smaller batch, the same pattern holds: cache/residual memory is much larger than quantized weight memory.

## Source-Level Cause

### 1. Conv MoDiff caches are full precision

`integration/kernels/int8_optimized.py` keeps:

- `a_hat_cache = torch.zeros_like(x)`
- `o_hat_cache = torch.zeros(... dtype=torch.float32)`
- `_residual_buf = torch.empty_like(x)`

In the profiled INT8 run, those three Conv buckets total about 10.34 GiB.

`integration/kernels/int4_optimized.py` keeps the same `a_hat_cache` and `o_hat_cache` structure. INT4 avoids the large persistent Conv residual bucket in this run, but its Conv caches still total about 6.65 GiB.

### 2. Quantized Conv output returns to FP32

Both optimized Conv paths quantize inputs for CUTLASS, run low-bit convolution, and then scale the output back into FP32:

- INT8 calls `conv2d_int8_fprop(...)`, then scales with `weight_scale_channel`.
- INT4 calls `conv2d_int4_fprop(...)`, then scales with `weight_scale_channel`.
- MoDiff updates write into FP32 `o_hat_cache`.

The next layer therefore usually starts from FP32 again, which forces repeated quantize/copy/dequantize work.

### 3. Linear layers duplicate representations

`integration/kernels/int8_linear.py` stores both `weight_fp16` and `weight_int8_t`.

`integration/kernels/int4_linear.py` stores both `weight_fp16` and `weight_packed_t`.

This is not the largest term in this profile, but it explains why low-bit conversion is not a simple FP16-weight replacement.

### 4. Layout normalization creates D2D copies

The Conv modules repeatedly call:

```python
x = x.contiguous(memory_format=torch.channels_last)
```

The model is also converted to channels-last, while some packed weights are restored to standard contiguous buffers for raw CUTLASS access. Nsight confirms the runtime effect: INT8/INT4 spend gigabytes of device-to-device copy traffic that FP16 does not.

## Corrected Root Cause

The memory increase is not a mystery in CUDA allocation, and it is not because INT8/INT4 weights are bigger. The current implementation is a hybrid:

1. Low-bit weights and low-bit GEMM/Conv payloads save memory locally.
2. Full-precision MoDiff caches and residual buffers spend much more memory globally.
3. Linear layers keep fallback FP16 weights as well as packed/int weights.
4. Runtime quantization and layout conversions add large transient D2D traffic.
5. Peak memory is sampled after warmup/sampling, exactly when lazy caches have been created.

## Why INT8/INT4 Do Not Reach 2x/4x Speedup

The expected 2x/4x speedup assumes the runtime is dominated by dense Conv/GEMM arithmetic and that INT8/INT4 replaces FP16 math without adding much overhead. Nsight shows that assumption is false for this implementation.

Measured end-to-end timing:

| Mode | Time/sample | Time/step | Relative to FP16 |
|---|---:|---:|---:|
| fp16 | 0.0777s | 1.55 ms | 1.00x |
| int8 | 0.0823s | 1.65 ms | 0.94x |
| int4 | 0.0825s | 1.65 ms | 0.94x |

Kernel-category totals from the Nsight SQLite exports:

| Category | FP16 ms | INT8 ms | INT4 ms | What it means |
|---|---:|---:|---:|---|
| FP/cuDNN Conv | 38,239.0 | 5,768.8 | 5,764.2 | Low-bit conversion removes most FP Conv time. |
| Low-bit CUTLASS Conv | 0.0 | 3,315.8 | 1,642.2 | The low-bit Conv kernels are faster than the FP Conv work they replace. |
| Quantize/pack/update | 0.0 | 2,236.5 | 2,143.2 | Extra kernels introduced only by quantization. |
| Scale accumulate | 0.0 | 1,608.5 | 1,608.0 | Extra FP32 MoDiff accumulation work. |
| Elementwise/norm/copy | 18,608.0 | 17,832.2 | 20,846.1 | This remains large and is not accelerated by low-bit Conv. |
| Attention | 8,740.9 | 8,580.1 | 8,572.3 | Attention remains FP-style and unchanged. |

These category totals can exceed wall time because kernels overlap and Nsight sums kernel durations, but the ratios explain where the work moved.

The key point: INT8/INT4 did accelerate the convolution part, but the run is not just convolution anymore. The saved FP Conv time is replaced by:

1. Quantization and packing kernels.
2. FP32 cache update kernels.
3. FP32 scale accumulation.
4. More elementwise/copy work.
5. Much more D2D memory movement.
6. More CUDA API and synchronization pressure.

CUDA API counts also show the overhead shift:

| Mode | CUDA kernel launches | `cudaMemcpyAsync` calls | D2D MiB |
|---|---:|---:|---:|
| fp16 | 144,211 | 2,104 | 111.7 |
| int8 | 196,680 | 7,640 | 15,268.2 |
| int4 | 211,698 | 7,500 | 7,901.1 |

That is why INT4 does not become 4x faster than FP16 even though its packed Conv kernel is smaller. The surrounding work is still full precision or memory-bound, and INT4 adds pack/unpack/update overhead. In this profile, INT4's low-bit Conv total is lower than INT8's, but INT4 has more elementwise/norm/copy time and many more kernels, so the end-to-end result is roughly tied with INT8 and slightly slower than FP16.

The practical speed fix is the same direction as the memory fix:

1. Keep activations/cache state low-bit or FP16 between compatible layers instead of returning to FP32 after every Conv.
2. Fuse quantize/pack, Conv, scale, and cache update more aggressively so the low-bit path does not add thousands of extra kernels.
3. Reduce layout conversions and D2D copies by enforcing one channels-last contract.
4. Avoid persistent FP32 residual/cache work where the update can be represented in lower precision.
5. Optimize only after checking the non-Conv ceiling: attention and norm/elementwise work already consume enough time that Conv-only speedups cannot deliver 2x/4x end-to-end.

## Updated Fix

The instrumentation/reporting fix is now in place and should stay: every benchmark can report resident quant memory after warmup, so future changes can tell whether they reduce actual module state or only move allocator peaks around.

The implementation fix should target Conv cache residency first:

1. Convert Conv `a_hat_cache` and `o_hat_cache` from FP32 to FP16 as the first low-risk memory fix. At batch 168, that would target roughly 6.65 GiB of INT4 cache and 6.65 GiB of INT8 cache.
2. Remove or narrow persistent INT8 `_residual_buf`; in this profile it alone costs 3.68 GiB.
3. Only after that, remove duplicate Linear storage where possible:
   - for FP16 fallback-only mode, skip packed/int Linear weights;
   - for true INT GEMM mode, make FP16 fallback storage opt-in.
4. Reduce D2D traffic by enforcing one channels-last layout contract at module boundaries and preallocating quantized activation buffers. The current buffer-pool log reports `0.0 MB`, so it is not absorbing the large activation/copy churn.
5. Longer term, pass `(quantized_tensor, scale)` between compatible quantized layers instead of returning FP32 from every Conv island.

The data says the first real memory win is not further weight packing. It is making MoDiff caches and residual buffers smaller or less persistent.

## Implemented FP16 Cache Optimization

After the first analysis, I implemented the first four requested fixes where they are compatible with the current UNet:

1. Converted calibrated MoDiff Conv `a_hat_cache` and `o_hat_cache` residency from FP32 to FP16.
2. Removed the unnecessary persistent INT8 `_residual_buf` allocation on the calibrated static path.
3. Extended the existing fused `step1_static_quantize_*` and `conv2d_*_fprop_o_hat` CUDA boundaries to operate directly on FP16 resident caches.
4. Kept resident Conv output cache in FP16, but returned FP32 at module boundaries because the surrounding fused ResBlock/GroupNorm path currently expects FP32 weights and activations.

Code touched:

- `csrc/cuda_kernels.cu`
- `integration/kernels/int8_optimized.py`
- `integration/kernels/int4_optimized.py`

The extension was rebuilt with:

```bash
python setup.py build_ext --inplace
```

I then reran the full Nsight experiment:

```bash
OUT_DIR=integration/results/nsys_memory_fp16_cache \
STEPS=50 BATCH_SIZE=168 NUM_SAMPLES=168 LINEAR_BACKEND=int_gemm \
  integration/benchmarks/run_nsys_memory_redo.sh
```

### Old vs New Results

| Mode | Metric | Before | After | Change |
|---|---|---:|---:|---:|
| int8 | Time/step | 1.647 ms | 1.617 ms | 1.8% faster |
| int8 | Tracked quant memory | 10,911.2 MiB | 3,901.4 MiB | -7,009.7 MiB |
| int8 | Cache/residual memory | 10,377.9 MiB | 3,368.2 MiB | -7,009.7 MiB |
| int8 | D2D traffic | 15,268.2 MiB | 8,615.8 MiB | -6,652.4 MiB |
| int4 | Time/step | 1.651 ms | 1.632 ms | 1.2% faster |
| int4 | Tracked quant memory | 6,988.8 MiB | 3,662.6 MiB | -3,326.2 MiB |
| int4 | Cache/residual memory | 6,694.4 MiB | 3,368.2 MiB | -3,326.2 MiB |
| int4 | D2D traffic | 7,901.1 MiB | 1,963.4 MiB | -5,937.8 MiB |

Optimized warmup memory buckets at batch 168:

| Mode | Bucket | MiB |
|---|---|---:|
| int8 | `conv_a_hat_cache_mib` | 1,841.8 |
| int8 | `conv_o_hat_cache_mib` | 1,484.4 |
| int8 | `conv_quant_weights_mib` | 450.6 |
| int4 | `conv_a_hat_cache_mib` | 1,841.8 |
| int4 | `conv_o_hat_cache_mib` | 1,484.4 |
| int4 | `conv_quant_weights_mib` | 225.3 |

### Why Speed Only Improved Slightly

The optimization did what it was meant to do for memory and D2D traffic. Speed improved only modestly because the current graph still returns FP32 to the surrounding ResBlock/GroupNorm path.

Nsight kernel-category deltas:

| Category | INT8 before | INT8 after | INT4 before | INT4 after |
|---|---:|---:|---:|---:|
| Quantize/pack/update | 2,236.5 ms | 1,578.2 ms | 2,143.2 ms | 769.3 ms |
| Scale accumulate | 1,608.5 ms | 1,049.8 ms | 1,608.0 ms | 1,049.8 ms |
| Elementwise/norm/copy | 17,832.2 ms | 18,677.4 ms | 20,846.1 ms | 21,717.6 ms |
| Attention | 8,580.1 ms | 8,583.9 ms | 8,572.3 ms | 8,574.9 ms |
| Low-bit CUTLASS Conv | 3,315.8 ms | 3,311.9 ms | 1,642.2 ms | 1,640.9 ms |

So the cache kernels got cheaper, and D2D traffic fell sharply, but some of that speed win is spent by FP16-cache-to-FP32 boundary casts and unchanged FP32/FP16 non-Conv work. To get a larger speedup, the next step must move the surrounding ResBlock/GroupNorm/activation path to accept FP16 activations or fuse the Conv output cast with the following normalization path.

## Added INT8/INT4 Baselines Without MoDiff

I extended the Nsight runner and analyzer to include `int8_baseline` and `int4_baseline`, which use the same optimized INT kernels, static calibration, and `LINEAR_BACKEND=int_gemm`, but disable MoDiff temporal caching.

Updated tooling:

- `integration/benchmarks/run_nsys_memory_redo.sh`: profiles `fp16 int8 int8_baseline int4 int4_baseline` by default and accepts `MODES="..."`.
- `integration/benchmarks/analyze_nsys_memory.py`: reports the same five-mode default.

Reproduction command for the added baseline profiles:

```bash
OUT_DIR=integration/results/nsys_memory_repro_20260624 \
MODES="int8_baseline int4_baseline" \
STEPS=50 BATCH_SIZE=168 NUM_SAMPLES=168 LINEAR_BACKEND=int_gemm \
  integration/benchmarks/run_nsys_memory_redo.sh
```

Then the combined five-mode report was regenerated with:

```bash
python integration/benchmarks/analyze_nsys_memory.py \
  --profile-dir integration/results/nsys_memory_repro_20260624/profiles \
  --benchmark-dir integration/results/nsys_memory_repro_20260624/benchmarks \
  --output-json integration/results/nsys_memory_repro_20260624/nsys_memory_summary.json \
  --output-md integration/results/nsys_memory_repro_20260624/NSYS_MEMORY_REDO_REPORT.md \
  --modes fp16 int8 int8_baseline int4 int4_baseline
```

Artifacts:

- Combined report: `integration/results/nsys_memory_repro_20260624/NSYS_MEMORY_REDO_REPORT.md`
- Combined summary: `integration/results/nsys_memory_repro_20260624/nsys_memory_summary.json`
- Baseline profiles: `integration/results/nsys_memory_repro_20260624/profiles/*_baseline_s50_b168.{nsys-rep,sqlite}`

### Five-Mode Comparison

| Mode | Time/sample | Time/step | Tracked quant MiB | Cache/residual MiB | Weight MiB | D2D MiB | D2D count |
|---|---:|---:|---:|---:|---:|---:|---:|
| fp16 | 0.0784s | 1.568 ms | 0.0 | 0.0 | 0.0 | 111.7 | 210 |
| int8 | 0.0812s | 1.625 ms | 3,901.4 | 3,368.2 | 532.0 | 8,615.8 | 1,264 |
| int8_baseline | 0.0744s | 1.487 ms | 533.3 | 0.0 | 532.0 | 1,915.2 | 910 |
| int4 | 0.0778s | 1.557 ms | 3,662.6 | 3,368.2 | 293.1 | 1,963.4 | 984 |
| int4_baseline | 0.0701s | 1.402 ms | 294.4 | 0.0 | 293.1 | 1,915.2 | 910 |

### What The Baselines Show

The baseline runs confirm that the large resident memory overhead is specifically MoDiff temporal cache state, not INT weight storage:

- INT8 baseline tracked only 533.3 MiB, almost exactly its 532.0 MiB of weights.
- INT4 baseline tracked only 294.4 MiB, almost exactly its 293.1 MiB of weights.
- Both baseline modes had 0.0 MiB cache/residual residency.
- MoDiff INT8 and INT4 both retained 3,368.2 MiB of cache/residual state after the FP16-cache optimization.

The baselines are faster in this run because they skip MoDiff cache/update work:

- INT8 baseline: 1.487 ms/step vs INT8 MoDiff: 1.625 ms/step.
- INT4 baseline: 1.402 ms/step vs INT4 MoDiff: 1.557 ms/step.

However, the baselines still do not collapse D2D traffic to FP16 levels. Both baseline modes report 1,915.2 MiB of D2D copies, versus 111.7 MiB for FP16. That residual D2D cost comes from the low-bit execution path itself: quantized activation materialization, layout normalization, pack/unpack movement, and FP32/FP16 boundaries around the quantized Conv islands.

So the baseline comparison sharpens the root cause:

1. MoDiff temporal caching is the source of the multi-GiB resident cache memory.
2. Low-bit execution, even without MoDiff, still introduces much more D2D traffic than FP16.
3. Removing or narrowing MoDiff caches fixes resident memory, but larger speedups still require reducing quantization/layout/boundary movement and moving more of the surrounding graph into a compatible low-bit or FP16 activation contract.

## FP16 Baseline Boundary Rerun

I then removed the avoidable FP32 module-boundary returns from the non-MoDiff INT8/INT4 baselines. The current extension still performs some internal FP32 work for scale/dequant math, but the baseline Conv and Linear wrappers now hand FP16 tensors back to the surrounding UNet path.

Code changes:

- `csrc/cuda_kernels.cu`: allows `conv2d_int8_fprop_no_ohat_prealloc` and `conv2d_int4_fprop_no_ohat_prealloc` to write scaled Conv outputs directly into FP16 output buffers via `scale_store_half_kernel`.
- `integration/kernels/int8_optimized.py` and `integration/kernels/int4_optimized.py`: baseline standard paths avoid unconditional FP32 input promotion and use preallocated FP16 output buffers when enabled.
- `integration/kernels/int8_linear.py` and `integration/kernels/int4_linear.py`: baseline Linear paths no longer force `.float()` outputs when FP16 output mode is enabled.
- `integration/benchmarks/benchmark_ldm.py`: enables FP16 output mode only for `int8_baseline` and `int4_baseline`.
- `integration/fused_ops/fused_resblock.py`: casts functional GroupNorm affine tensors to the activation dtype so FP16 activations can safely flow through fused ResBlocks.

Validation before profiling:

```bash
python -m py_compile integration/benchmarks/analyze_nsys_memory.py integration/benchmarks/benchmark_ldm.py integration/kernels/int8_optimized.py integration/kernels/int4_optimized.py integration/kernels/int8_linear.py integration/kernels/int4_linear.py integration/fused_ops/fused_resblock.py
bash -n integration/benchmarks/run_nsys_memory_redo.sh
python setup.py build_ext --inplace
python integration/benchmarks/benchmark_ldm.py --mode int8_baseline --steps 1 --batch_size 1 --num_samples 1 --output_dir /tmp/modiff_smoke_int8_base_fp16 --linear_backend int_gemm
python integration/benchmarks/benchmark_ldm.py --mode int4_baseline --steps 1 --batch_size 1 --num_samples 1 --output_dir /tmp/modiff_smoke_int4_base_fp16 --linear_backend int_gemm
```

Full rerun command:

```bash
OUT_DIR=integration/results/nsys_memory_repro_20260624_fp16_baseline \
STEPS=50 BATCH_SIZE=168 NUM_SAMPLES=168 LINEAR_BACKEND=int_gemm \
  integration/benchmarks/run_nsys_memory_redo.sh
```

Artifacts:

- Report: `integration/results/nsys_memory_repro_20260624_fp16_baseline/NSYS_MEMORY_REDO_REPORT.md`
- Summary: `integration/results/nsys_memory_repro_20260624_fp16_baseline/nsys_memory_summary.json`
- Profiles: `integration/results/nsys_memory_repro_20260624_fp16_baseline/profiles/*_s50_b168.{nsys-rep,sqlite}`

### FP16 Boundary Results

| Mode | Time/sample | Time/step | Tracked quant MiB | Cache/residual MiB | Weight MiB | D2D MiB | D2D count |
|---|---:|---:|---:|---:|---:|---:|---:|
| fp16 | 0.0778s | 1.556 ms | 0.0 | 0.0 | 0.0 | 111.7 | 210 |
| int8 | 0.0814s | 1.628 ms | 3,901.4 | 3,368.2 | 532.0 | 8,615.8 | 1,264 |
| int8_baseline | 0.0734s | 1.467 ms | 533.3 | 0.0 | 532.0 | 1,915.2 | 910 |
| int4 | 0.0767s | 1.534 ms | 3,662.6 | 3,368.2 | 293.1 | 1,963.4 | 984 |
| int4_baseline | 0.0683s | 1.365 ms | 294.4 | 0.0 | 293.1 | 1,915.2 | 910 |

### Total CUDA I/O

I updated `integration/benchmarks/analyze_nsys_memory.py` to compute total CUDA memcpy I/O directly from each Nsight SQLite export:

```text
total_cuda_io = Host-to-Device + Device-to-Host + Device-to-Device
```

Measured totals from `integration/results/nsys_memory_repro_20260624_fp16_baseline/nsys_memory_summary.json`:

| Mode | Total CUDA I/O MiB | Total memcpy count | Total memcpy ms | D2D MiB | D2D count |
|---|---:|---:|---:|---:|---:|
| fp16 | 2,807.7 | 2,104 | 285.3 | 111.7 | 210 |
| int8 | 11,312.5 | 7,640 | 315.2 | 8,615.8 | 1,264 |
| int8_baseline | 4,611.8 | 3,438 | 440.4 | 1,915.2 | 910 |
| int4 | 4,660.1 | 7,360 | 439.1 | 1,963.4 | 984 |
| int4_baseline | 4,611.8 | 3,438 | 392.0 | 1,915.2 | 910 |

Relative to FP16, each baseline adds:

```text
baseline_total_extra_io = 4,611.8 - 2,807.7 = 1,804.1 MiB
baseline_total_extra_count = 3,438 - 2,104 = 1,334 memcpy events
baseline_d2d_extra = 1,915.2 - 111.7 = 1,803.5 MiB
```

So almost all extra baseline CUDA I/O bytes are D2D:

```text
1,803.5 / 1,804.1 = 99.97%
```

### Updated Interpretation

The previous question was why the INT8/INT4 baselines were still higher than FP16. The answer was the FP32 boundary contract: the baseline low-bit Conv and Linear islands were returning FP32 tensors into a mostly FP16 benchmark path, so the graph paid extra cast/copy and memory-bandwidth cost around each quantized island.

After moving those baseline boundaries to FP16, both baselines are now faster than FP16:

| Comparison | Before | After |
|---|---:|---:|
| INT8 baseline time/step | 1.487 ms | 1.467 ms |
| INT4 baseline time/step | 1.402 ms | 1.365 ms |
| FP16 time/step | 1.568 ms | 1.556 ms |

The baselines still report much higher D2D traffic than FP16 (`1,915.2 MiB` vs `111.7 MiB`) because low-bit execution still materializes quantized activations, normalizes layouts, packs INT4 activations, and crosses quantized Conv/Linear island boundaries. D2D means GPU device-to-device memory copies: copies or layout movements where both source and destination are on the GPU, not CPU transfer. In this profile it is mainly a symptom of extra on-GPU tensor movement introduced by the quantized path.

So the refined conclusion is:

1. The old baseline speed gap was mostly an avoidable FP32 boundary issue.
2. The baseline resident memory is already low and dominated by quantized weights.
3. Baseline D2D remains high because the low-bit path still has quantize/pack/layout/boundary movement.
4. MoDiff modes remain memory-heavy because they intentionally keep temporal `a_hat` and `o_hat` caches.

## Redone CUDA I/O Usage Analysis

I remeasured CUDA memcpy I/O from the latest Nsight SQLite exports and generated tables plus plots under:

- Analysis report: `integration/results/nsys_memory_repro_20260624_fp16_baseline/io_usage/CUDA_IO_USAGE_ANALYSIS.md`
- Summary JSON: `integration/results/nsys_memory_repro_20260624_fp16_baseline/io_usage/cuda_io_usage_summary.json`
- CSV tables: `integration/results/nsys_memory_repro_20260624_fp16_baseline/io_usage/cuda_io_total.csv`, `cuda_io_by_kind.csv`, `cuda_io_by_runtime.csv`
- Plots: `integration/results/nsys_memory_repro_20260624_fp16_baseline/io_usage/plots/`

The measurement command was:

```bash
python integration/benchmarks/analyze_nsys_io_usage.py \
  --profile-dir integration/results/nsys_memory_repro_20260624_fp16_baseline/profiles \
  --output-dir integration/results/nsys_memory_repro_20260624_fp16_baseline/io_usage \
  --steps 50 \
  --batch-size 168 \
  --modes fp16 int8 int8_baseline int4 int4_baseline
```

### Total CUDA I/O, Redone

| Mode | Total CUDA I/O MiB | Total memcpy count | Total memcpy ms | D2D MiB | D2D count | Extra MiB vs FP16 | Extra D2D MiB vs FP16 |
|---|---:|---:|---:|---:|---:|---:|---:|
| fp16 | 2,807.7 | 2,104 | 285.3 | 111.7 | 210 | 0.0 | 0.0 |
| int8 | 11,312.5 | 7,640 | 315.2 | 8,615.8 | 1,264 | 8,504.8 | 8,504.1 |
| int8_baseline | 4,611.8 | 3,438 | 440.4 | 1,915.2 | 910 | 1,804.1 | 1,803.5 |
| int4 | 4,660.1 | 7,360 | 439.1 | 1,963.4 | 984 | 1,852.4 | 1,851.7 |
| int4_baseline | 4,611.8 | 3,438 | 392.0 | 1,915.2 | 910 | 1,804.1 | 1,803.5 |

### Where The I/O Is Used

| Mode | H2D MiB | H2D count | D2H MiB | D2H count | D2D MiB | D2D count |
|---|---:|---:|---:|---:|---:|---:|
| fp16 | 2,570.0 | 1,461 | 126.0 | 433 | 111.7 | 210 |
| int8 | 2,570.6 | 5,943 | 126.0 | 433 | 8,615.8 | 1,264 |
| int8_baseline | 2,570.6 | 2,095 | 126.0 | 433 | 1,915.2 | 910 |
| int4 | 2,570.6 | 5,943 | 126.0 | 433 | 1,963.4 | 984 |
| int4_baseline | 2,570.6 | 2,095 | 126.0 | 433 | 1,915.2 | 910 |

### Runtime Count And Time

Nsight maps every CUDA memcpy event in this capture to the same CUDA runtime API name, `cudaMemcpyAsync_v3020`:

| Mode | Runtime name | MiB | Count | ms |
|---|---|---:|---:|---:|
| fp16 | `cudaMemcpyAsync_v3020` | 2,807.7 | 2,104 | 285.3 |
| int8 | `cudaMemcpyAsync_v3020` | 11,312.5 | 7,640 | 315.2 |
| int8_baseline | `cudaMemcpyAsync_v3020` | 4,611.8 | 3,438 | 440.4 |
| int4 | `cudaMemcpyAsync_v3020` | 4,660.1 | 7,360 | 439.1 |
| int4_baseline | `cudaMemcpyAsync_v3020` | 4,611.8 | 3,438 | 392.0 |

The host/device transfer volume is effectively fixed across modes: about `2,570 MiB` H2D and `126 MiB` D2H. The mode-specific I/O delta is therefore almost entirely Device-to-Device.

For both baselines:

```text
extra total I/O vs FP16 = 4,611.8 - 2,807.7 = 1,804.1 MiB
extra D2D vs FP16       = 1,915.2 - 111.7   = 1,803.5 MiB
extra bytes explained by D2D = 1,803.5 / 1,804.1 = 99.97%
```

That is why INT8 and INT4 baselines still use significantly more CUDA I/O even after all baseline FP32 outputs were moved to FP16. The remaining extra I/O is not CPU-GPU traffic; it is on-GPU tensor movement created by the low-bit implementation: quantized activation materialization, INT4 packing, layout normalization, and copies at quantized Conv/Linear island boundaries.

The most useful plots are:

### Total CUDA memcpy I/O

![Total CUDA memcpy I/O](nsys_memory_repro_20260624_fp16_baseline/io_usage/plots/total_cuda_io.png)

### CUDA memcpy I/O by transfer kind

![CUDA memcpy I/O by transfer kind](nsys_memory_repro_20260624_fp16_baseline/io_usage/plots/cuda_io_by_kind.png)

### D2D memcpy event count

![D2D memcpy event count](nsys_memory_repro_20260624_fp16_baseline/io_usage/plots/d2d_count.png)

### Baseline D2D copies by repeated tensor size

![Baseline D2D copies by repeated tensor size](nsys_memory_repro_20260624_fp16_baseline/io_usage/plots/d2d_top_sizes_baselines.png)

### D2D traffic by repeated tensor-size bucket

![D2D traffic by repeated tensor-size bucket](nsys_memory_repro_20260624_fp16_baseline/io_usage/plots/d2d_size_heatmap.png)

### INT4 Versus INT4 Baseline

`int4` and `int4_baseline` still have almost the same total CUDA memcpy volume, but they do not have the same resident memory:

| Mode | Tracked quant MiB | Cache/residual MiB | Weight MiB | Total CUDA I/O MiB |
|---|---:|---:|---:|---:|
| int4 | 3,662.6 | 3,368.2 | 293.1 | 4,660.1 |
| int4_baseline | 294.4 | 0.0 | 293.1 | 4,611.8 |

The implementation check explains the split: the baseline owns packed INT4/INT8 weights but no MoDiff temporal caches, while MoDiff owns the same low-bit weights plus persistent `a_hat`/`o_hat` cache state. CUDA memcpy I/O is dominated by the shared low-bit island movement, so the byte totals stay close. Resident memory is dominated by MoDiff cache tensors, so memory usage separates sharply.

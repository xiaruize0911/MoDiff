# AWQ Attention Experiment And INT8 IO Analysis

Date: 2026-06-30
GPU: NVIDIA A40

## 1. AWQ GEMM / Attention-Layer Experiment

### What Was Actually Used

AWQ provides:

- W8A8 GEMM kernels: used.
- `single_query_attention`: not used for LDM full self-attention.

The reason `single_query_attention` is not used is semantic, not just plumbing. AWQ's kernel is an autoregressive decode kernel for one query plus KV cache. LDM `AttentionBlock` attends all spatial tokens at once. The core attention in this repo is already PyTorch `scaled_dot_product_attention`, which dispatches fused attention kernels for these shapes.

For attention layers, this experiment therefore uses:

- AWQ W8A8 GEMM for `qkv` and `proj_out` Conv1d-1x1 projections.
- Existing PyTorch fused SDPA for the full-token attention core.
- Existing MoDiff/CUTLASS INT8 Conv2d for spatial Conv2d.

Layer inventory:

- Full inventory: `integration/results/awq_attention_and_io/AWQ_ATTENTION_LAYER_INVENTORY.md`
- JSON: `integration/results/awq_attention_and_io/awq_attention_layer_inventory.json`

### Layer Ownership Summary

| Owner / kernel path | Count | Notes |
|---|---:|---|
| AWQ GEMM for attention Conv1d-1x1 | 42 | `qkv` and `proj_out` projections |
| AWQ GEMM backend for `nn.Linear` | 37 | timestep/embedding linear layers |
| PyTorch `scaled_dot_product_attention` | 21 | full spatial-token attention core |
| MoDiff/CUTLASS INT8 Conv2d | 70 | spatial 3x3 Conv2d layers |

### Benchmark Command

```bash
PYTHONPATH=/workspace/MoDiff:/workspace/llm-awq/awq/kernels \
python integration/benchmarks/benchmark_ldm.py \
  --mode int8_awq_full_baseline \
  --steps 10 --batch_size 4 --num_samples 8 \
  --output_dir integration/results/awq_attention_and_io/awq_attention_rerun \
  --skip_calibration \
  --calibration integration/calibration/int8_calibration.pt
```

### Benchmark Result

| Mode | Attention projections | Attention core | Time/sample | Time/step |
|---|---|---|---:|---:|
| INT8 AWQ attention/full-applicable baseline | AWQ W8A8 GEMM | PyTorch SDPA | 0.1068 s | 10.68 ms |

Comparison to the previous same-setup runs:

| Mode | Linear | Attention Conv1d projections | Time/sample | Time/step |
|---|---|---|---:|---:|
| FP16 | FP16 | FP16 | 0.1557 s | 15.57 ms |
| INT8 baseline | FP16 | FP16 | 0.1054 s | 10.54 ms |
| INT8 AWQ linear-only baseline | AWQ | FP16 | 0.0962 s | 9.62 ms |
| INT8 AWQ attention/full-applicable baseline | AWQ | AWQ | 0.1068 s | 10.68 ms |

Takeaway: AWQ is useful for the regular `nn.Linear` layers, but converting attention Conv1d projections to AWQ does not help this LDM run. It is slightly slower than the current INT8 baseline and clearly slower than AWQ-linear-only. The attention projection shapes include small token counts, so quantization/GEMM launch overhead eats the benefit.

## 2. Nsight / Nsight Systems IO Analysis, Ignoring AWQ

Primary source checked first: `integration/results/nsys_detailed_report.md`.

Nsight Systems artifacts used:

- `integration/results/nsys_memory_repro_20260624_fp16_baseline/profiles/fp16_s50_b168.sqlite`
- `integration/results/nsys_memory_repro_20260624_fp16_baseline/profiles/int8_baseline_s50_b168.sqlite`
- `integration/results/nsys_memory_repro_20260624_fp16_baseline/profiles/int8_s50_b168.sqlite`
- `integration/results/nsys_memory_repro_20260624_fp16_baseline/io_usage/CUDA_IO_USAGE_ANALYSIS.md`

Nsight Compute (`ncu`) was also attempted, but hardware counters are blocked in this container:

```text
ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters
```

So the byte-level IO evidence below comes from Nsight Systems CUDA memcpy traces, plus source mapping and kernel names.

### CUDA IO Summary

50 DDIM steps, batch size 168.

| Mode | Total CUDA memcpy | D2D memcpy | D2D count | Extra D2D vs FP16 |
|---|---:|---:|---:|---:|
| FP16 | 2,807.7 MiB | 111.7 MiB | 210 | 0.0 MiB |
| INT8 baseline | 4,611.8 MiB | 1,915.2 MiB | 910 | +1,803.5 MiB |
| INT8 MoDiff | 11,312.5 MiB | 8,615.8 MiB | 1,264 | +8,504.1 MiB |

The extra IO is almost entirely Device-to-Device GPU traffic. H2D/D2H bytes are basically unchanged; this is not dataset/image/file IO.

### Components Creating Extra IO

| Component | Evidence | IO introduced |
|---|---|---|
| Channels-last/layout normalization | `OptimizedInt8Conv2d.forward` calls `x.contiguous(memory_format=torch.channels_last)` when layout does not match | D2D copies of activation tensors |
| FP32 activation materialization before quantization | INT8 Conv standard path casts `x.float()` before `scale_quantize_int8` | extra activation reads/writes and possible D2D copies |
| INT8 activation quantization island | `scale_quantize_int8_kernel`, 7,000 launches in INT8 baseline | reads FP32/FP16 activation, writes INT8 activation |
| INT8 Conv output scale/store | `scale_store_half_kernel`, 7,000 launches in INT8 baseline | reads INT32/FP32 conv output/scales, writes FP16 output |
| Spatial INT8 Conv2d itself | CUTLASS implicit GEMM conv, 7,000 launches in INT8 baseline | reads INT8 activation + INT8 weights, writes output buffer |
| Bias/add and dtype/layout copies after Conv | PyTorch add/copy kernels remain high in INT8 baseline | extra elementwise reads/writes after the quantized island |
| MoDiff activation cache update | `static_quantize_and_update_ahat_kernel_int8_half_cache`, 6,860 launches in INT8 MoDiff | reads current activation + `a_hat_cache`, writes INT8 delta and updates cache |
| MoDiff output cache accumulation | `scale_accumulate_half_cache_kernel`, 6,860 launches in INT8 MoDiff | reads Conv result/scales and updates `o_hat_cache` |
| Full-precision resident caches | `conv_a_hat_cache`, `conv_o_hat_cache`, linear caches in benchmark memory report | large resident tensors and repeated cache traffic |
| Linear true INT8 path | `round`, `clamp`, `.to(int8)`, `abs/amax` in `OptimizedInt8Linear._int8_gemm_linear` | many small PyTorch kernels around GEMM when `linear_backend=int_gemm` |

### Repeated D2D Size Buckets

Largest INT8 baseline D2D buckets:

| Copy size | Count | Total | Approx count/step | Likely source |
|---:|---:|---:|---:|---|
| 20.25 MiB | 46 | 931.5 MiB | 0.92 | large Conv weight/layout or activation materialization |
| 40.50 MiB | 10 | 405.0 MiB | 0.20 | largest Conv tensors |
| 5.06 MiB | 42 | 212.6 MiB | 0.84 | medium Conv tensors |
| 10.13 MiB | 10 | 101.3 MiB | 0.20 | medium-large Conv tensors |
| 30.38 MiB | 2 | 60.8 MiB | 0.04 | setup/layout copy of large tensors |

Largest INT8 MoDiff D2D buckets:

| Copy size | Count | Total | Approx count/step | Likely source |
|---:|---:|---:|---:|---|
| 63.0 MiB | 38 | 2,394.0 MiB | 0.76 | activation/cache tensors |
| 126.0 MiB | 12 | 1,512.0 MiB | 0.24 | high-resolution activation/cache tensors |
| 31.5 MiB | 40 | 1,260.0 MiB | 0.80 | activation/cache tensors |
| 20.25 MiB | 46 | 931.5 MiB | 0.92 | Conv/layout tensors |
| 189.0 MiB | 2 | 378.0 MiB | 0.04 | largest setup/cache/layout tensors |

The size pattern is consistent with the source: baseline INT8 pays for low-bit Conv islands and layout/materialization; MoDiff INT8 adds full activation/cache tensors.

### Kernel Evidence

INT8 baseline top extra kernels:

| Kernel/category | Count | Total time |
|---|---:|---:|
| CUTLASS INT8 implicit GEMM Conv2d | 7,000 | 3,117.1 ms |
| `scale_quantize_int8_kernel` | 7,000 | 797.7 ms |
| `scale_store_half_kernel` | 7,000 | 825.2 ms |
| PyTorch direct-copy / elementwise copy kernels | many | several seconds total |

INT8 MoDiff extra kernels:

| Kernel/category | Count | Total time |
|---|---:|---:|
| CUTLASS INT8 implicit GEMM Conv2d | 7,280 | 3,179.0 ms |
| `static_quantize_and_update_ahat_kernel_int8_half_cache` | 6,860 | 1,529.5 ms |
| `scale_accumulate_half_cache_kernel` | 6,860 | 1,050.1 ms |
| PyTorch copy/binary/add kernels | many | several seconds total |

Attention is not the cause of the extra INT8 IO. The main fused attention kernel count/time is basically unchanged:

| Mode | Attention kernel count | Total time |
|---|---:|---:|
| FP16 | 750 | 8,680.8 ms |
| INT8 baseline | 750 | 8,553.9 ms |
| INT8 MoDiff | 750 | 8,518.1 ms |

### Source Pointers

- `integration/kernels/int8_optimized.py:270`: INT8 Conv forward starts.
- `integration/kernels/int8_optimized.py:275`: layout normalization to channels-last.
- `integration/kernels/int8_optimized.py:320`: FP32 activation materialization before quantization.
- `integration/kernels/int8_optimized.py:321`: `scale_quantize_int8`.
- `integration/kernels/int8_optimized.py:336`: INT8 Conv2d fused output-buffer path.
- `integration/kernels/int8_optimized.py:431`: MoDiff static quantize/update `a_hat`.
- `integration/kernels/int8_optimized.py:440`: MoDiff INT8 Conv2d accumulating into `o_hat_cache`.
- `integration/kernels/int8_linear.py:140`: linear activation absmax/scale.
- `integration/kernels/int8_linear.py:154`: linear round/clamp/to-int8 chain.

### Bottom Line

Our INT8 path has higher IO than FP16 because it is not a low-bit-resident graph. It repeatedly crosses quantized islands:

1. Keep activations in FP16/FP32.
2. Convert/layout-copy them for INT8 Conv.
3. Quantize to INT8.
4. Run INT8 Conv.
5. Scale/store back to FP16/FP32.
6. Run FP-style norm/activation/attention.
7. Repeat.

MoDiff adds another layer of IO by maintaining and updating activation/output caches. That explains the jump from `1,915.2 MiB` D2D in INT8 baseline to `8,615.8 MiB` D2D in INT8 MoDiff.

The fixes should target:

1. Keep compatible activations as `(int8_tensor, scale)` across adjacent quantized layers.
2. Fuse quantize + Conv + scale/store + bias where possible.
3. Eliminate avoidable `contiguous(memory_format=channels_last)` copies.
4. Store MoDiff caches in FP16 or quantized form where quality allows.
5. Avoid the true INT8 linear backend until its PyTorch quantization chain is fused.

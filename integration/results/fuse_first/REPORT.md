# INT8 Conv2d Fuse-First Report

## Plan

1. Inspect the CUTLASS INT8 Conv2d path used by `benchmark_ldm.py --mode int8_baseline`.
2. Check whether the per-channel weight scale can be folded directly into the existing CUTLASS Conv2d epilogue.
3. Implement the first low-risk fusion in the current path.
4. Rebuild the extension and run correctness checks.
5. Benchmark and profile the changed kernel sequence with Nsight Systems.

## What Was Fused

The current Conv2d baseline path is:

1. `scale_quantize_int8_kernel`: FP32 activation -> INT8 activation.
2. CUTLASS INT8 Conv2d: INT8 x INT8 -> FP32 conv output.
3. `scale_store_half_kernel`: FP32 conv output * per-output-channel weight scale -> FP16 output.
4. PyTorch elementwise add: FP16 output + bias -> FP16 output.

This patch fuses steps 3 and 4 for biased INT8 Conv2d layers:

`scale_bias_store_half_kernel`: FP32 conv output * per-channel weight scale + bias -> FP16 output.

The CUTLASS Conv2d still writes the temporary FP32 output. The reason is that the current code uses legacy `DefaultConv2dFprop` with `LinearCombination`, whose epilogue handles scalar alpha/beta scaling but not this project's per-output-channel `weight_scale[ch]` without a custom epilogue or a newer visitor-based Conv2d path.

## Code Changes

- Added `scale_bias_store_half_kernel` and `scale_bias_store_kernel` in `csrc/cuda_kernels.cu`.
- Added `conv2d_int8_fprop_no_ohat_prealloc_bias(...)` in `csrc/cuda_kernels.cu`.
- Bound the new function in `csrc/pybind.cpp`.
- Updated `OptimizedInt8Conv2d._forward_standard()` in `integration/kernels/int8_optimized.py` to call the fused-bias path for `standard_output_fp16` when bias exists.

## Correctness Check

Command: focused random Conv2d A/B test against the old path plus Python bias add.

Result:

| Bias dtype | Max abs diff |
|---|---:|
| FP32 | 0.0078125 |
| FP16 | 0.0078125 |

The result is not bit-exact because the old path rounds after scale-store to FP16 and then adds bias, while the fused path adds bias in FP32 and rounds once to FP16.

## Microbenchmark

Each result times the post-quantized Conv2d call. Old path is `conv2d_int8_fprop_no_ohat_prealloc(...)` followed by PyTorch bias add. New path is `conv2d_int8_fprop_no_ohat_prealloc_bias(...)`.

| Shape `[N,C,H,W] -> K` | Old + Torch bias | Fused bias | Speedup |
|---|---:|---:|---:|
| `[2,320,64,64] -> 320` | 0.1834 ms | 0.1624 ms | 1.130x |
| `[2,640,32,32] -> 640` | 0.0854 ms | 0.0763 ms | 1.120x |
| `[2,1280,16,16] -> 1280` | 0.1249 ms | 0.1198 ms | 1.042x |

## Nsight Systems Check

Capture:

- `integration/results/fuse_first/int8_bias_fuse_micro.nsys-rep`
- `integration/results/fuse_first/int8_bias_fuse_micro.sqlite`

Kernel summary for the mixed old/new micro profile:

| Kernel | Instances | Avg time |
|---|---:|---:|
| CUTLASS INT8 Conv2d | 20 | 132.3 us |
| `scale_store_half_kernel` | 10 | 29.1 us |
| `scale_bias_store_half_kernel<__half>` | 10 | 28.9 us |
| PyTorch elementwise add | 10 | 20.2 us |

Interpretation: the fused path replaces `scale_store_half_kernel + PyTorch elementwise add` with one `scale_bias_store_half_kernel`. This removes one full FP16 output read and write per biased Conv2d call.

## Further Fuse Update

The second pass removes the FP32 temporary for selected INT8 baseline Conv2d layers. Instead of launching CUTLASS INT8 Conv2d into FP32 and then launching a scale/store kernel, the new path uses a CUTLASS epilogue that reads a broadcasted per-channel FP16 scale vector as the epilogue source operand:

`CUTLASS INT8 Conv2d + dequant epilogue`: INT8 x INT8 accumulator * activation inverse scale * per-channel weight scale -> FP16 output.

This is exposed as `conv2d_int8_fprop_dequant_fp16_prealloc(...)`.

Bias is not included in this deeper epilogue because the legacy CUTLASS device wrapper gives the epilogue only one source tensor. For biased layers, the benchmark path now uses a measured heuristic:

- Use the deeper CUTLASS FP16 dequant epilogue when there is no bias.
- Use it for large biased outputs, currently `output.numel() >= 2_000_000`.
- Keep the first fused-bias postprocess path for smaller biased outputs, where fusing bias was faster than the deeper epilogue plus a separate bias add.

## Further Fuse Code Changes

- Added `Int8DequantScaleSource<8>` epilogue op in `csrc/cuda_kernels.cu`.
- Added `Conv2dInt8DequantFp16Op` using `DefaultConv2dFprop` with FP16 output.
- Added `conv2d_int8_fprop_dequant_fp16_prealloc(...)`.
- Bound the new symbol in `csrc/pybind.cpp`.
- Added `weight_scale_channel_half` in `OptimizedInt8Conv2d` and refreshed it after SmoothQuant calibration.
- Updated `_forward_standard()` to use the deeper fuse where it wins and fallback to the fused-bias path otherwise.

## Further Fuse Correctness

Focused no-bias A/B against old `conv2d_int8_fprop_no_ohat_prealloc(...)`:

| Check | Result |
|---|---:|
| Max abs diff | 0.0078125 |
| Mean abs diff | 0.000302 |

The difference is from using cached FP16 per-channel scales in the CUTLASS epilogue. The old postprocess path multiplies by FP32 scales and then stores FP16.

## Further Fuse Microbenchmark

No-bias comparison, old scale-store path versus direct FP16 dequant epilogue:

| Shape `[N,C,H,W] -> K` | Old scale-store | Deep fused | Speedup |
|---|---:|---:|---:|
| `[2,320,64,64] -> 320` | 0.1613 ms | 0.1206 ms | 1.337x |
| `[2,640,32,32] -> 640` | 0.0759 ms | 0.0639 ms | 1.187x |
| `[2,1280,16,16] -> 1280` | 0.1212 ms | 0.1160 ms | 1.045x |

Biased comparison, previous fused-bias path versus current heuristic:

| Shape `[N,C,H,W] -> K` | Fused-bias path | Current heuristic | Speedup | Deep used |
|---|---:|---:|---:|---|
| `[2,320,64,64] -> 320` | 0.1619 ms | 0.1422 ms | 1.138x | yes |
| `[2,640,32,32] -> 640` | 0.0765 ms | 0.0766 ms | 0.999x | no |
| `[2,1280,16,16] -> 1280` | 0.1195 ms | 0.1195 ms | 1.000x | no |

## Further Fuse Nsight Systems Check

Capture:

- `integration/results/fuse_first/int8_deep_fuse_micro.nsys-rep`
- `integration/results/fuse_first/int8_deep_fuse_micro.sqlite`

Kernel summary for the mixed fused-bias/deep-fused micro profile:

| Kernel | Instances | Avg time |
|---|---:|---:|
| Old CUTLASS INT8 Conv2d to FP32 | 10 | 132.2 us |
| Deep CUTLASS INT8 Conv2d to FP16 | 10 | 120.0 us |
| `scale_bias_store_half_kernel<__half>` | 10 | 29.5 us |
| PyTorch bias add after deep fuse | 10 | 21.8 us |

Interpretation: on the deep-fused calls, `scale_store_half_kernel` is gone. The dequant scale/store work is inside the CUTLASS epilogue, and the remaining extra kernel is only the bias add for biased layers.

## LDM Smoke Benchmark

Command:

```bash
python integration/benchmarks/benchmark_ldm.py --mode int8_baseline --steps 1 --num_samples 1 --batch_size 1 --skip_calibration --output_dir integration/results/fuse_first/ldm_smoke
```

Result:

| Mode | Samples | Steps | Time/sample | Time/step |
|---|---:|---:|---:|---:|
| `int8_baseline` | 1 | 1 | 0.242 s | 241.57 ms |

Output directory:

- `integration/results/fuse_first/ldm_smoke/`

## LDM Benchmark

Command:

```bash
python integration/benchmarks/benchmark_ldm.py --mode fp16 --steps 20 --num_samples 4 --batch_size 1 --output_dir integration/results/fuse_first/ldm_benchmark_fp16_s20_n4
python integration/benchmarks/benchmark_ldm.py --mode int8_baseline --steps 20 --num_samples 4 --batch_size 1 --skip_calibration --output_dir integration/results/fuse_first/ldm_benchmark_int8_baseline_s20_n4
```

Environment:

- GPU: NVIDIA A40
- Config: `configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml`
- Checkpoint: `models/ldm/lsun_churches256/model.ckpt`
- Batch size: 1
- Samples: 4
- DDIM steps: 20
- INT8 linear backend: `fp16`
- INT8 calibration: `integration/calibration/int8_calibration.pt`

Results:

| Mode | Total time | Time/sample | Time/step | Speedup vs FP16 |
|---|---:|---:|---:|---:|
| `fp16` | 3.796 s | 0.949 s | 47.45 ms | 1.00x |
| `int8_baseline` | 3.005 s | 0.751 s | 37.56 ms | 1.26x |

INT8 tracked quant memory after warmup:

| Component | MiB |
|---|---:|
| Total tracked | 533.265 |
| Quant weights | 531.984 |
| Scale and bias state | 1.280 |
| Cache/residual | 0.000 |

Raw result files:

- `integration/results/fuse_first/ldm_benchmark_fp16_s20_n4/results.json`
- `integration/results/fuse_first/ldm_benchmark_int8_baseline_s20_n4/results.json`

## LDM Benchmark, Large Batch

Command:

```bash
python integration/benchmarks/benchmark_ldm.py --mode fp16 --steps 200 --num_samples 168 --batch_size 42 --output_dir integration/results/fuse_first/ldm_benchmark_fp16_s200_n168_b42
python integration/benchmarks/benchmark_ldm.py --mode int8_baseline --steps 200 --num_samples 168 --batch_size 42 --skip_calibration --output_dir integration/results/fuse_first/ldm_benchmark_int8_baseline_s200_n168_b42
```

Environment:

- GPU: NVIDIA A40
- Config: `configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml`
- Checkpoint: `models/ldm/lsun_churches256/model.ckpt`
- Batch size: 42
- Samples: 168
- DDIM steps: 200
- INT8 linear backend: `fp16`
- INT8 calibration: `integration/calibration/int8_calibration.pt`

Results:

| Mode | Total time | Time/sample | Time/step | Speedup vs FP16 |
|---|---:|---:|---:|---:|
| `fp16` | 50.596 s | 0.301 s | 1.506 ms | 1.00x |
| `int8_baseline` | 49.305 s | 0.293 s | 1.467 ms | 1.03x |

INT8 tracked quant memory after warmup:

| Component | MiB |
|---|---:|
| Total tracked | 533.265 |
| Quant weights | 531.984 |
| Scale and bias state | 1.280 |
| Cache/residual | 0.000 |

Raw result files:

- `integration/results/fuse_first/ldm_benchmark_fp16_s200_n168_b42/results.json`
- `integration/results/fuse_first/ldm_benchmark_int8_baseline_s200_n168_b42/results.json`

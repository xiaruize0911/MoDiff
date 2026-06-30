# AWQ Whole-Applicable Model Benchmark Report

Date: 2026-06-30
GPU: NVIDIA A40
Benchmark: `integration/benchmarks/benchmark_ldm.py`

## What Was Converted

AWQ's upstream extension provides W8A8 GEMM kernels, not Conv2d kernels. For this LDM UNet, "whole-model AWQ" therefore means every AWQ-compatible GEMM-like projection:

- 37/37 `nn.Linear` layers converted to AWQ W8A8.
- 42/42 Conv1d-1x1 attention projection layers converted to AWQ W8A8.
- Spatial Conv2d layers remain on the existing MoDiff/CUTLASS INT8 Conv2d path.
- MoDiff temporal caching remains disabled; this is a baseline mode.

The new mode is `int8_awq_full_baseline`.

## Command

```bash
PYTHONPATH=/workspace/MoDiff:/workspace/llm-awq/awq/kernels \
python integration/benchmarks/benchmark_ldm.py \
  --mode int8_awq_full_baseline \
  --steps 10 --batch_size 4 --num_samples 8 \
  --output_dir integration/results/ldm_awq_full_report/int8_awq_full_baseline \
  --skip_calibration \
  --calibration integration/calibration/int8_calibration.pt
```

## Result

| Mode | AWQ coverage | Total time | Time/sample | Time/step |
|---|---|---:|---:|---:|
| INT8 AWQ full baseline | Linear + Conv1d-1x1 projections | 1.036 s | 0.1295 s | 12.95 ms |

## Comparison To Previous Same-Setup Runs

All rows use `steps=10`, `batch_size=4`, `num_samples=8`, A40, and the same LDM checkpoint/config.

| Mode | Linear backend | Conv1d-1x1 backend | Time/sample | Time/step | Relative to current INT8 baseline |
|---|---|---|---:|---:|---:|
| FP16 | FP16 | FP16 | 0.1557 s | 15.57 ms | 0.68x |
| INT8 baseline | FP16 | FP16 | 0.1054 s | 10.54 ms | 1.00x |
| INT8 baseline | our `int_gemm` | FP16 | 0.1150 s | 11.50 ms | 0.92x |
| INT8 AWQ baseline | AWQ W8A8 | FP16 | 0.0962 s | 9.62 ms | 1.10x |
| INT8 AWQ full baseline | AWQ W8A8 | AWQ W8A8 | 0.1295 s | 12.95 ms | 0.81x |

## Takeaways

Full AWQ-applicable conversion is slower than linear-only AWQ for this UNet benchmark.

The linear-only AWQ baseline remains the best result: `0.0962 s/sample`. Converting attention Conv1d-1x1 projections to AWQ makes the run slower: `0.1295 s/sample`.

The likely reason is shape regime. Attention Conv1d projections flatten to `[B * sequence_length, C]`; some attention resolutions have small `M`, including tiny cases that need the correctness fallback in the AWQ wrapper. For these small/medium projection GEMMs, AWQ's quantize + GEMM launch overhead is larger than the benefit.

So the recommended baseline adoption is:

- Use AWQ for UNet `nn.Linear` layers.
- Keep attention Conv1d-1x1 projections on the existing FP16/CUTLASS path for now.
- Do not convert spatial Conv2d layers to AWQ unless we add or import a real AWQ-compatible convolution kernel.

## Artifacts

- Results JSON: `integration/results/ldm_awq_full_report/int8_awq_full_baseline/results.json`
- Generated samples: `integration/results/ldm_awq_full_report/int8_awq_full_baseline/int8_awq_full_baseline/`

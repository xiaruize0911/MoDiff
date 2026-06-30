# AWQ Baseline Benchmark Report

Date: 2026-06-30
GPU: NVIDIA A40
Benchmark: `integration/benchmarks/benchmark_ldm.py`

## Setup

All runs used:

- `steps=10`
- `batch_size=4`
- `num_samples=8`
- LDM config: `configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml`
- Checkpoint: `models/ldm/lsun_churches256/model.ckpt`
- INT8 calibration: `integration/calibration/int8_calibration.pt`
- Warmup: one full 10-step pass at batch size 4 before timing

## Commands

```bash
PYTHONPATH=/workspace/MoDiff:/workspace/llm-awq/awq/kernels \
python integration/benchmarks/benchmark_ldm.py \
  --mode fp16 --steps 10 --batch_size 4 --num_samples 8 \
  --output_dir integration/results/ldm_awq_baseline_report/fp16

PYTHONPATH=/workspace/MoDiff:/workspace/llm-awq/awq/kernels \
python integration/benchmarks/benchmark_ldm.py \
  --mode int8_baseline --steps 10 --batch_size 4 --num_samples 8 \
  --output_dir integration/results/ldm_awq_baseline_report/int8_baseline_fp16_linear \
  --skip_calibration --calibration integration/calibration/int8_calibration.pt

PYTHONPATH=/workspace/MoDiff:/workspace/llm-awq/awq/kernels \
python integration/benchmarks/benchmark_ldm.py \
  --mode int8_baseline --linear_backend int_gemm \
  --steps 10 --batch_size 4 --num_samples 8 \
  --output_dir integration/results/ldm_awq_baseline_report/int8_baseline_int_gemm \
  --skip_calibration --calibration integration/calibration/int8_calibration.pt

PYTHONPATH=/workspace/MoDiff:/workspace/llm-awq/awq/kernels \
python integration/benchmarks/benchmark_ldm.py \
  --mode int8_awq_baseline --steps 10 --batch_size 4 --num_samples 8 \
  --output_dir integration/results/ldm_awq_baseline_report/int8_awq_baseline \
  --skip_calibration --calibration integration/calibration/int8_calibration.pt
```

## Results

| Mode | Linear backend | Total time | Time/sample | Time/step | Speedup vs FP16 | Speedup vs current INT8 baseline |
|---|---|---:|---:|---:|---:|---:|
| FP16 | FP16 | 1.246 s | 0.1557 s | 15.57 ms | 1.00x | 0.68x |
| INT8 baseline | FP16 | 0.843 s | 0.1054 s | 10.54 ms | 1.48x | 1.00x |
| INT8 baseline | our `int_gemm` | 0.920 s | 0.1150 s | 11.50 ms | 1.35x | 0.92x |
| INT8 AWQ baseline | AWQ W8A8 | 0.769 s | 0.0962 s | 9.62 ms | 1.62x | 1.10x |

## Takeaways

The new `int8_awq_baseline` mode is the fastest baseline in this run:

- 9.6% faster than the current `int8_baseline` default, which keeps linear layers on FP16 math.
- 19.6% faster than our true INT8 `int_gemm` linear backend.
- 1.62x faster than FP16 for this short LDM run.

The result matches the earlier kernel-level/Nsight Systems finding: our `int_gemm` backend loses time in activation quantization overhead around GEMM, while AWQ uses a fused activation quantizer plus a faster W8A8 GEMM kernel.

## Artifacts

Each run wrote `results.json` and 8 generated samples:

- `integration/results/ldm_awq_baseline_report/fp16/`
- `integration/results/ldm_awq_baseline_report/int8_baseline_fp16_linear/`
- `integration/results/ldm_awq_baseline_report/int8_baseline_int_gemm/`
- `integration/results/ldm_awq_baseline_report/int8_awq_baseline/`

## Caveats

This is a short throughput benchmark, not a quality evaluation. It does not compute FID. The AWQ backend currently applies to INT8 linear layers only; INT8 convolution remains the existing MoDiff/CUTLASS baseline kernel.

# LDM INT8 Kernel Comparison

Date: 2026-06-30
GPU: NVIDIA A40

This comparison uses the exact modules selected by `integration/benchmarks/benchmark_ldm.py`:

- Convolution baseline: `integration.kernels.int8_optimized.OptimizedInt8Conv2d`
- Linear baseline: `integration.kernels.int8_linear.OptimizedInt8Linear`
- AWQ linear backend: `benchmark_ldm.py --linear_backend awq`

## Commands

```bash
PYTHONPATH=/workspace/MoDiff:/workspace/llm-awq/awq/kernels \
python integration/benchmarks/benchmark_ldm_int8_kernels.py --warmup 10 --iters 50

env -u RUNPOD_API_KEY -u VSCODE_CLI_REQUIRE_TOKEN \
  PYTHONPATH=/workspace/MoDiff:/workspace/llm-awq/awq/kernels \
  nsys profile --force-overwrite=true --stats=true -t cuda,nvtx,osrt \
  -o integration/results/ldm_int8_kernel_compare/nsys/linear_int_gemm \
  python integration/benchmarks/benchmark_ldm_int8_kernels.py \
  --profile linear_int_gemm --profile-repeats 50

env -u RUNPOD_API_KEY -u VSCODE_CLI_REQUIRE_TOKEN \
  PYTHONPATH=/workspace/MoDiff:/workspace/llm-awq/awq/kernels \
  nsys profile --force-overwrite=true --stats=true -t cuda,nvtx,osrt \
  -o integration/results/ldm_int8_kernel_compare/nsys/linear_awq \
  python integration/benchmarks/benchmark_ldm_int8_kernels.py \
  --profile linear_awq --profile-repeats 50

env -u RUNPOD_API_KEY -u VSCODE_CLI_REQUIRE_TOKEN \
  PYTHONPATH=/workspace/MoDiff:/workspace/llm-awq/awq/kernels \
  nsys profile --force-overwrite=true --stats=true -t cuda,nvtx,osrt \
  -o integration/results/ldm_int8_kernel_compare/nsys/conv \
  python integration/benchmarks/benchmark_ldm_int8_kernels.py \
  --profile conv --profile-repeats 50
```

End-to-end smoke for the real LDM runner:

```bash
PYTHONPATH=/workspace/MoDiff:/workspace/llm-awq/awq/kernels \
python integration/benchmarks/benchmark_ldm.py \
  --mode int8_baseline --steps 1 --batch_size 1 --num_samples 1 \
  --output_dir integration/results/ldm_awq_smoke \
  --linear_backend awq --skip_calibration \
  --calibration integration/calibration/int8_calibration.pt
```

## Event-Timed Results

| kind | shape | backend | median ms | TOPS |
|---|---|---|---:|---:|
| conv | res_128_32 | OptimizedInt8Conv2d | 0.2570 | 37.60 |
| conv | res_256_16 | OptimizedInt8Conv2d | 0.1540 | 62.74 |
| conv | mid_512_8 | OptimizedInt8Conv2d | 0.0990 | 97.61 |
| conv | up_128_64 | OptimizedInt8Conv2d | 0.8957 | 43.15 |
| linear | attn_proj_512 | fp16 | 0.0488 | 57.80 |
| linear | attn_proj_512 | int_gemm | 0.3932 | 7.17 |
| linear | attn_proj_512 | awq | 0.2345 | 12.02 |
| linear | ffn_512_2048 | fp16 | 0.1464 | 76.99 |
| linear | ffn_512_2048 | int_gemm | 1.0977 | 10.27 |
| linear | ffn_512_2048 | awq | 0.1694 | 66.56 |
| linear | large_4096 | fp16 | 0.3249 | 105.77 |
| linear | large_4096 | int_gemm | 1.1141 | 30.84 |
| linear | large_4096 | awq | 0.2552 | 134.64 |
| linear | small_m_4096 | fp16 | 0.0808 | 53.13 |
| linear | small_m_4096 | int_gemm | 0.2261 | 19.00 |
| linear | small_m_4096 | awq | 0.4333 | 9.91 |

## Nsight Findings

Nsight Systems profiles were generated under `integration/results/ldm_int8_kernel_compare/nsys/` locally. The raw `.nsys-rep` and exported `.sqlite` files are intentionally not committed because Nsight records process environment metadata.

For the exact `OptimizedInt8Linear` `int_gemm` path, Nsight Systems shows the runtime is dominated by both the Triton GEMM and many surrounding PyTorch activation-quantization kernels:

- `ampere_igemm_int8_128x128_ldg4_nn`: about 624 us average in the profiled run
- PyTorch `abs`, `amax/reduce`, `round`, `clamp`, `to/copy`, and elementwise kernels appear around every GEMM

For the AWQ linear backend, the profile is much cleaner:

- `dense_kernel0_fuse_bias`: about 208 us average
- `vllm::quant_kernel`: about 41 us average
- Only a few tiny residual `aten` kernels

The main reason our true INT8 linear backend is slower is therefore not just GEMM math. The activation quantization is assembled from separate PyTorch ops, creating extra launches and memory traffic. AWQ fuses activation quantization and uses a faster W8A8 GEMM kernel for the large flattened LDM linear shapes.

The exact `OptimizedInt8Conv2d` path shows:

- CUTLASS implicit GEMM convolution: about 86 us average
- `scale_quantize_int8_kernel`: about 38 us average
- `scale_store_half_kernel`: about 47 us average
- PyTorch conversion/elementwise kernels: about 32-46 us average

So the convolution baseline also pays meaningful overhead outside the convolution math kernel.

## Tooling Notes

`nsys` was installed and is working. `ncu` is installed, but hardware performance counters are blocked by NVIDIA counter permissions in this container:

```text
ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters
```

That prevents roofline/counter metrics here, but Nsight Systems still provided useful launch-level insight.

## Integration Notes

`benchmark_ldm.py` defaults to `--linear_backend fp16`, preserving the previous behavior. The AWQ path can be selected explicitly with `--linear_backend awq`.

For baseline-first adoption, `benchmark_ldm.py` also provides a dedicated mode:

```bash
PYTHONPATH=/workspace/MoDiff:/workspace/llm-awq/awq/kernels \
python integration/benchmarks/benchmark_ldm.py \
  --mode int8_awq_baseline --skip_calibration \
  --calibration integration/calibration/int8_calibration.pt
```

This mode forces AWQ for INT8 linear layers and keeps MoDiff temporal caching disabled.

AWQ only replaces INT8 linear GEMM. The baseline convolution kernel remains the existing MoDiff/CUTLASS `OptimizedInt8Conv2d`.

The AWQ wrapper uses a safe fallback for `M <= 128` because the raw upstream AWQ W8A8 kernel produced incorrect results for small `M` on this A40/CUDA 12.4 setup.

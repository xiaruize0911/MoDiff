# Nsight Profiling Notes

Attempted Nsight Compute profiling with:

```bash
ncu --target-processes all --launch-skip 20 --launch-count 1 --set roofline \
  --export integration/results/awq_int8_baseline/ncu/awq_gemm_linear4096 \
  --force-overwrite \
  python integration/benchmarks/benchmark_awq_int8_baseline.py \
    --profile awq --profile-shape 1024,4096,4096 --profile-repeats 5
```

Nsight Compute connected to the process, but the driver denied access to GPU
performance counters:

```text
ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters
```

`nsys` was not available on `PATH` in this environment. As a fallback, CUDA
kernel-level profiling was collected with `torch.profiler` under
`integration/results/awq_int8_baseline/profiler/`.

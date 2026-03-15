# Optimization Layout

This directory now separates runtime-specific infrastructure from quantized layer implementations:

- `int8_optimized.py` / `int4_optimized.py`
  - INT8/INT4 convolution modules.
  - Support both baseline implementations:
    - `current`: quantize + CUTLASS conv + Python-side output dequant/bias.
    - `two_kernel_fused`: quantize + fused CUTLASS output dequant/bias wrapper.
- `runtime/`
  - Runtime-only helpers that are shared by benchmark entry points.
  - `cuda_graphs.py`: fixed-shape CUDA Graph capture/replay for LDM sampling.
- `benchmark_ldm.py`
  - End-to-end LDM benchmarking CLI with FP32 / FP16 / INT8 / INT4 / CUDA Graph / fused-baseline modes.
- `profiler.py`
  - CUDA-event profiler with machine-readable summaries for reports.

Low-level CUDA/CUTLASS kernels remain in `csrc/`:

- `pybind.cpp`: extension bindings.
- `cuda_kernels.cu`: quantization, CUTLASS wrappers, MoDiff fused kernels, and fused baseline output wrappers.

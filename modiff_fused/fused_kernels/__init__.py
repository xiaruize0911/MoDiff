# NOTE (2026-07-22): NON-SHIPPED PROTOTYPE. This `modiff_fused` package (Triton
# `fused_modiff_conv2d` / `fused_modiff_gemm` + `FusedMoDiffLinear`) is an early
# standalone experiment. It is NOT imported by the benchmarked pipeline
# (integration/benchmarks/benchmark_ldm.py) or by ldm/. The production quantized
# path is the CUTLASS csrc/kernels/* + integration/kernels/{int8,int4}_optimized.py
# stack. Kept for reference only; superseded.

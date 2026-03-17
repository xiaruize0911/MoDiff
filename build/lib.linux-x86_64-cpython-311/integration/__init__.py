"""
MoDiff Integration Package
==========================

Organized into the following subpackages:

    kernels/       - Quantized Conv2d and Linear layer implementations
                     (CUTLASS INT8/INT4 fused, PyTorch INT8, separate-kernel baselines)

    fused_ops/     - Fused operator implementations
                     (Triton GroupNorm+SiLU, fused residual blocks)

    utils/         - Infrastructure and utilities
                     (buffer pool, timestep cache, profiler)

    benchmarks/    - Benchmark scripts and evaluation tools
                     (LDM benchmarks, extended benchmarks, FID evaluation)

    calibration/   - Calibration data files (.pt)

    results/       - Benchmark output (images, JSON, reports, plots)
"""

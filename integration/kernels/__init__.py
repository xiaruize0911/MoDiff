"""
Quantized convolution and linear layer implementations.

Modules:
    int8_optimized   - CUTLASS INT8 fused conv with MoDiff temporal modulation
    int4_optimized   - CUTLASS INT4 fused conv with MoDiff temporal modulation
    int8_cudagraph   - PyTorch native INT8 conv with CUDA Graph support
    fused_baseline   - Separate-kernel (unfused) baseline for INT8/INT4
    int8_linear      - INT8 quantized linear layer
    int4_linear      - INT4 quantized linear layer
    modiff_layers    - Legacy CUTLASS conv implementation
"""

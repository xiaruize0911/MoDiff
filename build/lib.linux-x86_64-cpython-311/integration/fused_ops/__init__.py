"""
Fused operator implementations (Triton kernels and PyTorch modules).

Modules:
    fused_gn_silu    - Triton-based fused GroupNorm + SiLU kernel
    fused_resblock   - Fused residual block combining GN+SiLU+Conv
"""

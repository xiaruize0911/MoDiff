"""Fully-fused static-scale W8A8 GEMM for INT8 linear layers.

One Triton kernel does: inline activation quantize (with a precomputed static
scalar scale) -> INT8 tl.dot (INT32 accumulate) -> dequant (scale_a * scale_w)
+ bias -> FP16 output. No separate quantize/dequant kernels, no INT32 output
materialized to global memory, single read of the activation.

Motivation / when to use (measured on A40, LSUN + synthetic sweeps):
the raw INT8 tensor-core GEMM only out-throughputs fp16 cuBLAS once the
contraction dim K is large (crossover ~K>=1500-2048); below that, Ampere IMMA
can't reach its peak and fp16's mature small-shape kernels win. So callers should
K-gate this (see integration/kernels/int8_linear.py K_INT8_GATE) and use fp16 for
small-K linears. Above the gate this is ~1.1-1.9x faster than fp16.
"""
import torch
import triton
import triton.language as tl

_CONFIGS = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=3, num_warps=8),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=3, num_warps=8),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=5, num_warps=2),
]


@triton.autotune(configs=_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def _fused_static_w8a8_kernel(
    X, W, SW, B, O,
    qscale,              # scalar: 1/scale_a  (x_int8 = round(x * qscale))
    dq_a,                # scalar: scale_a    (activation dequant)
    M, N, K,
    sxm, sxk, swk, swn, som, son,
    HAS_BIAS: tl.constexpr, SW_VEC: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    npm = tl.cdiv(M, BLOCK_M)
    npn = tl.cdiv(N, BLOCK_N)
    npg = GROUP_M * npn
    gid = pid // npg
    fpm = gid * GROUP_M
    gsz = tl.minimum(npm - fpm, GROUP_M)
    pid_m = fpm + ((pid % npg) % gsz)
    pid_n = (pid % npg) // gsz

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    x_ptrs = X + (offs_m[:, None] * sxm + offs_k[None, :] * sxk)
    w_ptrs = W + (offs_k[:, None] * swk + offs_n[None, :] * swn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        kr = K - k * BLOCK_K
        km = offs_k < kr
        xf = tl.load(x_ptrs, mask=km[None, :], other=0.0).to(tl.float32)
        xi = tl.minimum(tl.maximum(tl.floor(xf * qscale + 0.5), -128.0), 127.0).to(tl.int8)
        wi = tl.load(w_ptrs, mask=km[:, None], other=0)
        acc = tl.dot(xi, wi, acc, out_dtype=tl.int32)
        x_ptrs += BLOCK_K * sxk
        w_ptrs += BLOCK_K * swk

    accf = acc.to(tl.float32) * dq_a
    if SW_VEC:
        sw = tl.load(SW + offs_n, mask=offs_n < N, other=1.0)
        accf = accf * sw[None, :]
    else:
        accf = accf * tl.load(SW)
    if HAS_BIAS:
        accf = accf + tl.load(B + offs_n, mask=offs_n < N, other=0.0)[None, :]

    ocm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    ocn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    o_ptrs = O + (ocm[:, None] * som + ocn[None, :] * son)
    tl.store(o_ptrs, accf.to(tl.float16), mask=(ocm[:, None] < M) & (ocn[None, :] < N))


def fused_static_w8a8_linear(x, weight_int8, scale_a, scale_w, bias=None):
    """Fused W8A8 linear. Returns FP16 [M, N].

    Args:
        x:           [M, K] activation (fp16/fp32).
        weight_int8: [K, N] INT8 weight (any strides; Triton handles layout).
        scale_a:     scalar activation dequant scale (absmax_x / 127).
        scale_w:     scalar or [N] weight dequant scale (absmax_w / 127).
        bias:        optional [N].
    """
    assert weight_int8.dtype == torch.int8
    M, K = x.shape
    K2, N = weight_int8.shape
    assert K == K2, f"K mismatch {K} vs {K2}"
    if not isinstance(scale_a, torch.Tensor):
        scale_a = torch.tensor(float(scale_a), device=x.device, dtype=torch.float32)
    scale_a = scale_a.to(device=x.device, dtype=torch.float32).reshape(())
    if not isinstance(scale_w, torch.Tensor):
        scale_w = torch.tensor(float(scale_w), device=x.device, dtype=torch.float32)
    scale_w = scale_w.to(device=x.device, dtype=torch.float32).contiguous()
    sw_vec = scale_w.numel() > 1
    out = torch.empty((M, N), device=x.device, dtype=torch.float16)
    qscale = 1.0 / float(scale_a)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    _fused_static_w8a8_kernel[grid](
        x, weight_int8, scale_w, bias if bias is not None else x, out,
        qscale, float(scale_a), M, N, K,
        x.stride(0), x.stride(1), weight_int8.stride(0), weight_int8.stride(1),
        out.stride(0), out.stride(1),
        HAS_BIAS=bias is not None, SW_VEC=sw_vec,
    )
    return out

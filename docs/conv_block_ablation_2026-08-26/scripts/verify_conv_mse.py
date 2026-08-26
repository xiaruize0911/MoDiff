"""Numerical correctness check for the four conv kernels bench_conv_block_ablation.py times.

That script only measures LATENCY -- it feeds the kernels random int8/int4 codes and never checks
the output is the right convolution. No existing test in integration/tests/ covers
conv2d_int{8,4}_evt_bias_residual_fp16 or conv2d_int{8,4}_evt_o_hat's numerics either (checked:
`grep -rl conv2d_int8 integration/` finds nothing). This closes that gap the same way
gn_vec2_2026-08-26/verify_mse.py closed it for the GroupNorm kernels: an independent float64
reference (dequantize, F.conv2d, compare) scored against the real CUDA kernel by MSE / max-abs.

DEQUANTIZATION CONVENTION, confirmed from integration/kernels/int8_optimized.py:682 (`w *
self.weight_scale_channel...`, i.e. weight_scale_channel MULTIPLIES the code to recover the
weight) and :1198 (alpha passed as "the reciprocal of the scale that quantized x_int8"):
    x_dequant = x_code * inv_scale            (inv_scale = 1/quantization_scale, a scalar)
    w_dequant = w_code * weight_scale[k]      (weight_scale = per-output-channel dequant multiplier)
    conv_output = conv2d(x_dequant, w_dequant)
which is exactly what conv2d_int8_evt_o_hat's own header comment says: "o_hat[elem] +=
acc*alpha*weight_scale[k]" (acc = the raw int32 dot product before either scale is applied).

Baseline (conv2d_int{8,4}_evt_bias_residual_fp16): output = conv_output + bias + residual.
MoDiff (conv2d_int{8,4}_evt_o_hat): o_hat_new = o_hat_old + conv_output (checked BEFORE/AFTER the
in-place accumulate, since the kernel mutates o_hat_cache and returns the same tensor).

int4 packing matches integration/tests/test_zpw_additive.py's pack_act/pack_w exactly: codes in
[-7,7], NCHW/KCRS int64, permuted to channels-last-style and packed two 4-bit codes per byte.
"""
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "/workspace/MoDiff/build/lib.linux-x86_64-cpython-311")
import modiff_cutlass as mc

torch.manual_seed(0)
N = 8
CL = torch.channels_last
# (Cin, Cout, H, W) -- a representative subset of the 20 real shapes, not all of them: this is a
# numerics check, not a timing one.
SHAPES = [(192, 192, 32, 32), (384, 384, 16, 16), (768, 768, 4, 4), (576, 192, 32, 32),
          (1152, 768, 4, 4), (384, 192, 32, 32)]


def mse(a, b):
    d = a.double() - b.double()
    return (d * d).mean().item(), d.abs().max().item()


def pack4(codes_nchw_or_kcrs):
    """[*, C, *, *] int64 codes in [-8,7] -> [*, *, *, C/2] int8. Verbatim convention from
    test_zpw_additive.py's pack_act/pack_w."""
    c = codes_nchw_or_kcrs.permute(0, 2, 3, 1).contiguous().to(torch.int64) & 0x0F
    lo, hi = c[..., 0::2], c[..., 1::2]
    v = lo | (hi << 4)
    return (v - 256 * (v > 127)).to(torch.int8).contiguous()


print("=== conv2d_int8_evt_bias_residual_fp16 / conv2d_int8_evt_o_hat vs fp64 reference ===")
print(f"{'shape':>18} {'arm':>8}  {'out MSE':>12} {'out maxerr':>12} {'rel maxerr':>11}")
for Cin, Cout, H, W in SHAPES:
    x_code = torch.randint(-100, 101, (N, Cin, H, W), device="cuda", dtype=torch.int64)
    w_code = torch.randint(-100, 101, (Cout, Cin, 3, 3), device="cuda", dtype=torch.int64)
    s_a = 32.0                                   # dequant multiplier convention: x = code / s_a
    inv_scale = torch.tensor([1.0 / s_a], device="cuda", dtype=torch.float32)
    ch_scale = (0.01 + 0.02 * torch.rand(Cout, device="cuda")).double()   # per-K dequant multiplier

    x_dq = (x_code.double() / s_a)
    w_dq = (w_code.double() * ch_scale.view(Cout, 1, 1, 1))
    conv_ref = F.conv2d(x_dq, w_dq, padding=1)   # [N,Cout,H,W], float64

    x8 = x_code.to(torch.int8).to(memory_format=CL)
    w8 = w_code.permute(0, 2, 3, 1).contiguous().to(torch.int8)   # [K,C,R,S] -> [K,R,S,C]
    wscale32 = ch_scale.float()
    empty_h = torch.empty(0, device="cuda", dtype=torch.float16)
    empty_f = torch.empty(0, device="cuda", dtype=torch.float32)
    out = torch.empty(N, Cout, H, W, device="cuda", dtype=torch.float16).to(memory_format=CL)
    mc.conv2d_int8_evt_bias_residual_fp16(x8, w8, inv_scale, wscale32, empty_f, empty_h, out,
                                          1, 1, 1, 1, 1, 1)
    torch.cuda.synchronize()
    m, mx = mse(out, conv_ref)
    scale_here = conv_ref.abs().max().item()
    print(f"{f'{Cin}->{Cout},{H}x{W}':>18} {'base':>8}  {m:>12.3e} {mx:>12.3e} {mx/scale_here:>10.2e}")

    o_hat0 = (0.1 * torch.randn(N, Cout, H, W, device="cuda", dtype=torch.float16)).to(memory_format=CL)
    o_hat = o_hat0.clone()
    mc.conv2d_int8_evt_o_hat(x8, w8, inv_scale, wscale32, o_hat, 1, 1, 1, 1, 1, 1)
    torch.cuda.synchronize()
    o_hat_ref = o_hat0.double() + conv_ref
    m, mx = mse(o_hat, o_hat_ref)
    scale_here = o_hat_ref.abs().max().item()
    print(f"{f'{Cin}->{Cout},{H}x{W}':>18} {'o_hat':>8}  {m:>12.3e} {mx:>12.3e} {mx/scale_here:>10.2e}")

print("\n=== conv2d_int4_evt_bias_residual_fp16 / conv2d_int4_evt_o_hat vs fp64 reference ===")
print(f"{'shape':>18} {'arm':>8}  {'out MSE':>12} {'out maxerr':>12} {'rel maxerr':>11}")
for Cin, Cout, H, W in SHAPES:
    x_code = torch.randint(-7, 8, (N, Cin, H, W), device="cuda", dtype=torch.int64)
    w_code = torch.randint(-7, 8, (Cout, Cin, 3, 3), device="cuda", dtype=torch.int64)
    s_a = 4.0
    inv_scale = torch.tensor([1.0 / s_a], device="cuda", dtype=torch.float32)
    ch_scale = (0.05 + 0.05 * torch.rand(Cout, device="cuda")).double()

    x_dq = (x_code.double() / s_a)
    w_dq = (w_code.double() * ch_scale.view(Cout, 1, 1, 1))
    conv_ref = F.conv2d(x_dq, w_dq, padding=1)

    x4 = pack4(x_code)                # [N,H,W,Cin/2] int8
    w4 = pack4(w_code)                # [Cout,3,3,Cin/2] int8
    wscale32 = ch_scale.float()
    empty_h = torch.empty(0, device="cuda", dtype=torch.float16)
    empty_f = torch.empty(0, device="cuda", dtype=torch.float32)
    out = torch.empty(N, Cout, H, W, device="cuda", dtype=torch.float16).to(memory_format=CL)
    mc.conv2d_int4_evt_bias_residual_fp16(x4, w4, inv_scale, wscale32, empty_f, empty_h, out,
                                          1, 1, 1, 1, 1, 1)
    torch.cuda.synchronize()
    m, mx = mse(out, conv_ref)
    scale_here = conv_ref.abs().max().item()
    print(f"{f'{Cin}->{Cout},{H}x{W}':>18} {'base':>8}  {m:>12.3e} {mx:>12.3e} {mx/scale_here:>10.2e}")

    o_hat0 = (0.1 * torch.randn(N, Cout, H, W, device="cuda", dtype=torch.float16)).to(memory_format=CL)
    o_hat = o_hat0.clone()
    mc.conv2d_int4_evt_o_hat(x4, w4, inv_scale, wscale32, o_hat, 1, 1, 1, 1, 1, 1)
    torch.cuda.synchronize()
    o_hat_ref = o_hat0.double() + conv_ref
    m, mx = mse(o_hat, o_hat_ref)
    scale_here = o_hat_ref.abs().max().item()
    print(f"{f'{Cin}->{Cout},{H}x{W}':>18} {'o_hat':>8}  {m:>12.3e} {mx:>12.3e} {mx/scale_here:>10.2e}")

print("\nDone. Expect relative maxerr at fp16-rounding scale (~1e-3) -- the int32 accumulate and")
print("the alpha*weight_scale multiply are exact; the ONLY lossy step is the final round to fp16")
print("on output/o_hat storage, which this reference does not itself replicate (it stays in")
print("float64 throughout), so ~1e-3 relative is the expected floor, not zero.")

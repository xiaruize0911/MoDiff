"""Prototype check: fuse GroupNorm+SiLU+quantize+2x resize into ONE kernel for updown ResBlocks.

The shipped updown path runs two kernels before the conv:

    group_norm_silu_nhwc            GN+SiLU, fp16 out (small)
    {upsample2x,avgpool2x}_quantize_pack_noahat_fprop   resize + quantize -> packed int4

GroupNorm emits fp16 there rather than quantizing in place, because with the quantize on the far
side of the resize the DOWN direction stays exact: averaging four already-quantized int4 codes is
not the same as averaging the real values and quantizing once. That argument only binds ACROSS
kernels. Inside a single kernel the 2x2 average can be taken on the fp32 post-SiLU values before
quantization, so the fusion is available in both directions -- which is what
group_norm_silu_quantize_pack_resize_nhwc does.

This measures whether that is worth wiring in. It reports, per real updown shape:
  * max |int4 code difference| between the two paths
  * rel-L2 of each path's dequantized output against an fp32 reference
  * per-call time of the fused kernel against the two-kernel sequence

Nothing here changes the pipeline; the fused entry point is not called by the model.
"""
import os
import statistics
import sys

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "integration/benchmarks/report"))

import torch
import torch.nn.functional as F
import modiff_cutlass as mc
from ck_bench_stats import cuda_bench_stats

# (C, H, W, direction) taken from the model's eight updown ResBlocks.
SHAPES = [(192, 32, 32, -1), (384, 16, 16, -1), (384, 8, 8, -1), (768, 4, 4, -1),
          (768, 2, 2, +1), (768, 4, 4, +1), (384, 8, 8, +1), (384, 16, 16, +1)]
BATCH, GROUPS, EPS = 128, 32, 1e-5


def unpack_int4(y, C):
    """[N,H,W,C/2] packed -> [N,H,W,C] signed nibbles."""
    lo = (y & 0x0F).to(torch.int16)
    hi = ((y >> 4) & 0x0F).to(torch.int16)
    lo = torch.where(lo > 7, lo - 16, lo)
    hi = torch.where(hi > 7, hi - 16, hi)
    return torch.stack([lo, hi], dim=-1).reshape(*y.shape[:-1], C)


def reference(x, gamma, beta, scale, direction):
    """fp32 GN -> SiLU -> resize -> quantize, the order the fused kernel implements."""
    x32 = x.float()
    N, C, H, W = x32.shape
    g = x32.reshape(N, GROUPS, C // GROUPS * H * W)
    mean = g.mean(-1, keepdim=True)
    var = g.var(-1, unbiased=False, keepdim=True)
    g = (g - mean) / torch.sqrt(var + EPS)
    y = g.reshape(N, C, H, W) * gamma.float().view(1, C, 1, 1) + beta.float().view(1, C, 1, 1)
    y = y * torch.sigmoid(y)
    y = (F.interpolate(y, scale_factor=2, mode="nearest") if direction > 0
         else F.avg_pool2d(y, 2))
    return torch.clamp(torch.round(y * scale), -7, 7)          # [N,C,Ho,Wo] fp32 codes


def main():
    dev = "cuda"
    torch.manual_seed(0)
    print("Fused GN+SiLU+quantize+resize vs the shipped two-kernel path, batch %d\n" % BATCH)
    print("| shape | dir | max code diff | rel-L2 two-kernel | rel-L2 fused | "
          "two-kernel µs | fused µs | speedup |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    sp = []
    for C, H, W, direction in SHAPES:
        x = torch.randn(BATCH, C, H, W, device=dev, dtype=torch.float16)
        x = x.contiguous(memory_format=torch.channels_last)
        gamma = torch.randn(C, device=dev, dtype=torch.float16)
        beta = torch.randn(C, device=dev, dtype=torch.float16)
        scale_t = torch.tensor([3.0], device=dev, dtype=torch.float32)
        empty = x.new_empty(0)
        empty_f = torch.empty(0, device=dev, dtype=torch.float32)

        resize_fn = (mc.upsample2x_quantize_pack_noahat_fprop if direction > 0
                     else mc.avgpool2x_quantize_pack_noahat_fprop)

        def two_kernel():
            h = mc.group_norm_silu_nhwc(x, gamma, beta, GROUPS, EPS, True, empty, empty)
            return resize_fn(h, scale_t, empty_f)

        def fused():
            return mc.group_norm_silu_quantize_pack_resize_nhwc(
                x, gamma, beta, GROUPS, EPS, True, scale_t, empty_f, empty, empty, 0, direction)

        with torch.inference_mode():
            y_two, y_fus = two_kernel(), fused()
            ref = reference(x, gamma, beta, float(scale_t.item()), direction)
            # packed [N,Ho,Wo,C/2] -> codes [N,C,Ho,Wo]
            c_two = unpack_int4(y_two, C).permute(0, 3, 1, 2).float()
            c_fus = unpack_int4(y_fus, C).permute(0, 3, 1, 2).float()
            maxdiff = (c_two - c_fus).abs().max().item()
            r_two = ((c_two - ref).norm() / ref.norm()).item()
            r_fus = ((c_fus - ref).norm() / ref.norm()).item()
            st2, _ = cuda_bench_stats(two_kernel, warm=20, iters=40, rounds=6)
            stf, _ = cuda_bench_stats(fused, warm=20, iters=40, rounds=6)
        sp.append(st2["mean"] / stf["mean"])
        print("| C%d/%dx%d | %s | %.0f | %.4f | %.4f | %.1f | %.1f | **%.2f×** |"
              % (C, H, W, "up" if direction > 0 else "down", maxdiff, r_two, r_fus,
                 st2["mean"], stf["mean"], st2["mean"] / stf["mean"]))
        del x, y_two, y_fus, ref, c_two, c_fus
        torch.cuda.empty_cache()
    print("\nspeedup over the eight shapes: %.2f-%.2f×, median %.2f×"
          % (min(sp), max(sp), statistics.median(sp)))


if __name__ == "__main__":
    main()

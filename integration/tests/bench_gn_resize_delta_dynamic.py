"""What the updown fusion is worth on a REFRESH step, at the eight real shapes.

At MODIFF_DELTA_REFRESH=1 -- the paper's configuration -- every step is a refresh, and before
2026-08-10 `_prequant_gn_resize_conv_modiff` declined on every one of them, so these eight
ResBlocks ran the four-kernel unfused route. This times that route against the now-available
fused one, per shape and summed, which is the per-step figure the fix is worth.

  unfused (what a refresh step ran)          fused dynamic (what it runs now)
    group_norm_silu_nhwc      GN+SiLU -> fp16    reduction-only launch   measures max|delta|
    interpolate / avg_pool2d  resize  -> fp16    quantizing launch       GN+SiLU+resize+quantize
    delta_absmax_fp16         the scale                                  +a_hat, one output
    step1_static_quantize_fprop  quantize +a_hat

The conv that consumes the codes is identical in both and is excluded. `fused static` is the
K>1 reuse path, timed as the floor: it is the same work without the reduction launch.

Run: python integration/tests/bench_gn_resize_delta_dynamic.py [batch]
"""
import os
import sys

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())

import torch
import torch.nn.functional as F
import modiff_cutlass as mc

SHAPES = [(192, 32, 32, -1), (384, 16, 16, -1), (384, 8, 8, -1), (768, 4, 4, -1),
          (768, 2, 2, +1), (768, 4, 4, +1), (384, 8, 8, +1), (384, 16, 16, +1)]
GROUPS, EPS = 32, 1e-5
DEV = "cuda"
Q_LEVEL = 127.0


def bench(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main():
    batch = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    torch.manual_seed(0)
    print(f"batch {batch}, {torch.cuda.get_device_name(0)}, int8 store, identity SmoothQuant\n")
    print("| shape | dir | unfused ms | fused dyn ms | fused static ms | dyn speedup |")
    print("|---|---|---:|---:|---:|---:|")
    tot_un = tot_dyn = tot_st = 0.0
    for C, H, W, direction in SHAPES:
        x = torch.randn(batch, C, H, W, device=DEV, dtype=torch.float16)
        x = x.contiguous(memory_format=torch.channels_last)
        gamma = torch.randn(C, device=DEV, dtype=torch.float16)
        beta = torch.randn(C, device=DEV, dtype=torch.float16)
        Ho, Wo = (H * 2, W * 2) if direction > 0 else (H // 2, W // 2)
        a_hat = (0.5 * torch.randn(batch, C, Ho, Wo, device=DEV, dtype=torch.float16)
                 ).contiguous(memory_format=torch.channels_last)
        ef = torch.empty(0, device=DEV, dtype=torch.float32)
        ei = torch.empty(0, device=DEV, dtype=torch.int32)
        eh = torch.empty(0, device=DEV, dtype=torch.float16)
        amax = torch.zeros(1, device=DEV, dtype=torch.float32)
        sc = torch.empty(1, device=DEV, dtype=torch.float32)
        inv = torch.empty(1, device=DEV, dtype=torch.float32)
        ret = torch.zeros(1, device=DEV, dtype=torch.int32)
        sc.fill_(Q_LEVEL / 4.0)

        def unfused():
            # GN+SiLU (fp16 out), then the resize, then the two delta passes on the resized tensor.
            normed = mc.group_norm_silu_nhwc(x, gamma, beta, GROUPS, EPS, True, eh, eh)
            r = (F.interpolate(normed, scale_factor=2, mode="nearest") if direction > 0
                 else F.avg_pool2d(normed, 2))
            r = r.contiguous(memory_format=torch.channels_last)
            mc.delta_absmax_fp16(r, a_hat, amax, sc, inv, ret, Q_LEVEL, ef, False)
            return mc.step1_static_quantize_fprop(r, a_hat, sc, ef, -1.0)

        def fused_dyn():
            return mc.group_norm_silu_delta_quantize_resize_nhwc(
                x, gamma, beta, GROUPS, EPS, True, sc, ef, eh, eh, 0, direction, False, a_hat,
                amax, sc, inv, ret, Q_LEVEL, False, 1.0, -1.0)

        def fused_static():
            return mc.group_norm_silu_delta_quantize_resize_nhwc(
                x, gamma, beta, GROUPS, EPS, True, sc, ef, eh, eh, 0, direction, False, a_hat)

        with torch.inference_mode():
            t_un = bench(unfused)
            t_dyn = bench(fused_dyn)
            t_st = bench(fused_static)
        tot_un += t_un
        tot_dyn += t_dyn
        tot_st += t_st
        print(f"| {C}x{H}x{W} | {direction:+d} | {t_un:.3f} | {t_dyn:.3f} | {t_st:.3f} | "
              f"{t_un / t_dyn:.2f}x |")
    print(f"| **all 8** | | **{tot_un:.3f}** | **{tot_dyn:.3f}** | **{tot_st:.3f}** | "
          f"**{tot_un / tot_dyn:.2f}x** |")
    print(f"\nper step at K=1: {tot_un - tot_dyn:+.3f} ms recovered "
          f"({tot_un:.3f} -> {tot_dyn:.3f})")
    print(f"the reduction launch costs {tot_dyn - tot_st:.3f} ms; a K>1 reuse step pays "
          f"{tot_st:.3f} ms")


if __name__ == "__main__":
    main()

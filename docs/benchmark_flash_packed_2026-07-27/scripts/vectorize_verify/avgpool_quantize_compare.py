"""Capture-then-compare regression check for avgpool2x_quantize_noahat_fprop /
_pack_noahat_fprop -- the new fused Downsample(avg_pool,2x2)+quantize kernel for
updown ResBlocks' down-transition in_conv, against the reference two-step path
(nn.AvgPool2d(2,2) exactly as Downsample.forward calls it, then the existing
cache-free static quantize kernel). This is the down-direction sibling of
noahat_quantize_vec2_compare.py / the upsample fusion's own gate.

  python avgpool_quantize_compare.py --capture   # against the reference two-step path
  python avgpool_quantize_compare.py --compare   # after a rebuild, re-run the fused kernel

Note: --capture computes the REFERENCE (unfused) path, not the fused kernel, so this
also serves as the from-scratch correctness check (not just a regression-vs-self check).
"""
import os, sys, argparse
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff")
import torch
import torch.nn as nn
import modiff_cutlass as mc

HERE = os.path.dirname(os.path.abspath(__file__))
CAPTURE_PATH = os.path.join(HERE, "avgpool_quantize_capture.pt")

# (N, C, H, W): H, W match this model's real pre-pool down-transition resolutions
# (32->16, 16->8, 8->4 for the 3 down-transitions at C=192/384/768).
SHAPES = [dict(N=2, C=192, H=32, W=32), dict(N=2, C=384, H=16, W=16),
          dict(N=4, C=768, H=8, W=8), dict(N=2, C=96, H=10, W=10)]  # H=10: non-power-of-2 smoke test


def cl(t):
    return t.contiguous(memory_format=torch.channels_last)


def reference(x, scale, smooth, use_int4):
    pooled = nn.functional.avg_pool2d(x, 2, 2)  # bit-identical algorithm to nn.AvgPool2d / Downsample
    fn = mc.step1_static_quantize_pack_int4_noahat_fprop if use_int4 else mc.step1_static_quantize_noahat_fprop
    return fn(cl(pooled), scale, smooth).clone()


def fused(x, scale, smooth, use_int4):
    fn = mc.avgpool2x_quantize_pack_noahat_fprop if use_int4 else mc.avgpool2x_quantize_noahat_fprop
    return fn(x, scale, smooth).clone()


def run_case(N, C, H, W, use_smooth, use_int4, dtype, seed, which):
    torch.manual_seed(seed)
    dev = "cuda"
    scale = torch.tensor([7.0 / 3.0 if use_int4 else 127.0 / 3.0], device=dev, dtype=torch.float32)
    smooth = (0.5 + torch.rand(C, device=dev, dtype=torch.float32)) if use_smooth \
        else torch.empty(0, device=dev, dtype=torch.float32)
    x = cl(torch.randn(N, C, H, W, device=dev, dtype=dtype))
    return (reference if which == "ref" else fused)(x, scale, smooth, use_int4)


def all_cases():
    for shape in SHAPES:
        for use_smooth in (False, True):
            for use_int4 in (False, True):
                for dtype in (torch.float16, torch.float32):
                    yield shape["N"], shape["C"], shape["H"], shape["W"], use_smooth, use_int4, dtype


def key_of(N, C, H, W, use_smooth, use_int4, dtype):
    return f"N{N}_C{C}_H{H}_smooth{int(use_smooth)}_int4{int(use_int4)}_{str(dtype).split('.')[-1]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", action="store_true")
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args()

    if a.capture or not os.path.exists(CAPTURE_PATH):
        results = {}
        for i, (N, C, H, W, use_smooth, use_int4, dtype) in enumerate(all_cases()):
            results[key_of(N, C, H, W, use_smooth, use_int4, dtype)] = run_case(
                N, C, H, W, use_smooth, use_int4, dtype, seed=i, which="ref")
        torch.save(results, CAPTURE_PATH)
        print(f"[capture] saved {len(results)} reference (unfused) cases -> {CAPTURE_PATH}")
        return 0

    ref = torch.load(CAPTURE_PATH)
    all_ok = True
    for i, (N, C, H, W, use_smooth, use_int4, dtype) in enumerate(all_cases()):
        key = key_of(N, C, H, W, use_smooth, use_int4, dtype)
        out = run_case(N, C, H, W, use_smooth, use_int4, dtype, seed=i, which="fused")
        ok = torch.equal(out, ref[key])
        all_ok &= ok
        print(f"[{'PASS' if ok else 'FAIL'}] {key:34s} equal={ok}")
    print("ALL PASS" if all_ok else "SOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

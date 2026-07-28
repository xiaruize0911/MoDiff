"""Capture-then-compare regression check for step1_static_quantize_noahat_fprop /
step1_static_quantize_pack_int4_noahat_fprop -- the cache-free "baseline" static
quantize kernels used by the updown (resize) ResBlocks' in_conv/skip_connection
when the resize output arrives as fp16 (static_quantize_int8_noahat_kernel /
static_quantize_pack_int4_noahat_kernel), which were scalar (1 element/thread)
before this fix.

  python noahat_quantize_vec2_compare.py --capture   # before any .cu change
  python noahat_quantize_vec2_compare.py --compare   # after a rebuild

No a_hat cache here (these are the cache-free "baseline" kernels), so a single
call per case is enough -- no multi-iteration cache-evolution concern.
"""
import os, sys, argparse
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff")
import torch
import modiff_cutlass as mc

HERE = os.path.dirname(os.path.abspath(__file__))
CAPTURE_PATH = os.path.join(HERE, "noahat_quantize_vec2_capture.pt")

SHAPES = [dict(N=2, C=128, H=16, W=16), dict(N=2, C=384, H=8, W=8),
          dict(N=4, C=192, H=32, W=32)]  # C=192, H=W=32: matches the real L0 updown resolution


def cl(t):
    return t.contiguous(memory_format=torch.channels_last)


def run_case(N, C, H, W, use_smooth, use_int4, dtype, seed):
    torch.manual_seed(seed)
    dev = "cuda"
    scale = torch.tensor([7.0 / 3.0 if use_int4 else 127.0 / 3.0], device=dev, dtype=torch.float32)
    smooth = (0.5 + torch.rand(C, device=dev, dtype=torch.float32)) if use_smooth \
        else torch.empty(0, device=dev, dtype=torch.float32)
    x = cl(torch.randn(N, C, H, W, device=dev, dtype=dtype))
    fn = mc.step1_static_quantize_pack_int4_noahat_fprop if use_int4 else mc.step1_static_quantize_noahat_fprop
    return fn(x, scale, smooth).clone()


def all_cases():
    for shape in SHAPES:
        for use_smooth in (False, True):
            for use_int4 in (False, True):
                for dtype in (torch.float16, torch.float32):
                    yield shape["N"], shape["C"], shape["H"], shape["W"], use_smooth, use_int4, dtype


def key_of(N, C, H, W, use_smooth, use_int4, dtype):
    return f"N{N}_C{C}_smooth{int(use_smooth)}_int4{int(use_int4)}_{str(dtype).split('.')[-1]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", action="store_true")
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args()

    if a.capture or not os.path.exists(CAPTURE_PATH):
        results = {}
        for i, (N, C, H, W, use_smooth, use_int4, dtype) in enumerate(all_cases()):
            results[key_of(N, C, H, W, use_smooth, use_int4, dtype)] = run_case(
                N, C, H, W, use_smooth, use_int4, dtype, seed=i)
        torch.save(results, CAPTURE_PATH)
        print(f"[capture] saved {len(results)} cases -> {CAPTURE_PATH}")
        return 0

    ref = torch.load(CAPTURE_PATH)
    all_ok = True
    for i, (N, C, H, W, use_smooth, use_int4, dtype) in enumerate(all_cases()):
        key = key_of(N, C, H, W, use_smooth, use_int4, dtype)
        out = run_case(N, C, H, W, use_smooth, use_int4, dtype, seed=i)
        ok = torch.equal(out, ref[key])
        all_ok &= ok
        print(f"[{'PASS' if ok else 'FAIL'}] {key:34s} equal={ok}")
    print("ALL PASS" if all_ok else "SOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

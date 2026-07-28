"""Capture-then-compare regression check for step1_static_quantize_fprop /
step1_static_quantize_fprop_silu (Cycle 2's
static_quantize_and_update_ahat_kernel_int8_half_cache[_silu]) and
step1_static_quantize_pack_int4_fprop / _silu (the follow-up fix for the
int4-pack sibling that Cycle 2 missed:
static_quantize_pack_and_update_ahat_kernel_int4_half_cache[_silu]).

  python modiff_delta_quantize_vec2_compare.py --capture   # before any .cu change
  python modiff_delta_quantize_vec2_compare.py --compare   # after a rebuild

Multi-iteration (5 steps) so the fp16 a_hat cache evolves -- a single-step
check would miss cache-drift bugs, mirroring gn_modiff_verify_kernel.py's
methodology. Test matrix: N=2,C=128,H=16,W=16 (matches that script's shape)
plus a real-model channel count (C=384), each with smooth_inv on/off (the
num_channels%2==0 gate), with/without SiLU, and int8 vs int4-pack.
"""
import os, sys, argparse
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff")
import torch
import modiff_cutlass as mc

HERE = os.path.dirname(os.path.abspath(__file__))
CAPTURE_PATH = os.path.join(HERE, "modiff_delta_quantize_vec2_capture.pt")

SHAPES = [dict(N=2, C=128, H=16, W=16), dict(N=2, C=384, H=8, W=8)]
ITERS = 5

FN_TABLE = {
    (False, False): mc.step1_static_quantize_fprop,
    (False, True): mc.step1_static_quantize_fprop_silu,
    (True, False): mc.step1_static_quantize_pack_int4_fprop,
    (True, True): mc.step1_static_quantize_pack_int4_fprop_silu,
}


def cl(t):
    return t.contiguous(memory_format=torch.channels_last)


def run_case(N, C, H, W, use_smooth, use_silu, use_int4, seed):
    torch.manual_seed(seed)
    dev = "cuda"
    scale = torch.tensor([7.0 / 3.0 if use_int4 else 127.0 / 3.0], device=dev, dtype=torch.float32)
    smooth = (0.5 + torch.rand(C, device=dev, dtype=torch.float32)) if use_smooth \
        else torch.empty(0, device=dev, dtype=torch.float32)
    a_hat = cl(torch.zeros(N, C, H, W, device=dev, dtype=torch.float16))
    fn = FN_TABLE[(use_int4, use_silu)]
    trace = []
    for it in range(ITERS):
        x = cl(torch.randn(N, C, H, W, device=dev, dtype=torch.float16) * (1.0 + 0.3 * it))
        out = fn(x, a_hat, scale, smooth)
        trace.append((out.clone(), a_hat.clone()))
    return trace


def all_cases():
    for shape in SHAPES:
        for use_smooth in (False, True):
            for use_silu in (False, True):
                for use_int4 in (False, True):
                    yield shape["N"], shape["C"], shape["H"], shape["W"], use_smooth, use_silu, use_int4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", action="store_true")
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args()

    if a.capture or not os.path.exists(CAPTURE_PATH):
        results = {}
        for i, (N, C, H, W, use_smooth, use_silu, use_int4) in enumerate(all_cases()):
            key = f"N{N}_C{C}_smooth{int(use_smooth)}_silu{int(use_silu)}_int4{int(use_int4)}"
            results[key] = run_case(N, C, H, W, use_smooth, use_silu, use_int4, seed=i)
        torch.save(results, CAPTURE_PATH)
        print(f"[capture] saved {len(results)} cases -> {CAPTURE_PATH}")
        return 0

    ref = torch.load(CAPTURE_PATH)
    all_ok = True
    for i, (N, C, H, W, use_smooth, use_silu, use_int4) in enumerate(all_cases()):
        key = f"N{N}_C{C}_smooth{int(use_smooth)}_silu{int(use_silu)}_int4{int(use_int4)}"
        trace = run_case(N, C, H, W, use_smooth, use_silu, use_int4, seed=i)
        ref_trace = ref[key]
        per_iter_ok = [torch.equal(o, ro) and torch.equal(ah, rah)
                       for (o, ah), (ro, rah) in zip(trace, ref_trace)]
        ok = all(per_iter_ok)
        all_ok &= ok
        print(f"[{'PASS' if ok else 'FAIL'}] {key:32s} per_iter_equal={per_iter_ok}")
    print("ALL PASS" if all_ok else "SOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

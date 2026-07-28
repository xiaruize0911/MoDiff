"""Capture-then-compare regression check for fp16_ncw_to_fp32_cl, fp32_cl_to_fp16_ncw,
and fp16_ncw_delta_to_int8_cl (both a_hat dtypes) -- the NCW<->channels-last transpose+cast
kernels used by MoDiffConv1dCUTLASS's QKV/output-proj layout conversion.

  python layout_transform_vec2_compare.py --capture   # before any .cu change
  python layout_transform_vec2_compare.py --compare   # after a rebuild

Test matrix: real churches attention shapes (L=1024/256/64, all even) to exercise the new
vec2 fast path, plus a synthetic odd-L shape (L=97) to exercise the scalar-fallback dispatch
branch that no real shape reaches.
"""
import os, sys, argparse
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff")
import torch
import modiff_cutlass as mc

HERE = os.path.dirname(os.path.abspath(__file__))
CAPTURE_PATH = os.path.join(HERE, "layout_transform_vec2_capture.pt")

SHAPES = [dict(N=2, C=192, L=1024), dict(N=2, C=384, L=256), dict(N=2, C=384, L=64),
          dict(N=2, C=192, L=97)]  # odd L: exercises the scalar-fallback dispatch branch
ITERS = 3  # a_hat evolves across iterations for the delta kernel


def cl(t):
    return t.contiguous(memory_format=torch.channels_last)


def run_case(N, C, L, cache_dtype, seed):
    torch.manual_seed(seed)
    dev = "cuda"
    scale = torch.tensor([127.0 / 3.0], device=dev, dtype=torch.float32)

    x_ncw = torch.randn(N, C, L, device=dev, dtype=torch.float16)
    fp32cl_out = mc.fp16_ncw_to_fp32_cl(x_ncw, N, C, L)
    ncw_back = mc.fp32_cl_to_fp16_ncw(fp32cl_out.contiguous(), N, C, L)

    a_hat = cl(torch.zeros(N * L, C, 1, 1, device=dev, dtype=cache_dtype))
    delta_trace = []
    for it in range(ITERS):
        x = torch.randn(N, C, L, device=dev, dtype=torch.float16) * (1.0 + 0.3 * it)
        out = mc.fp16_ncw_delta_to_int8_cl(x, a_hat, scale, N, C, L)
        delta_trace.append((out.clone(), a_hat.clone()))

    return fp32cl_out.clone(), ncw_back.clone(), delta_trace


def all_cases():
    for shape in SHAPES:
        for cache_dtype in (torch.float32, torch.float16):
            yield shape["N"], shape["C"], shape["L"], cache_dtype


def key_of(N, C, L, cache_dtype):
    return f"N{N}_C{C}_L{L}_{str(cache_dtype).split('.')[-1]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", action="store_true")
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args()

    if a.capture or not os.path.exists(CAPTURE_PATH):
        results = {}
        for i, (N, C, L, cache_dtype) in enumerate(all_cases()):
            results[key_of(N, C, L, cache_dtype)] = run_case(N, C, L, cache_dtype, seed=i)
        torch.save(results, CAPTURE_PATH)
        print(f"[capture] saved {len(results)} cases -> {CAPTURE_PATH}")
        return 0

    ref = torch.load(CAPTURE_PATH)
    all_ok = True
    for i, (N, C, L, cache_dtype) in enumerate(all_cases()):
        key = key_of(N, C, L, cache_dtype)
        fp32cl_out, ncw_back, delta_trace = run_case(N, C, L, cache_dtype, seed=i)
        r_fp32cl, r_ncw, r_delta = ref[key]
        ok1 = torch.equal(fp32cl_out, r_fp32cl)
        ok2 = torch.equal(ncw_back, r_ncw)
        per_iter_ok = [torch.equal(o, ro) and torch.equal(ah, rah)
                       for (o, ah), (ro, rah) in zip(delta_trace, r_delta)]
        ok = ok1 and ok2 and all(per_iter_ok)
        all_ok &= ok
        print(f"[{'PASS' if ok else 'FAIL'}] {key:20s} ncw_to_cl={ok1} cl_to_ncw={ok2} delta_per_iter={per_iter_ok}")
    print("ALL PASS" if all_ok else "SOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

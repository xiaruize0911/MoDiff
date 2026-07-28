"""Capture-then-compare regression check for group_norm_silu_quantize_nhwc /
group_norm_silu_quantize_pack_nhwc (Cycle 1's Tier-2 kernels).

  # before touching any .cu file:
  python gn_static_quantize_verify.py --capture
  # after a rebuild:
  python gn_static_quantize_verify.py --compare

Test matrix: the 3 real UNet channel counts (192/384/768, all num_groups=32,
CPG=6/12/24 -- all even) plus one synthetic odd-CPG shape (C=96, CPG=3) to
smoke-test the scalar-fallback dispatch wiring the plan adds for kernels with
no existing CPG-evenness TORCH_CHECK. Each shape x {mod on/off} x {smooth
on/off} x {int8, int4-pack}.
"""
import os, sys, argparse
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff")
import torch
import modiff_cutlass as mc

HERE = os.path.dirname(os.path.abspath(__file__))
CAPTURE_PATH = os.path.join(HERE, "gn_static_capture.pt")

SHAPES = [
    dict(C=192, ng=32), dict(C=384, ng=32), dict(C=768, ng=32),
    dict(C=96, ng=32),   # synthetic odd-CPG (CPG=3) -- exercises the scalar fallback
]
N, H, W = 2, 8, 8
EPS = 1e-5


def cl(t):
    return t.contiguous(memory_format=torch.channels_last)


def run_case(C, ng, use_mod, use_smooth, seed):
    torch.manual_seed(seed)
    dev = "cuda"
    x = cl(torch.randn(N, C, H, W, device=dev, dtype=torch.float16))
    gamma = torch.randn(C, device=dev, dtype=torch.float16)
    beta = torch.randn(C, device=dev, dtype=torch.float16)
    scale = torch.tensor([127.0 / 3.0], device=dev, dtype=torch.float32)
    smooth = (0.5 + torch.rand(C, device=dev, dtype=torch.float32)) if use_smooth \
        else torch.empty(0, device=dev, dtype=torch.float32)
    if use_mod:
        ms = torch.randn(N, C, device=dev, dtype=torch.float16)
        sh = torch.randn(N, C, device=dev, dtype=torch.float16)
    else:
        ms = sh = torch.empty(0, device=dev, dtype=torch.float16)

    out_i8 = mc.group_norm_silu_quantize_nhwc(x, gamma, beta, ng, EPS, True, scale, smooth, ms, sh)
    if C % 2 == 0 and (C // ng) % 2 == 0:
        out_i4 = mc.group_norm_silu_quantize_pack_nhwc(x, gamma, beta, ng, EPS, True, scale, smooth, ms, sh)
    else:
        out_i4 = None
    return out_i8.clone(), (out_i4.clone() if out_i4 is not None else None)


def all_cases():
    for shape in SHAPES:
        for use_mod in (False, True):
            for use_smooth in (False, True):
                yield shape["C"], shape["ng"], use_mod, use_smooth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", action="store_true")
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args()

    if a.capture or not os.path.exists(CAPTURE_PATH):
        results = {}
        for i, (C, ng, use_mod, use_smooth) in enumerate(all_cases()):
            out_i8, out_i4 = run_case(C, ng, use_mod, use_smooth, seed=i)
            key = f"C{C}_ng{ng}_mod{int(use_mod)}_smooth{int(use_smooth)}"
            results[key] = (out_i8, out_i4)
        torch.save(results, CAPTURE_PATH)
        print(f"[capture] saved {len(results)} cases -> {CAPTURE_PATH}")
        return 0

    ref = torch.load(CAPTURE_PATH)
    all_ok = True
    for i, (C, ng, use_mod, use_smooth) in enumerate(all_cases()):
        key = f"C{C}_ng{ng}_mod{int(use_mod)}_smooth{int(use_smooth)}"
        out_i8, out_i4 = run_case(C, ng, use_mod, use_smooth, seed=i)
        ref_i8, ref_i4 = ref[key]
        ok8 = torch.equal(out_i8, ref_i8)
        ok4 = True if (out_i4 is None and ref_i4 is None) else torch.equal(out_i4, ref_i4)
        ok = ok8 and ok4
        all_ok &= ok
        print(f"[{'PASS' if ok else 'FAIL'}] {key:32s} int8_equal={ok8} int4pack_equal={ok4}")
    print("ALL PASS" if all_ok else "SOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

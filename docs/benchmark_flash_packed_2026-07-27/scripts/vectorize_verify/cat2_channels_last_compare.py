"""Correctness check for cat2_channels_last_fp16 against torch.cat([a,b],dim=1) directly
-- not a before/after capture-compare like the other scripts here, since this op has no
numerics to drift: concatenation is pure data movement, so a fresh torch.equal check
against torch.cat is exactly as strong a guarantee as capturing a golden file would be.

  python cat2_channels_last_compare.py
"""
import os, sys
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff")
import torch
import modiff_cutlass as mc

# (N, C1, C2, H, W): real churches skip-concat shapes traced this session
# (768+768=1536, 768+384=1152, 384+192=576) plus a couple of edge-y shapes.
SHAPES = [
    (4, 768, 768, 2, 2), (4, 768, 384, 4, 4), (4, 384, 192, 8, 8),
    (4, 384, 192, 16, 16), (4, 192, 192, 32, 32),
    (2, 2, 2, 3, 5),      # tiny/odd spatial dims, still-even channels
]


def cl(t):
    return t.contiguous(memory_format=torch.channels_last)


def main():
    all_ok = True
    for (N, C1, C2, H, W) in SHAPES:
        for dtype in (torch.float16,):
            torch.manual_seed(0)
            a = cl(torch.randn(N, C1, H, W, device="cuda", dtype=dtype))
            b = cl(torch.randn(N, C2, H, W, device="cuda", dtype=dtype))
            ref = torch.cat([a, b], dim=1)
            out = mc.cat2_channels_last_fp16(a, b)
            ok_val = torch.equal(out, ref)
            ok_fmt = out.is_contiguous(memory_format=torch.channels_last)
            ok = ok_val and ok_fmt
            all_ok &= ok
            print(f"[{'PASS' if ok else 'FAIL'}] N{N}_C{C1}+{C2}_H{H}W{W}  "
                  f"values_equal={ok_val} channels_last_out={ok_fmt}")
    # Non-multiple-of-2 channel count: must raise (guarded in Python by openaimodel.py's
    # _skip_concat, but the raw kernel itself should refuse cleanly, not silently corrupt).
    try:
        a = cl(torch.randn(2, 3, 8, 8, device="cuda", dtype=torch.float16))
        b = cl(torch.randn(2, 4, 8, 8, device="cuda", dtype=torch.float16))
        mc.cat2_channels_last_fp16(a, b)
        print("[FAIL] odd C1 should have raised")
        all_ok = False
    except RuntimeError as e:
        print(f"[PASS] odd C1 cleanly rejected: {e}")

    print("ALL PASS" if all_ok else "SOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

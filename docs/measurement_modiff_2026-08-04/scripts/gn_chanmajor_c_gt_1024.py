"""Correctness, determinism and speed for the C>1024 channel-major GroupNorm statistics path.

Background. gn_stats_partials_chanmajor_kernel ran one thread per channel, so it required C <= 1024
and gn_launch_group_stats fell back to the historical group-major tree otherwise. An in-source
comment asserted "every channel count in this UNet is 192/384/576/768, all <= 1024" -- that is true
of ENCODER blocks only. A decoder ResBlock normalises cat([h, hs.pop()]), so its GroupNorm sees
1152 or 1536 channels and took the fallback. Measured 2026-08-04 on the MoDiff path:
gn_group_stats_kernel still at 142.3 ms/batch (0.71 ms/step).

The fix gives each thread K = ceil(C/1024) channels, block = C/K.

WHICH ENTRY POINT. gn_launch_group_stats has exactly two callers, both on the MoDiff delta path:
group_norm_silu_delta_quantize_nhwc and its _pack_ int4 twin. The plain group_norm_silu_nhwc
computes its statistics INSIDE the fused kernel and never reaches the launcher -- a first version of
this script benchmarked that entry and got byte-identical timings with and without
MODIFF_GN_STATS_ALT=0, which is what exposed the mistake. Only the delta entries exercise the
changed code, which is also consistent with gn_group_stats_kernel appearing only in MoDiff modes in
the end-to-end profile.

What is checked:
  1. accuracy   against an fp64 reference, NOT against the old kernel. Bit-exactness with the
                group-major tree was given up by design when the delta quantizer went dynamic; a
                different but equally valid fp32 summation order moves mean/var by ~1 ULP. The
                comparison is on the DEQUANTIZED codes, since the entry emits int8.
  2. determinism identical output over repeated launches, and across two spatial sizes so a
                different nblocks is exercised. This is the property that ruled out ALT=1/ALT=2.
  3. speed      against MODIFF_GN_STATS_ALT=0, which forces the group-major tree. Must be a
                SEPARATE PROCESS: the selector is a function-local static captured once per process,
                so setting the variable mid-process is silently ineffective.

Usage:  python <this> [--alt0]
"""
import argparse
import os
import sys

os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
sys.path.insert(0, os.getcwd())

ap = argparse.ArgumentParser()
ap.add_argument("--alt0", action="store_true", help="force the historical group-major tree")
ap.add_argument("--batch", type=int, default=128,
                help="N. At N=8 every call is launch-bound (~0.06 ms floor) and the "
                     "statistics kernel is invisible; the e2e figures are batch 128.")
a = ap.parse_args()
if a.alt0:
    os.environ["MODIFF_GN_STATS_ALT"] = "0"      # before the first launch, hence before any call

import torch                                                              # noqa: E402
import modiff_cutlass as mc                                               # noqa: E402

#: (C, H, W). 1152 and 1536 are the decoder ResBlock widths that took the fallback; the rest are
#: encoder widths that already used the channel-major path and must not regress.
SHAPES = [(192, 32, 32), (384, 16, 16), (576, 32, 32), (768, 16, 16),
          (1152, 8, 8), (1536, 4, 4)]
N, G, EPS, Q = a.batch, 32, 1e-6, 127.0
EMPTY_H = None


def make(C, H, W):
    x = (torch.randn(N, C, H, W, device="cuda", dtype=torch.float16)
         .contiguous(memory_format=torch.channels_last))
    w = torch.randn(C, device="cuda", dtype=torch.float16)
    b = torch.randn(C, device="cuda", dtype=torch.float16)
    # a_hat seeded non-zero: with a zero cache the delta is the activation itself and the cache
    # read/update path is not really exercised.
    a_hat = (torch.randn(N, C, H, W, device="cuda", dtype=torch.float16) * 0.1
             ).contiguous(memory_format=torch.channels_last)
    return x, w, b, a_hat


def run(x, w, b, a_hat, scale):
    """Dynamic-scale mode, which is what ships. Returns the int8 codes."""
    dev = x.device
    e_h = torch.empty(0, device=dev, dtype=torch.float16)
    e_f = torch.empty(0, device=dev, dtype=torch.float32)
    absmax = torch.zeros(1, device=dev, dtype=torch.float32)
    s_out = torch.zeros(1, device=dev, dtype=torch.float32)
    inv_out = torch.zeros(1, device=dev, dtype=torch.float32)
    retire = torch.zeros(1, device=dev, dtype=torch.int32)
    return mc.group_norm_silu_delta_quantize_nhwc(
        x, w, b, a_hat, G, EPS, True, scale, e_f, e_h, e_h,
        absmax, s_out, inv_out, retire, Q, False, 1.0)


def reference(x, w, b):
    """fp64 GroupNorm + SiLU from the same inputs -- the pre-quantize activation."""
    xd = x.double()
    y = torch.nn.functional.group_norm(xd, G, w.double(), b.double(), EPS)
    return y * torch.sigmoid(y)


def main():
    torch.manual_seed(0)
    print(f"{'C':>6}{'H':>4}{'W':>4}{'K':>3}{'blk':>6}   "
          f"{'rel L2 vs fp64':>15}   {'det':>5}  {'ms/call':>9}")
    bad, times = [], {}
    for C, H, W in SHAPES:
        K = (C + 1023) // 1024
        blk = C // K
        x, w, b, a_hat0 = make(C, H, W)
        scale = torch.full((1,), 8.0, device=x.device, dtype=torch.float32)

        # accuracy: (a_hat_after - a_hat_before) is the quantized delta; a_hat_after should track
        # the fp64 activation to within one quantizer step.
        a_hat = a_hat0.clone()
        run(x, w, b, a_hat, scale)
        ref = reference(x, w, b)
        rel = ((a_hat.double() - ref).norm() / ref.norm().clamp_min(1e-12)).item()

        # determinism: same inputs, fresh cache each time, 20 launches
        def once():
            ah = a_hat0.clone()
            codes = run(x, w, b, ah, scale)
            return codes.clone(), ah
        c0, h0 = once()
        det = True
        for _ in range(19):
            c, h = once()
            if not (torch.equal(c, c0) and torch.equal(h, h0)):
                det = False
                break

        ah = a_hat0.clone()
        for _ in range(10):
            run(x, w, b, ah, scale)
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(50):
            run(x, w, b, ah, scale)
        e.record()
        torch.cuda.synchronize()
        ms = s.elapsed_time(e) / 50
        times[(C, H, W)] = ms

        print(f"{C:>6}{H:>4}{W:>4}{K:>3}{blk:>6}   {rel:>15.3e}   "
              f"{'yes' if det else 'NO':>5}  {ms:>9.3f}")
        if rel > 0.2 or not det:
            bad.append((C, H, W, rel, det))

    print()
    if bad:
        print("FAIL:", bad)
        return 1
    print(f"path = {'ALT=0 group-major tree' if a.alt0 else 'channel-major (default)'}; "
          f"total {sum(times.values()):.3f} ms over the 6 shapes")
    return 0


if __name__ == "__main__":
    sys.exit(main())

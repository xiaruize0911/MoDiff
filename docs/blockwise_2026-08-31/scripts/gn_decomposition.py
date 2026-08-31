"""Does MoDiff's GN stage lack the baseline's fast-reduce? No -- it uses a better decomposition.

WHY THIS EXISTS. FINDINGS section 4 observed that MoDiff's GN stage (11.3 ms/step) is
0.58x the baseline's `_fast` GN stage (6.6 ms), and an earlier revision of this doc
recommended "port fast-reduce to the delta kernel, worth up to 4.7 ms/step". That was
wrong, twice over, and this script is the refutation.

What fast-reduce actually is (csrc/gn_block_size.h, csrc/baseline/norm/group_norm_silu.cu):
a BLOCK-SIZE policy for the GroupNorm group-statistics reduction, nothing more.

  generic:  block_size = 32, double until it covers group_size, cap 1024
  fast:     block_size = 128, double while block_size*12 < group_size, cap 512
            i.e. ~six pairs per thread on the pair-major pass 1

The grid is `N * num_groups` either way. The win is pure occupancy: 1024 threads are
catastrophic when one group cannot fill them, so the gain is largest at the smallest
shapes (4.5-4.9x at 8x8 and 4x4, 1.1x at 2x2 per docs/gn_fast_reduce_2026-08-16).

Why MoDiff does not want it. The shipped delta kernel does not use the group-major
decomposition at all -- `gn_launch_group_stats` defaults to CHANNEL-major (BLK = C/K),
whose block size is independent of batch, so it never has the occupancy problem
fast-reduce fixes. A group-major single-kernel delta variant WITH the fast heuristic
already exists -- `group_norm_silu_delta_quantize_nhwc_fused`, reachable via
MODIFF_GN_GROUPMAJOR=1 -- and is kept as measured-regression dead code. This times the
two head to head on the 20 real shapes at production batch, and also asks whether a
per-shape dispatch between them would beat either.

Run: source setup_cuda_env.sh
     python docs/blockwise_2026-08-31/scripts/gn_decomposition.py
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from integration.utils.preflight import preflight  # noqa: E402

preflight("torch", what="gn_decomposition.py")

import torch  # noqa: E402
import modiff_cutlass as mc  # noqa: E402

DEV, CL = "cuda", torch.channels_last
JSON_OUT = "docs/blockwise_2026-08-31/data/gn_decomposition.json"
G_NORM, EPS = 32, 1e-5
#: (Cin, H, W, freq) -- the GN stage is sized by the conv's INPUT, so only Cin matters
SHAPES = [
    (768, 2, 2, 12), (384, 8, 8, 8), (192, 32, 32, 7), (384, 16, 16, 7), (768, 4, 4, 7),
    (1536, 2, 2, 3), (1536, 4, 4, 2), (768, 8, 8, 2), (768, 16, 16, 2), (384, 32, 32, 2),
    (192, 16, 16, 1), (192, 16, 16, 1), (384, 4, 4, 1), (384, 4, 4, 1), (1152, 4, 4, 1),
    (768, 8, 8, 1), (1152, 8, 8, 1), (576, 16, 16, 1), (384, 32, 32, 1), (576, 32, 32, 1),
]

E32 = torch.empty(0, device=DEV, dtype=torch.float32)
E16 = torch.empty(0, device=DEV, dtype=torch.float16)
EI = torch.empty(0, device=DEV, dtype=torch.int32)


def cl(t):
    return t.contiguous(memory_format=CL)


def _time(fn, reps, warmup=8):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    ev0.record()
    for _ in range(reps):
        fn()
    ev1.record()
    torch.cuda.synchronize()
    return ev0.elapsed_time(ev1) / reps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--out", default=JSON_OUT)
    a = ap.parse_args()

    print(f"GPU {torch.cuda.get_device_name(0)}  B={a.batch}  GN groups={G_NORM}", flush=True)
    print(f"  {'Cin':>5s} {'HxW':>7s} {'f':>3s} {'chanmajor':>10s} {'groupmaj+fast':>14s} "
          f"{'ratio':>7s}", flush=True)
    rows = []
    for cin, h, w, freq in SHAPES:
        x = cl(torch.randn(a.batch, cin, h, w, device=DEV, dtype=torch.float16))
        gam = torch.randn(cin, device=DEV, dtype=torch.float16).abs() + 0.5
        bet = torch.randn(cin, device=DEV, dtype=torch.float16) * 0.1
        ah = cl(0.1 * torch.randn(a.batch, cin, h, w, device=DEV, dtype=torch.float16))
        sc = torch.tensor([16.0], device=DEV, dtype=torch.float32)

        def chanmajor():
            return mc.group_norm_silu_delta_quantize_nhwc(
                x, gam, bet, ah, G_NORM, EPS, True, sc, E32, E16, E16,
                E32, E32, E32, EI, 127.0, False, 1.0, False, True, E32)

        def groupmajor():
            return mc.group_norm_silu_delta_quantize_nhwc_fused(
                x, gam, bet, ah, G_NORM, EPS, True, sc, E32, E16, E16, False, True)

        t1 = min(_time(chanmajor, a.reps) for _ in range(a.trials))
        t2 = min(_time(groupmajor, a.reps) for _ in range(a.trials))
        rows.append({"cin": cin, "H": h, "W": w, "freq": freq, "HW": h * w,
                     "chanmajor_ms": t1, "groupmajor_fast_ms": t2, "ratio": t1 / t2})
        print(f"  {cin:5d} {f'{h}x{w}':>7s} {freq:3d} {t1:10.3f} {t2:14.3f} "
              f"{t1 / t2:6.3f}x", flush=True)

    ch = sum(r["chanmajor_ms"] * r["freq"] for r in rows)
    gm = sum(r["groupmajor_fast_ms"] * r["freq"] for r in rows)
    orc = sum(min(r["chanmajor_ms"], r["groupmajor_fast_ms"]) * r["freq"] for r in rows)
    # the dispatch rule that reproduces the oracle exactly on this shape set
    rule = sum((r["groupmajor_fast_ms"] if r["HW"] <= 16 else r["chanmajor_ms"]) * r["freq"]
               for r in rows)
    tot = {"chanmajor_ms": ch, "groupmajor_fast_ms": gm, "oracle_ms": orc,
           "rule_hw_le_16_ms": rule, "groupmajor_vs_chanmajor": ch / gm,
           "oracle_vs_chanmajor": ch / orc, "oracle_saving_ms": ch - orc,
           "rule_matches_oracle": abs(rule - orc) < 1e-9}

    print(f"\nfreq-weighted (ms/step):", flush=True)
    print(f"  chanmajor (shipped)          {ch:6.2f}", flush=True)
    print(f"  group-major + fast-reduce    {gm:6.2f}   {ch / gm:.3f}x  "
          f"<- the 'port fast-reduce' idea, and it LOSES", flush=True)
    print(f"  per-shape best (oracle)      {orc:6.2f}   {ch / orc:.3f}x  "
          f"saves {ch - orc:.2f} ms/step", flush=True)
    print(f"  rule: HW<=16 -> group-major  {rule:6.2f}   "
          f"reproduces the oracle exactly: {tot['rule_matches_oracle']}", flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"gpu": torch.cuda.get_device_name(0), "batch": a.batch, "reps": a.reps,
               "trials": a.trials, "gn_groups": G_NORM,
               "note": "chanmajor = group_norm_silu_delta_quantize_nhwc (shipped). "
                       "groupmajor_fast = group_norm_silu_delta_quantize_nhwc_fused, the "
                       "group-major single kernel carrying the same ~six-pairs-per-thread "
                       "block-size heuristic as the baseline's fast_reduce path.",
               "shapes": rows, "freq_weighted": tot}, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

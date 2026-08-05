"""How stale can the dynamic delta scale be? Sweep MODIFF_DELTA_REFRESH x MODIFF_DELTA_CLIP.

The exact per-call scale is the largest remaining MoDiff overhead: +4.93 ms/step at int8 and +7.38
at int4 (batch 128, 2026-08-04), all of it one extra read pass over `x` and `a_hat`. Two ways to
reduce it:

  A. refresh every Kth step (MODIFF_DELTA_REFRESH=K) -- pure Python, cost drops to 1/K, but the
     reused scale can clip when the delta's range grows between refreshes.
  B. have the quantize kernels report the absmax they already compute, giving a one-step-stale
     scale for free -- needs ~6 kernel edits.

This measures A first, deliberately. If quality survives K=4, then B (which is K=1 staleness, the
mildest possible) is clearly safe and worth the kernel work. If quality dies at K=2, B is a dead end
and that was learned for 20 lines of Python instead of 6 kernel edits.

MODIFF_DELTA_CLIP < 1 coarsens the grid on refresh steps, leaving headroom for growth -- the natural
partner to a stale scale, so both axes are swept together.
"""

import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers"),
                os.path.join(ROOT, "integration/benchmarks/report"),
                os.path.dirname(os.path.abspath(__file__))]

import torch

from dynamic_delta_ab import CALIB, build, latent

REFRESH = [int(v) for v in os.environ.get("SWEEP_REFRESH", "1,2,4,8,25").split(",")]
CLIPS = [float(v) for v in os.environ.get("SWEEP_CLIPS", "1.0,0.7").split(",")]


def main():
    os.environ["MODIFF_DELTA_REFRESH"] = "1"
    os.environ["MODIFF_DELTA_CLIP"] = "1.0"
    r, m, s = build("fp16", None, "dynamic")
    latent(r, m, s)
    ref, _ = latent(r, m, s)
    del m, s, r
    torch.cuda.empty_cache()
    print(f"fp16 reference |x|max {float(ref.abs().max()):.4f}\n", flush=True)

    out = {}
    for bits in ("int8", "int4"):
        print(f"{'=' * 74}\n{bits}   (K=1 is the exact per-call scale, the reference)\n{'=' * 74}",
              flush=True)
        for clip in CLIPS:
            for k in REFRESH:
                os.environ["MODIFF_DELTA_REFRESH"] = str(k)
                os.environ["MODIFF_DELTA_CLIP"] = str(clip)
                r, m, s = build(bits, CALIB[bits], "dynamic")
                latent(r, m, s)                 # warm-up
                lat, ms = latent(r, m, s)
                rel = float((lat - ref).norm() / ref.norm())
                out[f"{bits}|K{k}|clip{clip}"] = {"rel_l2_vs_fp16": rel, "ms_per_step": ms,
                                                  "refresh": k, "clip": clip}
                print(f"  clip {clip:4.2f}  K={k:<3d}  relL2 {rel:.4f}   {ms:7.2f} ms/step",
                      flush=True)
                del m, s, r
                torch.cuda.empty_cache()
            print(flush=True)
        os.environ["MODIFF_DELTA_REFRESH"] = "1"
        os.environ["MODIFF_DELTA_CLIP"] = "1.0"

    print(f"{'=' * 74}\nQuality cost of staleness, relative to K=1 at the same clip\n{'=' * 74}")
    for bits in ("int8", "int4"):
        for clip in CLIPS:
            base = out.get(f"{bits}|K1|clip{clip}")
            if not base:
                continue
            row = "  ".join(
                f"K{k}:{out[f'{bits}|K{k}|clip{clip}']['rel_l2_vs_fp16'] / base['rel_l2_vs_fp16']:.2f}x"
                for k in REFRESH if f"{bits}|K{k}|clip{clip}" in out)
            print(f"  {bits} clip {clip:4.2f}   base relL2 {base['rel_l2_vs_fp16']:.4f}   {row}")

    with open("docs/modiff_correctness_2026-08-03/data/delta_refresh_sweep.json", "w") as f:
        json.dump({"refresh": REFRESH, "clips": CLIPS, "results": out}, f, indent=2)
    print("\nwrote docs/modiff_correctness_2026-08-03/data/delta_refresh_sweep.json")


if __name__ == "__main__":
    main()

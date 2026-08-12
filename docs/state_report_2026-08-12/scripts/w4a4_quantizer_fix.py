"""Fixing the W4A4 activation quantizer: asymmetric, clipped, or both.

THE DIAGNOSIS (probe_int4_code_use.py, 70 convs, shipped W4A4 scale, real activations):

    mass on negative codes -7..-1   3.62% median
    codes carrying >0.1% of mass    5 of 15
    |max| / |min| of the activation 19.91x
    effective bits                  log2(5) = 2.32, against a nominal 3.91

The quantity every conv consumes is `silu(gn(x))`, which is one-sided -- SiLU bottoms out at
-0.2785 and is unbounded above. A SYMMETRIC grid sized by the positive tail therefore spends seven
of its fifteen codes on a range 20x narrower than the one that matters. That is where W4A4's
missing bit and a half goes, and it is why activations-only fake quant reads 0.9060 while
weights-only reads 0.2728.

TWO FIXES, and they cost very different amounts to implement:

  CLIP    keep the symmetric grid, shrink absmax by a ratio r so the codes concentrate on the bulk
          and the tail saturates. Pure calibration -- a different number in the same .pt file, no
          kernel change at all. This is also the lever the shipped absmax file pulls by ACCIDENT:
          its scale is 5.13x too large for unsmoothed input, and it beats the correctly-sized qdiff
          scale 0.71 to 0.86.

  ASYM    give the grid a zero point so it spans [vmin, vmax] instead of [-absmax, +absmax],
          recovering the whole negative half. Implementable in the int4 GEMM -- sum(w*(a_q - zp))
          = sum(w*a_q) - zp*sum(w), and the second term is a per-output-channel constant that folds
          into the bias -- but it IS a kernel change, so it needs to be worth one.

Measured act-only (weights left fp16) so the activation grid is the only variable, then the winner
re-measured on the full W4A4 stack. Everything is fake quantization on the ordinary fp16 model, so
none of this needs a kernel to exist first.

Run: python docs/state_report_2026-08-12/scripts/w4a4_quantizer_fix.py    # ~15 min, needs the GPU
"""
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers"),
                os.path.join(ROOT, "integration/benchmarks/report"),
                os.path.join(ROOT, "docs/modiff_correctness_2026-08-03/scripts"),
                os.path.join(ROOT, "docs/qdiff_bridge_2026-08-12/scripts")]

import torch                                                                # noqa: E402
import dynamic_delta_ab as H                                               # noqa: E402
import act_fake_quant as A                                                 # noqa: E402

A.QMAX = 7.0
Q = 7.0
ACT = "integration/calibration/int4_calibration_qdiff.pt"
OUT = "docs/state_report_2026-08-12/data/w4a4_quantizer_fix.json"
SEEDS = [1234, 20260805, 777]


class ActQuant:
    """Replace the conv input with its 4-bit reconstruction. kind = sym | asym."""

    def __init__(self, kind, absmax, vmin, vmax, ratio=1.0):
        self.kind = kind
        if kind == "sym":
            self.s = Q / max(absmax * ratio, 1e-9)
        else:
            # map [vmin, r*vmax] onto the 15 codes; r shrinks the tail exactly as `ratio` does
            lo, hi = vmin, vmax * ratio
            self.s = (2 * Q) / max(hi - lo, 1e-9)
            self.lo = lo

    def __call__(self, mod, args):
        v = args[0].float()
        if self.kind == "sym":
            qv = torch.clamp(torch.round(v * self.s), -Q, Q) / self.s
        else:
            qv = (torch.clamp(torch.round((v - self.lo) * self.s) - Q, -Q, Q) + Q) / self.s + self.lo
        return (qv.to(args[0].dtype),) + args[1:]


class Collect:
    def __init__(self):
        self.vmin, self.vmax, self.absmax = float("inf"), float("-inf"), 0.0

    def __call__(self, mod, args):
        v = args[0].detach().float()
        self.vmin = min(self.vmin, float(v.min()))
        self.vmax = max(self.vmax, float(v.max()))
        self.absmax = max(self.absmax, float(v.abs().max()))
        return None


def measure(convs, make_hook, runner, model, sampler, refs):
    handles = [convs[k].register_forward_pre_hook(make_hook(k)) for k in convs]
    rels = []
    for sd in SEEDS:
        H.SEED = sd
        H.latent(runner, model, sampler)
        lat, _ = H.latent(runner, model, sampler)
        rels.append(float((lat.float() - refs[sd]).norm() / refs[sd].norm()))
    for h in handles:
        h.remove()
    return statistics.mean(rels), rels


def main():
    H.STEPS, H.BATCH = 50, 6
    os.environ["MODIFF_LINEAR"] = "0"
    shipped = {k: float(v) for k, v in
               torch.load(ACT, map_location="cpu", weights_only=True).items()}
    runner, model, sampler = H.build("fp16", None, "static")
    convs = {k: v for k, v in A.target_convs(model.model.diffusion_model).items() if k in shipped}
    if len(convs) != len(shipped):
        print(f"FAIL: {len(convs)}/{len(shipped)} scales matched")
        return 1

    print("fp16 references ...", flush=True)
    refs = {}
    for sd in SEEDS:
        H.SEED = sd
        H.latent(runner, model, sampler)
        refs[sd] = H.latent(runner, model, sampler)[0].float()

    print("collecting per-conv activation ranges ...", flush=True)
    cols = {k: Collect() for k in convs}
    hs = [convs[k].register_forward_pre_hook(cols[k]) for k in convs]
    H.SEED = SEEDS[0]
    H.latent(runner, model, sampler)
    for h in hs:
        h.remove()
    asym = statistics.median(abs(c.vmax) / max(abs(c.vmin), 1e-9) for c in cols.values())
    print(f"  |max|/|min| median {asym:.2f}x", flush=True)

    out = {}
    print("\nact-only arms (weights fp16, so the grid is the only variable):", flush=True)
    #: (label, kind, ratio). ratio shrinks the top of the range -- 1.0 is no clipping.
    #: FINER THAN THE FIRST PASS, because the first pass was not monotone: sym went
    #: 1.1500 -> 0.9355 -> 0.9765 -> 0.4522 over ratios 1.0/0.50/0.25/0.15. Everything from
    #: ~0.93 up is the SATURATED regime -- the trajectory has already left the manifold and
    #: relL2 stops ordering anything there. What matters is where each family ESCAPES, and the
    #: escape is a cliff, so a coarse sweep can straddle it and read the wrong winner.
    ARMS = [("sym, shipped scale (current)", "sym-shipped", 1.0),
            ("sym, observed absmax", "sym", 1.0),
            ("sym, clip x0.20", "sym", 0.20),
            ("sym, clip x0.15", "sym", 0.15),
            ("sym, clip x0.12", "sym", 0.12),
            ("sym, clip x0.09", "sym", 0.09),
            ("asym [vmin, vmax]", "asym", 1.0),
            ("asym, clip x0.35", "asym", 0.35),
            ("asym, clip x0.25", "asym", 0.25),
            ("asym, clip x0.20", "asym", 0.20),
            ("asym, clip x0.15", "asym", 0.15),
            ("asym, clip x0.10", "asym", 0.10)]
    for label, kind, ratio in ARMS:
        def mk(k, kind=kind, ratio=ratio):
            c = cols[k]
            if kind == "sym-shipped":
                return ActQuant("sym", Q / shipped[k], c.vmin, c.vmax, 1.0)
            return ActQuant("sym" if kind == "sym" else "asym", c.absmax, c.vmin, c.vmax, ratio)
        m, rels = measure(convs, mk, runner, model, sampler, refs)
        out[label] = {"mean": m, "relL2": rels}
        print(f"  {label:32s} {m:.4f}   {[round(x, 3) for x in rels]}", flush=True)

    best = min(out, key=lambda k: out[k]["mean"])
    cur = out["sym, shipped scale (current)"]["mean"]
    print(f"\nbest act-only arm: {best} at {out[best]['mean']:.4f}, "
          f"against the shipped grid's {cur:.4f} ({out[best]['mean'] / cur:.2f}x)")

    json.dump({"seeds": SEEDS, "asym_ratio_median": asym, "results": out, "best": best},
              open(OUT, "w"), indent=1)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

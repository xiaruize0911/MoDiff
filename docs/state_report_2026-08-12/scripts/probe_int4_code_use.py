"""How many of int4's 15 activation codes does the model actually use?

W4A4's damage is almost entirely the activation grid (fake-quant: activations-only 0.9060 against
weights-only 0.2728). This asks the next question directly: given that grid, how much of it carries
information?

THE SUSPICION. The quantity being quantized is `silu(gn(x))`, the fused GroupNorm+SiLU output that
every one of the 70 convs consumes. SiLU is ONE-SIDED: min(silu) = -0.2785 at x = -1.2785, and it is
unbounded above. The quantizer is SYMMETRIC -- `clamp(round(v*s), -7, 7)`, with s = 7/absmax and
absmax set by the positive tail. If the positive tail reaches, say, 3.9 while the negative side stops
at -0.28, then the entire negative half of the grid spans a range 14x smaller than the positive half:
seven codes covering almost nothing, and everything negative collapsing onto code 0 or -1.

That would cost about a bit. At 8 bits, losing one of eight does not show. At 4 bits it is losing one
of FOUR, and export_qdiff_scales.py already records the same concern from the other direction --
"the export has to symmetrise, which costs up to ~1 bit on the one-sided post-SiLU activations all
70 convs consume".

WHAT IS MEASURED, on real activations at the shipped W4A4 scale, per conv:
  * the fraction of elements that land on a NEGATIVE code
  * how many of the 15 codes are ever used, and how many carry >0.1% of the mass
  * the observed min/max, to confirm the asymmetry is real rather than assumed

If the negative codes are near-empty this is a one-line-of-theory fix with a real payoff: an
asymmetric (zero-point) quantizer, or equivalently folding SiLU's known lower bound into the grid.

Run: python docs/state_report_2026-08-12/scripts/probe_int4_code_use.py    # ~3 min, needs the GPU
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

ACT = "integration/calibration/int4_calibration_qdiff.pt"
OUT = "docs/state_report_2026-08-12/data/int4_code_use.json"
QMAX = 7.0


class CodeProbe:
    """Record the int4 code histogram of the conv's input. Does not modify it."""

    def __init__(self, name, s):
        self.name, self.s = name, float(s)
        self.hist = torch.zeros(int(2 * QMAX + 1), dtype=torch.float64)   # codes -7..+7
        self.n = 0
        self.vmin, self.vmax = float("inf"), float("-inf")

    def __call__(self, mod, args):
        v = args[0].detach().float()
        self.vmin = min(self.vmin, float(v.min()))
        self.vmax = max(self.vmax, float(v.max()))
        c = torch.clamp(torch.round(v * self.s), -QMAX, QMAX).flatten()
        self.hist += torch.bincount((c + QMAX).to(torch.int64),
                                    minlength=self.hist.numel()).double().cpu()
        self.n += c.numel()
        return None                       # observe only


def main():
    H.STEPS, H.BATCH, H.SEED = 50, 4, 1234
    os.environ["MODIFF_LINEAR"] = "0"
    scales = {k: float(v) for k, v in
              torch.load(ACT, map_location="cpu", weights_only=True).items()}
    runner, model, sampler = H.build("fp16", None, "static")
    convs = A.target_convs(model.model.diffusion_model)
    matched = set(convs) & set(scales)
    if len(matched) != len(scales):
        print(f"FAIL: {len(matched)}/{len(scales)} scales matched to convs")
        return 1

    probes, handles = {}, []
    for k in matched:
        p = CodeProbe(k, scales[k])
        probes[k] = p
        handles.append(convs[k].register_forward_pre_hook(p))
    H.latent(runner, model, sampler)
    for h in handles:
        h.remove()

    rows = []
    for k, p in probes.items():
        h = p.hist / max(p.n, 1)
        neg = float(h[:int(QMAX)].sum())            # codes -7..-1
        zero = float(h[int(QMAX)])                  # code 0
        used = int((p.hist > 0).sum())
        live = int((h > 1e-3).sum())                # codes carrying >0.1% of the mass
        rows.append({"layer": k, "neg_frac": neg, "zero_frac": zero, "pos_frac": 1 - neg - zero,
                     "codes_used": used, "codes_live": live,
                     "vmin": p.vmin, "vmax": p.vmax,
                     "asym_ratio": (abs(p.vmax) / max(abs(p.vmin), 1e-9))})
    del runner, model, sampler
    torch.cuda.empty_cache()

    negs = [r["neg_frac"] for r in rows]
    lives = [r["codes_live"] for r in rows]
    useds = [r["codes_used"] for r in rows]
    asym = [r["asym_ratio"] for r in rows]
    print(f"{len(rows)} convs, int4 symmetric grid of 15 codes, shipped W4A4 activation scale\n")
    print(f"  mass on NEGATIVE codes (-7..-1) : median {statistics.median(negs) * 100:6.2f}%"
          f"   range {min(negs) * 100:.2f}-{max(negs) * 100:.2f}%")
    print(f"  codes ever used                 : median {statistics.median(useds):6.1f} / 15")
    print(f"  codes carrying >0.1% of mass    : median {statistics.median(lives):6.1f} / 15")
    print(f"  |max| / |min| of the activation : median {statistics.median(asym):6.2f}x")
    eff = statistics.median(lives)
    print(f"\n  effective bits ~ log2({eff:.0f}) = {torch.log2(torch.tensor(eff)).item():.2f}"
          f"   against the nominal log2(15) = 3.91")
    json.dump({"scale_file": ACT, "layers": rows,
               "summary": {"neg_frac_median": statistics.median(negs),
                           "codes_used_median": statistics.median(useds),
                           "codes_live_median": statistics.median(lives),
                           "asym_ratio_median": statistics.median(asym)}},
              open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

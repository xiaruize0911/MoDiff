"""What is an asymmetric activation grid WORTH, measured on the quantity it acts on?

The end-to-end asymmetric arms lose (+70% PTQ, +170% MoDiff, docs/zero_point_2026-08-13/data/
zp_measured.json) and integration/tests/test_int4_zp_padding.py explains why: zero-padded convolution
makes the per-channel bias fold wrong on the border ring, by exactly -z*sum(missing w_q)*ws/s. So those
numbers price a BROKEN IMPLEMENTATION, not an asymmetric grid.

Before building the border correction that would fix it, price the ceiling. The zero point acts on one
thing -- the quantization error of silu(gn(x)) -- and that can be measured with no kernel, no conv, no
padding and no sampler:

    err(grid) = || a - dequant(quantize(a, grid)) || / || a ||

on the REAL tensors, per conv. This has no instrument to distrust: it is the definition of the
quantizer applied to captured activations. It bounds what any correct implementation could deliver,
because everything downstream only sees the reconstruction.

WHY THIS IS TRUSTWORTHY WHERE THE FAKE-QUANT HARNESS WAS NOT. That harness (zp_headroom.py) predicted
end-to-end relL2 by substituting reconstructed activations into an fp16 network -- a different network
from the one that ships (70 of 149 quantized modules perturbed), which is why it got the ORDERING wrong
twice. This script makes no end-to-end claim at all. It measures the reconstruction error itself and
reports it as a bound, which is the only thing the measurement supports.

HOOKED ON THE fp16 MODEL, on nn.Conv2d, which is where silu(gn(x)) is an ordinary argument -- the
collection route that measured |max|/|min| = 19.91x. On the int4 model the fused path enters the conv
by direct method call and a pre-hook sees the wrong tensor (docs/zero_point_2026-08-13/FINDINGS.md,
mistake #3), so that route is not used here.

Run: python docs/zp_coverage_2026-08-13/scripts/zp_activation_error.py    # ~3 min, needs the GPU
"""
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(ROOT)
sys.path[:0] = [os.path.join(ROOT, "docs/qdiff_bridge_2026-08-12/scripts"),
                ROOT, os.path.join(ROOT, "src/taming-transformers"),
                os.path.join(ROOT, "integration/benchmarks/report"),
                os.path.join(ROOT, "docs/modiff_correctness_2026-08-03/scripts")]
os.environ["MODIFF_LINEAR"] = "0"

import torch                                                              # noqa: E402
import dynamic_delta_ab as H                                             # noqa: E402

D = "docs/zp_coverage_2026-08-13"
Q = 7.0
#: the shipped symmetric clip ratio (int4_optimized.ACT_CLIP_RATIO), swept to 4.5 on 2026-08-12
SYM_RATIOS = [1.0, 3.0, 4.5, 6.7]
ASYM_RATIOS = [1.0, 3.0, 4.5, 6.7]


class Cap:
    """Capture the running range AND a bounded sample of the real values.

    A sample rather than everything: 70 convs x 50 steps x batch 8 does not fit, and the error metric
    converges long before that. RESERVOIR-FREE and deterministic -- take the first N elements of every
    K-th call -- because a random sample would make the numbers unreproducible for no benefit.
    """

    def __init__(self, keep=200_000, every=7):
        self.lo, self.hi = float("inf"), float("-inf")
        self.buf, self.keep, self.every, self.n = [], keep, every, 0
        self.held = 0

    def __call__(self, mod, args):
        v = args[0].detach()
        self.lo = min(self.lo, float(v.min()))
        self.hi = max(self.hi, float(v.max()))
        if self.n % self.every == 0 and self.held < self.keep:
            f = v.reshape(-1).float()
            take = min(self.keep - self.held, f.numel())
            self.buf.append(f[:take].clone())
            self.held += take
        self.n += 1
        return None

    def values(self):
        return torch.cat(self.buf) if self.buf else torch.empty(0, device="cuda")


def err_sym(a, absmax, r):
    s = Q / max(absmax / r, 1e-9)
    aq = torch.clamp(torch.round(a * s), -Q, Q) / s
    return float((a - aq).norm() / a.norm()), s


def err_asym(a, lo, hi, r):
    hi_c = hi / r
    if not (hi_c > lo):
        return None, None
    s = (2 * Q) / (hi_c - lo)
    z = -round(lo * s) - Q
    aq = (torch.clamp(torch.round(a * s) + z, -Q, Q) - z) / s
    return float((a - aq).norm() / a.norm()), s


def main():
    import act_fake_quant as A
    H.STEPS, H.BATCH = 50, 4
    r, m, s = H.build("fp16", None, "static")
    convs = A.target_convs(m.model.diffusion_model)
    caps = {k: Cap() for k in convs}
    hs = [convs[k].register_forward_pre_hook(caps[k]) for k in convs]
    H.SEED = 1234
    H.latent(r, m, s)
    for h in hs:
        h.remove()

    asym_ratio = statistics.median(
        abs(c.hi) / max(abs(c.lo), 1e-9) for c in caps.values())
    print(f"{len(convs)} convs, median |max|/|min| = {asym_ratio:.2f}x", flush=True)
    # Same instrument gate as export_and_measure_zp.py: silu is bounded below by -0.2785 and unbounded
    # above, and this model measures 19.91x. A value near 1 means the hook is not seeing silu(gn(x)).
    if asym_ratio < 5.0:
        print(f"REFUSING TO CONTINUE: |max|/|min| = {asym_ratio:.2f}x, expected ~19.91x. The "
              f"collection is not seeing silu(gn(x)).")
        return 1

    rows = {}
    for k, c in caps.items():
        a = c.values()
        if a.numel() == 0:
            continue
        absmax = float(a.abs().max())
        e = {}
        for rr in SYM_RATIOS:
            e[f"sym_r{rr:g}"] = err_sym(a, absmax, rr)[0]
        for rr in ASYM_RATIOS:
            v, _ = err_asym(a, c.lo, c.hi, rr)
            if v is not None:
                e[f"asym_r{rr:g}"] = v
        rows[k] = {"lo": c.lo, "hi": c.hi, "n": int(a.numel()), "err": e}
        del a
    torch.cuda.empty_cache()

    keys = [f"sym_r{r_:g}" for r_ in SYM_RATIOS] + [f"asym_r{r_:g}" for r_ in ASYM_RATIOS]
    print(f"\nrelative reconstruction error of silu(gn(x)), median over {len(rows)} convs\n")
    print(f"{'grid':12s}{'median':>10}{'p90':>10}{'worst':>10}")
    med = {}
    for kk in keys:
        vals = sorted(r_["err"][kk] for r_ in rows.values() if kk in r_["err"])
        if not vals:
            continue
        med[kk] = statistics.median(vals)
        p90 = vals[int(0.9 * (len(vals) - 1))]
        print(f"{kk:12s}{med[kk]:10.4f}{p90:10.4f}{vals[-1]:10.4f}")

    best_sym = min((v, k) for k, v in med.items() if k.startswith("sym"))
    best_asym = min((v, k) for k, v in med.items() if k.startswith("asym"))
    gain = best_sym[0] / best_asym[0]
    print(f"\nbest symmetric   {best_sym[1]:10s} {best_sym[0]:.4f}")
    print(f"best asymmetric  {best_asym[1]:10s} {best_asym[0]:.4f}")
    print(f"the zero point reduces the activation reconstruction error by {gain:.2f}x")

    # PER-CONV, not just the median: a lever that helps the median and hurts a third of the layers is a
    # different proposition from one that helps everywhere.
    wins = sum(1 for r_ in rows.values()
               if min(v for k, v in r_["err"].items() if k.startswith("asym"))
               < min(v for k, v in r_["err"].items() if k.startswith("sym")))
    print(f"asymmetric is better on {wins}/{len(rows)} convs")

    os.makedirs(f"{D}/data", exist_ok=True)
    json.dump({"convs": len(rows), "median": med, "gain": gain, "wins": wins,
               "asym_ratio": asym_ratio, "rows": rows},
              open(f"{D}/data/zp_activation_error.json", "w"), indent=1)
    print(f"wrote {D}/data/zp_activation_error.json")

    # THE DECISION RULE, and it is about a CEILING. This gain is what a perfectly correct
    # implementation could deliver on the activation term alone; the end-to-end relL2 also carries
    # weight, attention and linear error, so the realised gain is smaller. The border correction that
    # would make the implementation correct costs a [K, 9] table plus a border-only epilogue pass.
    print()
    if gain < 1.15:
        print(f"CEILING TOO LOW: {gain:.2f}x on the activation term, before any downstream dilution. "
              f"Fix #2 is answered NEGATIVELY on evidence -- not because of the padding defect, but "
              f"because the grid it would install is barely better than the clipped symmetric one.")
    else:
        print(f"CEILING IS REAL: {gain:.2f}x on the activation term. The padding defect is worth "
              f"fixing (per-output-pixel correction, [K, 9] border table for 3x3/pad=1), and the "
              f"end-to-end question can only be re-asked after that.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

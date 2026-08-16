"""B5 step 3: does AdaRound's 1.58x survive on the REAL kernels, without the retired fake-quant harness?

WHY THIS EXISTS. A11/B5's headline figure -- AdaRound W4 beating our RTN+MSE by 1.58x end-to-end -- comes
from `zp_coverage_2026-08-13/scripts/weight_zp_end_to_end.py`, which states in its first line that it
measures "with no kernel at all" and which imports `act_fake_quant`, the harness **P4 retired** after it
"failed a third self-check" and "would have said 'implement' where the truth is negative". So the prize is
quoted from an instrument that was retired for overstating exactly this kind of prize.

This measures it on the real int4 datapath instead, which is possible because
/workspace/quant_models/church_w4a8_ckpt.pth is on disk.

HOW. `convert_model_to_optimized_int4(model.model.diffusion_model)` is the single point where fp weights
become packed int4. Substituting the fp weight just BEFORE that call makes the conversion itself do the
re-quantisation onto our symmetric MSE grid -- so the symmetric route (which drops AdaRound's zero point,
measured at +6.8% on ||W-Q(W)||) needs no separate implementation.

AdaRound's rounding is reconstructed from qdiff/adaptive_rounding.py:49-61 at soft_targets=False:

    x_int   = floor(W/delta) + (alpha >= 0)
    x_quant = clamp(x_int + zero_point, 0, 15)
    W_q     = (x_quant - zero_point) * delta

Validated twice already (docs/OPEN_ITEMS.md B5): exactly 16 discrete values per output channel, and our
own baseline reproduces `_int4_weight_scale`'s docstring to 0.2%.

THE METRIC. Per B3/B4, relL2 SCREENS and FID DECIDES -- relL2 pointed the opposite way to FID once already
this session. This script produces the screen. It must not be read as the verdict, and in particular a
||W-Q(W)|| regression is expected and is not evidence against AdaRound: that is the one metric AdaRound
trades away, and measuring it is how A11 was wrongly deprioritised in the first place.

NON-VACUITY. The substitution is counted and asserted -- 89 convs, verified as a clean bijection. A run
that silently substituted 0 layers would report "no difference" and look like a result.

Run: python integration/tests/b5_adaround_e2e.py [--seeds 4]
"""
import argparse
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "integration/benchmarks/report"))
sys.path.insert(0, os.path.join(ROOT, "docs/modiff_correctness_2026-08-03/scripts"))

import torch                                                             # noqa: E402
from integration.utils.preflight import preflight, MODEL                 # noqa: E402
preflight(*MODEL, what="b5_adaround_e2e.py")
import dynamic_delta_ab as H                                             # noqa: E402
import integration.benchmarks.benchmark_ldm as B                         # noqa: E402
import integration.kernels.int4_optimized as I4                          # noqa: E402

CKPT = "/workspace/quant_models/church_w4a8_ckpt.pth"
D = "docs/gn_fast_reduce_2026-08-16"


def adaround_weights():
    """{relative-name -> W_q} for every conv, names relative to diffusion_model."""
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    base = {m.group(1) for k in ck if (m := re.match(r"(.+)\.weight_quantizer\.alpha$", k))}
    out = {}
    for b in sorted(base):
        W = ck[b + ".weight"]
        if W.dim() != 4:
            continue
        W = W.float()
        a = ck[b + ".weight_quantizer.alpha"].float()
        d = ck[b + ".weight_quantizer.delta"].float()
        z = ck[b + ".weight_quantizer.zero_point"].float()
        x_int = torch.floor(W / d) + (a >= 0).float()
        x_q = torch.clamp(x_int + z, 0, 15)          # W4 -> n_levels = 16
        assert b.startswith("model."), b
        out[b[len("model."):]] = ((x_q - z) * d)
    return out


def install(weights, tally):
    """Substitute into the fp16 state_dict as it is LOADED -- the most upstream point there is.

    A first version patched convert_model_to_optimized_int4 and matched by module name. It substituted
    2 of 89 convs, because the UNet reaching that call has already been restructured (FusedResBlock and
    the rest) so named_modules() paths no longer match the checkpoint's -- and it still printed a
    plausible -8.87% improvement, which the counter below is what caught. The verified bijection
    (`model.diffusion_model.X`, 89/89, 0 shape mismatches) applies to the STATE DICT, so that is where it
    has to be applied.
    """
    orig = torch.load

    def patched(path, *a, **kw):
        sd = orig(path, *a, **kw)
        if isinstance(path, str) and path.endswith("model.ckpt"):
            d = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
            n = 0
            for rel, Wq in weights.items():
                k = "model.diffusion_model." + rel + ".weight"
                t = d.get(k)
                if t is not None and tuple(t.shape) == tuple(Wq.shape):
                    d[k] = Wq.to(t.dtype)
                    n += 1
            tally["n"] = n
        return sd

    torch.load = patched
    return orig


def arm(use_adaround, seeds, refs, weights):
    tally = {"n": 0}
    orig = install(weights, tally) if use_adaround else None
    try:
        r, m, s = H.build("int4", H.CALIB["int4"], "dynamic")
        H.latent(r, m, s)                                  # discard: not steady state
        rel = {}
        for seed in seeds:
            H.SEED = seed
            lat, _ = H.latent(r, m, s)
            rel[seed] = float((lat - refs[seed]).norm() / refs[seed].norm())
        del r, m, s
        torch.cuda.empty_cache()
    finally:
        if orig is not None:
            torch.load = orig
    return rel, tally["n"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    args = ap.parse_args()
    seeds = [1234 + i for i in range(args.seeds)]

    weights = adaround_weights()
    print(f"reconstructed {len(weights)} AdaRound conv weights from {CKPT}")

    r, m, s = H.build("fp16", None, "dynamic")
    H.latent(r, m, s)
    refs = {}
    for seed in seeds:
        H.SEED = seed
        refs[seed], _ = H.latent(r, m, s)
    del r, m, s
    torch.cuda.empty_cache()

    base, n_base = arm(False, seeds, refs, weights)
    ada, n_ada = arm(True, seeds, refs, weights)

    import statistics
    b = [base[x] for x in seeds]
    a = [ada[x] for x in seeds]
    per = [(y - x) / x * 100.0 for x, y in zip(b, a)]
    mb, ma = statistics.mean(b), statistics.mean(a)
    sd = statistics.stdev(per) if len(per) > 1 else 0.0
    sem = sd / len(per) ** 0.5 if len(per) > 1 else 0.0

    print(f"\n=== W4A4 latent relL2 vs fp16 (lower is better), {len(seeds)} seeds ===")
    print(f"{'arm':>26} | " + " ".join(f"{x:>8}" for x in seeds) + f" | {'mean':>8}")
    print(f"{'ours (RTN + MSE scale)':>26} | " + " ".join(f"{v:>8.4f}" for v in b) + f" | {mb:>8.4f}")
    print(f"{'AdaRound -> our sym grid':>26} | " + " ".join(f"{v:>8.4f}" for v in a) + f" | {ma:>8.4f}")
    print(f"paired per-seed diff: {statistics.mean(per):+.2f}% +- {sem:.2f}% (SEM)")
    print(f"  ratio ours/AdaRound = {mb / ma:.3f}x   (the fake-quant harness claimed 1.58x)")

    print(f"\nnon-vacuity: substituted convs -- baseline arm {n_base}, AdaRound arm {n_ada} (expect 0 / 89)")
    vacuous = not (n_base == 0 and n_ada == 89)
    print("  FAIL: substitution did not happen as expected -- this verdict means NOTHING."
          if vacuous else "  PASS: 89 convs substituted in the AdaRound arm, 0 in the baseline.")

    os.makedirs(f"{D}/data", exist_ok=True)
    json.dump({"seeds": seeds, "ours": base, "adaround": ada, "mean_ours": mb, "mean_ada": ma,
               "ratio_ours_over_ada": mb / ma, "paired_diff_pct": statistics.mean(per),
               "paired_sem": sem, "n_substituted": n_ada, "vacuous": bool(vacuous),
               "note": "relL2 SCREENS; FID decides (B3/B4). A ||W-Q(W)|| regression is expected."},
              open(f"{D}/data/b5_adaround_e2e.json", "w"), indent=1)
    print(f"wrote {D}/data/b5_adaround_e2e.json")
    return 1 if vacuous else 0


if __name__ == "__main__":
    sys.exit(main())

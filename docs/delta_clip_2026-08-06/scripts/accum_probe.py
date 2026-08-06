"""Why the clip ratio wins on one-step MSE and loses end to end: it is biased, and a_hat integrates.

clip_probe.py measured the delta quantizer in isolation and found the MSE optimum well below r=1
(26% better at A8/r=0.6, 55% at A4/r=0.25). clip_e2e.py then measured the same knob end to end at A8,
where it is a faithful clip, and every ratio below 0.7 LOST -- monotonically, up to +17.5% relL2 at
r=0.4. A 26% reduction in the quantity being optimised made the latent worse.

The suspected mechanism is that one-step MSE is the wrong objective for this quantizer. MoDiff does
not use Q(delta) and discard it: it accumulates, `a_hat_t = a_hat_{t-1} + Q(a_t - a_hat_{t-1})`, and
the conv consumes a_hat. Rounding error is roughly zero-mean, so it averages down across steps.
Clipping error is not: every saturated element is pulled toward zero, so the error has a consistent
sign wherever the tail is, and an accumulator integrates a systematic term instead of cancelling it.
If that is what is happening, the r that minimises per-step error is not the r that minimises the
error of a_hat after 50 steps, and the gap should widen with the number of steps.

This runs the recursion on a captured trajectory: the same eligible convs as clip_probe.py, but
their input at EVERY DDIM step, then the MoDiff update iterated in fp32 for each (bits, r). Reported
as relL2 of a_hat against the true activation, averaged over the trajectory, plus a bias diagnostic
(mean signed error over rms error -- 0 for symmetric rounding noise, negative-definite for clipping).

Teacher-forced, and that bounds the claim: a_t comes from an fp16 trajectory, so this measures how
the accumulator handles a fixed activation sequence, not how the sampler's own drift interacts with
it. The end-to-end sweep is what measures the latter; this is here to explain its sign.
"""

import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers"),
                os.path.join(ROOT, "integration/benchmarks/report"),
                os.path.join(ROOT, "docs/modiff_correctness_2026-08-03/scripts"),
                os.path.join(ROOT, "docs/act_bits_2026-08-05/scripts")]

import torch                                                                    # noqa: E402
import dynamic_delta_ab as H                                                    # noqa: E402
from weight_ceiling import conv_eligible                                        # noqa: E402

RATIOS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2]
BITS = [(int(b), 2.0 ** (int(b) - 1) - 1) for b in os.environ.get("ACCUM_BITS", "8,4,3").split(",")]
#: A subset, not all 70: this iterates 50 steps x |RATIOS| x |BITS| per layer, and the point is the
#: SHAPE of accumulated-vs-one-step, which does not need every layer. Spread across the UNet by
#: taking every Nth eligible conv.
NLAYERS = int(os.environ.get("ACCUM_LAYERS", "10"))
#: MoDiff's t=T warm-up, post-fix: 5 rounds, each on a fresh dynamic scale (MODIFF_WARMUP_STEPS).
WARMUP = int(os.environ.get("MODIFF_WARMUP_STEPS", "5"))
OUT = os.environ.get("ACCUM_OUT", "docs/delta_clip_2026-08-06/data/accum_probe.json")


def quant(d, Q, r):
    """The delta quantizer with a true clip ratio: step r*absmax/Q, saturating at +-Q."""
    s = Q / (r * d.abs().max().clamp_min(1e-8))
    return (d * s).round().clamp(-Q, Q) / s


def warm_up(a0, Q, r):
    """a_hat after the t=T warm-up: WARMUP residual rounds, each on its own dynamic scale.

    Uses the same clip ratio as the steps that follow, since the knob is not per-phase.
    """
    a_hat = torch.zeros_like(a0)
    for _ in range(WARMUP):
        a_hat = a_hat + quant(a0 - a_hat, Q, r)
    return a_hat


def trajectory(traj, Q, r):
    """Iterate a_hat_t = a_hat_{t-1} + Q(a_t - a_hat_{t-1}) over the captured steps.

    Returns (per-step relL2 of a_hat vs a, per-step signed-bias ratio).
    """
    a_hat = warm_up(traj[0], Q, r)
    rel, bias = [], []
    for t in range(1, len(traj)):
        a = traj[t]
        a_hat = a_hat + quant(a - a_hat, Q, r)
        e = a_hat - a
        rel.append(float(e.norm() / a.norm()))
        # mean signed error / rms error. Symmetric rounding noise gives ~0; a saturating quantizer
        # pulls every clipped element toward zero, which shows up as a consistent sign.
        bias.append(float(e.mean() / e.pow(2).mean().sqrt().clamp_min(1e-12)))
    return rel, bias


def capture():
    """{layer: [activation per DDIM step]} for NLAYERS eligible convs, from one fp16 run."""
    r, m, s = H.build("fp16", None, "static")
    unet = m.model.diffusion_model
    names = [n for n, mod in unet.named_modules() if conv_eligible(n, mod)]
    pick = names[:: max(1, len(names) // NLAYERS)][:NLAYERS]
    mods = dict(unet.named_modules())
    grab = {n: [] for n in pick}
    hooks = [mods[n].register_forward_pre_hook(
        (lambda nm: lambda mod, inp: grab[nm].append(inp[0].detach().half().cpu()))(n))
        for n in pick]
    H.SEED = 4242
    H.latent(r, m, s)
    for h in hooks:
        h.remove()
    del m, s, r
    torch.cuda.empty_cache()
    return pick, grab


def main():
    print(f"batch {H.BATCH}, DDIM {H.STEPS}, {NLAYERS} layers, warm-up {WARMUP} rounds\n", flush=True)
    pick, grab = capture()
    print("captured: " + ", ".join(f"{n.split('.')[-3] if '.' in n else n}({len(grab[n])})"
                                   for n in pick[:4]) + " ...\n", flush=True)

    out = {"batch": H.BATCH, "steps": H.STEPS, "warmup": WARMUP, "ratios": RATIOS,
           "layers": pick, "per_layer": {}}
    for name in pick:
        traj = [t.cuda().float() for t in grab[name]]
        out["per_layer"][name] = {}
        for bits, Q in BITS:
            out["per_layer"][name][bits] = {}
            for r in RATIOS:
                rel, bias = trajectory(traj, Q, r)
                out["per_layer"][name][bits][r] = {
                    "rel_mean": sum(rel) / len(rel), "rel_final": rel[-1],
                    "rel_step1": rel[0], "bias_mean": sum(bias) / len(bias)}
        del traj
        torch.cuda.empty_cache()

    for bits, Q in BITS:
        print(f"=== A{bits} (Q={int(Q)}), accumulated over {H.STEPS - 1} steps, "
              f"mean over {len(pick)} layers {'=' * 12}", flush=True)
        print(f"{'clip r':>7} | {'step 1':>8} | {'traj mean':>10} | {'final':>8} | "
              f"{'bias':>7} | {'vs r=1 (traj)':>13}", flush=True)
        print("-" * 70, flush=True)

        def col(r, key):
            v = [out["per_layer"][n][bits][r][key] for n in pick]
            return sum(v) / len(v)

        for r in RATIOS:
            print(f"{r:>7.2f} | {col(r, 'rel_step1'):>8.4f} | {col(r, 'rel_mean'):>10.4f} | "
                  f"{col(r, 'rel_final'):>8.4f} | {col(r, 'bias_mean'):>+7.3f} | "
                  f"{col(r, 'rel_mean') / col(1.0, 'rel_mean'):>12.3f}x", flush=True)
        b1 = min(RATIOS, key=lambda r: col(r, "rel_step1"))
        bt = min(RATIOS, key=lambda r: col(r, "rel_mean"))
        print(f"       best r on step 1: {b1:.2f}   best r on the trajectory: {bt:.2f}\n",
              flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()

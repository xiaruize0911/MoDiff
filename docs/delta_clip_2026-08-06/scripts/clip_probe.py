"""Would clipping the MoDiff delta grid help at A4/A3? Measured on real deltas, offline.

docs/act_bits_2026-08-05/FINDINGS.md listed "sweep MODIFF_DELTA_CLIP at A4 and A3" as the cheapest
quality win available, described as zero code. It is not, and the reason is worth recording before
anyone spends a GPU-hour on it.

`MODIFF_DELTA_CLIP=r` is implemented as `Q_level = act_q / r`, and Q_level only sets the quantizer's
SCALE (`scale = Q_level / absmax`). The code ceiling is a hardcoded literal in the kernels --
`fmaxf(-127.0f, fminf(127.0f, ...))` in every int8 path of csrc/kernels/quantize/modiff_delta_quantize.cu
and csrc/kernels/norm/group_norm_silu.cu, 7 sites, and 7.0f in the int4 ones. So at A4
(`MODIFF_ACT_Q=7`) a ratio of 0.5 gives Q_level 14, every code lands well inside +-127, and NOTHING
saturates: the grid simply gets finer. Sweeping the knob down at A4 measures effective activation
PRECISION (a non-integer number of bits), not clipping, and it would show a monotone "win" that is
just A5 wearing an A4 label. A real clip needs the ceiling to move with Q_b, which is a CUDA change
plus a rebuild -- so measure first whether the win exists at all.

That is what this does, in fp32 PyTorch on real captured deltas:

    shipped   scale = Q/(r*absmax), codes clamped at +-127   -- what the knob does today
    clipped   scale = Q/(r*absmax), codes clamped at +-Q      -- what a b-bit clip quantizer is

r=1 is the shipped default and is identical under both rules (codes cannot exceed Q when the scale is
built from the tensor's own absmax). Everything below r=1 is where they diverge.

Deltas come from an fp16 model: the conv input at DDIM step t minus the same conv's input at step
t+1. In the ResBlock/attention datapath the conv's input is already post-GroupNorm-SiLU, which is
exactly the tensor the fused kernel quantizes. Two approximations, both stated because they bound the
claim: the model's real subtrahend is a_hat (the DEQUANTIZED previous activation) rather than a, and
these deltas come from an fp16 trajectory rather than a quantized one. Both perturb the delta by one
quantization error, i.e. by the quantity whose distribution we are studying -- fine for "does a clip
ratio below 1 reduce quantization MSE", which is a question about tail shape, and not fine for
predicting an end-to-end relL2. The kernel change is what would answer that.
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

#: Which DDIM step pairs to form deltas from. Early, middle and late: the delta's magnitude and
#: tail shape both evolve along the trajectory, and a clip ratio tuned on one end can lose on the
#: other. (t index counts UNet forwards, 0-based, out of H.STEPS.)
PAIRS = [(int(a), int(a) + 1) for a in os.environ.get("CLIP_STEPS", "2,24,46").split(",")]
#: r=1.0 is the shipped default and the control. Below it the two clamp rules diverge.
RATIOS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1]
BITS = [(int(b), 2.0 ** (int(b) - 1) - 1) for b in os.environ.get("CLIP_BITS", "8,4,3,2").split(",")]
OUT = os.environ.get("CLIP_OUT", "docs/delta_clip_2026-08-06/data/clip_probe.json")


def capture(steps_wanted):
    """{layer_name: {step_index: activation}} for every eligible conv, from one fp16 sampling run.

    Captured on the fp16 model for the same reason probe_warmup.py does it: the question is about
    the delta's distribution, and an fp16 trajectory gives it without having to replicate which of
    the three quantized forward paths (GN-fused, modulated, first-step) each of the 70 layers takes.
    """
    r, m, s = H.build("fp16", None, "static")
    unet = m.model.diffusion_model
    names = [n for n, mod in unet.named_modules() if conv_eligible(n, mod)]
    mods = dict(unet.named_modules())
    grab, seen = {}, {}

    def hook(nm):
        def f(mod, inp):
            k = seen.get(nm, 0)
            seen[nm] = k + 1
            if k in steps_wanted:
                # fp16 on CPU: 70 layers x |steps| captures at batch 8 is ~350 MB at fp32, and the
                # arithmetic below promotes to fp32 per layer anyway.
                grab.setdefault(nm, {})[k] = inp[0].detach().half().cpu()
        return f

    hooks = [mods[n].register_forward_pre_hook(hook(n)) for n in names]
    H.SEED = 4242
    H.latent(r, m, s)                        # ONE run: fp16 has no self-calibrating state to warm up
    for h in hooks:
        h.remove()
    del m, s, r
    torch.cuda.empty_cache()
    return names, grab


def sweep(d, Q):
    """Relative L2 of the dequantized delta, per ratio, under both clamp rules.

    Both share `scale = Q/(r*absmax)`. `shipped` clamps at +-127 (the literal in the kernels today),
    `clipped` clamps at +-Q (what a b-bit quantizer with a clip ratio actually is).
    """
    am = d.abs().max().clamp_min(1e-8)
    n = d.norm()
    out = {}
    for r in RATIOS:
        s = Q / (r * am)
        codes = (d * s).round()
        out[r] = {
            "shipped": float((codes.clamp(-127.0, 127.0) / s - d).norm() / n),
            "clipped": float((codes.clamp(-Q, Q) / s - d).norm() / n),
        }
    return out


def main():
    steps_wanted = sorted({s for p in PAIRS for s in p})
    print(f"batch {H.BATCH}, DDIM {H.STEPS}, capturing steps {steps_wanted}, "
          f"pairs {PAIRS}\n", flush=True)
    names, grab = capture(set(steps_wanted))
    have = [n for n in names if all(s in grab.get(n, {}) for s in steps_wanted)]
    print(f"{len(have)} of {len(names)} eligible convs captured at every step\n", flush=True)

    out = {"batch": H.BATCH, "steps": H.STEPS, "pairs": PAIRS, "ratios": RATIOS,
           "layers": len(have), "per_pair": {}}

    for t0, t1 in PAIRS:
        per_layer, weight = {}, {}
        for name in have:
            d = grab[name][t0].cuda().float() - grab[name][t1].cuda().float()
            if float(d.norm()) == 0.0:                  # a layer whose input did not move
                continue
            weight[name] = float(d.norm()) ** 2
            per_layer[name] = {bits: sweep(d, Q) for bits, Q in BITS}
            del d
            torch.cuda.empty_cache()
        out["per_pair"][f"{t0}->{t1}"] = {"per_layer": per_layer, "weight_l2sq": weight}

        def agg(bits, r, rule):
            """Signal-weighted rms of the per-layer relative errors.

            = sqrt(sum_l ||err_l||^2 / sum_l ||d_l||^2), since err_l is stored relative to ||d_l||.
            Weighted, so a 224-channel layer is not outvoted by a 32-channel one -- an unweighted
            mean over 70 layers answers a different question than "how much quantization noise does
            the network absorb".
            """
            num = sum(weight[n] * per_layer[n][bits][r][rule] ** 2 for n in per_layer)
            return (num / sum(weight.values())) ** 0.5

        for bits, Q in BITS:
            print(f"=== delta t{t0}->t{t1}, A{bits} (Q={int(Q)}) {'=' * 30}", flush=True)
            print(f"{'ratio':>6} | {'clipped (clamp +-Q)':>19} | "
                  f"{'shipped (clamp +-127)':>21} | {'layers best here':>16}", flush=True)
            print("-" * 72, flush=True)
            for r in RATIOS:
                nbest = sum(1 for n in per_layer
                            if min(RATIOS, key=lambda rr: per_layer[n][bits][rr]["clipped"]) == r)
                print(f"{r:>6.2f} | {agg(bits, r, 'clipped'):>19.5f} | "
                      f"{agg(bits, r, 'shipped'):>21.5f} | {nbest:>16d}", flush=True)
            best = min(RATIOS, key=lambda r: agg(bits, r, "clipped"))
            print(f"       best global r = {best:.2f}: "
                  f"{agg(bits, 1.0, 'clipped'):.5f} -> {agg(bits, best, 'clipped'):.5f} "
                  f"({100 * (1 - agg(bits, best, 'clipped') / agg(bits, 1.0, 'clipped')):+.1f}%)\n",
                  flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()

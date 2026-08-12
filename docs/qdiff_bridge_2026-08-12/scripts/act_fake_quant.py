"""A7: judge activation-scale quality WITHOUT the CUDA kernels.

This is the advisor's explicit first step -- "先测一下 fake quantization 下表现怎么样" -- and it gates
every step after it. If Q-Diffusion's scales do not beat the shipped ones here, the calibration
hypothesis is dead and no kernel work is justified.

THE TRICK THAT MAKES THIS 100 LINES INSTEAD OF A REWRITE. A convolution is linear, so MoDiff's
recursion telescopes:

    o_hat_T = A(a_hat_T) + b
    o_hat_t = o_hat_{t-1} + A(d_dq_t) = A(a_hat_{t-1} + d_dq_t) + b = A(a_hat_t) + b

So substituting a_hat_t for the conv's INPUT reproduces the entire MoDiff datapath exactly, without
touching the conv at all. The model runs in fp16 mode -- no OptimizedInt8Conv2d, no modiff_cutlass,
no rebuild -- and a forward_pre_hook does the whole job.

WHAT IS MIRRORED, from integration/kernels/int8_optimized.py (the SHIPPED path, not
fused_baseline.py, which uses a dynamic scale at t=T and 3 warm-up rounds rather than 5):
  t=T    a_hat = q(x, s_act)                                        (:1428, the STATIC scale)
         then warmup_steps-1 = 4 rounds of
           r = x - a_hat ; a_hat += q(r, 127/absmax(r))             (:1440-1452, DYNAMIC per round)
  t<T    d = x - a_hat  ; a_hat += q(d, s_delta)                    (s_delta dynamic or from a table)
  q(v,s) = clamp(round(v*s), -127, 127) / s                         (:722-726)

THE ONE IDEALISATION, stated because it is the thing that could make this harness disagree with the
kernels: the kernels accumulate o_hat in fp16, this accumulates a_hat in fp32 and does one conv. If
A9's kernel A/B ever contradicts this harness's ordering, that gap is the first suspect.

Run: python docs/qdiff_bridge_2026-08-12/scripts/act_fake_quant.py [--steps 50] [--seeds 3]
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers"),
                os.path.join(ROOT, "integration/benchmarks/report"),
                os.path.join(ROOT, "docs/modiff_correctness_2026-08-03/scripts")]

import torch                                                                # noqa: E402
import dynamic_delta_ab as H                                               # noqa: E402

SHIPPED = "integration/calibration/int8_calibration_realckpt.pt"
SHIPPED_DELTA = "integration/calibration/int8_delta_calibration.pt"
QD_SYM = "docs/qdiff_bridge_2026-08-12/data/qdiff_act_sym.pt"
QD_MSE = "docs/qdiff_bridge_2026-08-12/data/qdiff_act_mse.pt"
QD_DELTA = "docs/qdiff_bridge_2026-08-12/data/qdiff_delta.pt"
QD_DELTA_FLAT = "docs/qdiff_bridge_2026-08-12/data/qdiff_delta_flat.pt"
OUT = "docs/qdiff_bridge_2026-08-12/data/act_fake_quant.json"

#: int8_optimized.py:113 -- 1 initial quantize + 4 refinement rounds at t=T
WARMUP_ROUNDS = 5
QMAX = 127.0

#: The bar, from docs/modiff_correctness_2026-08-03/FINDINGS.md's 2026-08-04 headline table
#: (real ckpt, DDIM S=50, batch 8, seed 1234, latent relL2 vs fp16, measured AT STEADY STATE).
#:
#: DO NOT use data/dynamic_delta_ab.json for this. That file is the FIRST version of that A/B and it
#: is stale: its int8-dynamic row reads 10.32 with latent_absmax 47.8, a diverged first-run capture
#: taken before the warm-up discipline existed. FINDINGS carries an explicit correction saying the
#: first version "reported the opposite for W8A8". gn_stats_ab.json (int8 default 0.0412) and
#: int4_delta_table.json (int8_baseline 0.2378) corroborate the table below.
REFERENCE = {"baseline": 0.2378,          # int8_baseline, MoDiff off
             "modiff_static_notable": 0.1878,
             "modiff_static_table": 0.0422,
             "modiff_dynamic": 0.0393}


def q(v, s):
    """clamp(round(v*s), -127, 127) / s -- integration's _dequantize_activation, :722-726."""
    return torch.clamp(torch.round(v * s), -QMAX, QMAX) / s


class DeltaHook:
    """Per-conv MoDiff state, applied by replacing the conv's input.

    mode 'baseline' : a_hat = q(x, s_act) every step, no state
    mode 'modiff'   : the t=T warm-up then the t<T delta recursion
    """

    def __init__(self, name, s_act, s_delta=None, mode="modiff"):
        self.name, self.s_act, self.s_delta, self.mode = name, s_act, s_delta, mode
        self.a_hat = None
        self.step = 0
        self.calls = 0
        self.clip_hits = 0          # elements that saturated at +-127
        self.elems = 0
        self.max_code = 0.0

    def reset(self):
        self.a_hat, self.step = None, 0

    def _note(self, v, s):
        c = (v * s).abs()
        self.max_code = max(self.max_code, float(c.max()))
        self.clip_hits += int((c > QMAX).sum())
        self.elems += c.numel()

    def __call__(self, mod, args):
        x = args[0].float()
        self.calls += 1
        if self.mode == "baseline":
            self._note(x, self.s_act)
            return (q(x, self.s_act).to(args[0].dtype),) + args[1:]

        if self.a_hat is None or self.a_hat.shape != x.shape:
            self._note(x, self.s_act)
            a = q(x, self.s_act)
            for _ in range(WARMUP_ROUNDS - 1):
                r = x - a
                rs = QMAX / r.abs().max().clamp_min(1e-6)
                a = a + q(r, rs)
            self.a_hat, self.step = a, 1
        else:
            d = x - self.a_hat
            if self.s_delta is None:                       # today's DYNAMIC default: per-call absmax
                s = QMAX / d.abs().max().clamp_min(1e-6)
            else:
                s = float(self.s_delta[min(self.step - 1, self.s_delta.numel() - 1)])
            self._note(d, s)
            self.a_hat = self.a_hat + q(d, s)
            self.step += 1
        return (self.a_hat.to(args[0].dtype),) + args[1:]


def target_convs(unet):
    """The 70 live convs, reached through FusedResBlock.

    NOT named_modules(): FusedResBlock keeps self.original, whose in_layers[-1]/out_layers[-1] are the
    SAME objects as in_conv/out_conv, so a name walk registers each conv twice and the hook would run
    twice per call. Walking the blocks also yields integration's key names for free.
    """
    import integration.fused_ops.fused_resblock as FR
    out = {}
    for name, rb in unet.named_modules():
        if not isinstance(rb, FR.FusedResBlock):
            continue
        for attr, suffix in (("in_conv", "in_conv"), ("out_conv", "out_conv")):
            m = getattr(rb, attr, None)
            if m is not None:
                out[f"{name}.{suffix}"] = m
    return out


def run_arm(label, scales, delta_table, mode, steps, seed, runner, model, sampler, unet):
    convs = target_convs(unet)
    hooks, handles = {}, []
    for key, mod in convs.items():
        s = scales.get(key)
        if s is None:
            continue
        dt = None
        if delta_table is not None and key in delta_table:
            dt = delta_table[key].float()
        h = DeltaHook(key, float(s), dt, mode)
        hooks[key] = h
        handles.append(mod.register_forward_pre_hook(h))
    H.SEED = seed
    lat, ms = H.latent(runner, model, sampler)
    for hd in handles:
        hd.remove()
    return lat, hooks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    H.STEPS = a.steps
    H.BATCH = 8      # match the reference table above; do not change without re-basing it

    def load(p):
        return torch.load(p, map_location="cpu", weights_only=False) if os.path.exists(p) else None

    shipped, qd_sym, qd_mse = load(SHIPPED), load(QD_SYM), load(QD_MSE)
    shipped_delta, qd_delta, qd_flat = load(SHIPPED_DELTA), load(QD_DELTA), load(QD_DELTA_FLAT)
    if shipped is None or qd_sym is None:
        print("FAIL: need the shipped and qdiff_act_sym scale files")
        return 1

    arms = [("baseline/shipped", shipped, None, "baseline"),
            ("baseline/qdiff_sym", qd_sym, None, "baseline")]
    if qd_mse is not None:
        arms.append(("baseline/qdiff_mse", qd_mse, None, "baseline"))
    arms += [("modiff/shipped+dynamic", shipped, None, "modiff"),
             ("modiff/qdiff_sym+dynamic", qd_sym, None, "modiff")]
    if shipped_delta is not None:
        arms.append(("modiff/shipped+nativetable", shipped, shipped_delta, "modiff"))
    if qd_delta is not None:
        arms.append(("modiff/qdiff_sym+qdifftable", qd_sym, qd_delta, "modiff"))
    if qd_flat is not None:
        arms.append(("modiff/qdiff_sym+qdifftable_flat", qd_sym, qd_flat, "modiff"))

    print(f"building fp16 model (batch {H.BATCH}, {a.steps} steps) ...", flush=True)
    runner, model, sampler = H.build("fp16", None, "static")
    unet = model.model.diffusion_model
    convs = target_convs(unet)
    print(f"  {len(convs)} conv hook targets; {len(set(convs) & set(shipped))} match the shipped keys")
    if len(set(convs) & set(shipped)) != 70:
        print("FAIL: hook targets do not cover the 70 calibrated convs")
        return 1

    seeds = [1234, 20260805, 777][:a.seeds]
    results = {}
    for seed in seeds:
        H.SEED = seed
        H.latent(runner, model, sampler)                    # warm-up, discarded
        ref, _ = H.latent(runner, model, sampler)           # per-seed fp16 reference
        for label, sc, dt, mode in arms:
            lat, hooks = run_arm(label, sc, dt, mode, a.steps, seed, runner, model, sampler, unet)
            rel = float((lat - ref).norm() / ref.norm())
            calls = sum(h.calls for h in hooks.values())
            clip = sum(h.clip_hits for h in hooks.values())
            elems = max(1, sum(h.elems for h in hooks.values()))
            n_clipping = sum(1 for h in hooks.values() if h.max_code > QMAX)
            r = results.setdefault(label, {"relL2": [], "clip_frac": [], "n_layers_clipping": [],
                                           "calls": calls})
            r["relL2"].append(rel)
            r["clip_frac"].append(clip / elems)
            r["n_layers_clipping"].append(n_clipping)
            print(f"  seed {seed}  {label:34s} relL2 {rel:.4f}  "
                  f"clip {100*clip/elems:.3f}%  layers clipping {n_clipping}/70  calls {calls}",
                  flush=True)

    print(f"\n{'arm':36s} {'relL2 mean':>11} {'stdev':>8} {'clip%':>8} {'layers':>7}")
    import statistics
    summary = {}
    for label, r in results.items():
        m = statistics.mean(r["relL2"])
        sd = statistics.stdev(r["relL2"]) if len(r["relL2"]) > 1 else 0.0
        cf = 100 * statistics.mean(r["clip_frac"])
        nl = statistics.mean(r["n_layers_clipping"])
        summary[label] = {"relL2_mean": m, "relL2_stdev": sd, "clip_pct": cf,
                          "layers_clipping": nl, "per_seed": r["relL2"], "calls": r["calls"]}
        print(f"{label:36s} {m:11.4f} {sd:8.4f} {cf:8.3f} {nl:7.1f}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"steps": a.steps, "batch": H.BATCH, "seeds": seeds,
               "warmup_rounds": WARMUP_ROUNDS, "reference": REFERENCE,
               "summary": summary}, open(OUT, "w"), indent=1)
    print("\n  reference (FINDINGS 2026-08-04, same steps/batch/seed):")
    for k, v in REFERENCE.items():
        print(f"    {k:26s} {v:.4f}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

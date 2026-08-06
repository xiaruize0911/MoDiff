"""How much of MoDiff's remaining error lives OUTSIDE the conv path? Attention and Linear, ablated.

The question this answers: MoDiff compensates the ResBlock conv activations, and `MODIFF_ACT_Q`
reaches only those (70 of 89 converted layers). It does NOT reach the 21 quantized attention blocks
(`quantized_std_attention.py` hardcodes lvl = 127.0 / 7.0) or the 42 quantized Linear layers
(`wxax_linear.py`), which stay at A8 in every row of every sweep in this project. So a natural
hypothesis for why W8A8+MoDiff sits at relL2 ~0.06 rather than near zero, and why pushing the conv
path to A4 only gets to ~0.09 with a clip, is that the floor is not in the conv path at all -- it is
attention, quantized and uncompensated.

Ablation by REMOVAL, which needs no new code and no mode composition:

    shipped            as measured everywhere else in this directory
    attn fp16          MODIFF_QUANT_ATTN=0   -- attention reverts to fp16 SDPA
    linear fp16        MODIFF_QUANT_LINEAR=0 -- the 42 Linears revert to fp16

(A both-off arm is not reachable through these env vars -- see the note above ARMS.)

Each arm removes a candidate error source entirely, so its relL2 drop is the UPPER BOUND on what any
amount of work on that component (MoDiff inside attention included) could ever buy. If "attn fp16"
lands on top of "shipped", the hypothesis is dead and no attention work can help quality. If it drops
a lot, attention is the ceiling and `int8_attn_modiff` (which exists: benchmark_ldm.py applies
convert_attention_to_modiff to the qkv/proj_out Conv1d) becomes the next thing to measure.

Run at A8 and A4, because the two can differ: at A8 everything is 8-bit and attention is a peer of the
conv path, while at A4 the conv path is deliberately the worst part of the network, so attention's
share should shrink. The A4 arm is measured at the clip optimum r=0.40 as well as r=1.0 -- with the
conv path at its best, whatever is left is more likely to be the other components.

Protocol as everywhere else here: one warm-up sampling run per arm discarded, paired over seeds
against a per-seed fp16 reference, real-checkpoint calibration.
"""

import json
import os
import statistics
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers"),
                os.path.join(ROOT, "integration/benchmarks/report"),
                os.path.join(ROOT, "docs/modiff_correctness_2026-08-03/scripts")]

import torch                                                                    # noqa: E402
import dynamic_delta_ab as H                                                    # noqa: E402
import kernel_suites_bench as ks                                                # noqa: E402

SEEDS = [int(s) for s in os.environ.get("ABL_SEEDS", "1234,20260805,777").split(",")]
OUT = os.environ.get("ABL_OUT", "docs/delta_clip_2026-08-06/data/attn_ablation.json")
CALIB = H.CALIB["int8"]

#: (label, {env overrides applied AFTER ks.set_env})
#:
#: The two knobs are NOT independent, which the first run of this script discovered the hard way.
#: benchmark_ldm.py:689 gates quantized attention on
#:     std_attn_bits in (4, 8) and (not quant_lin or _force_qattn)
#: so with MODIFF_QUANT_LINEAR=0 the `not quant_lin` term is already true and attention is converted
#: no matter what MODIFF_QUANT_ATTN says. A "both fp16" arm is therefore unreachable through these env
#: vars: it came back bit-identical to `linear fp16`, per seed, at all three configs, which is how the
#: gate was found. Dropped rather than kept as a duplicate. `attn fp16` is unaffected -- there
#: quant_lin is 1, so the gate reduces to _force_qattn and MODIFF_QUANT_ATTN=0 does revert attention.
ARMS = [
    ("shipped", {}),
    ("attn fp16", {"MODIFF_QUANT_ATTN": "0"}),
    ("linear fp16", {"MODIFF_QUANT_LINEAR": "0"}),
]
#: (label, MODIFF_ACT_Q, MODIFF_DELTA_CLIP)
CONFIGS = [("A8", "127", "1.0"), ("A4 r=1.0", "7", "1.0"), ("A4 r=0.40", "7", "0.4")]

_OVERRIDES = {}
_orig_set_env = ks.set_env


def _set_env_patched(mode):
    """ks.set_env writes the whole QUANT_ENV block unconditionally for any non-fp16 mode, including
    MODIFF_QUANT_ATTN=1 -- so an override has to be re-applied after it and before _setup_model reads
    it. Patching the module attribute rather than copying H.build keeps ONE definition of the harness;
    the duplicated-harness route is how the earlier sweeps in this project drifted apart."""
    _orig_set_env(mode)
    for k, v in _OVERRIDES.items():
        os.environ[k] = v


ks.set_env = _set_env_patched


def runs(mode, delta_mode, refs, calib):
    r, m, s = H.build(mode, calib, delta_mode)
    H.SEED = SEEDS[0]
    H.latent(r, m, s)                                   # warm-up, discarded
    rel, ms, lat = {}, [], {}
    for seed in SEEDS:
        H.SEED = seed
        cur, t = H.latent(r, m, s)
        ms.append(t)
        if refs is None:
            lat[seed] = cur
        else:
            rel[seed] = float((cur - refs[seed]).norm() / refs[seed].norm())
    del m, s, r
    torch.cuda.empty_cache()
    return (lat if refs is None else rel), sum(ms) / len(ms)


def stat(d):
    v = list(d.values())
    return (statistics.mean(v), (statistics.stdev(v) if len(v) > 1 else 0.0), min(v), max(v))


def main():
    os.environ["MODIFF_DELTA_REPORT"] = "0"
    os.environ["MODIFF_DELTA_REFRESH"] = os.environ.get("MODIFF_DELTA_REFRESH", "4")

    print(f"batch {H.BATCH}, DDIM {H.STEPS}, seeds {SEEDS}, "
          f"refresh={os.environ['MODIFF_DELTA_REFRESH']}\n", flush=True)
    _OVERRIDES.clear()
    refs, fp16_ms = runs("fp16", "static", None, None)
    print(f"fp16 reference: {fp16_ms:6.2f} ms/step\n", flush=True)

    out = {"batch": H.BATCH, "steps": H.STEPS, "seeds": SEEDS,
           "delta_refresh": os.environ["MODIFF_DELTA_REFRESH"], "rows": []}
    print(f"{'config':>10} {'arm':>12} | {'MoDiff relL2':>34} | {'vs shipped':>10} | {'ms/step':>8}",
          flush=True)
    print("-" * 84, flush=True)

    for cfg, q, clip in CONFIGS:
        base = None
        for label, ov in ARMS:
            os.environ["MODIFF_ACT_Q"], os.environ["MODIFF_DELTA_CLIP"] = q, clip
            _OVERRIDES.clear()
            _OVERRIDES.update(ov)
            rel, ms = runs("int8", "dynamic", refs, CALIB)
            st = stat(rel)
            base = st[0] if base is None else base
            out["rows"].append({"config": cfg, "act_q": q, "clip_ratio": float(clip),
                                "arm": label, "overrides": ov, "mean": st[0], "stdev": st[1],
                                "per_seed": rel, "ms_per_step": ms, "ratio_to_shipped": st[0] / base})
            print(f"{cfg:>10} {label:>12} | {st[0]:>8.4f} +- {st[1]:<6.4f} "
                  f"[{st[2]:.4f},{st[3]:.4f}] | {st[0] / base:>9.3f}x | {ms:>7.1f}", flush=True)
            with open(OUT, "w") as f:
                json.dump(out, f, indent=2)
        print(flush=True)

    _OVERRIDES.clear()
    os.environ["MODIFF_ACT_Q"], os.environ["MODIFF_DELTA_CLIP"] = "127", "1.0"
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()

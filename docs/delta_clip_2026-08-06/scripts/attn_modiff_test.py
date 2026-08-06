"""Does applying MoDiff to attention's qkv/proj_out projections help? The direct test.

`attn_ablation.py` split attention into its two quantized halves and removed each:

  * the QK^T/AV MATH (MODIFF_QUANT_ATTN, quantized_std_attention) -- removing it entirely changes
    nothing: 1.005x at A8, 1.019x at A4/r=1.0, 1.009x at A4/r=0.40, all slightly WORSE than shipped.
    So the half MoDiff structurally cannot compensate is not costing anything to begin with.
  * the qkv/proj PROJECTIONS (MODIFF_QUANT_LINEAR, _QuantLinearWxAx) -- removing those moved A4/r=1.0
    from 0.1725 to 0.1466, i.e. -15%.

The projections are exactly what `benchmark_ldm.py`'s `int8_attn_modiff` mode extends MoDiff to
(convert_attention_to_modiff over the qkv/proj_out Conv1d). So the -15% is the headroom, and this
measures whether MoDiff collects it -- keeping the projections QUANTIZED rather than reverting them,
which is what a real deployment needs.

Prediction to check the result against, from the ablation: up to ~15% at A4/r=1.0, and ~nothing at
A4/r=0.40, where removing the projections entirely was worth only 0.7%.

Reports the conversion count the mode prints, because if the qkv/proj have already become
_QuantLinearWxAx there may be no Conv1d left for convert_attention_to_modiff to wrap -- in which case
the arm is void rather than negative, and the count is the only way to tell those apart.
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

SEEDS = [int(s) for s in os.environ.get("AM_SEEDS", "1234,20260805,777").split(",")]
OUT = os.environ.get("AM_OUT", "docs/delta_clip_2026-08-06/data/attn_modiff_test.json")
CALIB = H.CALIB["int8"]
CONFIGS = [("A8", "127", "1.0"), ("A4 r=1.0", "7", "1.0"), ("A4 r=0.40", "7", "0.4")]
MODES = [("shipped int8", "int8"), ("int8_attn_modiff", "int8_attn_modiff")]


def runs(mode, delta_mode, refs, calib):
    r, m, s = H.build(mode, calib, delta_mode)
    H.SEED = SEEDS[0]
    H.latent(r, m, s)
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
    os.environ.setdefault("MODIFF_DELTA_REFRESH", "4")
    print(f"batch {H.BATCH}, DDIM {H.STEPS}, seeds {SEEDS}, "
          f"refresh={os.environ['MODIFF_DELTA_REFRESH']}\n", flush=True)
    refs, fp16_ms = runs("fp16", "static", None, None)
    print(f"fp16 reference: {fp16_ms:6.2f} ms/step\n", flush=True)

    out = {"batch": H.BATCH, "steps": H.STEPS, "seeds": SEEDS, "rows": []}
    print(f"{'config':>10} {'mode':>18} | {'MoDiff relL2':>34} | {'vs shipped':>10} | {'ms/step':>8}",
          flush=True)
    print("-" * 90, flush=True)
    for cfg, q, clip in CONFIGS:
        base = None
        for label, mode in MODES:
            os.environ["MODIFF_ACT_Q"], os.environ["MODIFF_DELTA_CLIP"] = q, clip
            rel, ms = runs(mode, "dynamic", refs, CALIB)
            st = stat(rel)
            base = st[0] if base is None else base
            out["rows"].append({"config": cfg, "act_q": q, "clip_ratio": float(clip),
                                "mode": mode, "label": label, "mean": st[0], "stdev": st[1],
                                "per_seed": rel, "ms_per_step": ms, "ratio_to_shipped": st[0] / base})
            print(f"{cfg:>10} {label:>18} | {st[0]:>8.4f} +- {st[1]:<6.4f} "
                  f"[{st[2]:.4f},{st[3]:.4f}] | {st[0] / base:>9.3f}x | {ms:>7.1f}", flush=True)
            with open(OUT, "w") as f:
                json.dump(out, f, indent=2)
        print(flush=True)
    os.environ["MODIFF_ACT_Q"], os.environ["MODIFF_DELTA_CLIP"] = "127", "1.0"
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()

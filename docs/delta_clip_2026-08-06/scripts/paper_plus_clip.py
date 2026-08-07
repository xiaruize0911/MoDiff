"""The one cell nothing has measured: the paper's configuration AND the clip, together, at A4/A3.

Two things independently reduce error, in disjoint regimes:

  * the CLIP (r < 1) -- a departure from the paper, whose dynamic quantizer exists precisely to avoid
    clipping. Worth 2x at A4 and 2.9x at A3, and worth nothing at A8.
  * MoDiff on the attention PROJECTIONS (the paper's "A(.) is any linear operator") -- worth 21%/19%
    at A8/A7 on 3 of 3 seeds, and nothing from A5 down.

Their regimes look complementary: the clip fixes the conv path, the projections fix what is left once
the conv path is good. So combining them at A4/A3 could go either way -- the projections might reappear
once the clip has removed the conv-path error that was masking them, or they might stay invisible
because at A4 the conv path still dominates even after clipping. The prediction on record before
running this was "still invisible", from the pattern in act_bit_sweep_paper_cfg.

Everything is K=1 (the paper's per-step dynamic scale) and every arm sits in the SAME process against
the SAME per-seed fp16 reference, because cross-script relL2 in this project disagrees by up to ~20%
and the effect being looked for is smaller than that.
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

SEEDS = [int(s) for s in os.environ.get("PC_SEEDS", "1234,20260805,777").split(",")]
OUT = os.environ.get("PC_OUT", "docs/delta_clip_2026-08-06/data/paper_plus_clip.json")
CALIB = H.CALIB["int8"]
#: (label, MODIFF_ACT_Q, MODIFF_DELTA_CLIP). r=1.0 rows are the in-process anchor; the others are each
#: row's own measured optimum from clip_e2e_bits at K=1 (A4 -> 0.40, A3 -> 0.20).
CONFIGS = [("A4 r=1.0", "7", "1.0"), ("A4 r=0.40", "7", "0.4"),
           ("A3 r=1.0", "3", "1.0"), ("A3 r=0.20", "3", "0.2")]
#: (label, MODIFF_LINEAR) -- conv-only vs conv + the 42 attention qkv/proj projections
ARMS = [("conv only", "0"), ("conv+proj", "1")]


def runs(mode, delta_mode, refs, calib):
    r, m, s = H.build(mode, calib, delta_mode)
    H.SEED = SEEDS[0]
    H.latent(r, m, s)                       # warm-up, discarded (H.latent resets all MoDiff families)
    rel, ms, lat, bad = {}, [], {}, 0
    for seed in SEEDS:
        H.SEED = seed
        cur, t = H.latent(r, m, s)
        ms.append(t)
        if not bool(torch.isfinite(cur).all()):
            bad += 1
            continue
        if refs is None:
            lat[seed] = cur
        else:
            rel[seed] = float((cur - refs[seed]).norm() / refs[seed].norm())
    del m, s, r
    torch.cuda.empty_cache()
    return (lat if refs is None else rel), sum(ms) / len(ms), bad


def stat(d):
    v = [x for x in d.values() if x == x]
    if not v:
        return (float("nan"),) * 4
    return (statistics.mean(v), (statistics.stdev(v) if len(v) > 1 else 0.0), min(v), max(v))


def main():
    os.environ["MODIFF_DELTA_REPORT"] = "0"
    os.environ["MODIFF_DELTA_REFRESH"] = "1"            # the paper's per-step dynamic scale
    os.environ["MODIFF_WARMUP_STEPS"] = os.environ.get("MODIFF_WARMUP_STEPS", "5")
    print(f"batch {H.BATCH}, DDIM {H.STEPS}, seeds {SEEDS}, K=1, warmup="
          f"{os.environ['MODIFF_WARMUP_STEPS']}\n", flush=True)
    os.environ["MODIFF_LINEAR"] = "0"
    refs, fp16_ms, _ = runs("fp16", "static", None, None)
    print(f"fp16 reference: {fp16_ms:6.2f} ms/step\n", flush=True)

    out = {"batch": H.BATCH, "steps": H.STEPS, "seeds": SEEDS, "delta_refresh": "1", "rows": []}
    print(f"{'config':>11} {'arm':>10} | {'MoDiff relL2':>34} | {'vs conv only':>12} | "
          f"{'ms/step':>8} | nonfinite", flush=True)
    print("-" * 92, flush=True)
    for cfg, q, clip in CONFIGS:
        base = None
        for label, lin in ARMS:
            os.environ["MODIFF_ACT_Q"], os.environ["MODIFF_DELTA_CLIP"] = q, clip
            os.environ["MODIFF_LINEAR"] = lin
            rel, ms, bad = runs("int8", "dynamic", refs, CALIB)
            st = stat(rel)
            base = st[0] if base is None else base
            out["rows"].append({"config": cfg, "act_q": q, "clip_ratio": float(clip),
                                "arm": label, "modiff_linear": lin, "mean": st[0], "stdev": st[1],
                                "per_seed": rel, "ms_per_step": ms, "nonfinite_seeds": bad,
                                "ratio_to_conv_only": st[0] / base})
            print(f"{cfg:>11} {label:>10} | {st[0]:>8.4f} +- {st[1]:<6.4f} "
                  f"[{st[2]:.4f},{st[3]:.4f}] | {st[0] / base:>11.3f}x | {ms:>7.1f} | {bad}",
                  flush=True)
            with open(OUT, "w") as f:
                json.dump(out, f, indent=2)
        print(flush=True)
    os.environ["MODIFF_LINEAR"] = "0"
    os.environ["MODIFF_ACT_Q"], os.environ["MODIFF_DELTA_CLIP"] = "127", "1.0"
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()

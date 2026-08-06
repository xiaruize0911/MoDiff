"""MoDiff on the Linears and on the attention projections too, quality and speed, accepting the cost.

Where MoDiff actually applies in the shipped `int8` mode, established by reading the wiring:

  * ResBlock convs      -- YES (70 calibrated layers), this is what every sweep in this directory moves
  * OptimizedInt8Linear -- `enable_modiff_mode_linear(True)` is called, but it only reaches
                           int8_linear.py's class, and when MODIFF_QUANT_LINEAR=1 (the default for
                           quantized modes) the 42 Linears are wxax_linear._QuantLinearWxAx instead
  * wxax Linears        -- NO. `convert_linears_to_wxax(..., modiff=is_modiff)` with
                           is_modiff gated on MODIFF_LINEAR, which defaults to 0
  * attention QK^T/AV   -- NO, and structurally cannot (docs/attn_modiff_profile_2026-08-04 §1)
  * attention qkv/proj  -- NO in `int8`; YES in `int8_attn_modiff` (convert_attention_to_modiff)

So "MoDiff everywhere" is two switches: MODIFF_LINEAR=1 and the int8_attn_modiff mode. Both were off
for stated reasons. The Linear one is a SPEED reason and says so at benchmark_ldm.py:715 -- the linear
MoDiff path has no GEMM o_hat-accumulate epilogue, so it costs three extra full-tensor launches per
linear per step; the correctness reason it originally carried (rel-err diverging 0.06 -> 3.2) was
Bug 2, fixed 2026-08-03.

A harness fix this needs. `dynamic_delta_ab.latent()` resets conv MoDiff state only (r8/r4), which is
all the shipped `int8` mode has. With attention or Linear MoDiff on, their temporal caches survive
into the next sampling run: measured directly, int8_attn_modiff gives a finite latent on run 1 and an
ALL-NaN latent on run 2, and since the protocol here discards run 1 as warm-up, that is what a sweep
would have recorded. `_reset_all` below resets every MoDiff-bearing family (conv, int8 Linear, wxax
Linear, attention), which is the sequence benchmark_ldm.py's own calibration loop uses.

Arms are cumulative so each row attributes one switch:

    shipped         int8            MODIFF_LINEAR=0
    +linear         int8            MODIFF_LINEAR=1
    +attn           int8_attn_modiff MODIFF_LINEAR=0
    +both           int8_attn_modiff MODIFF_LINEAR=1

Predictions to check against, from attn_ablation.py: removing the Linears entirely was worth 15% at
A4/r=1.0 and 0.7% at A4/r=0.40, so +linear cannot beat those and should track them. Removing the
attention MATH was worth nothing at all, but that is the half MoDiff cannot reach -- the projections
were never separately ablatable, so +attn is genuinely unmeasured territory.
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

SEEDS = [int(s) for s in os.environ.get("ME_SEEDS", "1234,20260805,777").split(",")]
OUT = os.environ.get("ME_OUT", "docs/delta_clip_2026-08-06/data/modiff_everywhere.json")
CALIB = H.CALIB["int8"]
CONFIGS = [("A8", "127", "1.0"), ("A4 r=1.0", "7", "1.0"), ("A4 r=0.40", "7", "0.4")]
#: (label, mode, MODIFF_LINEAR)
#:
#: The "42 wxax Linear layers" are NOT generic Linears: enumerated by name they are exactly the 21
#: attention `qkv` plus the 21 attention `proj` (input_blocks.1.1.qkv, ...). So MODIFF_LINEAR=1 is
#: "MoDiff on the attention projections" via the wxax route, and mode int8_attn_modiff is the same
#: thing via the modiff_attention Conv1d route -- in that mode wxax reports "Quantized 0 Linear
#: layers" because convert_attention_to_modiff has already taken those modules. Two routes to one
#: set of layers, so a "+both" arm is meaningless (MODIFF_LINEAR is a no-op under int8_attn_modiff)
#: and is not measured.
ARMS = [("shipped", "int8", "0"),
        ("proj:wxax", "int8", "1"),
        ("proj:conv1d", "int8_attn_modiff", "0")]


def _reset_all(model):
    """Reset every MoDiff-bearing module family. Partial resets leak state across sampling runs and
    show up as NaN on the second run (measured for int8_attn_modiff), not as a small error."""
    unet = model.model.diffusion_model
    from integration.kernels.int8_optimized import reset_modiff_state as r8
    from integration.kernels.int4_optimized import reset_modiff_state as r4
    fns = [r8, r4]
    for mod, name in (("integration.kernels.int8_linear", "reset_modiff_state_linear"),
                      ("integration.kernels.wxax_linear", "reset_wxax_modiff"),
                      ("integration.kernels.modiff_attention", "reset_attention_modiff")):
        try:
            fns.append(getattr(__import__(mod, fromlist=[name]), name))
        except Exception as e:
            print(f"  (no {name}: {e})", flush=True)
    for fn in fns:
        try:
            fn(unet)
        except Exception:
            pass


def runs(mode, delta_mode, refs, calib):
    r, m, s = H.build(mode, calib, delta_mode)
    H.SEED = SEEDS[0]
    _reset_all(m)
    H.latent(r, m, s)                                   # warm-up, discarded
    rel, ms, lat, bad = {}, [], {}, 0
    for seed in SEEDS:
        H.SEED = seed
        _reset_all(m)
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
    v = [x for x in d.values() if x == x]                # guard: a NaN would raise inside stdev
    if not v:
        return (float("nan"),) * 4
    return (statistics.mean(v), (statistics.stdev(v) if len(v) > 1 else 0.0), min(v), max(v))


def main():
    os.environ["MODIFF_DELTA_REPORT"] = "0"
    os.environ.setdefault("MODIFF_DELTA_REFRESH", "4")
    print(f"batch {H.BATCH}, DDIM {H.STEPS}, seeds {SEEDS}, "
          f"refresh={os.environ['MODIFF_DELTA_REFRESH']}\n", flush=True)
    os.environ["MODIFF_LINEAR"] = "0"
    refs, fp16_ms, _ = runs("fp16", "static", None, None)
    print(f"fp16 reference: {fp16_ms:6.2f} ms/step\n", flush=True)

    out = {"batch": H.BATCH, "steps": H.STEPS, "seeds": SEEDS,
           "delta_refresh": os.environ["MODIFF_DELTA_REFRESH"], "rows": []}
    print(f"{'config':>10} {'arm':>9} | {'MoDiff relL2':>34} | {'vs shipped':>10} | "
          f"{'ms/step':>8} | nonfinite", flush=True)
    print("-" * 92, flush=True)

    for cfg, q, clip in CONFIGS:
        base = None
        for label, mode, lin in ARMS:
            os.environ["MODIFF_ACT_Q"], os.environ["MODIFF_DELTA_CLIP"] = q, clip
            os.environ["MODIFF_LINEAR"] = lin
            rel, ms, bad = runs(mode, "dynamic", refs, CALIB)
            st = stat(rel)
            base = st[0] if base is None else base
            out["rows"].append({"config": cfg, "act_q": q, "clip_ratio": float(clip),
                                "arm": label, "mode": mode, "modiff_linear": lin,
                                "mean": st[0], "stdev": st[1], "per_seed": rel,
                                "ms_per_step": ms, "nonfinite_seeds": bad,
                                "ratio_to_shipped": st[0] / base if base else None})
            print(f"{cfg:>10} {label:>9} | {st[0]:>8.4f} +- {st[1]:<6.4f} "
                  f"[{st[2]:.4f},{st[3]:.4f}] | {st[0] / base:>9.3f}x | {ms:>7.1f} | {bad}",
                  flush=True)
            with open(OUT, "w") as f:
                json.dump(out, f, indent=2)
        print(flush=True)

    os.environ["MODIFF_LINEAR"] = "0"
    os.environ["MODIFF_ACT_Q"], os.environ["MODIFF_DELTA_CLIP"] = "127", "1.0"
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()

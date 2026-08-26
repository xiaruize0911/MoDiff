"""Same instrument as docs/bench_report_2026-08-13_postzp/scripts/warmup_cost.py, run at
MODIFF_WARMUP_STEPS=1 instead of the default 5, so its output can be diffed against the committed
warmup_cost.json (measured at 5) for e2e speedup and warm-up % per mode.

MODIFF_WARMUP_STEPS must be set before this process imports anything that builds a model (the conv
wrappers read it in __init__), so it is set via the environment before `python` starts, not inside
this file: run as `MODIFF_WARMUP_STEPS=1 python docs/warmup1_speedup_2026-08-25/scripts/warmup_cost_w1.py`.

Writes docs/warmup1_speedup_2026-08-25/data/warmup_cost_w1.json -- a distinct path from the original
so the warmup=5 baseline stays intact for the diff.

Run: MODIFF_WARMUP_STEPS=1 python docs/warmup1_speedup_2026-08-25/scripts/warmup_cost_w1.py
"""
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(ROOT)
sys.path[:0] = [os.path.join(ROOT, "docs/attn_modiff_2026-08-13/scripts"),
                os.path.join(ROOT, "docs/qdiff_bridge_2026-08-12/scripts"),
                ROOT, os.path.join(ROOT, "src/taming-transformers"),
                os.path.join(ROOT, "integration/benchmarks/report"),
                os.path.join(ROOT, "docs/modiff_correctness_2026-08-03/scripts")]
os.environ["MODIFF_LINEAR"] = "0"
os.environ["MODIFF_DELTA_MODE"] = "static"

import torch                                                              # noqa: E402
import dynamic_delta_ab as H                                             # noqa: E402

D = "docs/warmup1_speedup_2026-08-25"
MODES = [("fp16", "fp16"), ("int8_baseline", "W8A8 PTQ"), ("int8", "W8A8 MoDiff"),
         ("int4_baseline", "W4A4 PTQ"), ("int4", "W4A4 MoDiff")]
STEPS = int(os.environ.get("WARMUP_STEPS_N", "200"))
BATCH = int(os.environ.get("WARMUP_BATCH", "128"))
WARMUP_STEPS_SETTING = int(os.environ.get("MODIFF_WARMUP_STEPS", "5"))


def timed_steps(mode):
    """Per-UNet-forward ms for one sample, after a discarded sample."""
    import integration.benchmarks.benchmark_ldm as B
    H.STEPS, H.BATCH = STEPS, BATCH
    H.AUTO_DELTA_TABLE = True
    r, m, s = H.build(mode, B._default_calibration_path(mode), "static")

    unet = m.model.diffusion_model
    orig = unet.forward
    marks = []

    def timed(*a, **k):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        out = orig(*a, **k)
        e1.record()
        marks.append((e0, e1))
        return out

    unet.forward = timed
    H.SEED = 1234
    H.latent(r, m, s)                      # discarded: autotune + attention scale freeze settle here
    marks.clear()
    H.latent(r, m, s)                      # the measured sample
    torch.cuda.synchronize()
    ms = [e0.elapsed_time(e1) for e0, e1 in marks]
    unet.forward = orig
    del r, m, s
    torch.cuda.empty_cache()
    return ms


def main():
    out = {"steps": STEPS, "batch": BATCH, "gpu": torch.cuda.get_device_name(0),
           "modiff_warmup_steps": WARMUP_STEPS_SETTING, "modes": {}}
    print(f"MODIFF_WARMUP_STEPS={WARMUP_STEPS_SETTING}")
    print(f"{out['gpu']}, batch {BATCH}, {STEPS} steps, one discarded sample first\n")
    print(f"{'mode':14}{'step0':>9}{'step1':>9}{'steady med':>12}{'excess':>9}"
          f"{'excess/sample':>15}{'% of sample':>13}")
    for key, label in MODES:
        ms = timed_steps(key)
        if len(ms) < 5:
            print(f"{label:14}  only {len(ms)} forwards recorded -- skipped")
            continue
        steady = statistics.median(ms[1:])
        excess = ms[0] - steady
        total = sum(ms)
        out["modes"][key] = {"label": label, "n_forwards": len(ms), "step0_ms": ms[0],
                             "step1_ms": ms[1], "steady_median_ms": steady,
                             "excess_ms": excess, "total_ms": total,
                             "excess_pct_of_sample": 100.0 * excess / total}
        print(f"{label:14}{ms[0]:9.1f}{ms[1]:9.1f}{steady:12.2f}{excess:9.1f}"
              f"{excess:14.1f}ms{100 * excess / total:12.2f}%")

    #: the PTQ arms have no MoDiff warm-up, so their excess is the floor every arm pays
    floor = [v["excess_ms"] for k, v in out["modes"].items()
             if k in ("fp16", "int8_baseline", "int4_baseline")]
    base = statistics.median(floor) if floor else 0.0
    out["first_step_floor_ms"] = base
    print(f"\nfirst-step floor (median of fp16 / W8A8 PTQ / W4A4 PTQ excess): {base:.1f} ms")
    print(f"{'mode':14}{'excess':>9}{'minus floor':>13}{'= MoDiff warm-up':>19}{'% of sample':>13}")
    for k, v in out["modes"].items():
        attributable = v["excess_ms"] - base
        v["modiff_warmup_ms"] = attributable
        v["modiff_warmup_pct"] = 100.0 * attributable / v["total_ms"]
        tag = "" if k in ("int8", "int4") else "   (no MoDiff warm-up by construction)"
        print(f"{v['label']:14}{v['excess_ms']:9.1f}{base:13.1f}{attributable:19.1f}"
              f"{100 * attributable / v['total_ms']:12.2f}%{tag}")

    os.makedirs(f"{D}/data", exist_ok=True)
    out_path = f"{D}/data/warmup_cost_w{WARMUP_STEPS_SETTING}.json"
    json.dump(out, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")
    print(f"\nNOTE 1: the share scales as 1/steps -- it is a per-COLD-SAMPLE cost. At {STEPS} steps the "
          f"numbers above\napply; at 50 steps multiply the percentages by {STEPS / 50:.0f}.")
    print("NOTE 2: REPORT.md's ms/step does NOT include this. e2e_three_mode_bench never resets MoDiff\n"
          "state, so only its discarded warm-up samples pay the warm-up (measured call counts: 70, 0, 0)\n"
          "while harnesses that reset pay it every sample (70, 70, 70).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

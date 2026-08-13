"""How much of the pipeline is WARM-UP? Per-step wall time, so t=T is separated from steady state.

THERE ARE THREE DIFFERENT "WARM-UPS" IN THIS PIPELINE and conflating them gives a wrong answer, so this
file measures the one that costs per SAMPLE and names the other two:

  1. MoDiff's t=T warm-up (what this measures). At the first timestep every modulated conv runs
     _forward_first_step: one activation quantize on the calibrated grid plus warmup_steps-1 = 4
     residual rounds, each with its own conv, so 5 convs where a steady step runs 1 delta conv (paper
     Appendix D.5). It is paid ONCE PER SAMPLE, so its share of a run is inversely proportional to the
     step count -- at 200 steps it is diluted 4x versus 50.
  2. The BENCH's warm-up samples (--warmups 2), whole discarded samples. Measurement hygiene, not
     pipeline cost, and excluded from every published ms/step. Not measured here.
  3. Build-time attention self-calibration: a 5-step sample that freezes 42 linear activation scales.
     Once per PROCESS, not per sample. Not measured here.

METHOD. One discarded sample first, so CUTLASS/cuDNN autotune and the attention scale freeze are already
settled -- otherwise step 0 would also carry _ensure_tuned_config's timing loop and the answer would be
"warm-up plus autotune" attributed entirely to warm-up. Then every UNet forward in one sample is timed
individually with CUDA events. The DDIM sampler calls the UNet once per step (unconditional, no CFG), so
one forward == one step.

WHAT TO EXPECT, and it is the control that makes this readable: the PTQ arms have NO MoDiff warm-up at
all, so their step 0 should sit at the steady-state median. Whatever excess they do show is the
first-step-of-anything overhead (allocator, cache warmth) and has to be subtracted from the MoDiff arms'
excess before calling it warm-up.

WHO ACTUALLY PAYS IT -- and the answer is NOT "every sample", which is why REPORT.md's ms/step and the
number below are both right and are not the same number. _forward_first_step runs only when
`a_hat_cache is None or its shape != x.shape`, so it is paid per COLD sample, and whether a sample is
cold depends on the caller. Measured by counting the calls:

    e2e_three_mode_bench's pattern (sampler.sample() repeatedly, NO reset)   70, 0, 0
    the quality harnesses' pattern (dynamic_delta_ab.latent, WHICH RESETS)   70, 70, 70

So REPORT.md's 58.50 / 73.19 ms/step CONTAIN NO WARM-UP: the bench pays it once inside its discarded
warm-up samples and its timed repeats reuse the state. Every harness that resets -- which is all the
quality ones, because a leftover cache does not degrade gracefully (dynamic_delta_ab's own comment
records an ALL-NaN latent after an unreset run) -- pays it on every sample.

Read the numbers below as the cost of a COLD sample, which is what correct per-sample usage costs, and
read REPORT.md's ms/step as steady state with the warm-up excluded.

Run: python docs/bench_report_2026-08-13_postzp/scripts/warmup_cost.py    # ~6 min, needs an idle GPU
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
import dynamic_delta_ab as H                                              # noqa: E402

D = "docs/bench_report_2026-08-13_postzp"
MODES = [("fp16", "fp16"), ("int8_baseline", "W8A8 PTQ"), ("int8", "W8A8 MoDiff"),
         ("int4_baseline", "W4A4 PTQ"), ("int4", "W4A4 MoDiff")]
STEPS = int(os.environ.get("WARMUP_STEPS_N", "200"))
BATCH = int(os.environ.get("WARMUP_BATCH", "128"))


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
    out = {"steps": STEPS, "batch": BATCH, "gpu": torch.cuda.get_device_name(0), "modes": {}}
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
    json.dump(out, open(f"{D}/data/warmup_cost.json", "w"), indent=1)
    print(f"\nwrote {D}/data/warmup_cost.json")
    print(f"\nNOTE 1: the share scales as 1/steps -- it is a per-COLD-SAMPLE cost. At {STEPS} steps the "
          f"numbers above\napply; at 50 steps multiply the percentages by {STEPS / 50:.0f}.")
    print("NOTE 2: REPORT.md's ms/step does NOT include this. e2e_three_mode_bench never resets MoDiff\n"
          "state, so only its discarded warm-up samples pay the warm-up (measured call counts: 70, 0, 0)\n"
          "while harnesses that reset pay it every sample (70, 70, 70).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

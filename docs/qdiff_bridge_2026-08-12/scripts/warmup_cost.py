"""Stage C: what does warm-up actually cost, and is it inside the reported ms/step?

The advisor asked for this directly ("Warmup 花了多少时间"), and suspected a kernel relaunch
("你可能要换kernel重新launch"). There are THREE separate mechanisms and they answer differently.

  (1) ATTENTION self-calibration -- MODIFF_ATTN_CALIB_STEPS, default 8 forwards.
      EXCLUDED from every reported ms/step: each harness discards a full run before measuring
      (dynamic_delta_ab.py:107, differential_timing.py:286). Nothing resets the freeze flags, so it
      is once per process, not once per sample. Reported here, not re-measured.

  (2) CONV MoDiff warm-up rounds -- MODIFF_WARMUP_STEPS, default 5, i.e. 4 EXTRA
      quantize+conv passes over all 70 convs, on t=T only. INCLUDED in every reported ms/step and
      never broken out. Measured here by sweeping the knob.

  (3) POST-FREEZE one-shot -- on the first forward after the scales freeze, _ensure_route1,
      _qkv_inv_out_scale and _ensure_fused fold weights and build scale vectors, and the route
      selection changes so a DIFFERENT set of CUDA kernels runs from forward 9 onward. This is the
      advisor's suspicion. Measured here per-forward.

Run: python docs/qdiff_bridge_2026-08-12/scripts/warmup_cost.py [--steps 50] [--batch 8]
"""
import argparse
import json
import os
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers"),
                os.path.join(ROOT, "integration/benchmarks/report"),
                os.path.join(ROOT, "docs/modiff_correctness_2026-08-03/scripts")]

import torch                                                                # noqa: E402
import dynamic_delta_ab as H                                               # noqa: E402

OUT = "docs/qdiff_bridge_2026-08-12/data/warmup_cost.json"


def sweep_conv_warmup(steps, batch, pairs):
    """(2): what the 4 extra t=T rounds cost -- measured on FORWARD 1, where they are actually paid.

    Two earlier attempts failed, both for the same reason: the effect is 4 extra quantize+conv passes
    over 70 convs on ONE step in 50, so amortized into ms/step it is ~1% of the number and the
    measurement noise is larger than that.

      attempt 1, rebuild per setting : 27.6 / 19.3 / 27.9 for W=1/3/5 -- non-monotonic, the middle
                                       setting 8 ms FASTER than both ends.
      attempt 2, paired on one model : deltas +0.74 / -5.60 / +2.68 / +2.28, stdev 3.8 on a median
                                       of 1.5. Still noise.

    The cause is not contention -- the GPU was idle between runs. It is CLOCK RAMP: this A40 idles
    at 210 MHz, and 50-step batch-8 runs are short enough to bounce between clock states.

    So measure it where it is big. The warm-up rounds are paid ENTIRELY on the t=T forward, so time
    that forward directly on an already-warm model and alternate the setting. The per-step figure is
    then a division, not a measurement. warmup_steps is read in __init__ (:113) so the env var cannot
    move a live model, but the attribute can.
    """
    from integration.kernels.int8_optimized import OptimizedInt8Conv2d
    H.STEPS, H.BATCH, H.SEED = steps, batch, 1234
    r, m, s = H.build("int8", H.CALIB["int8"], "dynamic")
    unet = m.model.diffusion_model
    live = [c for c in unet.modules() if isinstance(c, OptimizedInt8Conv2d) and c.is_calibrated]
    print(f"  {len(live)} calibrated convs (of "
          f"{sum(1 for c in unet.modules() if isinstance(c, OptimizedInt8Conv2d))} modules; the rest "
          f"are the known shadow copies under FusedResBlock.original)")

    times, orig = [], unet.forward

    def timed(*aa, **kk):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = orig(*aa, **kk)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
        return out
    unet.forward = timed

    for _ in range(2):                       # attention self-calibration AND clock ramp
        H.latent(r, m, s)

    def run(w):
        for c in live:
            c.warmup_steps = w
        times.clear()
        H.SEED = 1234
        H.latent(r, m, s)
        return times[0], statistics.median(times[1:])

    acc = {1: [], 5: []}
    for i in range(pairs):
        f1_1, rest1 = run(1)
        f1_5, rest5 = run(5)
        acc[1].append(f1_1)
        acc[5].append(f1_5)
        print(f"    pair {i + 1}: forward1 W=1 {f1_1:7.2f}  W=5 {f1_5:7.2f}  "
              f"delta {f1_5 - f1_1:+7.2f}   (t<T median {rest1:.2f} / {rest5:.2f})", flush=True)
    unet.forward = orig
    del r, m, s
    torch.cuda.empty_cache()

    deltas = [b - a for a, b in zip(acc[1], acc[5])]
    med = statistics.median(deltas)
    return {"measured_on": "forward 1 (t=T), where the warm-up rounds are paid",
            "pairs": pairs, "forward1_w1": acc[1], "forward1_w5": acc[5], "deltas": deltas,
            "delta_median_ms_on_forward1": med,
            "delta_stdev": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
            "amortized_ms_per_step": med / steps, "steps": steps}


def per_forward(steps, batch, n_show=14):
    """(3): time each UNet forward on a FRESH model, so forwards 1..8 are the calibrating ones.

    Wraps the UNet's own forward rather than sampling repeatedly: the freeze happens at a forward
    count, not a sample count, and nothing ever resets it.
    """
    H.STEPS, H.BATCH, H.SEED = steps, batch, 1234
    r, m, s = H.build("int8", H.CALIB["int8"], "dynamic")
    unet = m.model.diffusion_model
    times, orig = [], unet.forward

    def timed(*a, **k):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = orig(*a, **k)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
        return out
    unet.forward = timed
    H.latent(r, m, s)
    unet.forward = orig
    del r, m, s
    torch.cuda.empty_cache()

    calib = int(os.environ.get("MODIFF_ATTN_CALIB_STEPS", "8"))
    print(f"\n  per-forward ms on a fresh model (attention freezes after forward {calib}):")
    for i, t in enumerate(times[:n_show], 1):
        mark = "  <- calibrating" if i <= calib else ("  <- FIRST FROZEN" if i == calib + 1 else "")
        print(f"    forward {i:2d}  {t:7.2f}{mark}")
    steady = times[calib + 2:]
    return {"attn_calib_steps": calib, "per_forward_ms": times,
            "calibrating_mean": statistics.mean(times[:calib]),
            "first_frozen": times[calib] if len(times) > calib else None,
            "steady_median": statistics.median(steady) if steady else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--pairs", type=int, default=4)
    a = ap.parse_args()

    print("(2) conv MoDiff warm-up rounds -- INCLUDED in ms/step")
    sweep = sweep_conv_warmup(a.steps, a.batch, a.pairs)
    dm, ds = sweep["delta_median_ms_on_forward1"], sweep["delta_stdev"]
    print(f"  4 extra rounds cost {dm:+.2f} ms ON FORWARD 1 (stdev {ds:.2f}), i.e. "
          f"{sweep['amortized_ms_per_step']:+.3f} ms/step amortized over {a.steps} steps")

    print("\n(3) post-freeze one-shot and route change")
    pf = per_forward(a.steps, a.batch)
    if pf["first_frozen"] is not None and pf["steady_median"]:
        print(f"\n  calibrating mean {pf['calibrating_mean']:.2f} ms | "
              f"first frozen forward {pf['first_frozen']:.2f} ms | "
              f"steady median {pf['steady_median']:.2f} ms")
        print(f"  one-shot cost at the freeze boundary: "
              f"{pf['first_frozen'] - pf['steady_median']:+.2f} ms, paid ONCE per process")
        # Decomposed on purpose: calibrating_mean INCLUDES forward 1, whose cost is cuDNN
        # autotuning and first-touch allocation, not the calibrating route. Reporting the lump
        # attributes ~2/3 of a one-time allocation cost to the route change.
        t, c, st = pf["per_forward_ms"], pf["attn_calib_steps"], pf["steady_median"]
        print(f"  decomposed: forward 1 {t[0] - st:+.2f} (cuDNN autotune + alloc), "
              f"forwards 2..{c} {statistics.mean(t[1:c]) - st:+.2f}/forward (calibrating route)")

    res = {"steps": a.steps, "batch": a.batch,
           "conv_warmup_sweep": sweep,
           "conv_warmup_cost_ms_per_step": sweep["amortized_ms_per_step"],
           "per_forward": pf,
           "attention_calibration": {
               "steps": pf["attn_calib_steps"],
               "included_in_reported_ms_per_step": False,
               "why": "every harness discards a full run before measuring; the freeze flags are "
                      "never reset, so it is once per process"}}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

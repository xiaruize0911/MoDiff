"""Can MoDiff's lower per-step error be spent as FEWER STEPS -- i.e. a real speedup vs baseline?

Three routes to "MoDiff faster than its quantized baseline" have been measured and closed:
  1. same steps, faster per step -- impossible (+2.2 ms/step floor from the a_hat/o_hat state tensors)
  2. W4A4 reaching W8A8-baseline quality -- no (92% of W4A4's error is 4-bit conv weights)
  3. W8A4 -- would fix quality but runs on the int8 datapath, so no GEMM speed advantage

This is the fourth and last route, and the only one where MoDiff's accuracy converts directly into
wall-clock: diffusion quality depends on the number of sampling steps, and MoDiff's per-step
quantization error is ~6x lower at W8A8. If MoDiff at N < 50 steps is as good as the baseline at 50,
then equal quality costs N * 76.23 ms against the baseline's 50 * 70.94 ms.

Measurement design, and the part that is easy to get wrong: relL2 is normally computed against an
fp16 reference at the SAME step count, which is useless here because changing the step count moves
the reference. So everything is measured against ONE fixed target -- the fp16 latent at 200 steps --
and each configuration's distance to it therefore contains BOTH its discretization error (grows as
steps drop) and its quantization error (constant per step). That is exactly the trade being tested:
baseline at 50 steps has low discretization error and high quantization error; MoDiff at fewer steps
has the reverse.
"""

import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers"),
                os.path.join(ROOT, "integration/benchmarks/report"),
                os.path.dirname(os.path.abspath(__file__))]

import time

import torch

import integration.benchmarks.benchmark_ldm as B
import kernel_suites_bench as ks

BATCH = int(os.environ.get("SQ_BATCH", "8"))
SEED = 1234
REF_STEPS = int(os.environ.get("SQ_REF_STEPS", "200"))
BASE_STEPS = int(os.environ.get("SQ_BASE_STEPS", "50"))
MODIFF_STEPS = [int(v) for v in os.environ.get("SQ_STEPS", "50,40,32,25,20,16").split(",")]
CALIB = {"int8": "integration/calibration/int8_calibration_realckpt.pt",
         "int4": "integration/calibration/int4_calibration_realckpt.pt"}
# ms/step at production batch 128, from e2e_wallclock (reporting OFF, the valid configuration).
MS = {"int8_baseline": 70.938, "int8": 76.230, "int4_baseline": 57.530, "int4": 62.967}


def build(mode, calib, delta_mode="dynamic"):
    ks.set_env(mode)
    os.environ["MODIFF_DELTA_MODE"] = delta_mode
    os.environ["MODIFF_DELTA_REPORT"] = "0"      # diverges at W4A4; see FINDINGS
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    runner = B.BenchmarkRunner(
        config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
        ckpt_path="models/ldm/lsun_churches256/model.ckpt",
        output_dir="docs/modiff_correctness_2026-08-03/tmp_out",
        batch_size=BATCH, steps=BASE_STEPS, shape=(4, 32, 32), calibration_path=calib)
    model, sampler = runner._setup_model(mode)
    return runner, model, sampler


def sample(runner, model, sampler, steps):
    from integration.kernels.int4_optimized import reset_modiff_state as r4
    from integration.kernels.int8_optimized import reset_modiff_state as r8
    for r in (r8, r4):
        try:
            r(model.model.diffusion_model)
        except Exception:
            pass
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    cond = runner._cond_kwargs(model, BATCH)
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        out = sampler.sample(S=steps, batch_size=BATCH, shape=runner.shape, eta=0.0,
                             verbose=False, **cond)
    lat = out[0] if isinstance(out, (tuple, list)) else out
    return lat.detach().float().cpu()


def run(mode, calib, steps_list, ref):
    """Returns {steps: distance-to-ref}. One warm-up sample per model (attention self-calibrates)."""
    r, m, s = build(mode, calib)
    sample(r, m, s, BASE_STEPS)                 # warm-up, discarded
    out = {}
    for st in steps_list:
        lat = sample(r, m, s, st)
        out[st] = float((lat - ref).norm() / ref.norm())
    del m, s, r
    torch.cuda.empty_cache()
    return out


def main():
    # Fixed target: fp16 at REF_STEPS. Every distance below is measured against this one tensor.
    r, m, s = build("fp16", None)
    sample(r, m, s, BASE_STEPS)
    ref = sample(r, m, s, REF_STEPS)
    del m, s, r
    torch.cuda.empty_cache()
    print(f"target = fp16 @ {REF_STEPS} steps, |x|max {float(ref.abs().max()):.4f}\n", flush=True)

    results = {}
    for bits in ("int8", "int4"):
        # Sweep the BASELINE over the same step counts too. Without this the comparison is
        # unsound: distance-to-fp16@200 contains discretization error as well as quantization
        # error, and at low step counts discretization dominates for BOTH configurations. Only
        # the baseline curve tells us how much of MoDiff's low-step distance is its own merit.
        base_all = run(f"{bits}_baseline", CALIB[bits], sorted(set(MODIFF_STEPS + [BASE_STEPS])), ref)
        base = base_all[BASE_STEPS]
        print(f"  baseline curve: " + "  ".join(f"{k}:{v:.4f}" for k, v in sorted(base_all.items())),
              flush=True)
        base_ms = BASE_STEPS * MS[f"{bits}_baseline"]
        print(f"{'=' * 78}\n{bits}: BAR = baseline @ {BASE_STEPS} steps, dist {base:.4f}, "
              f"{base_ms:.0f} ms/sample (batch 128)\n{'=' * 78}", flush=True)
        mod = run(bits, CALIB[bits], MODIFF_STEPS, ref)
        results[bits] = {"bar_dist": base, "bar_ms": base_ms, "modiff": {},
                         "baseline_curve": base_all}
        best = None
        for st in MODIFF_STEPS:
            d = mod[st]
            ms = st * MS[bits]
            speed = base_ms / ms
            hit = d <= base
            results[bits]["modiff"][st] = {"dist": d, "ms": ms, "speedup_vs_bar": speed,
                                           "matches_bar": hit}
            bd = base_all.get(st)
            print(f"  MoDiff @ {st:3d} steps   dist {d:.4f}   (baseline same steps "
                  f"{bd:.4f}{'  MoDiff better' if bd is not None and d < bd else '  baseline better'})"
                  f"   {ms:6.0f} ms/sample   {speed:.2f}x the bar's time"
                  f"   {'<= bar' if hit else ''}", flush=True)
            if hit and (best is None or st < best):
                best = st
        if best is not None:
            sp = base_ms / (best * MS[bits])
            print(f"\n  => MoDiff needs only {best} steps to match the baseline's {BASE_STEPS}-step "
                  f"quality: {sp:.2f}x SPEEDUP vs baseline at equal quality\n", flush=True)
            results[bits]["equal_quality_steps"] = best
            results[bits]["equal_quality_speedup"] = sp
        else:
            print(f"\n  => no step count in {MODIFF_STEPS} matches the bar; "
                  f"MoDiff @ {MODIFF_STEPS[0]} is {mod[MODIFF_STEPS[0]]:.4f} vs bar {base:.4f}\n",
                  flush=True)
            results[bits]["equal_quality_steps"] = None

    with open("docs/modiff_correctness_2026-08-03/data/steps_equal_quality.json", "w") as f:
        json.dump({"ref_steps": REF_STEPS, "base_steps": BASE_STEPS, "batch": BATCH,
                   "ms_per_step_batch128": MS, "results": results}, f, indent=2)
    print("wrote docs/modiff_correctness_2026-08-03/data/steps_equal_quality.json")


if __name__ == "__main__":
    main()

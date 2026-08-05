"""Static-Q for MoDiff: calibrate the per-step delta-scale table on the REAL checkpoint, then measure
whether the finer quantizer step actually buys latent fidelity.

Until now the table could only be evaluated by its own step-size ratio (12.5x median finer, i.e.
~155x lower squared error by Theorem 4.3) because the stub checkpoint made every latent identical.
With real weights the question becomes answerable: does a static, per-step delta quantizer beat
quantizing the delta on the activation grid?

  table OFF = the pre-2026-08-03 behaviour: delta quantized with static_input_scale, so the step is
              the ACTIVATION's step and Theorem 4.3 predicts no error reduction at all
  table ON  = static per-step delta scale, the paper's method

Both share one fp16 reference, one seed, one activation calibration, one process.
"""

import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers"),
                os.path.join(ROOT, "integration/benchmarks/report")]

import torch

import integration.benchmarks.benchmark_ldm as B
import kernel_suites_bench as ks
from integration.kernels.int8_optimized import (
    apply_int8_delta_scales, begin_delta_calibration_int8, delta_calibration_report_int8,
    end_delta_calibration_int8, export_int8_delta_scales, reset_modiff_state)

STEPS = int(os.environ.get("SQ_STEPS", "50"))
BATCH = int(os.environ.get("SQ_BATCH", "8"))
ROUNDS = int(os.environ.get("SQ_ROUNDS", "1"))   # single-shot: one pass is exact
SEED = 1234
ACT_CALIB = "integration/calibration/int8_calibration_realckpt.pt"
TABLE = "integration/calibration/int8_delta_calibration.pt"


def build(mode, calib):
    ks.set_env(mode)
    # The delta-table calibration observes max|q| emitted under a KNOWN scale and recovers the
    # delta's range as max|q|/scale_used. That inversion only holds on the static path; under
    # MODIFF_DELTA_MODE=dynamic the kernel picks its own scale per call (and drives max|q| to
    # exactly Q by construction), so the observation would carry no information about the range.
    # Pin static for the calibration regardless of the shipped default.
    os.environ["MODIFF_DELTA_MODE"] = "static"
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    runner = B.BenchmarkRunner(
        config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
        ckpt_path="models/ldm/lsun_churches256/model.ckpt",
        output_dir="docs/modiff_correctness_2026-08-03/tmp_out",
        batch_size=BATCH, steps=STEPS, shape=(4, 32, 32), calibration_path=calib)
    model, sampler = runner._setup_model(mode)
    return runner, model, sampler


def latent(runner, model, sampler, warm=False):
    """`warm=True` discards one sampling run first. Attention self-calibrates over its first
    forwards, so a single run is not steady state (FINDINGS 2026-08-04: 0.2107 then 0.0399)."""
    if warm:
        latent(runner, model, sampler)
    reset_modiff_state(model.model.diffusion_model)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    cond = runner._cond_kwargs(model, BATCH)
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        out = sampler.sample(S=STEPS, batch_size=BATCH, shape=runner.shape, eta=0.0,
                            verbose=False, **cond)
    lat = out[0] if isinstance(out, (tuple, list)) else out
    return lat.detach().float().cpu()


def main():
    # fp16 reference
    r, m, s = build("fp16", None)
    ref = latent(r, m, s, warm=True)
    del m, s, r; torch.cuda.empty_cache()
    print(f"fp16 reference |x|max {float(ref.abs().max()):.4f}\n", flush=True)

    # ---- calibrate the delta table on the real checkpoint ----
    r, m, s = build("int8", ACT_CALIB)
    print(f"calibrating the per-step delta table, {ROUNDS} pass(es) "
          f"(observed at act_scale/4, which provably cannot clip)", flush=True)
    for i in range(ROUNDS):
        begin_delta_calibration_int8(m, reset=(i == 0))
        latent(r, m, s)
        n = end_delta_calibration_int8(m)
        rep = delta_calibration_report_int8(m)
        g = sorted(x["step_gain_tail"] for x in rep if x.get("step_gain_tail"))
        clip = [x["obs_clipped_frac"] for x in rep if x.get("obs_clipped_frac") is not None]
        print(f"  round {i}: {n} tables set; step gain median {g[len(g)//2]:.1f}x "
              f"max {g[-1]:.1f}x; still clipping {sum(1 for c in clip if c > 0)}/{len(clip)}",
              flush=True)
    table = export_int8_delta_scales(m)
    torch.save(table, TABLE)
    rep = delta_calibration_report_int8(m)
    gains = sorted(x["step_gain_tail"] for x in rep if x.get("step_gain_tail"))
    print(f"  wrote {TABLE}: {len(table)} layers, median step gain "
          f"{gains[len(gains)//2]:.1f}x\n", flush=True)
    del m, s, r; torch.cuda.empty_cache()

    # ---- A/B: table OFF vs ON, identical everything else ----
    out = {}
    for tag, use in (("delta table OFF (activation grid)", False),
                     ("delta table ON  (static per-step delta grid)", True)):
        r, m, s = build("int8", ACT_CALIB)
        n = apply_int8_delta_scales(m, torch.load(TABLE, weights_only=True)) if use else 0
        lat = latent(r, m, s, warm=True)
        rel = float((lat - ref).norm() / ref.norm())
        out[tag] = {"rel_l2_vs_fp16": rel, "latent_absmax": float(lat.abs().max()),
                    "table_layers": n}
        print(f"  {tag:44s} relL2 vs fp16 = {rel:.4f}"
              f"{f'   (table on {n} layers)' if use else ''}", flush=True)
        del m, s, r; torch.cuda.empty_cache()

    a = out["delta table OFF (activation grid)"]["rel_l2_vs_fp16"]
    b = out["delta table ON  (static per-step delta grid)"]["rel_l2_vs_fp16"]
    print(f"\n{'=' * 74}")
    print(f"  static delta-Q effect: {a:.4f} -> {b:.4f}   "
          f"({(a - b) / a * 100:+.1f}% error, {a / b:.3f}x better)")
    print(f"  (int8_baseline for context: 0.2717; MoDiff with table OFF should ~match it)")
    with open("docs/modiff_correctness_2026-08-03/data/static_delta_q.json", "w") as f:
        json.dump({"steps": STEPS, "batch": BATCH, "rounds": ROUNDS,
                   "fp16_absmax": float(ref.abs().max()), "ab": out,
                   "median_step_gain": gains[len(gains) // 2]}, f, indent=2)


if __name__ == "__main__":
    main()

"""Does a non-clipping (dynamic) delta quantizer beat a static one? W8A8 and W4A4.

Motivation, measured 2026-08-04 on the real LSUN-churches checkpoint: at the production static
setting the MoDiff delta quantizer CLIPS on 49 of 70 conv layers -- median max|q| lands exactly
on the 127 ceiling. Clipping is unrecoverable error, and MoDiff's feedback term then carries it
forward. It also explains the otherwise-odd static_delta_q.py result, where making the delta grid
*finer* made latents worse (0.1778 -> 0.2135): a finer grid clips more.

The paper's Theorem 4.3 bound assumes a dynamic quantizer explicitly ("to avoid clipping error"),
so this measures the faithful implementation against the two static ones.

Modes compared, all sharing one fp16 reference, one seed, one activation calibration, one process:

  <bits>_baseline           MoDiff off -- the reference point for "did MoDiff help at all"
  <bits> static, table off  delta quantized on the ACTIVATION grid (pre-2026-08-03 behaviour)
  <bits> static, table on   per-step static delta table (int8 only; int4 never had a table)
  <bits> dynamic            Q/max|delta| per call, cannot clip (delta_absmax_fp16 +
                            gn_delta_absmax_flat_kernel)

Reports latent relative L2 vs fp16. Also times each mode so the extra reduction pass dynamic
mode costs is visible rather than assumed.
"""

import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers"),
                os.path.join(ROOT, "integration/benchmarks/report")]

import torch

STEPS = int(os.environ.get("AB_STEPS", "50"))
BATCH = int(os.environ.get("AB_BATCH", "8"))
SEED = 1234
#: Left pointing at the absmax file ON PURPOSE: every number this report has committed was measured
#: against it, and silently switching would change what those numbers mean without re-measuring them.
#: Set AB_CALIB8=integration/calibration/int8_calibration_qdiff.pt to score the Q-Diffusion scales
#: (2026-08-12; benchmark_ldm.py now prefers that file for production runs).
CALIB = {"int8": os.environ.get("AB_CALIB8",
                                "integration/calibration/int8_calibration_realckpt.pt"),
         "int4": os.environ.get("AB_CALIB4",
                                "integration/calibration/int4_calibration_realckpt.pt")}
TABLE = "integration/calibration/int8_delta_calibration.pt"


def build(mode, calib, delta_mode):
    """Construct a fresh model. MODIFF_DELTA_MODE is read in the conv wrappers' __init__, so it
    must be set before _setup_model, not before sampling."""
    import integration.benchmarks.benchmark_ldm as B
    import kernel_suites_bench as ks
    ks.set_env(mode)
    os.environ["MODIFF_DELTA_MODE"] = delta_mode
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    runner = B.BenchmarkRunner(
        config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
        ckpt_path="models/ldm/lsun_churches256/model.ckpt",
        output_dir="docs/modiff_correctness_2026-08-03/tmp_out",
        batch_size=BATCH, steps=STEPS, shape=(4, 32, 32), calibration_path=calib)
    model, sampler = runner._setup_model(mode)
    return runner, model, sampler


def latent(runner, model, sampler):
    # Reset EVERY MoDiff-bearing family, not just the convs. This used to reset r8/r4 only, which
    # was sufficient while the conv path was the only one carrying temporal state. It stopped being
    # sufficient when MODIFF_LINEAR defaulted to 1 (2026-08-06): the 42 attention qkv/proj then hold
    # an a_hat/o_hat cache too, and a leftover cache does not degrade gracefully -- measured, a run
    # after an unreset run returns an ALL-NaN latent. Every protocol built on this harness discards
    # run 1 as warm-up and measures run 2, so a partial reset here corrupts exactly the run that
    # gets recorded. Each entry is a no-op when its family is absent or not in MoDiff mode --
    # which is why this stays comprehensive even though the default flipped BACK to 0 on
    # 2026-08-12: any run that sets MODIFF_LINEAR=1 explicitly still needs the wxax reset,
    # and paying for a no-op is free.
    from integration.kernels.int4_optimized import reset_modiff_state as r4
    from integration.kernels.int8_optimized import reset_modiff_state as r8
    resets = [r8, r4]
    for mod, name in (("integration.kernels.int8_linear", "reset_modiff_state_linear"),
                      ("integration.kernels.wxax_linear", "reset_wxax_modiff"),
                      ("integration.kernels.modiff_attention", "reset_attention_modiff")):
        try:
            resets.append(getattr(__import__(mod, fromlist=[name]), name))
        except Exception:
            pass
    for r in resets:
        try:
            r(model.model.diffusion_model)
        except Exception:
            pass
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    cond = runner._cond_kwargs(model, BATCH)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        out = sampler.sample(S=STEPS, batch_size=BATCH, shape=runner.shape, eta=0.0,
                             verbose=False, **cond)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000.0 / STEPS
    lat = out[0] if isinstance(out, (tuple, list)) else out
    return lat.detach().float().cpu(), ms


def measure(mode, calib, delta_mode, table, ref, label, out):
    """One sampling run is NOT steady state. Measured 2026-08-04: for int8 dynamic, the first run
    after model construction gives relL2 0.2107 and the second gives 0.0399 -- a 5.3x difference.
    The quantized attention blocks self-calibrate their static scales over the first
    MODIFF_ATTN_CALIB_STEPS forwards, so run 1 is measuring a model that is still calibrating.
    Every number here is therefore run 2, with run 1 discarded as warm-up."""
    r, m, s = build(mode, calib, delta_mode)
    n = 0
    if table:
        from integration.kernels.int8_optimized import apply_int8_delta_scales
        n = apply_int8_delta_scales(m, torch.load(table, weights_only=True))
    latent(r, m, s)                 # warm-up: lets attention finish self-calibrating
    lat, ms = latent(r, m, s)
    rel = float((lat - ref).norm() / ref.norm())
    out[label] = {"rel_l2_vs_fp16": rel, "ms_per_step": ms, "latent_absmax": float(lat.abs().max()),
                  "delta_mode": delta_mode, "table_layers": n}
    print(f"  {label:34s} relL2 {rel:.4f}   {ms:7.2f} ms/step"
          + (f"   (table on {n} layers)" if n else ""), flush=True)
    del m, s, r
    torch.cuda.empty_cache()
    return rel


def main():
    out = {}
    r, m, s = build("fp16", None, "static")
    latent(r, m, s)                 # same warm-up discipline for the reference
    ref, ms = latent(r, m, s)
    out["fp16"] = {"rel_l2_vs_fp16": 0.0, "ms_per_step": ms}
    print(f"fp16 reference: |x|max {float(ref.abs().max()):.4f}, {ms:.2f} ms/step\n", flush=True)
    del m, s, r
    torch.cuda.empty_cache()

    print(f"{'=' * 78}\nW8A8\n{'=' * 78}")
    measure("int8_baseline", CALIB["int8"], "static", None, ref, "int8_baseline (MoDiff off)", out)
    a = measure("int8", CALIB["int8"], "static", None, ref, "int8 MoDiff static, table off", out)
    if os.path.exists(TABLE):
        measure("int8", CALIB["int8"], "static", TABLE, ref, "int8 MoDiff static, table on", out)
    b = measure("int8", CALIB["int8"], "dynamic", None, ref, "int8 MoDiff DYNAMIC", out)
    print(f"\n  dynamic vs static(table off): {a:.4f} -> {b:.4f}  "
          f"({(a - b) / a * 100:+.1f}% error, {a / b:.3f}x)")

    print(f"\n{'=' * 78}\nW4A4\n{'=' * 78}")
    measure("int4_baseline", CALIB["int4"], "static", None, ref, "int4_baseline (MoDiff off)", out)
    c = measure("int4", CALIB["int4"], "static", None, ref, "int4 MoDiff static (activation grid)", out)
    d = measure("int4", CALIB["int4"], "dynamic", None, ref, "int4 MoDiff DYNAMIC", out)
    print(f"\n  dynamic vs static: {c:.4f} -> {d:.4f}  "
          f"({(c - d) / c * 100:+.1f}% error, {c / d:.3f}x)")

    os.makedirs("docs/modiff_correctness_2026-08-03/data", exist_ok=True)
    with open("docs/modiff_correctness_2026-08-03/data/dynamic_delta_ab.json", "w") as f:
        json.dump({"steps": STEPS, "batch": BATCH, "seed": SEED, "results": out}, f, indent=2)
    print("\nwrote docs/modiff_correctness_2026-08-03/data/dynamic_delta_ab.json")


if __name__ == "__main__":
    main()

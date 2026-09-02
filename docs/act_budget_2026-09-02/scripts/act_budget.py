"""Error budget: which quantizer actually sets W8A8 latent relL2, and does conv-input
granularity move it once the other sources are removed?

WHY. docs/delta_dynamic_2026-09-02 measured a 3.5x improvement in the conv-input quantizer
(per-tensor static 0.1838 -> dynamic 0.0451, per-layer) buying ZERO end-to-end relL2
(0.0999 -> 0.1091, inside noise). The whole accuracy case for the blockwise conv mainloop is
stated in those same per-layer units (B=64 = 0.0102, "16x better"), so it inherits the doubt.
Two readings were left unseparated:

  1. the conv-input quantizer is not the dominant error term, and is masked by W8 weights
     and/or W8A8 attention;
  2. latent relL2 saturates and cannot resolve the difference at all.

This separates them by turning each source off independently.

At MODIFF_LINEAR=0 there are exactly three quantizers in the loop, confirmed from the
delta_dynamic log: the conv-input activation quantizer, the W8 per-output-channel conv
weights, and W8A8 static standard attention (21 blocks). Linears stay fp16.

KNOBS (all in integration/kernels/int8_optimized.py):
  MODIFF_ACT_BLOCK      -3 exact | -2 per-tensor static | -1 per-tensor dynamic | N blockwise
  MODIFF_ACT_SIM_EXACT_W=1  exact instead of W8 conv weights (must precede model build)
  MODIFF_ACT_SIM_QMAX=7     coarser activation grid -- the needle control
  MODIFF_STD_ATTN_BITS=0   leave attention fp16 (token-major math SDPA); unset = W8A8 STATIC

MODIFF_QUANT_ATTN=0 is NOT the knob for this and was tried first: with MODIFF_LINEAR=0 the
gate is `std_attn_bits in (4,8) and (not quant_lin or _force_qattn)`, and `not quant_lin` is
already True, so attention stays quantized -- the flag only flips it from STATIC to dynamic.
That would have measured a different attention arm under an "fp16 attention" label.

THE FLOOR ARM IS LOAD-BEARING. The sim runs convs in fp32 while the fp16 reference runs in
fp16, so "everything exact" is not 0 relL2 but the harness's own numerical floor. Every other
arm has to be read against that floor, not against zero.

THE NEEDLE ARM IS LOAD-BEARING TOO. A flat sweep only means "granularity does not matter" if
the same metric visibly responds to a coarser grid. If int4 activations are also flat, the
metric is saturated and this whole measurement approach is void.

Quality only -- the sim bypasses every fused kernel, so its time means nothing.
n=6, seed 20260805, latent relL2 vs the fp16 arm.

Run: source /workspace/MoDiff/setup_cuda_env.sh
     python docs/act_budget_2026-09-02/scripts/act_budget.py
"""
from __future__ import annotations

import gc
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers")]

# Held fixed across arms.
os.environ["MODIFF_LINEAR"] = "0"
os.environ["MODIFF_CACHE_SKIP_K"] = "1"
os.environ["MODIFF_REPLAY_K"] = "1"
os.environ["MODIFF_AHAT_BITS"] = "16"
os.environ["MODIFF_AHAT_REFRESH"] = "0"
os.environ["MODIFF_IMODE"] = "0"
os.environ["MODIFF_AHAT_BLOCK"] = "0"
os.environ["MODIFF_DELTA_MODE"] = "static"
# _sim_guard: every conv must reach forward(), so EVERY fusion must be off. There are five
# kill switches, not the two the guard's error message names -- the o_hat-residual fold
# (forward_modiff_fused_silu_residual) and the Upsample/AvgPool quantize folds
# (forward_to_int8 / forward_from_int8) have their own. All are read at fused_resblock IMPORT
# time, hence before the benchmark_ldm import below.
os.environ["MODIFF_DISABLE_GN_MODIFF_FUSION"] = "1"
os.environ["MODIFF_DISABLE_GN_INT8_FUSION"] = "1"
os.environ["MODIFF_DISABLE_O_HAT_RESIDUAL_FUSION"] = "1"
os.environ["MODIFF_DISABLE_UPSAMPLE_QUANTIZE_FUSION"] = "1"
os.environ["MODIFF_DISABLE_AVGPOOL_QUANTIZE_FUSION"] = "1"

from integration.utils.preflight import preflight, MODEL  # noqa: E402
preflight(*MODEL, what="act_budget.py")

import torch  # noqa: E402
import integration.benchmarks.benchmark_ldm as B  # noqa: E402

SHAPE = (4, 32, 32)
NQ, STEPS, SEED = 6, 50, 20260805
OUT_JSON = "docs/act_budget_2026-09-02/data/act_budget.json"

# (label, group, mode, ACT_BLOCK, EXACT_W, ATTN_BITS, QMAX)   ATTN_BITS 0 = fp16, 8 = W8A8 static
ARMS = [
    ("fp16 reference",                             "ref",   "fp16", None, 0, 0, 127),

    # --- the harness floor, and each non-activation source alone -------------------
    ("exact A, exact W, fp16 attn  (FLOOR)",       "floor", "int8",   -3, 1, 0, 127),
    ("exact A, W8,      fp16 attn",                "solo",  "int8",   -3, 0, 0, 127),
    ("exact A, exact W, W8A8 attn",                "solo",  "int8",   -3, 1, 8, 127),
    ("exact A, W8,      W8A8 attn",                "solo",  "int8",   -3, 0, 8, 127),

    # --- the sweep with every masking source REMOVED ------------------------------
    ("A per-tensor static, exact W, fp16 attn",    "sweep", "int8",   -2, 1, 0, 127),
    ("A per-tensor dyn,    exact W, fp16 attn",    "sweep", "int8",   -1, 1, 0, 127),
    ("A blockwise B=64,    exact W, fp16 attn",    "sweep", "int8",   64, 1, 0, 127),
    ("A blockwise B=16,    exact W, fp16 attn",    "sweep", "int8",   16, 1, 0, 127),

    # --- needle control: does relL2 respond to a coarser grid at all? -------------
    ("A static on INT4 grid, exact W, fp16 attn",  "needle", "int8",  -2, 1, 0, 7),

    # --- full stack, to connect back to the real path -----------------------------
    ("A per-tensor static, W8, W8A8 attn (shipped)", "full", "int8",  -2, 0, 8, 127),
    ("A blockwise B=64,    W8, W8A8 attn",           "full", "int8",  64, 0, 8, 127),
]


def sample(model, sampler, n, quantized):
    if quantized:
        B.reset_modiff_state_int8(model.model.diffusion_model)
    B._reset_wxax_modiff_safe(model)
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        out = sampler.sample(S=STEPS, batch_size=n, shape=SHAPE, eta=0.0, verbose=False)
    return out[0] if isinstance(out, (tuple, list)) else out


def main():
    print(f"GPU {torch.cuda.get_device_name(0)}  n={NQ} steps={STEPS}", flush=True)
    recs, ref = [], None

    for label, group, mode, ablk, exw, abits, qmax in ARMS:
        os.environ["MODIFF_ACT_BLOCK"] = "0" if ablk is None else str(ablk)
        os.environ["MODIFF_ACT_SIM_EXACT_W"] = "1" if exw else "0"
        os.environ["MODIFF_STD_ATTN_BITS"] = str(abits)
        os.environ.pop("MODIFF_QUANT_ATTN", None)   # default -> STATIC, as shipped
        os.environ["MODIFF_ACT_SIM_QMAX"] = str(qmax)
        quantized = mode != "fp16"
        print(f"\n===== [{group}] {label} =====", flush=True)

        runner = B.BenchmarkRunner(
            config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
            ckpt_path="models/ldm/lsun_churches256/model.ckpt",
            output_dir="docs/act_budget_2026-09-02/tmp",
            batch_size=NQ, steps=STEPS, shape=SHAPE,
            calibration_path=B._default_calibration_path(mode),
            auto_delta_table=True)
        model, sampler = runner._setup_model(mode)

        sample(model, sampler, NQ, quantized)          # warmup -> attention self-calibrates
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        lat = sample(model, sampler, NQ, quantized).detach().float().cpu()

        if ref is None:
            ref, rel = lat.clone(), 0.0
        else:
            rel = float((lat - ref).norm() / ref.norm())
        recs.append({"label": label, "group": group, "act_block": ablk, "exact_w": bool(exw),
                     "attn_bits": abits, "qmax": qmax, "relL2_vs_fp16": rel})
        print(f"  relL2 {rel:.4f}", flush=True)

        del model, sampler, runner
        gc.collect()
        torch.cuda.empty_cache()

    print("\n===== error budget (latent relL2 vs fp16) =====", flush=True)
    for g in ["ref", "floor", "solo", "sweep", "needle", "full"]:
        rows = [r for r in recs if r["group"] == g]
        if not rows:
            continue
        print(f"  -- {g} --", flush=True)
        for r in rows:
            print(f"     {r['label']:52s} {r['relL2_vs_fp16']:8.4f}", flush=True)

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump({"gpu": torch.cuda.get_device_name(0), "n_quality": NQ, "steps": STEPS,
               "seed": SEED, "arms": recs}, open(OUT_JSON, "w"), indent=1)
    print(f"\nwrote {OUT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

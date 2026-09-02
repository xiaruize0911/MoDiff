"""n=6 relL2 + overflow for I-MoDiff vs frozen-s* fp16 control vs W8A8 full.

Arms isolate freeze-the-table from integer math:
  w8a8_full   per-step δ, fp16 a_hat (ship)
  frozen_s    scale[0] only, fp16 a_hat, IMODE=0
  imode16/8/4 integer a_hat, frozen s*

Run: source setup_cuda_env.sh
     python docs/cache_schemes_report_2026-08-28/scripts/imode_quality.py
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers")]

os.environ.setdefault("MODIFF_DELTA_MODE", "static")
os.environ["MODIFF_LINEAR"] = "0"
os.environ["MODIFF_CACHE_SKIP_K"] = "1"
os.environ["MODIFF_REPLAY_K"] = "1"
os.environ["MODIFF_AHAT_BITS"] = "16"
os.environ["MODIFF_AHAT_REFRESH"] = "0"
os.environ["MODIFF_IMODE"] = "0"
os.environ["MODIFF_DELTA_FREEZE"] = "0"

from integration.utils.preflight import preflight, MODEL  # noqa: E402
preflight(*MODEL, what="imode_quality.py")

import torch  # noqa: E402
import integration.benchmarks.benchmark_ldm as B  # noqa: E402
from integration.kernels.int8_optimized import OptimizedInt8Conv2d  # noqa: E402

SHAPE = (4, 32, 32)
OUT = "docs/cache_schemes_report_2026-08-28/data/imode.json"


def _knobs(imode, bits, freeze):
    os.environ["MODIFF_IMODE"] = "1" if imode else "0"
    os.environ["MODIFF_AHAT_BITS"] = str(bits)
    os.environ["MODIFF_DELTA_FREEZE"] = "1" if freeze else "0"
    os.environ["MODIFF_REPLAY_K"] = "1"
    os.environ["MODIFF_CACHE_SKIP_K"] = "1"


def _overflow(model):
    fracs = []
    for m in model.modules():
        if isinstance(m, OptimizedInt8Conv2d) and m.a_hat_cache is not None:
            if m.a_hat_cache.dtype in (torch.int8, torch.int16):
                fracs.append(m.ahat_sat_frac())
    if not fracs:
        return {"n_int": 0, "max_sat": 0.0, "n_over": 0}
    return {"n_int": len(fracs), "max_sat": max(fracs),
            "n_over": sum(1 for f in fracs if f >= 1.0 - 1e-6)}


def _reset(model):
    B.reset_modiff_state_int8(model.model.diffusion_model)
    B._reset_wxax_modiff_safe(model)


def sample(runner, model, sampler, n, seed, steps):
    _reset(model)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cond = runner._cond_kwargs(model, n)
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        out = sampler.sample(S=steps, batch_size=n, shape=SHAPE, eta=0.0,
                             verbose=False, **cond)
        lat = out[0] if isinstance(out, (tuple, list)) else out
    return lat.float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20260805)
    a = ap.parse_args()

    print(f"GPU {torch.cuda.get_device_name(0)}  n={a.n} steps={a.steps}", flush=True)
    runner = B.BenchmarkRunner(
        config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
        ckpt_path="models/ldm/lsun_churches256/model.ckpt",
        output_dir="docs/cache_schemes_report_2026-08-28/tmp_imode",
        batch_size=a.n, steps=a.steps, shape=SHAPE,
        calibration_path=B._default_calibration_path("int8"),
        auto_delta_table=True)

    print("===== fp16 =====", flush=True)
    _knobs(False, 16, False)
    model, sampler = runner._setup_model("fp16")
    sample(runner, model, sampler, a.n, a.seed, a.steps)
    ref = sample(runner, model, sampler, a.n, a.seed, a.steps)
    del model, sampler
    torch.cuda.empty_cache()

    arms = [
        ("w8a8_full", False, 16, False),
        ("frozen_s", False, 16, True),
        ("imode16", True, 16, False),
        ("imode8", True, 8, False),
        ("imode4", True, 4, False),
    ]
    quality = {"fp16": {"relL2_vs_fp16": 0.0}}
    print("===== int8 I-MoDiff =====", flush=True)
    model, sampler = runner._setup_model("int8")
    for name, imode, bits, freeze in arms:
        _knobs(imode, bits, freeze)
        sample(runner, model, sampler, a.n, a.seed, a.steps)
        lat = sample(runner, model, sampler, a.n, a.seed, a.steps)
        rel = float((lat - ref).norm() / (ref.norm() + 1e-12))
        ov = _overflow(model)
        quality[name] = {"relL2_vs_fp16": rel, **ov}
        print(f"  {name:12s} relL2 {rel:.4f}  sat_max={ov['max_sat']:.3f}  "
              f"n_over={ov['n_over']}/{ov['n_int']}", flush=True)
    del model, sampler
    torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    payload = {"seed": a.seed, "steps": a.steps, "n": a.n, "relL2": quality}
    # merge into imode.json if speed/FID already present
    if os.path.exists(OUT):
        prev = json.load(open(OUT))
        prev.update(payload)
        payload = prev
    json.dump(payload, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}")
    _knobs(False, 16, False)
    return 0


if __name__ == "__main__":
    sys.exit(main())

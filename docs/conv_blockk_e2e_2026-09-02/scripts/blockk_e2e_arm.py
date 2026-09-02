"""One arm of the conv-blockwise E2E measurement. Run via blockk_e2e.sh, not directly.

Each arm is its own PROCESS because the five fusion kill switches are read at
integration/fused_ops/fused_resblock IMPORT time -- they cannot be toggled per arm in one
process, and a stale one would silently mix fusion levels across arms.

argv: <arm> <mode> <fusions:on|off> <blockk:0|32|64> <ctrl:0|1>

Timing: batch 128, 50 DDIM, CUDA events, median of 2 after 1 warmup.
Quality: n=6, seed 20260805, latent relL2 vs the fp16 arm (read from data/ref_fp16.pt).
Emits one JSON line prefixed ARMJSON: for the driver to collect.
"""
from __future__ import annotations
import json, os, statistics, sys

ROOT = "/workspace/MoDiff"
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers")]

arm, mode, fusions, blockk, ctrl = sys.argv[1:6]

BASE = {"MODIFF_LINEAR": "0", "MODIFF_CACHE_SKIP_K": "1", "MODIFF_REPLAY_K": "1",
        "MODIFF_AHAT_BITS": "16", "MODIFF_AHAT_REFRESH": "0", "MODIFF_IMODE": "0",
        "MODIFF_AHAT_BLOCK": "0", "MODIFF_DELTA_MODE": "static", "MODIFF_ACT_BLOCK": "0",
        "MODIFF_CONV_BLOCKK": blockk, "MODIFF_CONV_BLOCKK_CTRL": ctrl}
if fusions == "off":
    BASE.update({"MODIFF_DISABLE_GN_MODIFF_FUSION": "1", "MODIFF_DISABLE_GN_INT8_FUSION": "1",
                 "MODIFF_DISABLE_O_HAT_RESIDUAL_FUSION": "1",
                 "MODIFF_DISABLE_UPSAMPLE_QUANTIZE_FUSION": "1",
                 "MODIFF_DISABLE_AVGPOOL_QUANTIZE_FUSION": "1"})
elif fusions == "gn":
    # The FUSED blockwise path: keep the GN->conv fusion ON (that is where
    # blockk_gn_fused injects) and turn off only the three folds it cannot serve, so those
    # convs fall through to forward() instead of tripping the guard. Measured coverage:
    # 98.7% of conv calls take the fused path.
    BASE.update({"MODIFF_DISABLE_O_HAT_RESIDUAL_FUSION": "1",
                 "MODIFF_DISABLE_UPSAMPLE_QUANTIZE_FUSION": "1",
                 "MODIFF_DISABLE_AVGPOOL_QUANTIZE_FUSION": "1"})
os.environ.update(BASE)

from integration.utils.preflight import preflight, MODEL  # noqa: E402
preflight(*MODEL, what="blockk_e2e_arm.py")
import torch  # noqa: E402
import integration.benchmarks.benchmark_ldm as B  # noqa: E402

SHAPE, BATCH, STEPS, NQ, SEED = (4, 32, 32), 128, 50, 6, 20260805
REF = "docs/conv_blockk_e2e_2026-09-02/data/ref_fp16.pt"


def sample(model, sampler, n, quantized):
    if quantized:
        B.reset_modiff_state_int8(model.model.diffusion_model)
    B._reset_wxax_modiff_safe(model)
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        out = sampler.sample(S=STEPS, batch_size=n, shape=SHAPE, eta=0.0, verbose=False)
    return out[0] if isinstance(out, (tuple, list)) else out


runner = B.BenchmarkRunner(
    config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
    ckpt_path="models/ldm/lsun_churches256/model.ckpt",
    output_dir="docs/conv_blockk_e2e_2026-09-02/tmp",
    batch_size=BATCH, steps=STEPS, shape=SHAPE,
    calibration_path=B._default_calibration_path(mode), auto_delta_table=True)
model, sampler = runner._setup_model(mode)
quantized = mode != "fp16"

sample(model, sampler, BATCH, quantized)                 # warmup -> steady state
torch.cuda.synchronize()
xs = []
for _ in range(2):
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record(); sample(model, sampler, BATCH, quantized); e.record()
    torch.cuda.synchronize(); xs.append(s.elapsed_time(e) / STEPS)
ms = statistics.median(xs)

torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
lat = sample(model, sampler, NQ, quantized).detach().float().cpu()
if mode == "fp16":
    torch.save(lat, REF)
    rel = 0.0
else:
    ref = torch.load(REF)
    rel = float((lat - ref).norm() / ref.norm())

print("ARMJSON:" + json.dumps({"arm": arm, "mode": mode, "fusions": fusions,
                               "blockk": int(blockk), "ctrl": ctrl == "1",
                               "ms_step": ms, "trials": xs, "relL2_vs_fp16": rel}), flush=True)

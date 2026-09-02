"""One arm of the W/A precision x granularity error budget. Run via wa_budget.sh.

QUESTION. Should the WEIGHT scale be blockwise along C (the reduction axis), as DeepSeek-V3 does
at 128x128, instead of the per-output-channel scale we ship? Per-output-channel is the FREE axis
-- it factors out of the reduction and is applied once in the epilogue. Blocking along C does not
factor out, so it permanently moves the weight scale into the mainloop flush. It is therefore only
worth it if it buys measurable error. docs/act_budget_2026-09-02 already showed W8 per-channel
weights land BELOW the measurement floor at 8 bits; this asks the same at 4 bits, where weight
error is expected to be dominant (AWQ/GPTQ group at 128 for int4 for exactly this reason).

INSTRUMENT. The MODIFF_ACT_BLOCK simulation forward, so bit width and granularity are qmax and
grouping changes rather than new kernels. This measures the QUANTIZERS and how their error
propagates -- which is what decides the granularity question. It is NOT the shipped int4 kernel
path (no zero-point, no packing), so it must not be read as an int4 speed or fidelity number.

Attention is left fp16 in every arm (MODIFF_STD_ATTN_BITS=0) because act_budget showed quantized
attention is 2.9x the floor and would mask everything else.

argv: <arm> <act_block> <act_qmax> <wbits> <wblock>
  act_block -3 exact | -2 per-tensor static | -1 per-tensor dynamic | N blockwise along C
  wbits     -1 exact | 8 | 4        wblock  0 per-output-channel | N blockwise along C
"""
from __future__ import annotations
import json, os, sys

ROOT = "/workspace/MoDiff"
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers")]

arm, ablk, aqmax, wbits, wblock = sys.argv[1:6]

os.environ.update({
    "MODIFF_LINEAR": "0", "MODIFF_CACHE_SKIP_K": "1", "MODIFF_REPLAY_K": "1",
    "MODIFF_AHAT_BITS": "16", "MODIFF_AHAT_REFRESH": "0", "MODIFF_IMODE": "0",
    "MODIFF_AHAT_BLOCK": "0", "MODIFF_DELTA_MODE": "static", "MODIFF_CONV_BLOCKK": "0",
    "MODIFF_STD_ATTN_BITS": "0",                       # attention fp16 -- see docstring
    "MODIFF_ACT_BLOCK": ablk, "MODIFF_ACT_SIM_QMAX": aqmax,
    "MODIFF_ACT_SIM_WBITS": wbits, "MODIFF_ACT_SIM_WBLOCK": wblock,
    # every fusion off: the sim requires every conv to reach forward()
    "MODIFF_DISABLE_GN_MODIFF_FUSION": "1", "MODIFF_DISABLE_GN_INT8_FUSION": "1",
    "MODIFF_DISABLE_O_HAT_RESIDUAL_FUSION": "1",
    "MODIFF_DISABLE_UPSAMPLE_QUANTIZE_FUSION": "1",
    "MODIFF_DISABLE_AVGPOOL_QUANTIZE_FUSION": "1",
})

from integration.utils.preflight import preflight, MODEL  # noqa: E402
preflight(*MODEL, what="wa_budget_arm.py")
import torch  # noqa: E402
import integration.benchmarks.benchmark_ldm as B  # noqa: E402

SHAPE, NQ, STEPS, SEED = (4, 32, 32), 6, 50, 20260805
REF = "docs/wa_budget_2026-09-02/data/ref_fp16.pt"
mode = "fp16" if arm == "fp16" else "int8"


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
    output_dir="docs/wa_budget_2026-09-02/tmp",
    batch_size=NQ, steps=STEPS, shape=SHAPE,
    calibration_path=B._default_calibration_path(mode), auto_delta_table=True)
model, sampler = runner._setup_model(mode)
q = mode != "fp16"

sample(model, sampler, NQ, q)                      # warmup -> steady state
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
lat = sample(model, sampler, NQ, q).detach().float().cpu()
if mode == "fp16":
    torch.save(lat, REF); rel = 0.0
else:
    rel = float((lat - torch.load(REF)).norm() / torch.load(REF).norm())

print("ARMJSON:" + json.dumps({"arm": arm, "act_block": int(ablk), "act_qmax": int(aqmax),
                               "wbits": int(wbits), "wblock": int(wblock),
                               "relL2_vs_fp16": rel}), flush=True)

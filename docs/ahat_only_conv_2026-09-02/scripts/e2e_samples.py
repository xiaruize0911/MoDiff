"""E2E wall clock + peak memory + REAL DECODED SAMPLES for the a_hat-only configuration.

The conv-input quantizer is per-tensor (the original scheme); the only thing blockwise is the
a_hat cache storage (MODIFF_AHAT_BLOCK=32 -> int8 codes + fp32 scales [N,H,W,C/32]).

One process per arm so no state leaks between them. argv: arm name.
  fp16 | int8_ptq | int8_ahat0 | int8_ahat32 | int4_ptq | int4_ahat0 | int4_ahat32
env: E2E_BATCH (default 128), E2E_STEPS (50), E2E_TRIALS (2), E2E_NSAMP (8), E2E_SKIP_TIME
Samples use a FIXED seed so every arm decodes the same noise and the grids are comparable.
"""
import os, statistics, sys, json
ROOT = "/workspace/MoDiff"; os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers")]
ARM = sys.argv[1]
# arm name carries the a_hat block size: int8_ahat0 / int8_ahat16 / int8_ahat32 / int8_ahat64
import re as _re
_m = _re.search(r"ahat(\d+)", ARM)
BLOCK = _m.group(1) if _m else "0"
# ...and optionally the a_hat BIT WIDTH: int4_ahat0_bits4 => MODIFF_AHAT_BITS=4 (qmax 7).
# Note bits only bites on the PER-TENSOR path; the blockwise helpers hardcode lim=127.
_b = _re.search(r"bits(\d+)", ARM)
AHAT_BITS = _b.group(1) if _b else "16"
os.environ.update({
    "MODIFF_LINEAR": "0", "MODIFF_CACHE_SKIP_K": "1", "MODIFF_REPLAY_K": "1",
    "MODIFF_AHAT_BITS": AHAT_BITS, "MODIFF_AHAT_REFRESH": "0", "MODIFF_IMODE": "0",
    "MODIFF_DELTA_MODE": "static", "MODIFF_CONV_BLOCKK": "0", "MODIFF_ACT_BLOCK": "0",
    "MODIFF_AHAT_BLOCK": BLOCK,
    # sim(\d+) in the arm name => MODIFF_AHAT_SIM_BITS, the fp16-storage blockwise N-bit
    # SIMULATION. Quality-only: no memory saving, and the ms/step is not meaningful.
    "MODIFF_AHAT_SIM_BITS": (_re.search(r"sim(\d+)", ARM).group(1)
                             if _re.search(r"sim(\d+)", ARM) else "0")})
MODE = ("fp16" if ARM == "fp16" else
        "int8_baseline" if ARM == "int8_ptq" else
        "int4_baseline" if ARM == "int4_ptq" else
        "int8" if ARM.startswith("int8") else "int4")
import torch, torchvision.utils as tvu
import integration.benchmarks.benchmark_ldm as B

SHAPE = (4, 32, 32)
BATCH = int(os.environ.get("E2E_BATCH", "128"))
STEPS = int(os.environ.get("E2E_STEPS", "50"))
TRIALS = int(os.environ.get("E2E_TRIALS", "2"))
NSAMP = int(os.environ.get("E2E_NSAMP", "8"))
SEED = 1234
OUTDIR = "docs/ahat_only_conv_2026-09-02/samples"
os.makedirs(OUTDIR, exist_ok=True)

r = B.BenchmarkRunner(
    config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
    ckpt_path="models/ldm/lsun_churches256/model.ckpt",
    output_dir="docs/ahat_only_conv_2026-09-02/tmp",
    batch_size=BATCH, steps=STEPS, shape=SHAPE,
    calibration_path=B._default_calibration_path(MODE), auto_delta_table=True)
m, s = r._setup_model(MODE)

def reset():
    if MODE != "fp16":
        if MODE.startswith("int8"):
            B.reset_modiff_state_int8(m.model.diffusion_model)
        else:
            B.reset_modiff_state_int4(m.model.diffusion_model)
    B._reset_wxax_modiff_safe(m)

def run(bs, seed=None):
    reset()
    if seed is not None:
        torch.manual_seed(seed)
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        return s.sample(S=STEPS, batch_size=bs, shape=SHAPE, eta=0.0, verbose=False)

res = {"arm": ARM, "mode": MODE, "ahat_block": int(BLOCK), "batch": BATCH, "steps": STEPS}

if os.environ.get("E2E_SKIP_TIME", "0") != "1":
    run(BATCH)                                   # warmup: cuDNN autotune + first-step calib
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    ts = []
    for _ in range(TRIALS):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); run(BATCH); b.record(); torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) / STEPS)
    res["ms_step"] = statistics.median(ts)
    res["trials"] = ts
    res["peak_alloc_MB"] = torch.cuda.max_memory_allocated() / 2**20
    res["peak_reserved_MB"] = torch.cuda.max_memory_reserved() / 2**20

# ---- real decoded samples, fixed seed so the arms are directly comparable ----
lat, _ = run(NSAMP, seed=SEED)
with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
    img = m.decode_first_stage(lat)
img = torch.clamp((img.float() + 1.0) / 2.0, 0.0, 1.0)
path = f"{OUTDIR}/{ARM}.png"
tvu.save_image(img.cpu(), path, nrow=4)
res["samples"] = path
res["sample_finite"] = bool(torch.isfinite(img).all().item())
res["sample_mean"] = float(img.mean())
res["latent_absmax"] = float(lat.abs().max())
print("E2EJSON:" + json.dumps(res))

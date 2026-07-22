"""Fusion-fix per-component timing profile (torch.profiler, CUDA self-time bucketed by kernel).

One CONFIG per invocation (a fusion flag is toggled by the CALLER's env before import, because
several flags are read at module-load time). Builds the given MoDiff mode, warms up, records an
independent wall ms/step and a torch.profiler per-component breakdown (buckets mirror the
benchmark_5mode e2e_timing_profile.cat()), and appends one row to data/fusion_profile.csv.
Also counts the config's key fused kernel to PROVE the fusion engaged (or not).

Usage:  CFG=<label> MODE=<int8|int4> [flag envs...] python profile_fusions.py
  e.g.  CFG=int8_modiff.ohat_on   MODE=int8  MODIFF_DEEPFUSE_OHAT=1 python profile_fusions.py
        CFG=int8_modiff.ohat_off  MODE=int8  MODIFF_DEEPFUSE_OHAT=0 python profile_fusions.py
"""
import os, sys, csv, time
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
# driver wiring (do NOT clobber the fusion flags the caller set)
os.environ.setdefault("MODIFF_QUANT_LINEAR", "1"); os.environ.setdefault("MODIFF_QUANT_ATTN", "1")
os.environ.setdefault("MODIFF_QUANT_ATTN_STATIC", "1"); os.environ.setdefault("MODIFF_LINEAR_OUT_I8", "0")
os.environ.pop("MODIFF_FLASH_ATTN", None)
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
import torch
from torch.profiler import profile, ProfilerActivity
import modiff_cutlass as _mc
import integration.benchmarks.benchmark_ldm as B

CFG = os.environ["CFG"]; MODE = os.environ["MODE"]
BATCH = int(os.environ.get("BATCH", "64"))          # b64: modiff-cache memory headroom; deltas are batch-robust
WARMUP, STEPS, NP = 20, 20, 3
HERE = "docs/fusion_fix_2026-07-22"
CSV = f"{HERE}/data/fusion_profile.csv"
BUCKETS = ["attention", "conv (int GEMM)", "qkv/proj int GEMM", "attn bmm (fp16)", "other fp16 GEMM",
           "GroupNorm", "quantize/dequant", "modiff cache", "upsample/concat", "elementwise/copy", "other"]


def cat(name):
    l = name.lower()
    if "softmax" in l or "scaled_dot" in l or "flash" in l: return "attention"
    if "implicit" in l or "cudnn" in l or "fprop" in l or "conv2d" in l or "scudnn" in l or "wgrad" in l or "convolution" in l: return "conv (int GEMM)"
    if "gemm_w8a8" in l or "gemm_w4a4" in l or "awq" in l: return "qkv/proj int GEMM"
    if "wmma_tensorop_f16" in l or "bmm" in l: return "attn bmm (fp16)"
    if "cutlass" in l or "cublas" in l or "ampere_fp16" in l or "s1688gemm" in l: return "other fp16 GEMM"
    if "group_norm" in l or "groupnorm" in l or "gn_" in l or "fused_gn" in l: return "GroupNorm"
    if "scale_accumulate" in l or "o_hat" in l or "dequant_accumulate" in l or "accumulate_from_half" in l: return "modiff cache"
    if "quant" in l or "dequant" in l or "requant" in l or "absmax" in l or "ahat" in l or "step1" in l or "aq_" in l or "pack" in l: return "quantize/dequant"
    if "upsample" in l or "interpolate" in l or "catarray" in l.replace("_", ""): return "upsample/concat"
    if "elementwise" in l or "vectorized" in l or "silu" in l or "copy" in l or "cat" in l or "fill" in l or "add" in l or "index" in l or "store" in l or "clamp" in l or "round" in l: return "elementwise/copy"
    return "other"


# engagement counters: wrap the key fused kernels so the row records whether each engaged
ENG = {"ohat_deepfuse": 0, "gn_delta": 0, "gn_pack": 0}
for attr, key in [("conv2d_int8_dequant_fp16_o_hat_tuned", "ohat_deepfuse"),
                  ("conv2d_int4_dequant_fp16_o_hat_tuned", "ohat_deepfuse"),
                  ("group_norm_silu_delta_quantize_nhwc", "gn_delta"),
                  ("group_norm_silu_quantize_pack_nhwc", "gn_pack")]:
    if hasattr(_mc, attr):
        _o = getattr(_mc, attr)
        def mk(o, k):
            def w(*a, **kw):
                ENG[k] += 1; return o(*a, **kw)
            return w
        setattr(_mc, attr, mk(_o, key))

calib = "integration/calibration/int8_calibration.pt" if "int8" in MODE else "integration/calibration/int4_calibration.pt"
r = B.BenchmarkRunner("configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
                      "models/ldm/lsun_churches256/model.ckpt", output_dir="integration/results/fusionfix",
                      batch_size=BATCH, steps=STEPS, shape=(4, 32, 32), calibration_path=calib,
                      linear_backend="int_gemm")
model, sampler = r._setup_model(MODE); cond = r._cond_kwargs(model, BATCH)


def reset():
    B.reset_modiff_state_int8(model.model.diffusion_model) if "int8" in MODE else B.reset_modiff_state_int4(model.model.diffusion_model)
    if "int8" in MODE and B.HAS_INT8_LINEAR: B.reset_modiff_state_linear(model.model.diffusion_model)
    if "int4" in MODE and B.HAS_INT4_LINEAR: B.reset_modiff_state_int4_linear(model.model.diffusion_model)


def smp(S):
    reset()
    with torch.inference_mode(), torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
        sampler.sample(S=S, batch_size=BATCH, shape=r.shape, eta=0.0, verbose=False, **cond)


# GPU clock burn-in + warmup (wires/freezes fused paths, attn self-calibration)
bn = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
for _ in range(40): bn = bn @ bn * 1e-4 + 1.0
smp(WARMUP); torch.cuda.synchronize()

wall = []
for _ in range(3):
    torch.cuda.synchronize(); t0 = time.time(); smp(STEPS); torch.cuda.synchronize()
    wall.append((time.time() - t0) / STEPS * 1000)
wall = min(wall)

ENG.update(ohat_deepfuse=0, gn_delta=0, gn_pack=0)
with profile(activities=[ProfilerActivity.CUDA]) as p:
    smp(NP * STEPS)
torch.cuda.synchronize()
agg = {b: 0.0 for b in BUCKETS}
for e in p.key_averages():
    if e.self_device_time_total > 0:
        agg[cat(e.key)] += e.self_device_time_total / (NP * STEPS) / 1000.0
gpu = sum(agg.values())
eng = {k: round(v / (NP * STEPS)) for k, v in ENG.items()}   # per-step engagement count

row = dict(config=CFG, mode=MODE, batch=BATCH, **{b: round(agg[b], 3) for b in BUCKETS},
           gpu_busy=round(gpu, 3), wall=round(wall, 3),
           eng_ohat=eng["ohat_deepfuse"], eng_gn_delta=eng["gn_delta"], eng_gn_pack=eng["gn_pack"])
newfile = not os.path.exists(CSV)
with open(CSV, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(row.keys()))
    if newfile: w.writeheader()
    w.writerow(row)
print(f"{CFG:26} gpu_busy={gpu:6.2f} wall={wall:6.2f} ms/step  "
      f"eng(ohat={eng['ohat_deepfuse']},gn_delta={eng['gn_delta']},gn_pack={eng['gn_pack']})/step")

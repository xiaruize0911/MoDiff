"""Detailed profile of the int8/int4 baseline config at batch 128: (1) verify STATIC quantization +
which FUSED kernels are live per layer; (2) full torch.profiler per-kernel breakdown (top kernels + %,
and a quantized-vs-fp16 split) to show exactly where the time goes and why the e2e speedup is small.
Writes data/detailed_<mode>_b128.csv."""
import os, sys, csv, time
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
import torch
from torch.profiler import profile, ProfilerActivity
import integration.benchmarks.benchmark_ldm as B

BATCH = int(os.environ.get("E2E_BATCH", "128"))
MODE = os.environ.get("PROF_MODE", "int8_baseline")
STEPS, WARM_S, NP = 20, 3.0, 3


def cat(name):
    l = name.lower()
    if "softmax" in l or "scaled_dot" in l or "flash" in l: return "attention softmax"
    if "implicit" in l or "cudnn" in l or "fprop" in l or "conv2d" in l or "scudnn" in l or "wgrad" in l or "convolution" in l:
        return "conv (int GEMM)"                              # int8/int4 conv (CUTLASS ImplicitGemmConvolution)
    if "gemm_w8a8" in l or "gemm_w4a4" in l: return "qkv/proj int GEMM"
    if "wmma_tensorop_f16" in l or ("bmm" in l): return "attention QKᵀ/AV bmm (fp16)"
    if "cutlass" in l or "cublas" in l or "ampere_fp16" in l or "s1688gemm" in l: return "other fp16 GEMM"
    if "group_norm" in l or "groupnorm" in l or "gn_" in l or "fused_gn" in l: return "GroupNorm"
    if "quant" in l or "dequant" in l or "requant" in l or "absmax" in l or "ahat" in l or "step1" in l: return "quantize/dequant"
    if "scale_accumulate" in l or "o_hat" in l: return "modiff cache"
    if "upsample" in l or "interpolate" in l or "catarray" in l.replace("_", ""): return "upsample/concat"
    if "elementwise" in l or "vectorized" in l or "silu" in l or "copy" in l or "cat" in l or "fill" in l or "add" in l or "index" in l or "store" in l: return "elementwise/copy"
    return "other"


QUANTIZED = {"qkv/proj int GEMM", "conv (int GEMM)"}
FP16_HEAVY = {"attention softmax", "attention QKᵀ/AV bmm (fp16)", "GroupNorm", "elementwise/copy", "upsample/concat"}

_quant = MODE != "fp16"
os.environ["MODIFF_QUANT_LINEAR"] = "1" if _quant else "0"
os.environ["MODIFF_LINEAR_OUT_I8"] = "0"
os.environ["MODIFF_QUANT_ATTN"] = "0"
calib = ("integration/calibration/int8_calibration.pt" if "int8" in MODE else
         ("integration/calibration/int4_calibration.pt" if "int4" in MODE else None))
r = B.BenchmarkRunner("configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
                      "models/ldm/lsun_churches256/model.ckpt", output_dir="integration/results/detail",
                      batch_size=BATCH, steps=STEPS, shape=(4, 32, 32), calibration_path=calib,
                      linear_backend=("int_gemm" if _quant else "fp16"))
model, sampler = r._setup_model(MODE); cond = r._cond_kwargs(model, BATCH)

# ---- (1) verify static + fused paths ----
from integration.kernels.wxax_linear import QuantLinearWxAx
from integration.fused_ops.token_major_attention import TokenMajorAttentionBlock
lins = [m for m in model.model.diffusion_model.modules() if isinstance(m, QuantLinearWxAx)]
blks = [m for m in model.model.diffusion_model.modules() if isinstance(m, TokenMajorAttentionBlock)]
n_static = sum(1 for m in lins if m.a_scale is not None)
n_outi8 = sum(1 for m in lins if getattr(m, "_out_i8", False) and m._inv_out_scale is not None)
n_qkvfuse = sum(1 for m in blks if m._fuse_qkv_quant and isinstance(m.qkv, QuantLinearWxAx) and m.qkv.bits == 8 and m.qkv.a_scale is not None)
n_projfuse = sum(1 for m in blks if m._fuse_proj_quant and isinstance(m.proj, QuantLinearWxAx) and m.proj.bits == 8 and m.proj.a_scale is not None)
print(f"=== {MODE} b{BATCH}: config check ===")
print(f"  Linear layers: {len(lins)}  | static a_scale set: {n_static}/{len(lins)}  | output-fused (out_i8 ready): {n_outi8}")
print(f"  Attention blocks: {len(blks)} | GN->int8-quantize FUSED qkv: {n_qkvfuse}  | proj transpose+quant FUSED: {n_projfuse}")
print(f"  bits={lins[0].bits if lins else 'none (fp16)'}")


def smp(S):
    with torch.inference_mode(), torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
        sampler.sample(S=S, batch_size=BATCH, shape=r.shape, eta=0.0, verbose=False, **cond)


tw = time.time()
while time.time() - tw < WARM_S: smp(STEPS)
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CUDA]) as pr:
    for _ in range(NP): smp(STEPS)
    torch.cuda.synchronize()

kern = {}
for e in pr.key_averages():
    dt = e.self_device_time_total
    if dt <= 0: continue
    kern[e.key] = kern.get(e.key, 0.0) + dt / (NP * STEPS) / 1e3   # ms/step
gpu = sum(kern.values())
buckets = {}
for k, v in kern.items(): buckets[cat(k)] = buckets.get(cat(k), 0.0) + v

print(f"\n=== category breakdown (GPU-busy {gpu:.1f} ms/step) ===")
for b, v in sorted(buckets.items(), key=lambda kv: -kv[1]):
    print(f"  {v:7.2f} ms  {v/gpu*100:5.1f}%  {b}")
qz = sum(v for b, v in buckets.items() if b in QUANTIZED)
fh = sum(v for b, v in buckets.items() if b in FP16_HEAVY)
print(f"\n  quantized compute (conv+qkv/proj int GEMM): {qz:.1f} ms ({qz/gpu*100:.0f}%)  <- only this benefits from int8/int4")
print(f"  fp16/memory-bound (attn+GN+elementwise):     {fh:.1f} ms ({fh/gpu*100:.0f}%)  <- unchanged by quantization")

print(f"\n=== top 22 kernels ===")
top = sorted(kern.items(), key=lambda kv: -kv[1])[:22]
for k, v in top:
    print(f"  {v:6.2f} ms {v/gpu*100:4.1f}%  [{cat(k):26s}] {k[:60]}")

with open(f"docs/layer_roofline_2026-07-19/data/detailed_{MODE}_b{BATCH}.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["kernel", "ms_step", "pct", "category"])
    for k, v in sorted(kern.items(), key=lambda kv: -kv[1]):
        w.writerow([k[:90], round(v, 4), round(v / gpu * 100, 2), cat(k)])
print(f"\nWROTE detailed_{MODE}_b{BATCH}.csv")

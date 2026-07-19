"""Combined config: AWQ w8a8 / modified w4a4 LINEAR (qkv/proj) + dynamic quantized W8A8/W4A4
ATTENTION (quantized_std_attention: QKᵀ int-GEMM -> dynamic requant softmax -> AV int-GEMM,
materialized, no static-c) in ONE model, via MODIFF_QUANT_ATTN=1. Versions:
  1. fp16                          (reference; fp16 attention)
  2. int8_baseline + W8A8 attn     (AWQ w8a8 linear + int8 attention, no MoDiff cache)
  3. int8_modiff   + W8A8 attn
  4. int4_baseline + W4A4 attn     (w4a4 linear + int4 attention, no MoDiff cache)
  5. int4_modiff   + W4A4 attn
Measures per-version: quality rel-L2 vs fp16 (single UNet forward, fixed input), e2e latency, and a
torch.profiler bucket breakdown. Compare speed to bench5_speed_noflash (same linears, fp16 attention).
Writes data/combined_*.csv. Batch via E2E_BATCH (default 64)."""
import os, sys, csv, time, statistics
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
import torch
from torch.profiler import profile, ProfilerActivity
import integration.benchmarks.benchmark_ldm as B

BATCH = int(os.environ.get("E2E_BATCH", "64"))
STEPS, WARM_S, NP, RUNS = 20, 3.0, 2, 6
D = "docs/attention_fused_int8_int4_2026-07-19/data"
# Attention quantization mode: STATIC by default (calibrated Q/K/V scales + static-c softmax, no
# runtime reductions — consistent with the static conv/linear quant). Set STAT=dynamic for the A/B.
STAT = os.environ.get("MODIFF_QUANT_ATTN_STATIC", "1") == "1"
os.environ["MODIFF_QUANT_ATTN_STATIC"] = "1" if STAT else "0"
SUF = "" if STAT else "_dynamic"   # static is the default deliverable; dynamic kept as *_dynamic
VERSIONS = [  # (label, mode, backend, quant_attn)
    ("fp16",          "fp16",          "fp16",     False),
    ("int8_baseline", "int8_baseline", "int_gemm", True),
    ("int8_modiff",   "int8",          "int_gemm", True),
    ("int4_baseline", "int4_baseline", "int_gemm", True),
    ("int4_modiff",   "int4",          "int_gemm", True),
]

torch.manual_seed(0)
X = torch.randn(BATCH, 4, 32, 32, device="cuda")
TT = torch.randint(0, 1000, (BATCH,), device="cuda")


def bucket(name):
    l = name.lower()
    if "flash" in l or "fmha" in l or "scaled_dot" in l or "softmax" in l or "requant" in l: return "attention (int QKᵀ/softmax/AV)"
    if "group_norm" in l or "groupnorm" in l or "rowwisemoments" in l or "computefused" in l or "gn_" in l or "fused_gn" in l: return "GroupNorm"
    if "cudnn" in l or "implicit" in l or "fprop" in l or "scudnn" in l or "wgrad" in l or "conv2d" in l: return "conv (GEMM)"
    if "attn_qk" in l or "attn_av" in l: return "attention (int QKᵀ/softmax/AV)"
    if "gemm" in l or "cutlass" in l or "cublas" in l or "bmm" in l: return "qkv/proj GEMM (+ fp16 attn bmm)"
    if any(s in l for s in ["quantize", "dequant", "sub_absmax", "delta", "o_hat", "step1", "pack", "absmax", "aq_"]): return "quantize / absmax"
    if "store" in l or "epilogue" in l: return "conv store epilogue"
    if "upsample" in l or "catarray" in l.replace("_", "") or "cat_" in l: return "upsample / concat"
    if any(s in l for s in ["elementwise", "vectorized", "functor", "silu", "copy", "fill", "index", "add"]): return "elementwise / copy"
    return "other"


fp16_out = [None]


def run(label, mode, backend, qattn):
    quant = backend == "int_gemm"
    if quant: os.environ["MODIFF_QUANT_LINEAR"] = "1"
    else: os.environ.pop("MODIFF_QUANT_LINEAR", None)
    os.environ["MODIFF_QUANT_ATTN"] = "1" if qattn else "0"
    calib = "integration/calibration/int8_calibration.pt" if "int8" in mode else \
            ("integration/calibration/int4_calibration.pt" if "int4" in mode else None)
    runner = B.BenchmarkRunner("configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
                               "models/ldm/lsun_churches256/model.ckpt", output_dir="integration/results/combined",
                               batch_size=BATCH, steps=STEPS, shape=(4, 32, 32), calibration_path=calib,
                               linear_backend=backend)
    model, sampler = runner._setup_model(mode)
    from integration.fused_ops.quantized_std_attention import QuantizedStandardAttentionBlock
    nqa = sum(1 for m in model.model.diffusion_model.modules() if isinstance(m, QuantizedStandardAttentionBlock))
    cond = runner._cond_kwargs(model, BATCH)
    dm = model.model.diffusion_model

    def smp(S):
        with torch.inference_mode(), torch.amp.autocast('cuda', enabled=(mode != "fp32"), dtype=torch.float16):
            sampler.sample(S=S, batch_size=BATCH, shape=runner.shape, eta=0.0, verbose=False, **cond)

    # warmup first (lets any attention static-calib settle; dynamic needs none)
    tw = time.time()
    while time.time() - tw < WARM_S: smp(STEPS)
    torch.cuda.synchronize()
    # quality: single UNet forward vs fp16
    with torch.inference_mode(), torch.amp.autocast('cuda', enabled=(mode != "fp32"), dtype=torch.float16):
        out = dm(X, TT).float()
    if label == "fp16":
        fp16_out[0] = out.clone()
        rel = 0.0
    else:
        rel = ((out - fp16_out[0]).norm() / fp16_out[0].norm()).item()
    walls = []
    for _ in range(RUNS):
        torch.cuda.synchronize(); t0 = time.time(); smp(STEPS)
        torch.cuda.synchronize(); walls.append((time.time() - t0) / STEPS * 1000)
    wall = min(walls)
    with profile(activities=[ProfilerActivity.CUDA]) as pr:
        for _ in range(NP): smp(STEPS)
        torch.cuda.synchronize()
    buckets = {}
    for e in pr.key_averages():
        if e.self_device_time_total <= 0: continue
        buckets[bucket(e.key)] = buckets.get(bucket(e.key), 0.0) + e.self_device_time_total / (NP * STEPS) / 1e3
    gpu = sum(buckets.values())
    print(f"\n===== {label} (qattn blocks={nqa}) =====  wall={wall:.2f}  GPU={gpu:.2f} ms/step  rel_vs_fp16={rel:.4f}")
    for bk, vv in sorted(buckets.items(), key=lambda kv: -kv[1]):
        print(f"   {vv:7.3f} ms {vv/gpu*100:5.1f}%  {bk}")
    del model, sampler; torch.cuda.empty_cache()
    return wall, gpu, rel, buckets, nqa


def main():
    srows, brows = [], []
    for (label, mode, backend, qattn) in VERSIONS:
        wall, gpu, rel, buckets, nqa = run(label, mode, backend, qattn)
        srows.append({"version": label, "wall_ms_step": round(wall, 3), "gpu_busy_ms_step": round(gpu, 3),
                      "rel_vs_fp16": round(rel, 4), "qattn_blocks": nqa})
        for bk, v in buckets.items(): brows.append({"version": label, "bucket": bk, "ms_step": round(v, 4)})

    def w(path, rows, cols):
        with open(path, "w", newline="") as f:
            c = csv.DictWriter(f, fieldnames=cols); c.writeheader(); c.writerows(rows)
    w(f"{D}/combined_speed{SUF}_b{BATCH}.csv", srows, ["version", "wall_ms_step", "gpu_busy_ms_step", "rel_vs_fp16", "qattn_blocks"])
    w(f"{D}/combined_buckets{SUF}_b{BATCH}.csv", brows, ["version", "bucket", "ms_step"])
    fp = next(r["wall_ms_step"] for r in srows if r["version"] == "fp16")
    print(f"\n===== COMBINED (AWQ linear + quantized STATIC attention) SUMMARY, batch {BATCH} =====")
    for r in srows:
        print(f"  {r['version']:16s} wall={r['wall_ms_step']:7.2f}  {fp/r['wall_ms_step']:.3f}× vs fp16  "
              f"rel_vs_fp16={r['rel_vs_fp16']:.4f}  (qattn blocks={r['qattn_blocks']})")
    print(f"\nWROTE {D}/combined_*_b{BATCH}.csv")


if __name__ == "__main__":
    main()

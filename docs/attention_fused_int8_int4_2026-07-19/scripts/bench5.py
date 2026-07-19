"""5-version benchmark + profile of the churches UNet with the FUSED int8 flash attention
(flash_attn_int8: QKᵀ/softmax/AV in one kernel, online-softmax, no T×T materialization) wired into
the attention blocks. Versions:
  1. fp16                    - no quantization (baseline reference)
  2. int8_baseline + flash   - int8 conv/linear, NO MoDiff cache, int8 flash attention
  3. int8_modiff   + flash   - int8 conv/linear + MoDiff temporal caching, int8 flash attention
  4. int4_baseline + flash   - int4 conv/linear, NO MoDiff cache, int8 flash attention (no int4 flash kernel)
  5. int4_modiff   + flash   - int4 conv/linear + MoDiff caching, int8 flash attention
Flash attention (int8) is enabled for all 4 quant versions via MODIFF_FLASH_ATTN=1 (T>=512 -> the
dominant T=1024 block). Emits per-version e2e latency + torch.profiler bucket breakdown + top kernels.
Batch via E2E_BATCH (default 64). Writes data/*.csv.
"""
import os, sys, csv, time, statistics
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
import torch
from torch.profiler import profile, ProfilerActivity
import integration.benchmarks.benchmark_ldm as B

BATCH = int(os.environ.get("E2E_BATCH", "64"))
STEPS, WARM_S, NP, RUNS = 20, 3.0, 2, 6
D = "docs/attention_fused_int8_int4_2026-07-19/data"
NOFLASH = os.environ.get("BENCH5_NOFLASH") == "1"   # force fp16 SDPA attention everywhere (A/B reference)
SUF = "_noflash" if NOFLASH else ""

# (label, _setup_model mode, linear_backend, flash_attn_on)
VERSIONS = [
    ("fp16",          "fp16",          "fp16",     False),
    ("int8_baseline", "int8_baseline", "int_gemm", True),
    ("int8_modiff",   "int8",          "int_gemm", True),
    ("int4_baseline", "int4_baseline", "int_gemm", True),
    ("int4_modiff",   "int4",          "int_gemm", True),
]


def bucket(name):
    l = name.lower()
    if "flash" in l or "fmha" in l or "scaled_dot" in l or "softmax" in l: return "attention (softmax/flash)"
    if "group_norm" in l or "groupnorm" in l or "rowwisemoments" in l or "computefused" in l or "gn_" in l or "fused_gn" in l: return "GroupNorm"
    if "cudnn" in l or "implicit" in l or "fprop" in l or "scudnn" in l or "wgrad" in l or "conv2d" in l: return "conv (GEMM)"
    if "gemm" in l or "cutlass" in l or "cublas" in l or "bmm" in l: return "qkv/proj GEMM (+ fp16 attn bmm)"
    if any(s in l for s in ["quantize", "dequant", "sub_absmax", "delta", "o_hat", "step1", "pack", "absmax", "aq_"]): return "quantize / absmax"
    if "store" in l or "epilogue" in l or "requant" in l: return "conv store epilogue"
    if "upsample" in l or "catarray" in l.replace("_", "") or "cat_" in l: return "upsample / concat"
    if any(s in l for s in ["elementwise", "vectorized", "functor", "silu", "copy", "fill", "index", "add"]): return "elementwise / copy"
    return "other"


def run(label, mode, backend, flash):
    quant = backend == "int_gemm"
    if quant: os.environ["MODIFF_QUANT_LINEAR"] = "1"
    else: os.environ.pop("MODIFF_QUANT_LINEAR", None)
    flash = flash and not NOFLASH
    os.environ["MODIFF_FLASH_ATTN"] = "1" if flash else "0"
    calib = None
    if "int8" in mode: calib = "integration/calibration/int8_calibration.pt"
    elif "int4" in mode: calib = "integration/calibration/int4_calibration.pt"
    runner = B.BenchmarkRunner("configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
                               "models/ldm/lsun_churches256/model.ckpt",
                               output_dir="integration/results/bench5", batch_size=BATCH, steps=STEPS,
                               shape=(4, 32, 32), calibration_path=calib, linear_backend=backend)
    model, sampler = runner._setup_model(mode)
    # confirm flash engaged
    from integration.fused_ops.token_major_attention import TokenMajorAttentionBlock
    nfl = sum(1 for m in model.model.diffusion_model.modules()
              if isinstance(m, TokenMajorAttentionBlock) and getattr(m, "_flash", False))
    cond = runner._cond_kwargs(model, BATCH)

    def smp(S):
        with torch.inference_mode(), torch.amp.autocast('cuda', enabled=(mode != "fp32"), dtype=torch.float16):
            sampler.sample(S=S, batch_size=BATCH, shape=runner.shape, eta=0.0, verbose=False, **cond)

    tw = time.time()
    while time.time() - tw < WARM_S: smp(STEPS)
    torch.cuda.synchronize()
    walls = []
    for _ in range(RUNS):
        torch.cuda.synchronize(); t0 = time.time(); smp(STEPS)
        torch.cuda.synchronize(); walls.append((time.time() - t0) / STEPS * 1000)
    wall = min(walls)
    with profile(activities=[ProfilerActivity.CUDA]) as pr:
        for _ in range(NP): smp(STEPS)
        torch.cuda.synchronize()
    buckets, kern = {}, {}
    for e in pr.key_averages():
        dt = e.self_device_time_total
        if dt <= 0: continue
        v = dt / (NP * STEPS) / 1e3
        buckets[bucket(e.key)] = buckets.get(bucket(e.key), 0.0) + v
        kern[e.key] = kern.get(e.key, 0.0) + v
    gpu = sum(buckets.values())
    print(f"\n===== {label} (flash blocks={nfl}) =====  wall={wall:.2f}  GPU-busy={gpu:.2f} ms/step")
    for bk, vv in sorted(buckets.items(), key=lambda kv: -kv[1]):
        print(f"   {vv:7.3f} ms  {vv/gpu*100:5.1f}%  {bk}")
    top = sorted(kern.items(), key=lambda kv: -kv[1])[:12]
    del model, sampler; torch.cuda.empty_cache()
    return wall, gpu, buckets, top, nfl


def main():
    srows, brows, krows = [], [], []
    for (label, mode, backend, flash) in VERSIONS:
        wall, gpu, buckets, top, nfl = run(label, mode, backend, flash)
        srows.append({"version": label, "wall_ms_step": round(wall, 3), "gpu_busy_ms_step": round(gpu, 3),
                      "flash_blocks": nfl})
        for bk, v in buckets.items(): brows.append({"version": label, "bucket": bk, "ms_step": round(v, 4)})
        for k, v in top: krows.append({"version": label, "kernel": k[:90], "ms_step": round(v, 4), "bucket": bucket(k)})

    def w(path, rows, cols):
        with open(path, "w", newline="") as f:
            c = csv.DictWriter(f, fieldnames=cols); c.writeheader(); c.writerows(rows)
    w(f"{D}/bench5_speed{SUF}_b{BATCH}.csv", srows, ["version", "wall_ms_step", "gpu_busy_ms_step", "flash_blocks"])
    w(f"{D}/bench5_buckets{SUF}_b{BATCH}.csv", brows, ["version", "bucket", "ms_step"])
    w(f"{D}/bench5_topkernels{SUF}_b{BATCH}.csv", krows, ["version", "kernel", "ms_step", "bucket"])
    fp = next(r["wall_ms_step"] for r in srows if r["version"] == "fp16")
    print("\n===== SUMMARY (wall ms/step, batch %d) =====" % BATCH)
    for r in srows:
        print(f"  {r['version']:16s} {r['wall_ms_step']:8.2f}  {fp/r['wall_ms_step']:.3f}× vs fp16  (flash blocks={r['flash_blocks']})")
    print(f"\nWROTE {D}/bench5_*_b{BATCH}.csv")


if __name__ == "__main__":
    main()

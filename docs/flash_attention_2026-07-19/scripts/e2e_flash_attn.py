"""E2E effect of enabling int8 flash attention (MODIFF_FLASH_ATTN=8) on top of int8 linear
quant, vs int8 with fp16 MATH attention, vs fp16. b128, MATH attention default; flash is opt-in.
Writes data/e2e_flash_attn_b128.csv."""
import os, sys, csv, time, statistics
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
import torch
import integration.benchmarks.benchmark_ldm as B

BATCH = 128; WARMUP, TIMED, RUNS = 15, 50, 3
# (label, mode, flash_bits)
VERS = [("fp16", "fp16", 0),
        ("int8 (attn=MATH)", "int8_baseline", 0),
        ("int8 + flash_attn=8", "int8_baseline", 8)]

def run(mode, flash_bits):
    quant = mode != "fp16"
    os.environ["MODIFF_QUANT_LINEAR"] = "1" if quant else "0"
    os.environ["MODIFF_QUANT_ATTN"] = "0"          # attention via TokenMajorAttentionBlock (fp16 MATH / flash)
    os.environ["MODIFF_LINEAR_OUT_I8"] = "0"
    if flash_bits: os.environ["MODIFF_FLASH_ATTN"] = str(flash_bits)
    else: os.environ.pop("MODIFF_FLASH_ATTN", None)
    os.environ.pop("MODIFF_FUSE_QKV_QUANT", None); os.environ.pop("MODIFF_FUSE_PROJ_QUANT", None)
    calib = "integration/calibration/int8_calibration.pt" if "int8" in mode else None
    r = B.BenchmarkRunner("configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
                          "models/ldm/lsun_churches256/model.ckpt", output_dir="integration/results/e2e_flash",
                          batch_size=BATCH, steps=TIMED, shape=(4, 32, 32), calibration_path=calib,
                          linear_backend=("int_gemm" if quant else "fp16"))
    model, sampler = r._setup_model(mode); cond = r._cond_kwargs(model, BATCH)
    # confirm flash is engaged
    if flash_bits:
        from integration.fused_ops.token_major_attention import TokenMajorAttentionBlock
        blks = [m for m in model.model.diffusion_model.modules() if isinstance(m, TokenMajorAttentionBlock)]
        nfb = sum(1 for m in blks if getattr(m, "_flash_bits", 0) == flash_bits)
        print(f"    [flash engaged on {nfb}/{len(blks)} attention blocks]")
    def smp(S):
        # autocast fp16 ON for ALL modes (fixed 2026-07-20: was enabled=quant -> fp16 baseline ran fp32/tf32).
        with torch.inference_mode(), torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
            sampler.sample(S=S, batch_size=BATCH, shape=r.shape, eta=0.0, verbose=False, **cond)
    smp(WARMUP); torch.cuda.synchronize()
    ms = []
    for _ in range(RUNS):
        torch.cuda.synchronize(); t0 = time.time(); smp(TIMED); torch.cuda.synchronize()
        ms.append((time.time() - t0) / TIMED * 1000)
    del model, sampler; torch.cuda.empty_cache()
    return statistics.mean(ms), min(ms)

rows = []; fp16_mean = None
print(f"E2E b{BATCH}, {WARMUP} warm + {RUNS}x{TIMED} steps, MEAN ms/step\n")
print(f"{'version':22} {'mean ms/step':>12} {'vs fp16':>8} {'vs int8-MATH':>13}")
int8_mean = None
for (label, mode, fb) in VERS:
    mean, mn = run(mode, fb)
    if label == "fp16": fp16_mean = mean
    if label.startswith("int8 (attn=MATH"): int8_mean = mean
    sp = fp16_mean / mean if fp16_mean else 1.0
    vs_i8 = (int8_mean / mean) if int8_mean else float("nan")
    print(f"{label:22} {mean:12.2f} {sp:7.2f}x {vs_i8:12.3f}x")
    rows.append(dict(version=label, mean_ms=round(mean, 2), min_ms=round(mn, 2),
                     speedup_vs_fp16=round(sp, 3), vs_int8_math=round(vs_i8, 3)))
with open("docs/flash_attention_2026-07-19/data/e2e_flash_attn_b128.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("\nWROTE data/e2e_flash_attn_b128.csv")

"""End-to-end int8/int4 vs fp16 on the CURRENT code (flash removed, MATH attention,
static quant + fused linear kernels). Lighter step counts than bench5_confirm for a
faster-but-solid mean. b128. Writes data/e2e_int8_int4_vs_fp16_b128.csv."""
import os, sys, csv, time, statistics
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
import torch
import integration.benchmarks.benchmark_ldm as B

BATCH = 128; WARMUP, TIMED, RUNS = 15, 50, 3
VERS = [("fp16", "fp16"), ("int8_baseline", "int8_baseline"), ("int8_modiff", "int8"),
        ("int4_baseline", "int4_baseline"), ("int4_modiff", "int4")]

def run(mode):
    quant = mode != "fp16"
    os.environ["MODIFF_QUANT_LINEAR"] = "1" if quant else "0"
    os.environ["MODIFF_QUANT_ATTN"] = "0"          # fp16 MATH attention (flash removed)
    os.environ["MODIFF_LINEAR_OUT_I8"] = "0"
    os.environ.pop("MODIFF_FUSE_QKV_QUANT", None); os.environ.pop("MODIFF_FUSE_PROJ_QUANT", None)
    calib = "integration/calibration/int8_calibration.pt" if "int8" in mode else \
            ("integration/calibration/int4_calibration.pt" if "int4" in mode else None)
    r = B.BenchmarkRunner("configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
                          "models/ldm/lsun_churches256/model.ckpt", output_dir="integration/results/e2e_cmp",
                          batch_size=BATCH, steps=TIMED, shape=(4, 32, 32), calibration_path=calib,
                          linear_backend=("int_gemm" if quant else "fp16"))
    model, sampler = r._setup_model(mode); cond = r._cond_kwargs(model, BATCH)
    def smp(S):
        # autocast fp16 ON for ALL modes. (Bug fixed 2026-07-20: was enabled=quant, which left the
        # fp16 baseline running fp32/tf32 -> inflated the int8 "2x" to a precision artifact. See
        # docs/flash_attention_2026-07-19/scripts/true_fp16_vs_int8.py.)
        with torch.inference_mode(), torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
            sampler.sample(S=S, batch_size=BATCH, shape=r.shape, eta=0.0, verbose=False, **cond)
    smp(WARMUP); torch.cuda.synchronize()
    ms = []
    for _ in range(RUNS):
        torch.cuda.synchronize(); t0 = time.time(); smp(TIMED); torch.cuda.synchronize()
        ms.append((time.time() - t0) / TIMED * 1000)
    del model, sampler; torch.cuda.empty_cache()
    return statistics.mean(ms), min(ms)

rows = []
print(f"E2E b{BATCH}, {WARMUP} warm + {RUNS}x{TIMED} steps, MEAN ms/step (MATH attention, static quant + fused linear)\n")
print(f"{'version':16} {'mean ms/step':>12} {'min ms/step':>12} {'speedup vs fp16':>16}")
fp16_mean = None
for (label, mode) in VERS:
    mean, mn = run(mode)
    if label == "fp16": fp16_mean = mean
    sp = fp16_mean / mean if fp16_mean else 1.0
    print(f"{label:16} {mean:12.2f} {mn:12.2f} {sp:15.2f}x")
    rows.append(dict(version=label, mean_ms=round(mean, 2), min_ms=round(mn, 2), speedup_vs_fp16=round(sp, 3)))
with open("docs/flash_attention_2026-07-19/data/e2e_int8_int4_vs_fp16_b128.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("\nWROTE data/e2e_int8_int4_vs_fp16_b128.csv")

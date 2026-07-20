"""Corrected e2e table: int8/int4/modiff vs TRUE fp16 (autocast fp16 enabled for ALL modes,
fixing the harness bug where autocast was only on for the quant path). b128, DDIM.
Writes data/e2e_true_fp16_table_b128.csv."""
import os, sys, csv, time, statistics
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
import torch
import integration.benchmarks.benchmark_ldm as B

BATCH = 128; WARMUP, TIMED, RUNS = 15, 40, 3
VERS = [("fp16 (true, autocast)", "fp16"), ("int8_baseline", "int8_baseline"), ("int8_modiff", "int8"),
        ("int4_baseline", "int4_baseline"), ("int4_modiff", "int4")]

def run(mode):
    quant = mode != "fp16"
    os.environ["MODIFF_QUANT_LINEAR"] = "1" if quant else "0"
    os.environ["MODIFF_QUANT_ATTN"] = "0"; os.environ["MODIFF_LINEAR_OUT_I8"] = "0"; os.environ.pop("MODIFF_FLASH_ATTN", None)
    calib = "integration/calibration/int8_calibration.pt" if "int8" in mode else \
            ("integration/calibration/int4_calibration.pt" if "int4" in mode else None)
    r = B.BenchmarkRunner("configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
                          "models/ldm/lsun_churches256/model.ckpt", output_dir="integration/results/e2e_true",
                          batch_size=BATCH, steps=TIMED, shape=(4, 32, 32), calibration_path=calib,
                          linear_backend=("int_gemm" if quant else "fp16"))
    model, sampler = r._setup_model(mode); cond = r._cond_kwargs(model, BATCH)
    def smp(S):  # autocast fp16 ON for ALL modes (the fix)
        with torch.inference_mode(), torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
            sampler.sample(S=S, batch_size=BATCH, shape=r.shape, eta=0.0, verbose=False, **cond)
    smp(WARMUP); torch.cuda.synchronize()
    ms = []
    for _ in range(RUNS):
        torch.cuda.synchronize(); t0 = time.time(); smp(TIMED); torch.cuda.synchronize(); ms.append((time.time() - t0) / TIMED * 1000)
    del model, sampler; torch.cuda.empty_cache()
    return statistics.mean(ms), min(ms)

# GPU clock burn-in so the first (fp16) run isn't cold
bn = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
for _ in range(60): bn = bn @ bn * 1e-4 + 1.0
torch.cuda.synchronize()

rows = []; fp16 = None
print(f"CORRECTED e2e vs TRUE fp16 (autocast ON for all) @ b{BATCH}\n")
print(f"{'version':24} {'ms/step':>9} {'vs true fp16':>13}")
for (label, mode) in VERS:
    mean, mn = run(mode)
    if fp16 is None: fp16 = mean
    sp = fp16 / mean
    print(f"{label:24} {mean:9.1f} {sp:12.2f}x")
    rows.append(dict(version=label, mean_ms=round(mean, 1), min_ms=round(mn, 1), speedup_vs_true_fp16=round(sp, 3)))
with open("docs/flash_attention_2026-07-19/data/e2e_true_fp16_table_b128.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("\nWROTE data/e2e_true_fp16_table_b128.csv")

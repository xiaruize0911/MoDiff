"""E2E item 1 — DDIM sampling speed across the 5 modes (measured, b128, 200 steps).

fp16 / int8_baseline / int4_baseline / int8_modiff / int4_modiff. int8/int4 use fused-flash
quantized attention by default (MODIFF_QUANT_ATTN=1). autocast fp16 ON for all modes (fair true-fp16
baseline). GPU clock burn-in, 30 warmup steps, then 5 rounds x 200 timed steps, synchronize around
each timed region; report mean + min ms/step. Writes data/e2e_speed.csv.
"""
import os, sys, csv, time, statistics
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
import torch
import integration.benchmarks.benchmark_ldm as B

BATCH = 128
WARMUP, TIMED, RUNS = 50, 200, 7
HERE = "docs/benchmark_5mode_2026-07-23"
# (report label, _setup_model mode string)
VERS = [("fp16", "fp16"),
        ("int8_baseline", "int8_baseline"),
        ("int4_baseline", "int4_baseline"),
        ("int8_modiff", "int8"),
        ("int4_modiff", "int4")]


def run(mode):
    quant = mode != "fp16"
    os.environ["MODIFF_QUANT_LINEAR"] = "1" if quant else "0"
    os.environ["MODIFF_QUANT_ATTN"] = "1" if quant else "0"   # flash quant attention (default) for int8/int4
    os.environ["MODIFF_LINEAR_OUT_I8"] = "0"; os.environ.pop("MODIFF_FLASH_ATTN", None)
    calib = "integration/calibration/int8_calibration.pt" if "int8" in mode else \
            ("integration/calibration/int4_calibration.pt" if "int4" in mode else None)
    r = B.BenchmarkRunner("configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
                          "models/ldm/lsun_churches256/model.ckpt", output_dir="integration/results/bench5mode",
                          batch_size=BATCH, steps=TIMED, shape=(4, 32, 32), calibration_path=calib,
                          linear_backend=("int_gemm" if quant else "fp16"))
    model, sampler = r._setup_model(mode); cond = r._cond_kwargs(model, BATCH)

    def smp(S):
        with torch.inference_mode(), torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
            sampler.sample(S=S, batch_size=BATCH, shape=r.shape, eta=0.0, verbose=False, **cond)

    smp(WARMUP); torch.cuda.synchronize()
    ms = []
    for _ in range(RUNS):
        torch.cuda.synchronize(); t0 = time.time(); smp(TIMED); torch.cuda.synchronize()
        ms.append((time.time() - t0) / TIMED * 1000)
    del model, sampler; torch.cuda.empty_cache()
    return statistics.mean(ms), min(ms)


# GPU clock burn-in so the first (fp16) run isn't cold
bn = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
for _ in range(60):
    bn = bn @ bn * 1e-4 + 1.0
torch.cuda.synchronize()

rows = []; fp16 = None
print(f"E2E speed @ b{BATCH}  (30 warm + {RUNS}x{TIMED} steps, autocast fp16 all, flash quant attn)\n")
print(f"{'mode':16} {'ms/step':>9} {'min':>8} {'vs fp16':>9}")
for (label, mode) in VERS:
    mean, mn = run(mode)
    if fp16 is None:
        fp16 = mean
    sp = fp16 / mean
    print(f"{label:16} {mean:9.2f} {mn:8.2f} {sp:8.3f}x")
    rows.append(dict(mode=label, ms_step=round(mean, 2), min_ms=round(mn, 2), speedup_vs_fp16=round(sp, 3)))

with open(f"{HERE}/data/e2e_speed.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"\nWROTE {HERE}/data/e2e_speed.csv")

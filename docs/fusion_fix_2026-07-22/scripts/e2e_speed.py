"""E2E DDIM sampling speed across the 5 modes, CURRENT (fusion-fixed) code state, b128.

Same methodology as benchmark_5mode/e2e_speed.py (GPU clock burn-in, 30 warm + 5x200 timed steps,
synchronize around each timed region; mean + min ms/step, autocast fp16 all, flash quant attn) so
the numbers are directly comparable to benchmark_5mode_2026-07-21/data/e2e_speed.csv (the
"before fusion-fix" baseline). The landed fusions are ON by default: Phase 2 o_hat deep-fuse
(MODIFF_DEEPFUSE_OHAT) and Phase 5 int4 GN->pack (MODIFF_FUSE_GN_QKV_I4). Writes data/e2e_speed.csv.
"""
import os, sys, csv, time, statistics
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
import torch
import integration.benchmarks.benchmark_ldm as B

BATCH = 128
WARMUP, TIMED, RUNS = 30, 200, 5
HERE = "docs/fusion_fix_2026-07-22"
BEFORE = {"fp16": 188.1, "int8_baseline": 121.0, "int4_baseline": 114.9, "int8_modiff": 136.8, "int4_modiff": 141.5}
VERS = [("fp16", "fp16"), ("int8_baseline", "int8_baseline"), ("int4_baseline", "int4_baseline"),
        ("int8_modiff", "int8"), ("int4_modiff", "int4")]


def run(mode):
    quant = mode != "fp16"
    os.environ["MODIFF_QUANT_LINEAR"] = "1" if quant else "0"
    os.environ["MODIFF_QUANT_ATTN"] = "1" if quant else "0"
    os.environ["MODIFF_LINEAR_OUT_I8"] = "0"; os.environ.pop("MODIFF_FLASH_ATTN", None)
    calib = "integration/calibration/int8_calibration.pt" if "int8" in mode else \
            ("integration/calibration/int4_calibration.pt" if "int4" in mode else None)
    r = B.BenchmarkRunner("configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
                          "models/ldm/lsun_churches256/model.ckpt", output_dir="integration/results/fusionfix",
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


bn = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
for _ in range(60): bn = bn @ bn * 1e-4 + 1.0
torch.cuda.synchronize()

rows = []; fp16 = None
print(f"E2E speed @ b{BATCH} (30 warm + {RUNS}x{TIMED} steps) -- CURRENT fusion-fixed state\n")
print(f"{'mode':16} {'ms/step':>9} {'min':>8} {'vs fp16':>9} {'before':>8} {'Δ vs before':>12}")
for (label, mode) in VERS:
    mean, mn = run(mode)
    if fp16 is None: fp16 = mean
    sp = fp16 / mean
    b0 = BEFORE.get(label)
    dlt = (b0 - mean) if b0 else 0.0
    print(f"{label:16} {mean:9.2f} {mn:8.2f} {sp:8.3f}x {b0:8.1f} {dlt:+11.2f}")
    rows.append(dict(mode=label, ms_step=round(mean, 2), min_ms=round(mn, 2), speedup_vs_fp16=round(sp, 3),
                     before_ms_step=b0, delta_vs_before=round(dlt, 2)))

with open(f"{HERE}/data/e2e_speed.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"\nWROTE {HERE}/data/e2e_speed.csv")

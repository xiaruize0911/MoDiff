"""E2E test of the int8-score attention fix on int8_baseline (AWQ w8a8 linear + quantized dynamic
attention). Compares 3 configs at batch 64: fp16 (ref); int8_baseline + dynamic quant attn (fp16
scores); int8_baseline + dynamic quant attn with the int8-SCORE fix (MODIFF_ATTN_S8_SCORE=1).
Reports wall ms/step + rel-L2 vs fp16. Writes data/e2e_s8score_b64.csv."""
import os, sys, csv, time, statistics
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
import torch
import integration.benchmarks.benchmark_ldm as B

BATCH = 64; STEPS, WARM_S, RUNS = 20, 3.0, 6
torch.manual_seed(0)
X = torch.randn(BATCH, 4, 32, 32, device="cuda"); TT = torch.randint(0, 1000, (BATCH,), device="cuda")
fp16_out = [None]


def run(label, mode, backend, qattn, s8):
    if backend == "int_gemm": os.environ["MODIFF_QUANT_LINEAR"] = "1"
    else: os.environ.pop("MODIFF_QUANT_LINEAR", None)
    os.environ["MODIFF_QUANT_ATTN"] = "1" if qattn else "0"
    os.environ["MODIFF_QUANT_ATTN_STATIC"] = "0"          # dynamic softmax (quality-safe)
    os.environ["MODIFF_ATTN_S8_SCORE"] = "1" if s8 else "0"
    calib = "integration/calibration/int8_calibration.pt" if "int8" in mode else None
    r = B.BenchmarkRunner("configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
                          "models/ldm/lsun_churches256/model.ckpt", output_dir="integration/results/e2e_s8",
                          batch_size=BATCH, steps=STEPS, shape=(4, 32, 32), calibration_path=calib, linear_backend=backend)
    model, sampler = r._setup_model(mode); cond = r._cond_kwargs(model, BATCH); dm = model.model.diffusion_model

    def smp(S):
        with torch.inference_mode(), torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
            sampler.sample(S=S, batch_size=BATCH, shape=r.shape, eta=0.0, verbose=False, **cond)
    tw = time.time()
    while time.time() - tw < WARM_S: smp(STEPS)     # warms + calibrates sS
    torch.cuda.synchronize()
    with torch.inference_mode(), torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
        out = dm(X, TT).float()
    rel = 0.0 if label == "fp16" else ((out - fp16_out[0]).norm() / fp16_out[0].norm()).item()
    if label == "fp16": fp16_out[0] = out.clone()
    ms = []
    for _ in range(RUNS):
        torch.cuda.synchronize(); t0 = time.time(); smp(STEPS); torch.cuda.synchronize()
        ms.append((time.time() - t0) / STEPS * 1000)
    wall = min(ms)
    print(f"  {label:28s} wall={wall:7.2f} ms/step  rel_vs_fp16={rel:.4f}")
    del model, sampler; torch.cuda.empty_cache()
    return wall, rel


rows = []
for (label, mode, backend, qattn, s8) in [
    ("fp16", "fp16", "fp16", False, False),
    ("int8_baseline dyn-attn (fp16 S)", "int8_baseline", "int_gemm", True, False),
    ("int8_baseline dyn-attn (int8 S fix)", "int8_baseline", "int_gemm", True, True),
]:
    wall, rel = run(label, mode, backend, qattn, s8)
    rows.append(dict(config=label, wall_ms_step=round(wall, 3), rel_vs_fp16=round(rel, 4)))
with open("docs/layer_roofline_2026-07-19/data/e2e_s8score_b64.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
fp = rows[0]["wall_ms_step"]
print("\n=== summary (× vs fp16) ===")
for r in rows: print(f"  {r['config']:36s} {r['wall_ms_step']:7.2f}  {fp/r['wall_ms_step']:.3f}×  rel={r['rel_vs_fp16']}")
print("WROTE e2e_s8score_b64.csv")

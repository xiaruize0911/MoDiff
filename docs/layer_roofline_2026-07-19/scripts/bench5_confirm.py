"""Re-run the 5-version e2e benchmark after removing flash from the code, to confirm no regression.
Default config: static quant, fused kernels (GN->int quantize qkv, transpose+quant proj, int4 gap
fusions), fp16 SDPA attention. b128, 30 warm-up + 5x200 steps, MEAN. Compares to prior numbers."""
import os, sys, csv, time, statistics
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
import torch
import integration.benchmarks.benchmark_ldm as B

BATCH = 128; WARMUP, TIMED, RUNS = 30, 200, 5
# prior standard-config means (bench5_outi8_b128.csv "fix OFF" = same config) for regression check
PRIOR = {"fp16": 188.0, "int8_baseline": 178.1, "int8_modiff": 201.1, "int4_baseline": 177.7, "int4_modiff": 203.7}
VERS = [("fp16", "fp16"), ("int8_baseline", "int8_baseline"), ("int8_modiff", "int8"),
        ("int4_baseline", "int4_baseline"), ("int4_modiff", "int4")]


def run(mode):
    quant = mode != "fp16"
    os.environ["MODIFF_QUANT_LINEAR"] = "1" if quant else "0"
    os.environ["MODIFF_QUANT_ATTN"] = "0"          # fp16 SDPA attention (default best)
    os.environ["MODIFF_LINEAR_OUT_I8"] = "0"       # default (fusions engage instead)
    os.environ.pop("MODIFF_FUSE_QKV_QUANT", None); os.environ.pop("MODIFF_FUSE_PROJ_QUANT", None)  # default ON
    calib = "integration/calibration/int8_calibration.pt" if "int8" in mode else \
            ("integration/calibration/int4_calibration.pt" if "int4" in mode else None)
    r = B.BenchmarkRunner("configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
                          "models/ldm/lsun_churches256/model.ckpt", output_dir="integration/results/b5confirm",
                          batch_size=BATCH, steps=TIMED, shape=(4, 32, 32), calibration_path=calib,
                          linear_backend=("int_gemm" if quant else "fp16"))
    model, sampler = r._setup_model(mode); cond = r._cond_kwargs(model, BATCH)

    def smp(S):
        # autocast fp16 ON for ALL modes. FIXED 2026-07-20: was enabled=quant, which ran the fp16
        # baseline in fp32/tf32 -> the reported "int8 ~2x" was ~1.85x precision (fp32->fp16) x ~1.08x
        # quantization. With this fix the fp16 baseline is true fp16 and int8 is ~1.08x. See
        # docs/flash_attention_2026-07-19/scripts/true_fp16_vs_int8.py.
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
print(f"b{BATCH}, {WARMUP} warmup + {RUNS}x{TIMED} steps, MEAN ms/step (post flash-removal)\n")
for (label, mode) in VERS:
    mean, mn = run(mode)
    prior = PRIOR[label]; delta = (mean - prior) / prior * 100
    flag = "OK" if delta < 1.5 else "REGRESSED?"
    print(f"  {label:16s} mean={mean:7.2f}  min={mn:7.2f}  (prior {prior:.1f}, {delta:+.1f}%)  {flag}")
    rows.append(dict(version=label, mean_ms=round(mean, 3), min_ms=round(mn, 3), prior_ms=prior, delta_pct=round(delta, 2)))
fp = rows[0]["mean_ms"]
with open("docs/layer_roofline_2026-07-19/data/bench5_confirm_b128.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["version", "mean_ms", "min_ms", "prior_ms", "delta_pct"]); w.writeheader(); w.writerows(rows)
print(f"\n=== vs fp16 ===")
for r in rows: print(f"  {r['version']:16s} {r['mean_ms']:7.2f}  {fp/r['mean_ms']:.3f}x")
print("WROTE bench5_confirm_b128.csv")

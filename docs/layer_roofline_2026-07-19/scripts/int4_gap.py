"""Verify + benchmark closing the int4 unfused-quantize gap (int4 GN->pack qkv fusion +
int4 transpose+pack proj fusion). int4_baseline at b128: fusions OFF vs ON, fused-layer count,
quality rel-L2 vs fp16, and wall (30 warm-up + 5x200 steps, MEAN). Writes data/int4_gap_b128.csv."""
import os, sys, csv, time, statistics
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
import torch
import integration.benchmarks.benchmark_ldm as B

BATCH = 128; WARMUP, TIMED, RUNS = 30, 200, 5
torch.manual_seed(0); X = torch.randn(BATCH, 4, 32, 32, device="cuda"); TT = torch.randint(0, 1000, (BATCH,), device="cuda")
fp16_out = [None]


def run(label, mode, fuse):
    os.environ["MODIFF_QUANT_LINEAR"] = "1" if mode != "fp16" else "0"
    os.environ["MODIFF_QUANT_ATTN"] = "0"; os.environ["MODIFF_LINEAR_OUT_I8"] = "0"
    os.environ["MODIFF_FUSE_QKV_QUANT"] = "1" if fuse else "0"
    os.environ["MODIFF_FUSE_PROJ_QUANT"] = "1" if fuse else "0"
    calib = "integration/calibration/int4_calibration.pt" if "int4" in mode else \
            ("integration/calibration/int8_calibration.pt" if "int8" in mode else None)
    backend = "int_gemm" if mode != "fp16" else "fp16"
    r = B.BenchmarkRunner("configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
                          "models/ldm/lsun_churches256/model.ckpt", output_dir="integration/results/i4gap",
                          batch_size=BATCH, steps=TIMED, shape=(4, 32, 32), calibration_path=calib, linear_backend=backend)
    model, sampler = r._setup_model(mode); cond = r._cond_kwargs(model, BATCH); dm = model.model.diffusion_model
    from integration.fused_ops.token_major_attention import TokenMajorAttentionBlock
    from integration.kernels.wxax_linear import QuantLinearWxAx
    # count how many attention qkv/proj actually take the fused branch (bits eligible + K unpadded)
    nqf = nprf = 0
    for m in model.model.diffusion_model.modules():
        if isinstance(m, TokenMajorAttentionBlock):
            if fuse and isinstance(m.qkv, QuantLinearWxAx) and m.qkv.a_scale is not None and m.qkv._awqt_K == m.qkv.in_features: nqf += 1
            if fuse and isinstance(m.proj, QuantLinearWxAx) and m.proj.a_scale is not None and m.proj._awqt_K == m.proj.in_features: nprf += 1

    def smp(S):
        with torch.inference_mode(), torch.amp.autocast('cuda', enabled=(mode != "fp32"), dtype=torch.float16):
            sampler.sample(S=S, batch_size=BATCH, shape=r.shape, eta=0.0, verbose=False, **cond)
    smp(WARMUP); torch.cuda.synchronize()
    with torch.inference_mode(), torch.amp.autocast('cuda', enabled=(mode != "fp32"), dtype=torch.float16):
        out = dm(X, TT).float()
    rel = 0.0 if label == "fp16" else ((out - fp16_out[0]).norm() / fp16_out[0].norm()).item()
    if label == "fp16": fp16_out[0] = out.clone()
    ms = []
    for _ in range(RUNS):
        torch.cuda.synchronize(); t0 = time.time(); smp(TIMED); torch.cuda.synchronize()
        ms.append((time.time() - t0) / TIMED * 1000)
    mean = statistics.mean(ms)
    print(f"  {label:28s} fused(qkv/proj)={nqf}/{nprf}  mean={mean:7.2f} ms/step  rel_vs_fp16={rel:.4f}")
    del model, sampler; torch.cuda.empty_cache()
    return mean, rel, nqf, nprf


rows = []
for (label, mode, fuse) in [("fp16", "fp16", False),
                            ("int8_baseline fuse-on", "int8_baseline", True),
                            ("int4_baseline fuse-OFF", "int4_baseline", False),
                            ("int4_baseline fuse-ON", "int4_baseline", True)]:
    mean, rel, nqf, nprf = run(label, mode, fuse)
    rows.append(dict(config=label, mean_ms=round(mean, 3), rel_vs_fp16=round(rel, 4), qkv_fused=nqf, proj_fused=nprf))
with open("docs/layer_roofline_2026-07-19/data/int4_gap_b128.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
off = next(r["mean_ms"] for r in rows if "fuse-OFF" in r["config"])
on = next(r["mean_ms"] for r in rows if "fuse-ON" in r["config"] and "int4" in r["config"])
print(f"\n=== int4 gap close: OFF {off:.2f} -> ON {on:.2f} ms/step  = {off/on:.4f}x ({(off/on-1)*100:+.2f}%) ===")
print("WROTE int4_gap_b128.csv")

"""Verify the in-model fused GN->qkv-quantize (MODIFF_FUSE_GN_QKV_I8): same int8 model, flag OFF vs ON.
Checks (a) output correctness — rel-L2 of the final DDIM latent, ON vs OFF, from identical noise; and
(b) e2e ms/step. int8 only (the fused int8 path). Writes data/fuse_gn_qkv_e2e.csv."""
import os, sys, csv, time, statistics
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
import torch
import integration.benchmarks.benchmark_ldm as B

BATCH = 128
HERE = "docs/benchmark_5mode_2026-07-21"


def setup(mode):
    quant = mode != "fp16"
    os.environ["MODIFF_QUANT_LINEAR"] = "1"; os.environ["MODIFF_QUANT_ATTN"] = "1"
    os.environ["MODIFF_LINEAR_OUT_I8"] = "0"; os.environ.pop("MODIFF_FLASH_ATTN", None)
    calib = "integration/calibration/int8_calibration.pt"
    r = B.BenchmarkRunner("configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
                          "models/ldm/lsun_churches256/model.ckpt", output_dir="integration/results/bench5mode",
                          batch_size=BATCH, steps=50, shape=(4, 32, 32), calibration_path=calib, linear_backend="int_gemm")
    model, sampler = r._setup_model(mode); cond = r._cond_kwargs(model, BATCH)
    return r, model, sampler, cond


def sample(r, sampler, cond, S, seed):
    torch.manual_seed(seed)
    with torch.inference_mode(), torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
        out = sampler.sample(S=S, batch_size=BATCH, shape=r.shape, eta=0.0, verbose=False, **cond)
    return out[0] if isinstance(out, (tuple, list)) else out


def relL2(a, b): return (a.float() - b.float()).norm().item() / (b.float().norm().item() + 1e-9)


def ms_step(r, sampler, cond, S=200, warm=30, runs=3):
    torch.manual_seed(0)
    with torch.inference_mode(), torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
        sampler.sample(S=warm, batch_size=BATCH, shape=r.shape, eta=0.0, verbose=False, **cond)
    torch.cuda.synchronize()
    ms = []
    for _ in range(runs):
        torch.cuda.synchronize(); t0 = time.time()
        with torch.inference_mode(), torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
            sampler.sample(S=S, batch_size=BATCH, shape=r.shape, eta=0.0, verbose=False, **cond)
        torch.cuda.synchronize(); ms.append((time.time() - t0) / S * 1000)
    return min(ms)


bn = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
for _ in range(60): bn = bn @ bn * 1e-4 + 1.0
torch.cuda.synchronize()

rows = []
mode = "int8_baseline"
print(f"=== {mode}: fused GN->qkv-quantize OFF vs ON ===")
r, model, sampler, cond = setup(mode)

os.environ["MODIFF_FUSE_GN_QKV_I8"] = "0"; out_off = sample(r, sampler, cond, S=50, seed=1234)
os.environ["MODIFF_FUSE_GN_QKV_I8"] = "1"; out_on = sample(r, sampler, cond, S=50, seed=1234)
rl = relL2(out_on, out_off)
print(f"output rel-L2 (ON vs OFF, 50-step latent): {rl:.6f}")

os.environ["MODIFF_FUSE_GN_QKV_I8"] = "0"; t_off = ms_step(r, sampler, cond)
os.environ["MODIFF_FUSE_GN_QKV_I8"] = "1"; t_on = ms_step(r, sampler, cond)
print(f"e2e ms/step: OFF {t_off:.2f}  ON {t_on:.2f}  delta {t_off - t_on:+.2f}  ({t_off/t_on:.3f}x)")
rows.append(dict(mode=mode, output_relL2_on_vs_off=round(rl, 6), ms_step_off=round(t_off, 2),
                 ms_step_on=round(t_on, 2), delta_ms=round(t_off - t_on, 2), speedup=round(t_off / t_on, 3)))

with open(f"{HERE}/data/fuse_gn_qkv_e2e.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"WROTE {HERE}/data/fuse_gn_qkv_e2e.csv")

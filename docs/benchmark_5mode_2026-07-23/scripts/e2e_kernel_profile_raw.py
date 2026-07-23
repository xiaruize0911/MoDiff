"""E2E per-kernel timing profile — RAW, no bucketing. Every CUDA kernel's self_device_time_total,
normalized to ms/step, per mode. Union across modes, sorted by total. Writes data/e2e_kernel_profile_raw.csv."""
import os, sys, csv, time
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
import torch
from torch.profiler import profile, ProfilerActivity
import integration.benchmarks.benchmark_ldm as B

BATCH = 128
WARMUP, STEPS, NP = 40, 20, 5
HERE = "docs/benchmark_5mode_2026-07-23"
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
                          "models/ldm/lsun_churches256/model.ckpt", output_dir="integration/results/bench5mode",
                          batch_size=BATCH, steps=STEPS, shape=(4, 32, 32), calibration_path=calib,
                          linear_backend=("int_gemm" if quant else "fp16"))
    model, sampler = r._setup_model(mode); cond = r._cond_kwargs(model, BATCH)

    def smp(S):
        with torch.inference_mode(), torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
            sampler.sample(S=S, batch_size=BATCH, shape=r.shape, eta=0.0, verbose=False, **cond)

    smp(WARMUP); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        smp(NP * STEPS)
    torch.cuda.synchronize()
    per = {}
    for e in p.key_averages():
        t = e.self_device_time_total
        if t > 0:
            per[e.key] = per.get(e.key, 0.0) + t / (NP * STEPS) / 1000.0   # ms/step
    del model, sampler; torch.cuda.empty_cache()
    return per


bn = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
for _ in range(60):
    bn = bn @ bn * 1e-4 + 1.0
torch.cuda.synchronize()

MODE_LABELS = [lbl for lbl, _ in VERS]
data = {}
for (label, mode) in VERS:
    print(f"profiling {label} ...", flush=True)
    data[label] = run(mode)

names = set()
for d in data.values():
    names.update(d.keys())
rows = []
for n in names:
    vals = {m: data[m].get(n, 0.0) for m in MODE_LABELS}
    rows.append((n, vals, sum(vals.values())))
rows.sort(key=lambda x: -x[2])

with open(f"{HERE}/data/e2e_kernel_profile_raw.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["kernel"] + MODE_LABELS + ["total_all_modes"])
    for n, vals, tot in rows:
        w.writerow([n] + [round(vals[m], 4) for m in MODE_LABELS] + [round(tot, 4)])

print(f"\n{len(rows)} distinct CUDA kernels (ms/step per mode). Full CSV: {HERE}/data/e2e_kernel_profile_raw.csv\n")
hdr = f"{'kernel':<62} " + " ".join(f"{m:>13}" for m in MODE_LABELS)
print(hdr); print("-" * len(hdr))
for n, vals, tot in rows:
    disp = n if len(n) <= 60 else n[:57] + "..."
    print(f"{disp:<62} " + " ".join(f"{vals[m]:13.3f}" for m in MODE_LABELS))
print("\nRAW_PROFILE_DONE")

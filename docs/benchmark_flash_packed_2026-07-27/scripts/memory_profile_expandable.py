"""Same as memory_profile.py, but with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
to test whether PyTorch's allocator setting reduces the reserved/allocated gap found in
memory_profile.py (especially pronounced for _modiff modes: persistent per-layer cache
buffers fragment the default caching allocator).
"""
import os, sys, json, time, statistics
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
import torch
import integration.benchmarks.benchmark_ldm as B

BATCH = 128
WARMUP, TIMED, RUNS = 10, 30, 3
HERE = "docs/benchmark_flash_packed_2026-07-27"
VERS = [("fp16", "fp16"), ("int8_baseline", "int8_baseline"), ("int4_baseline", "int4_baseline"),
        ("int8_modiff", "int8"), ("int4_modiff", "int4")]


def run(mode):
    quant = mode != "fp16"
    os.environ["MODIFF_QUANT_LINEAR"] = "1" if quant else "0"
    os.environ["MODIFF_QUANT_ATTN"] = "1" if quant else "0"
    os.environ["MODIFF_LINEAR_OUT_I8"] = "0"; os.environ.pop("MODIFF_FLASH_ATTN", None)
    os.environ.pop("MODIFF_FLASH_PACKED", None)
    os.environ.pop("MODIFF_SDPA_BACKEND", None)
    calib = "integration/calibration/int8_calibration.pt" if "int8" in mode else \
            ("integration/calibration/int4_calibration.pt" if "int4" in mode else None)
    r = B.BenchmarkRunner("configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
                          "models/ldm/lsun_churches256/model.ckpt", output_dir=f"{HERE}/tmp_out",
                          batch_size=BATCH, steps=TIMED, shape=(4, 32, 32), calibration_path=calib,
                          linear_backend=("int_gemm" if quant else "fp16"))
    model, sampler = r._setup_model(mode); cond = r._cond_kwargs(model, BATCH)

    def smp(S):
        with torch.inference_mode(), torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
            sampler.sample(S=S, batch_size=BATCH, shape=r.shape, eta=0.0, verbose=False, **cond)

    smp(WARMUP); torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    ms = []
    for _ in range(RUNS):
        torch.cuda.synchronize(); t0 = time.time(); smp(TIMED); torch.cuda.synchronize()
        ms.append((time.time() - t0) / TIMED * 1000)
    peak_alloc_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    peak_reserved_mb = torch.cuda.max_memory_reserved() / (1024 ** 2)

    del model, sampler; torch.cuda.empty_cache()
    return dict(ms_step=round(statistics.mean(ms), 2),
                peak_alloc_mb=round(peak_alloc_mb, 1),
                peak_reserved_mb=round(peak_reserved_mb, 1))


bn = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
for _ in range(60):
    bn = bn @ bn * 1e-4 + 1.0
torch.cuda.synchronize()
del bn; torch.cuda.empty_cache()

results = {}
print(f"Peak memory (expandable_segments:True) @ b{BATCH}\n{'mode':16} {'ms/step':>9} {'peak_alloc(MB)':>15} {'peak_reserved(MB)':>18}")
for label, mode in VERS:
    r = run(mode)
    print(f"{label:16} {r['ms_step']:9.2f} {r['peak_alloc_mb']:15.1f} {r['peak_reserved_mb']:18.1f}")
    results[label] = r

with open(f"{HERE}/data/memory_profile_expandable.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nWROTE {HERE}/data/memory_profile_expandable.json")

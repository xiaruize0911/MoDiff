"""Clean re-benchmark focused on identifying and timing every quantize-related
kernel. Fixes the cold-start confound seen in layer_profile_v3.py (fp16 and
int8_baseline, the first two modes in that run, came out ~2x their known-clean
values) by running one throwaway fp16 pass first (discarded) before any real
measurement -- this lets cudnn/flash-kernel algorithm search converge once,
process-wide, before anything is timed.
"""
import os, sys, json, time, statistics
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
import torch
from torch.profiler import profile, ProfilerActivity, DeviceType
import integration.benchmarks.benchmark_ldm as B

BATCH = 128
WARMUP, TIMED, RUNS = 30, 150, 5
PROF_STEPS = 15
HERE = "docs/benchmark_flash_packed_2026-07-27"
VERS = [("int8_baseline", "int8_baseline"), ("int4_baseline", "int4_baseline"),
        ("int8_modiff", "int8"), ("int4_modiff", "int4")]

QUANTIZE_KEYWORDS = ["aq_qtok", "aq_vscale", "aq_vquant", "aq_kquant", "scale_quantize",
                     "quantize_and_pack", "quantize_act_int8", "gn_apply_delta_quantize",
                     "gn_group_stats", "static_quantize_and_update_ahat", "from_i8_qtok",
                     "from_i8_vtrans", "group_norm_silu_quantize", "group_norm_silu_delta_quantize",
                     "upsample2x_quantize"]


def is_quantize_kernel(name):
    low = name.lower()
    return any(k.lower() in low for k in QUANTIZE_KEYWORDS)


def run(mode):
    quant = mode != "fp16"
    os.environ["MODIFF_QUANT_LINEAR"] = "1" if quant else "0"
    os.environ["MODIFF_QUANT_ATTN"] = "1" if quant else "0"
    os.environ["MODIFF_LINEAR_OUT_I8"] = "0"; os.environ.pop("MODIFF_FLASH_ATTN", None)
    os.environ.pop("MODIFF_FLASH_PACKED", None)
    os.environ.pop("MODIFF_SDPA_BACKEND", None)   # shipped default (math) -- this audit is about quantize kernels, not the SDPA question
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
    ms = []
    for _ in range(RUNS):
        torch.cuda.synchronize(); t0 = time.time(); smp(TIMED); torch.cuda.synchronize()
        ms.append((time.time() - t0) / TIMED * 1000)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        smp(PROF_STEPS)
        torch.cuda.synchronize()
    kernel_time, total = {}, 0.0
    for evt in prof.key_averages():
        if evt.device_type != DeviceType.CUDA:
            continue
        t = evt.self_device_time_total
        if t <= 0:
            continue
        kernel_time[evt.key] = kernel_time.get(evt.key, 0.0) + t
        total += t
    quantize_kernels = {k: v for k, v in kernel_time.items() if is_quantize_kernel(k)}
    quantize_total = sum(quantize_kernels.values())

    del model, sampler, prof; torch.cuda.empty_cache()
    return (statistics.mean(ms), min(ms), total, quantize_total,
            sorted(quantize_kernels.items(), key=lambda x: -x[1]))


bn = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
for _ in range(60):
    bn = bn @ bn * 1e-4 + 1.0
torch.cuda.synchronize()

print("throwaway warmup pass (discarded, lets algo search converge)...")
_ = run("int8_baseline")
print("  done\n")

results = {}
print(f"{'mode':16} {'ms/step':>9} {'min':>8} {'quantize %':>11}")
for label, mode in VERS:
    mean, mn, total, qtotal, qkernels = run(mode)
    pct = qtotal / total * 100 if total else 0
    print(f"{label:16} {mean:9.2f} {mn:8.2f} {pct:10.2f}%")
    results[label] = dict(ms_step=round(mean, 2), min_ms=round(mn, 2),
                          quantize_pct_of_total=round(pct, 2),
                          quantize_kernels_us=[dict(name=n[:150], us=round(v, 1), pct_of_total=round(v/total*100, 3)) for n, v in qkernels])

with open(f"{HERE}/data/quantize_kernel_audit.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nWROTE {HERE}/data/quantize_kernel_audit.json")

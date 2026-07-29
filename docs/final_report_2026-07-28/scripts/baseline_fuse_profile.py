"""Per-kernel time breakdown for the two BASELINE modes, next to the fp16 counterpart.

Why this exists separately from profile_tree_by_caller.py: that script force-sets
MODIFF_QUANT_LINEAR=1 for every quantized mode, and that switch in turn force-disables
MODIFF_FUSE_GN_QKV (benchmark_ldm.py forces it off so the quantized-linear path owns the qkv
projection). So its numbers describe a DIFFERENT configuration than the one you get by running
the benchmark normally. Both configurations are real, so this script measures both:

  linear=fp16   -- the default: --linear_backend fp16, MODIFF_QUANT_LINEAR unset. Convs and
                   attention are quantized; every nn.Linear (qkv, proj, emb) runs fp16 cuBLAS.
                   GroupNorm+qkv IS fused (fused_gn_qkv). The proj-side quantize/GEMM/residual
                   fusions are all dormant because they require a _QuantLinearWxAx instance.
  linear=int    -- MODIFF_QUANT_LINEAR=1 + linear_backend=int_gemm: linears become int8/int4
                   AWQ GEMMs, which switches ON the proj quantize+GEMM+bias+residual fusion and
                   switches OFF the GroupNorm+qkv fusion.

Output is per-KERNEL (not per-layer-type): kernel name, ms/step, % of step, and for each
quantized kernel the fp16 kernel(s) that occupy the same position in the pipeline.
"""
import collections
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")

import torch
from torch.profiler import ProfilerActivity, profile

import integration.benchmarks.benchmark_ldm as B

HERE = "docs/final_report_2026-07-28"
BATCH, WARMUP, TIMED, RUNS, PROF_STEPS = 128, 30, 150, 3, 10

# (label, mode, quant_linear)
VERS = [
    ("fp16",                 "fp16",          False),
    ("int8_baseline",        "int8_baseline",  False),
    ("int4_baseline",        "int4_baseline",  False),
    ("int8_baseline_qlin",   "int8_baseline",  True),
    ("int4_baseline_qlin",   "int4_baseline",  True),
]


def short(name):
    """Collapse a mangled/templated kernel symbol to a readable identity.

    Keeps distinct CUTLASS GEMM/conv configs distinct (they are different kernels with very
    different performance), but strips template noise. Mirrors profile_tree_by_caller.py's
    short_kernel_name so the two reports name the same kernel the same way.
    """
    n = name
    for pre in ("void ", "at::native::"):
        if n.startswith(pre):
            n = n[len(pre):]
    # generic launchers wrap the real functor in <...>
    for launcher in ("cutlass::device_kernel<", "cutlass::Kernel2<", "cutlass::Kernel<",
                     "elementwise_kernel<", "vectorized_elementwise_kernel<",
                     "unrolled_elementwise_kernel<"):
        if launcher in n:
            n = n.split(launcher, 1)[1]
    n = n.split("(")[0]
    # ATen functors: keep the functor name, drop the lambda/type soup
    for marker in ("::{lambda", "<::"):
        n = n.split(marker)[0]
    if n.startswith("_Z"):                     # still mangled: take the first name-like token
        i = 2
        while i < len(n) and n[i].isdigit():
            i += 1
        ln = n[2:i]
        if ln.isdigit():
            n = n[i:i + int(ln)]
    n = n.split("<")[0].strip(" &*:")
    return n[:58] if n else name[:58]


def run(mode, quant_linear):
    quant = mode != "fp16"
    if quant and quant_linear:
        os.environ["MODIFF_QUANT_LINEAR"] = "1"
        os.environ["MODIFF_QUANT_ATTN"] = "1"
    else:
        os.environ.pop("MODIFF_QUANT_LINEAR", None)
        os.environ.pop("MODIFF_QUANT_ATTN", None)
    os.environ["MODIFF_LINEAR_OUT_I8"] = "0"
    for k in ("MODIFF_FLASH_ATTN", "MODIFF_FLASH_PACKED", "MODIFF_SDPA_BACKEND",
              "MODIFF_FLASH_GATE"):
        os.environ.pop(k, None)
    calib = ("integration/calibration/int8_calibration.pt" if "int8" in mode else
             "integration/calibration/int4_calibration.pt" if "int4" in mode else None)
    r = B.BenchmarkRunner("configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
                          "models/ldm/lsun_churches256/model.ckpt", output_dir=f"{HERE}/tmp_out",
                          batch_size=BATCH, steps=TIMED, shape=(4, 32, 32),
                          calibration_path=calib,
                          linear_backend=("int_gemm" if (quant and quant_linear) else "fp16"))
    model, sampler = r._setup_model(mode)
    cond = r._cond_kwargs(model, BATCH)

    def smp(S):
        with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
            sampler.sample(S=S, batch_size=BATCH, shape=r.shape, eta=0.0, verbose=False, **cond)

    smp(WARMUP)
    torch.cuda.synchronize()
    ms = []
    for _ in range(RUNS):
        torch.cuda.synchronize()
        t0 = time.time()
        smp(TIMED)
        torch.cuda.synchronize()
        ms.append((time.time() - t0) / TIMED * 1000)
    # UNPROFILED wall clock: the profiler adds per-launch overhead that scales with kernel
    # count, so a profiled window is not a valid denominator for a GPU-busy fraction.
    mean_ms = statistics.mean(ms)

    trace = f"/tmp/trace_bfp_{mode}_{int(quant_linear)}.json"
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        smp(PROF_STEPS)
        torch.cuda.synchronize()
    prof.export_chrome_trace(trace)
    ev = json.load(open(trace))["traceEvents"]
    os.remove(trace)

    agg = collections.defaultdict(lambda: {"us": 0.0, "n": 0})
    for e in ev:
        if e.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        dur = e.get("dur", 0.0)
        if dur <= 0:
            continue
        k = short(e["name"]) if e.get("cat") == "kernel" else "[memcpy/memset]"
        agg[k]["us"] += dur
        agg[k]["n"] += 1

    kern = {k: {"ms_step": v["us"] / 1e3 / PROF_STEPS,
                "calls_per_step": v["n"] / PROF_STEPS} for k, v in agg.items()}
    gpu_ms = sum(v["ms_step"] for v in kern.values())
    for v in kern.values():
        v["pct_of_gpu"] = 100.0 * v["ms_step"] / gpu_ms if gpu_ms else 0.0
        v["pct_of_step"] = 100.0 * v["ms_step"] / mean_ms if mean_ms else 0.0
    return {"ms_step": round(mean_ms, 2), "gpu_ms_step": round(gpu_ms, 2),
            "gpu_busy_pct": round(100.0 * gpu_ms / mean_ms, 1),
            "kernels": dict(sorted(kern.items(), key=lambda kv: -kv[1]["ms_step"]))}


def main():
    # settle clocks so the first mode is not measured on a cold GPU
    bn = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    for _ in range(60):
        bn = bn @ bn * 1e-4 + 1.0
    torch.cuda.synchronize()
    del bn
    torch.cuda.empty_cache()

    out = {}
    for label, mode, qlin in VERS:
        out[label] = run(mode, qlin)
        r = out[label]
        print(f"\n=== {label}: {r['ms_step']} ms/step | GPU busy {r['gpu_ms_step']} ms "
              f"({r['gpu_busy_pct']}%) ===")
        print(f"{'kernel':60s} {'ms/step':>8s} {'%GPU':>6s} {'calls':>7s}")
        for k, v in list(r["kernels"].items())[:22]:
            print(f"{k:60s} {v['ms_step']:8.2f} {v['pct_of_gpu']:6.1f} {v['calls_per_step']:7.1f}")

    with open(f"{HERE}/data/baseline_fuse_profile.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWROTE {HERE}/data/baseline_fuse_profile.json")


if __name__ == "__main__":
    main()

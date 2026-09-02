"""Kernel-time split: W8A8 fp16 a_hat vs along-C B=32 int8. 20 DDIM steps, batch 128."""
from __future__ import annotations
import os, sys, re
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers")]
os.environ.setdefault("MODIFF_DELTA_MODE", "static")
os.environ["MODIFF_LINEAR"] = "0"
os.environ["MODIFF_CACHE_SKIP_K"] = "1"
os.environ["MODIFF_REPLAY_K"] = "1"
os.environ["MODIFF_AHAT_BITS"] = "16"
os.environ["MODIFF_AHAT_REFRESH"] = "0"
os.environ["MODIFF_IMODE"] = "0"
os.environ["MODIFF_AHAT_BLOCK"] = "0"

from integration.utils.preflight import preflight, MODEL
preflight(*MODEL, what="block32_why_slower.py")
import torch
from torch.profiler import profile, ProfilerActivity, DeviceType
import integration.benchmarks.benchmark_ldm as B

# 5 steps let the t=T first step (which has no a_hat read) dominate and understated
# the per-step delta by ~10x against the 50-step wall clock; 20 tracks it.
SHAPE, BATCH, STEPS = (4, 32, 32), 128, int(os.environ.get("PROF_STEPS", "20"))


def bucket(name: str) -> str:
    n = name.lower()
    if "ahat_commit" in n:
        return "ahat_commit_block"
    if "gn_apply_delta" in n or "gn_apply" in n:
        return "gn_apply (delta+a_hat)"
    if "gn_stats" in n or "gn_group_stats" in n or "gn_delta_absmax" in n:
        return "gn_stats/absmax"
    if "resize" in n or "gndqr" in n:
        return "gn_resize"
    if "upsample2x" in n:
        return "upsample2x_quant"
    if "static_quantize_and_update_ahat" in n or "update_ahat" in n:
        return "step1 quant+a_hat"
    if "cutlass" in n or "implicit_gemm" in n or "conv2d" in n:
        return "conv"
    if "flash" in n or "attn" in n:
        return "attn"
    return "other"


def set_block(b: int) -> None:
    os.environ["MODIFF_AHAT_BLOCK"] = str(b)
    os.environ["MODIFF_AHAT_BITS"] = "16"
    os.environ["MODIFF_IMODE"] = "0"
    os.environ["MODIFF_AHAT_REFRESH"] = "0"


def reset(model) -> None:
    B.reset_modiff_state_int8(model.model.diffusion_model)
    B._reset_wxax_modiff_safe(model)


def sample(model, sampler):
    reset(model)
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        sampler.sample(S=STEPS, batch_size=BATCH, shape=SHAPE, eta=0.0, verbose=False)


def prof_arm(model, sampler, label):
    sample(model, sampler)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        sample(model, sampler)
        torch.cuda.synchronize()
    buckets, kernels = {}, {}
    total = 0.0
    for evt in prof.key_averages():
        if evt.device_type != DeviceType.CUDA:
            continue
        t = evt.self_device_time_total / 1e3  # us -> ms for whole window
        if t <= 0:
            continue
        total += t
        kernels[evt.key] = kernels.get(evt.key, 0.0) + t
        b = bucket(evt.key)
        buckets[b] = buckets.get(b, 0.0) + t
    print(f"\n===== {label}  GPU-kernel {total/STEPS:.2f} ms/step =====", flush=True)
    for k, v in sorted(buckets.items(), key=lambda kv: -kv[1]):
        print(f"  {k:28s} {v/STEPS:7.3f} ms/step  ({100*v/total:5.1f}%)", flush=True)
    print("  -- kernels with ahat/gn/quant in the name --", flush=True)
    for k, v in sorted(kernels.items(), key=lambda kv: -kv[1]):
        if re.search(r"ahat|gn_apply|gn_stats|quantize|commit|upsample2x|resize", k, re.I):
            short = k if len(k) < 90 else k[:87] + "..."
            print(f"    {v/STEPS:7.3f}  {short}", flush=True)
    return ({k: v / STEPS for k, v in buckets.items()}, total / STEPS,
            {k: v / STEPS for k, v in kernels.items()})


def main():
    print(f"GPU {torch.cuda.get_device_name(0)}  batch={BATCH} steps={STEPS}", flush=True)
    runner = B.BenchmarkRunner(
        config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
        ckpt_path="models/ldm/lsun_churches256/model.ckpt",
        output_dir="docs/ahat_blockwise_2026-09-01/tmp_why",
        batch_size=BATCH, steps=STEPS, shape=SHAPE,
        calibration_path=B._default_calibration_path("int8"),
        auto_delta_table=True)
    set_block(0)
    model, sampler = runner._setup_model("int8")
    recs = []
    for block, label in [(0, "fp16 a_hat"), (32, "int8 B=32")]:
        set_block(block)
        recs.append((label, *prof_arm(model, sampler, label)))
    print("\n===== per-kernel delta, top 20 (ms/step) =====", flush=True)
    kf, ki = recs[0][3], recs[1][3]
    for k in sorted(set(kf) | set(ki), key=lambda k: -abs(ki.get(k, 0) - kf.get(k, 0)))[:20]:
        a, b = kf.get(k, 0.0), ki.get(k, 0.0)
        short = k if len(k) < 84 else k[:81] + "..."
        print(f"  {b-a:+7.3f}  fp16 {a:7.3f}  i8 {b:7.3f}  {short}", flush=True)

    print("\n===== delta vs fp16 a_hat (ms/step) =====", flush=True)
    fp = recs[0][1]
    i8 = recs[1][1]
    keys = sorted(set(fp) | set(i8), key=lambda k: -(i8.get(k, 0) - fp.get(k, 0)))
    for k in keys:
        a, b = fp.get(k, 0.0), i8.get(k, 0.0)
        print(f"  {k:28s}  fp16 {a:7.3f}  i8 {b:7.3f}  Δ {b-a:+7.3f}", flush=True)
    print(f"  {'TOTAL':28s}  fp16 {recs[0][2]:7.3f}  i8 {recs[1][2]:7.3f}  "
          f"Δ {recs[1][2]-recs[0][2]:+7.3f}", flush=True)


if __name__ == "__main__":
    main()

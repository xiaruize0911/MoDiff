"""How much of the fp16 baseline is PyTorch, and how much is this repo?

The fp16 "baseline" every speedup in this project is measured against is NOT vanilla
PyTorch. benchmark_ldm._setup_model applies four things to it:

  1. FusedResBlock            -> replaces GroupNorm+SiLU with this repo's CUDA kernel
                                 (group_norm_silu_nhwc_kernel, ~21.6 ms/step in fp16)
  2. token-major AttentionBlock -> reshapes attention and pins it to the MATH SDPA backend
  3. convert_upsample_to_fused  -> wraps Upsample modules
  4. MODIFF_FP16_MATERIALIZED=1 -> forces materialized (MATH) attention
  plus channels_last for the whole model.

That matters because a baseline restructured by the thing being measured is not a neutral
reference. This script times fp16 with those switched off, one axis at a time, so the
contribution of each is visible instead of assumed:

  vanilla          stock nn.Module graph, PyTorch picks the SDPA backend (flash allowed)
  vanilla_math     same, but SDPA forced to MATH  -> isolates the SDPA-backend effect
  vanilla_nchw     vanilla without channels_last  -> isolates the layout effect
  repo_fp16        the current baseline (all four applied) for reference

Writes data/fp16_baseline_variants.json.
"""
import os, sys, json, time, statistics
os.chdir("/workspace/MoDiff")
sys.path.insert(0, "/workspace/MoDiff")
sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
import torch
from torch.profiler import profile, ProfilerActivity, DeviceType

HERE = "docs/final_report_2026-07-28"
BATCH, WARMUP, TIMED, RUNS = 128, 30, 150, 5
PROF_STEPS = 10


def build(variant):
    """Load the model and apply only what `variant` asks for."""
    import integration.benchmarks.benchmark_ldm as B
    for k in ("MODIFF_FP16_MATERIALIZED", "MODIFF_STATIC_SOFTMAX", "MODIFF_QUANT_LINEAR",
              "MODIFF_QUANT_ATTN", "MODIFF_FLASH_ATTN", "MODIFF_FLASH_PACKED",
              "MODIFF_SDPA_BACKEND"):
        os.environ.pop(k, None)
    os.environ["MODIFF_QUANT_LINEAR"] = "0"
    os.environ["MODIFF_QUANT_ATTN"] = "0"

    r = B.BenchmarkRunner("configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
                          "models/ldm/lsun_churches256/model.ckpt", output_dir=f"{HERE}/tmp_out",
                          batch_size=BATCH, steps=TIMED, shape=(4, 32, 32),
                          calibration_path=None, linear_backend="fp16")
    if variant == "repo_fp16":
        model, sampler = r._setup_model("fp16")
        return r, model, sampler

    # --- vanilla paths: bypass _setup_model's conversions entirely ---
    from integration.benchmarks.benchmark_ldm import load_model
    from ldm.models.diffusion.ddim import DDIMSampler
    model, _ = load_model(r.config_path, r.ckpt_path)
    torch.backends.cudnn.benchmark = True
    if variant != "vanilla_nchw":
        model = model.to(memory_format=torch.channels_last)
    # inference hygiene that is NOT a custom kernel: disable grad checkpointing
    for m in model.modules():
        if hasattr(m, "use_checkpoint"):
            m.use_checkpoint = False
    from ldm.modules.diffusionmodules.openaimodel import AttentionBlock
    AttentionBlock.forward = lambda self, x: self._forward(x)
    if variant == "vanilla_math":
        os.environ["MODIFF_FP16_MATERIALIZED"] = "1"
    sampler = DDIMSampler(model)
    return r, model, sampler


def measure(variant):
    r, model, sampler = build(variant)
    cond = r._cond_kwargs(model, BATCH)
    ctx = (torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False,
                                          enable_math=True)
           if variant == "vanilla_math" else None)

    def smp(S):
        with torch.inference_mode(), torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
            if ctx is not None:
                with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False,
                                                    enable_math=True):
                    sampler.sample(S=S, batch_size=BATCH, shape=r.shape, eta=0.0,
                                   verbose=False, **cond)
            else:
                sampler.sample(S=S, batch_size=BATCH, shape=r.shape, eta=0.0,
                               verbose=False, **cond)

    smp(WARMUP); torch.cuda.synchronize()
    ms = []
    for _ in range(RUNS):
        torch.cuda.synchronize(); t0 = time.time(); smp(TIMED); torch.cuda.synchronize()
        ms.append((time.time() - t0) / TIMED * 1000)
    mean_ms = statistics.mean(ms)

    # which kernels ran, and how much of the time is this repo's own code
    OURS = ("group_norm_silu", "gn_accum", "gn_finalize", "gn_group_stats", "gn_apply",
            "ImplicitGemmConvolutionFusionPerSample", "ImplicitGemmConvolutionEVT",
            "flash_attn_int", "aq_qtok", "aq_vquant", "cat2_channels_last", "upsample2x_quantize",
            "avgpool2x_quantize", "static_quantize", "gemm_w8a8", "gemm_w4a4", "scale_quantize",
            "quant_attn_out", "quant_act_int4", "layout_transform", "dequant_accumulate")
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        smp(PROF_STEPS)
        torch.cuda.synchronize()
    tot = ours = 0.0
    top = []
    for e in prof.key_averages():
        if e.device_type != DeviceType.CUDA or e.self_device_time_total <= 0:
            continue
        t = e.self_device_time_total
        tot += t
        if any(o.lower() in e.key.lower() for o in OURS):
            ours += t
        top.append((t / PROF_STEPS / 1000, e.key[:70], e.count / PROF_STEPS))
    top.sort(reverse=True)
    del model, sampler, prof
    torch.cuda.empty_cache()
    return dict(variant=variant, ms_step=round(mean_ms, 2),
                repo_kernel_ms_step=round(ours / tot * mean_ms, 3) if tot else 0.0,
                repo_kernel_pct=round(ours / tot * 100, 2) if tot else 0.0,
                top_kernels=[{"ms_step": round(a, 3), "kernel": b, "calls_per_step": round(c, 1)}
                             for a, b, c in top[:12]])


def main():
    bn = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    for _ in range(60):
        bn = bn @ bn * 1e-4 + 1.0
    torch.cuda.synchronize(); del bn; torch.cuda.empty_cache()

    out = {}
    for v in ("vanilla", "vanilla_math", "vanilla_nchw", "repo_fp16"):
        try:
            res = measure(v)
        except Exception as e:
            print(f"{v}: FAILED {e!r}")
            out[v] = {"variant": v, "error": repr(e)[:300]}
            continue
        out[v] = res
        print(f"{v:14s} {res['ms_step']:8.2f} ms/step   repo kernels "
              f"{res['repo_kernel_ms_step']:6.2f} ms ({res['repo_kernel_pct']:.1f}%)")
    with open(f"{HERE}/data/fp16_baseline_variants.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWROTE {HERE}/data/fp16_baseline_variants.json")


if __name__ == "__main__":
    main()

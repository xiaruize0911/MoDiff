"""Run the real sampler in all three modes and assert the updown fusion is both used and used
correctly.

This exists because of a bug that every kernel-level test missed. `group_norm_silu_quantize_
resize_nhwc` was verified standalone at all eight updown shapes, both directions, both
quantizations, against an fp32 reference -- all exact -- and INT4 ran end to end. But the INT8
consumer, `_conv_from_int8(x_q)`, reads its spatial extents off `x_q.shape[2]` and `x_q.shape[3]`,
so it needs a channels_last `[N, C, Ho, Wo]`, while the kernel returned a literal `[N, Ho, Wo, C]`.
Same bytes, same numbers, different logical shape: the conv took Wo for its height and C for its
width and read ~128 KiB past the end of the activation. Nothing that checked kernel OUTPUT VALUES
could see it -- only running the consumer could.

So the check here is deliberately not numerical. It is:

  1. the sampler completes in fp16, int8_baseline and int4_baseline without a CUDA fault
     (run under `compute-sanitizer --tool memcheck` to make out-of-bounds reads fatal rather
     than luck-dependent -- an OOB read only raises if it leaves the allocator's segment)
  2. the fused entry actually fires 8x per step, i.e. the eight updown ResBlocks took it
  3. the three kernels it replaced fire zero times, i.e. the old path is gone rather than
     merely bypassed

Batch 4 / 5 steps: this is a correctness and wiring check, not a benchmark.
"""
import os
import sys

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "integration/benchmarks/report"))

import torch
import modiff_cutlass as mc
import integration.benchmarks.benchmark_ldm as B
from kernel_suites_bench import set_env, CALIB

BATCH, STEPS = 4, 5
UPDOWN_BLOCKS = 8
FUSED = "group_norm_silu_quantize_resize_nhwc"
# Replaced by FUSED on this path. group_norm_silu_nhwc is still legitimately called by OTHER
# paths, so it is only required to be absent in the quantized modes where the fusion applies.
RETIRED = ["upsample2x_quantize_pack_noahat_fprop", "upsample2x_quantize_noahat_fprop",
           "avgpool2x_quantize_pack_noahat_fprop", "avgpool2x_quantize_noahat_fprop"]


def run(mode, counts):
    set_env(mode)
    r = B.BenchmarkRunner("configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
                          "models/ldm/lsun_churches256/model.ckpt",
                          output_dir="docs/final_report_2026-07-28/tmp_out",
                          batch_size=BATCH, steps=STEPS, shape=(4, 32, 32),
                          calibration_path=CALIB.get(mode), linear_backend="fp16")
    model, sampler = r._setup_model(mode)
    cond = r._cond_kwargs(model, BATCH)
    counts.clear()          # discard whatever the calibration/observer pass fired
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        sampler.sample(S=STEPS, batch_size=BATCH, shape=r.shape, eta=0.0, verbose=False, **cond)
    torch.cuda.synchronize()
    return dict(counts)


def main():
    counts = {}
    watched = [FUSED, "group_norm_silu_nhwc"] + RETIRED
    for name in watched:
        if not hasattr(mc, name):
            continue
        orig = getattr(mc, name)

        def wrap(orig=orig, name=name):
            def w(*a, **kw):
                counts[name] = counts.get(name, 0) + 1
                return orig(*a, **kw)
            return w
        setattr(mc, name, wrap())

    ok = True
    print("| mode | %s | retired kernels | verdict |" % FUSED)
    print("|---|---:|---:|---|")
    for mode in ("fp16", "int8_baseline", "int4_baseline"):
        c = run(mode, counts)
        fused = c.get(FUSED, 0)
        retired = sum(c.get(k, 0) for k in RETIRED)
        if mode == "fp16":
            # No calibration -> _prequant_gn_resize_conv declines and the fp16 two-step path
            # runs. Asserting that keeps the fusion from silently swallowing the fp16 mode.
            good = fused == 0
            note = "fp16: fusion correctly not taken"
        else:
            good = fused == UPDOWN_BLOCKS * STEPS and retired == 0
            note = "expect %d fused, 0 retired" % (UPDOWN_BLOCKS * STEPS)
        ok &= good
        print("| %s | %d | %d | %s -- %s |"
              % (mode, fused, retired, "PASS" if good else "FAIL", note))
    print("\n%s" % ("ALL PASS" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

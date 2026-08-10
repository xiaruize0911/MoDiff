"""Assert the updown fusion actually fires under MoDiff at MODIFF_DELTA_REFRESH=1.

The kernel-level test next door proves the dynamic-scale resize kernel is correct. This proves the
model reaches it, which is the half that was broken: `_prequant_gn_resize_conv_modiff` declined on
every refresh step, and at K=1 -- the paper's own configuration -- every step is a refresh, so all
eight updown ResBlocks ran the unfused four-kernel route. The 2026-08-07 traces recorded 6/8 fused
at K=4 and 0/8 at K=1.

So this is a counting test, in the same spirit as test_updown_fusion_pipeline.py:

  1. the sampler completes in MoDiff mode at K=1 and at K=4
  2. `group_norm_silu_delta_quantize_resize_nhwc` fires 8x per step at BOTH refresh settings

Counting the fused entry alone would not be enough: it was already firing 6/8 at K=4, so a test
that only checked "> 0" passed against the bug. It has to be 8/8, and the K=1 row is the one that
matters.

"Per step" is derived rather than assumed. The sampler runs its own number of timesteps (warm-up
and the DDIM schedule both move it), so the count is normalised against
`group_norm_silu_delta_quantize_nhwc`, which serves a fixed 62 conv layers every modulated step.
The ratio 8/62 is what is asserted, so the test cannot be fooled by a step count that shifts.

A "the unfused route is gone" counter was tried and dropped: `step1_static_quantize_fprop` and
`group_norm_silu_nhwc` both serve other layers (18 and several respectively), so neither is a
marker for these eight specifically. The ratio above is the unambiguous statement.

Batch 4 / 6 steps: wiring, not throughput. 6 steps so K=4 sees both a refresh and a reuse step
after the warm-up.

Run: python integration/tests/test_updown_fusion_modiff_k1.py
"""
import os
import sys

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "integration/benchmarks/report"))

import torch
import modiff_cutlass as mc

BATCH, STEPS = 4, 6
UPDOWN_BLOCKS = 8
FUSED = "group_norm_silu_delta_quantize_resize_nhwc"

# modiff_full_k1's environment, from
# docs/component_attribution_2026-08-07/data/differential_timing.json.
MODIFF_ENV = {
    "MODIFF_QUANT_LINEAR": "1", "MODIFF_QUANT_ATTN": "1", "MODIFF_QUANT_ATTN_STATIC": "1",
    "MODIFF_QATTN_FLASH": "1", "MODIFF_FLASH_GATE": "on", "MODIFF_QUANT_ATTN_ALLT": "0",
    "MODIFF_LINEAR_OUT_I8": "0", "MODIFF_FUSE_PROJ_QUANT": "1", "MODIFF_LINEAR": "1",
    "MODIFF_ACT_BITS": "8", "MODIFF_DELTA_MODE": "dynamic",
    "MODIFF_WARMUP_STEPS": "2",
}


def run(refresh, counts):
    for k, v in MODIFF_ENV.items():
        os.environ[k] = v
    os.environ["MODIFF_DELTA_REFRESH"] = str(refresh)
    for k in ("MODIFF_FLASH_ATTN", "MODIFF_FLASH_PACKED", "MODIFF_SDPA_BACKEND"):
        os.environ.pop(k, None)

    import integration.benchmarks.benchmark_ldm as B
    from kernel_suites_bench import CALIB
    r = B.BenchmarkRunner("configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
                          "models/ldm/lsun_churches256/model.ckpt",
                          output_dir="docs/component_attribution_2026-08-07/tmp_out",
                          batch_size=BATCH, steps=STEPS, shape=(4, 32, 32),
                          calibration_path=CALIB.get("int8"), linear_backend="fp16")
    model, sampler = r._setup_model("int8")
    cond = r._cond_kwargs(model, BATCH)

    # Count UNet invocations rather than assuming the DDIM step count. `_setup_model` already ran a
    # calibration pass and a warm-up pass, and the scheduler adds its own steps, so S=STEPS is not
    # the number of forwards -- and the fused count has to be divided by the real one.
    unet = model.model.diffusion_model
    orig_fwd = unet.forward
    box = {"n": 0}

    def counting_forward(*a, **kw):
        box["n"] += 1
        return orig_fwd(*a, **kw)

    unet.forward = counting_forward
    counts.clear()          # discard the calibration / warm-up passes
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        sampler.sample(S=STEPS, batch_size=BATCH, shape=r.shape, eta=0.0, verbose=False, **cond)
    torch.cuda.synchronize()
    unet.forward = orig_fwd
    return dict(counts), box["n"]


def main():
    counts = {}
    for name in (FUSED,):
        if not hasattr(mc, name):
            print(f"missing kernel {name} -- rebuild the extension")
            return 1
        orig = getattr(mc, name)

        def wrap(orig=orig, name=name):
            def w(*a, **kw):
                counts[name] = counts.get(name, 0) + 1
                return orig(*a, **kw)
            return w
        setattr(mc, name, wrap())

    fails = []
    print(f"\n| K | UNet forwards | fused calls | fused/step | want |")
    print("|---|---:|---:|---:|---:|")
    for refresh in (1, 4):
        c, steps = run(refresh, counts)
        fused = c.get(FUSED, 0)
        if steps == 0:
            fails.append(f"K={refresh}: the UNet never ran")
            continue
        per_step = fused / steps
        print(f"| {refresh} | {steps} | {fused} | {per_step:.2f} | {UPDOWN_BLOCKS} |")
        if abs(per_step - UPDOWN_BLOCKS) > 1e-6:
            fails.append(f"K={refresh}: {per_step:.2f} of {UPDOWN_BLOCKS} updown blocks fused")

    print()
    if fails:
        print(f"FAILED ({len(fails)}):")
        for f in fails:
            print("  -", f)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

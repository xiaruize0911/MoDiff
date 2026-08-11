"""Per-COMPONENT time attribution for K=1 full MoDiff: conv vs attention vs the projections.

STATUS: SUPERSEDED, and still unverified -- the patch below was never re-run. Do not quote any
number this script prints. The replacement landed 2026-08-07 and answers the same question with
no profiler in the measured region: docs/component_attribution_2026-08-07 (differential wall
clock, plus a trace bucketed offline). Its two methods agree to 0.01 ms/step on the projection
delta, against this script's factor-of-2.2 error (235.74 reported, 106.30 measured).

Kept in the tree because its two defects are worth not repeating: module forward hooks miss the
62 convs the ResBlock dispatches directly, and summing self_device_time_total over record_function
scopes double-counts the kernels inside them.

`fusion_profile.py` buckets by kernel NAME, which cannot separate the components: both the conv
modulated path and the wxax projections call `step1_static_quantize_fprop` and `delta_absmax_fp16`, so
those kernels land in one bucket no matter who launched them. This attributes by MODULE instead, by
wrapping each family in a `record_function` scope so every kernel launched inside is charged to it.

One subtlety that makes the raw numbers misleading if ignored: **the 42 projections live INSIDE the 21
attention blocks**, so the attention scope's inclusive time already contains them. Both are reported,
plus attention-minus-projections, which is the number to use when apportioning blame.

Configuration profiled: K=1 (MODIFF_DELTA_REFRESH=1) with MoDiff on conv AND the projections
(MODIFF_LINEAR=1) -- the paper's datapath, and the slowest of the ones measured.
"""

import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers"),
                os.path.join(ROOT, "integration/benchmarks/report")]

import torch                                                                    # noqa: E402
import kernel_suites_bench as ks                                                # noqa: E402
import integration.benchmarks.benchmark_ldm as B                                # noqa: E402
from torch.profiler import profile, ProfilerActivity, record_function           # noqa: E402

BATCH = int(os.environ.get("CP_BATCH", "128"))
STEPS = int(os.environ.get("CP_STEPS", "12"))
OUT = os.environ.get("CP_OUT", "docs/delta_clip_2026-08-06/data/component_profile.json")
CALIB = "integration/calibration/int8_calibration.pt"


def scope_wrap(model):
    """Wrap the METHODS each family actually enters, and return the unwrap thunks.

    Module forward hooks do not work here. 62 of the 70 MoDiff conv layers are invoked by the ResBlock
    calling `forward_gn_fused_modiff` directly, bypassing `__call__`, so a forward hook never fires for
    them -- the first version of this script charged the whole conv family 5.55 ms of a 191 ms step and
    the number was nonsense. The fusion audit had already established which methods are live, so wrap
    exactly those.

    Scope count is kept small on purpose: one record_function per module per step (240 modules x 12
    steps in the first attempt) inflated the measured total from ~110 to 191 ms/step, i.e. the
    instrument dominated what it measured.
    """
    from integration.kernels.int8_optimized import OptimizedInt8Conv2d
    unet = model.model.diffusion_model
    try:
        from integration.kernels.wxax_linear import QuantLinearWxAx
    except Exception:
        QuantLinearWxAx = ()
    undo = []

    def patch(obj, meth, scope):
        fn = getattr(obj, meth, None)
        if fn is None:
            return

        def inner(*a, **kw):
            with record_function(f"MODIFF::{scope}"):
                return fn(*a, **kw)
        setattr(obj, meth, inner)
        undo.append((obj, meth, fn))

    counts = {"conv": 0, "attn": 0, "proj": 0}
    for m in unet.modules():
        if isinstance(m, OptimizedInt8Conv2d):
            counts["conv"] += 1
            for meth in ("forward_gn_fused_modiff", "_forward_modulated", "_forward_first_step",
                         "forward_modiff_fused_silu_residual", "forward_from_int8",
                         "forward_from_int8_dual", "forward"):
                patch(m, meth, "conv")
        elif type(m).__name__ == "QuantizedStandardAttentionBlock":
            counts["attn"] += 1
            patch(m, "_forward_routes", "attn")
        elif QuantLinearWxAx and isinstance(m, QuantLinearWxAx):
            counts["proj"] += 1
            patch(m, "forward", "proj")
    return undo, counts


def main():
    os.environ["MODIFF_LINEAR"] = "1"
    ks.set_env("int8")
    os.environ["MODIFF_LINEAR"] = "1"            # ks.set_env rewrites the quant block
    os.environ["MODIFF_DELTA_REFRESH"] = "1"     # K=1, the paper's per-step scale
    os.environ["MODIFF_DELTA_REPORT"] = "0"
    os.environ["MODIFF_ACT_Q"], os.environ["MODIFF_DELTA_CLIP"] = "127", "1.0"
    print(f"batch {BATCH}, {STEPS} steps, K=1, MoDiff on conv + projections\n", flush=True)

    runner = B.BenchmarkRunner(
        config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
        ckpt_path="models/ldm/lsun_churches256/model.ckpt",
        output_dir="integration/results/component_profile",
        batch_size=BATCH, steps=STEPS, shape=(4, 32, 32), calibration_path=CALIB)
    model, sampler = runner._setup_model("int8")
    cond = runner._cond_kwargs(model, BATCH)

    def sample():
        with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
            sampler.sample(S=STEPS, batch_size=BATCH, shape=runner.shape, eta=0.0,
                           verbose=False, **cond)

    sample()                                     # warm-up (kernel selection + self-calibration)
    undo, counts = scope_wrap(model)
    torch.cuda.synchronize()
    # CUDA only. ProfilerActivity.CPU records EVERY aten op, which was most of the 2.2x
    # inflation and also stalls the CPU's kernel queueing, leaving real GPU gaps.
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        sample()
        torch.cuda.synchronize()
    for obj, meth, fn in undo:
        setattr(obj, meth, fn)

    rows, total = {}, 0.0
    for e in prof.key_averages():
        if e.key.startswith("MODIFF::"):
            rows[e.key.split("::")[1]] = float(getattr(e, "device_time_total", 0.0)) / 1000.0 / STEPS
            continue
        else:
            # Excluded above: a MODIFF:: scope reports the device time of the kernels INSIDE it, so
            # adding it alongside those kernels counted them twice. That double count, not overhead
            # alone, is why the first run reported 235.74 ms/step against a measured 106.30.
            total += float(getattr(e, "self_device_time_total", 0.0) or 0.0)
    total = total / 1000.0 / STEPS

    attn = rows.get("attn", 0.0)
    proj = rows.get("proj", 0.0)
    out = {"batch": BATCH, "steps": STEPS, "module_counts": counts,
           "total_gpu_ms_per_step": total, "inclusive_ms_per_step": rows,
           "attn_excluding_projections_ms_per_step": attn - proj}
    print(f"module counts: {counts}")
    print(f"total GPU self time: {total:.2f} ms/step\n")
    print(f"{'component':28s} {'ms/step':>9}  {'share':>7}   note")
    print("-" * 78)
    for k, v in sorted(rows.items(), key=lambda kv: -kv[1]):
        note = "INCLUDES the projections" if k == "attn" else ""
        print(f"{k:28s} {v:9.2f}  {v/total*100:6.1f}%   {note}")
    print(f"{'attn MINUS projections':28s} {attn-proj:9.2f}  {(attn-proj)/total*100:6.1f}%   "
          f"the attention math itself")
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()

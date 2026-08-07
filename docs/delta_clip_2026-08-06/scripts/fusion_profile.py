"""Where does the time actually go, per kernel, at production batch? Fusion vs bandwidth.

The fusion audit established WHICH paths engage (`data/fusion_audit.json`):

    int8 PTQ      62 convs on forward_from_int8 (int8->int8 chained), 21/21 attention qout-fused
    conv MoDiff   62 convs on forward_gn_fused_modiff, 8 on the UNFUSED _forward_modulated,
                  21/21 attention qout-fused
    conv+proj     same convs, but 0/21 attention qout-fused -- the gate term that flips is
                  proj._use_bias_res, which wxax_linear sets as `_HAS_BIAS_RES and not modiff and
                  not self._out_i8`, so MoDiff on a projection structurally disqualifies the fused
                  flash+proj path on every block

That says fusion coverage changes, but not what it costs. Two candidate explanations for
1.46x -> 1.38x -> 0.98x compete, and they call for different fixes:

  (a) LOST FUSION -- 21 blocks fall back to fp16 attn-output round-trip + a separate
      quantize_attn_out kernel. Fixable by making the fused path MoDiff-aware.
  (b) STATE BANDWIDTH -- a_hat/o_hat traffic that Eqs 9-10 require. Stage 3.3's accounting put it at
      10.21 GB/step (~14.6 ms at 700 GB/s) for the projections alone. NOT fixable by fusion.

This buckets every CUDA kernel's self time so the two can be told apart: if (a) dominates, the growth
is in attention/quantize kernels that the fused path would have eliminated; if (b) dominates, the
growth is in the delta absmax/quantize/accumulate passes, whose cost is proportional to state size.
"""

import json
import os
import re
import sys
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers"),
                os.path.join(ROOT, "integration/benchmarks/report")]

import torch                                                                    # noqa: E402
import kernel_suites_bench as ks                                                # noqa: E402
import integration.benchmarks.benchmark_ldm as B                                # noqa: E402

BATCH = int(os.environ.get("FP_BATCH", "128"))
STEPS = int(os.environ.get("FP_STEPS", "12"))       # enough for steady state; the profiler is heavy
OUT = os.environ.get("FP_OUT", "docs/delta_clip_2026-08-06/data/fusion_profile.json")
CALIB = "integration/calibration/int8_calibration.pt"
CONFIGS = [("fp16", "fp16", "0"), ("int8 PTQ", "int8_baseline", "0"),
           ("conv MoDiff", "int8", "0"), ("conv+proj MoDiff", "int8", "1")]

#: Buckets, in priority order -- first match wins. Names are matched case-insensitively against the
#: CUDA kernel name, so the CUTLASS conv/GEMM entries land in one bucket regardless of tile config.
BUCKETS = [
    ("conv/gemm (tensor core)", r"cutlass|implicit_gemm|s8_?gemm|gemm_wxax|awq|dp4a|i8gemm|conv2d"),
    ("attention/flash",         r"flash|attn|softmax|score"),
    ("delta absmax/quantize",   r"absmax|quantize|delta|pack|dequant"),
    ("groupnorm/silu",          r"group_norm|groupnorm|gn_|silu|layer_norm"),
    ("elementwise/copy",        r"elementwise|vectorized|copy|cat|add|mul|fill|index|slice|transpose|permute"),
    ("reduce/other",            r"reduce|sum|mean|norm"),
]


def bucket_of(name):
    low = name.lower()
    for label, pat in BUCKETS:
        if re.search(pat, low):
            return label
    return "unclassified"


def run(label, mode, lin):
    os.environ["MODIFF_LINEAR"] = lin
    ks.set_env(mode)
    os.environ["MODIFF_LINEAR"] = lin            # ks.set_env rewrites the quant block; re-assert
    os.environ["MODIFF_DELTA_REPORT"] = "0"
    os.environ["MODIFF_DELTA_REFRESH"] = os.environ.get("MODIFF_DELTA_REFRESH", "4")
    runner = B.BenchmarkRunner(
        config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
        ckpt_path="models/ldm/lsun_churches256/model.ckpt",
        output_dir="integration/results/fusion_profile",
        batch_size=BATCH, steps=STEPS, shape=(4, 32, 32),
        calibration_path=None if mode == "fp16" else CALIB)
    model, sampler = runner._setup_model(mode)
    cond = runner._cond_kwargs(model, BATCH)

    def sample():
        with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
            sampler.sample(S=STEPS, batch_size=BATCH, shape=runner.shape, eta=0.0,
                           verbose=False, **cond)

    sample()                                     # warm-up: cuDNN/CUTLASS selection + self-calibration
    torch.cuda.synchronize()
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        sample()
        torch.cuda.synchronize()

    tot = defaultdict(float)
    per_kernel = defaultdict(float)
    for e in prof.key_averages():
        us = float(getattr(e, "self_device_time_total", 0.0) or 0.0)
        if us <= 0:
            continue
        tot[bucket_of(e.key)] += us
        per_kernel[e.key] += us
    total_us = sum(tot.values())
    row = {"config": label, "mode": mode, "modiff_linear": lin, "batch": BATCH, "steps": STEPS,
           "total_ms_per_step": total_us / 1000.0 / STEPS,
           "buckets_ms_per_step": {k: v / 1000.0 / STEPS for k, v in
                                   sorted(tot.items(), key=lambda kv: -kv[1])},
           "top_kernels_ms_per_step": {k: v / 1000.0 / STEPS for k, v in
                                       sorted(per_kernel.items(), key=lambda kv: -kv[1])[:12]}}
    del model, sampler, runner
    torch.cuda.empty_cache()
    return row


def main():
    print(f"batch {BATCH}, {STEPS} DDIM steps profiled (after one warm-up sample)\n", flush=True)
    rows = []
    for label, mode, lin in CONFIGS:
        row = run(label, mode, lin)
        rows.append(row)
        print(f"=== {label} === total {row['total_ms_per_step']:.2f} ms/step (GPU self time)",
              flush=True)
        for k, v in row["buckets_ms_per_step"].items():
            print(f"    {k:26s} {v:7.2f} ms", flush=True)
        print(flush=True)
        with open(OUT, "w") as f:
            json.dump(rows, f, indent=2)
    os.environ["MODIFF_LINEAR"] = "0"
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()

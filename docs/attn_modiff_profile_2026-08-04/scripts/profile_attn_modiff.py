"""Profile the attention layer UNDER MoDiff, and find where the projection regression lives.

Motivation. MEASUREMENT_REPORT_2026-08-01's stage table shows QKV/output projection going
1773.5 ms (fp16) -> 1850.5 ms (int8) for the whole model, i.e. quantizing the projections made them
SLOWER. That was carried into the 08-04 report without being chased. Meanwhile the attention core
only reaches 1.28x. Both say attention is not done, so this measures it properly:

  * five modes, including MoDiff -- the earlier aligned trace only had fp16/int8/int4 baselines, so
    it could not say whether MoDiff changes anything inside attention (it should not: the qkv
    epilogue emits quantized codes and MoDiff has no codes to emit, so attention is structurally
    excluded -- but "should not" is a claim, and this checks it)
  * one Perfetto trace with all five modes time-aligned per shape, so the same attention forward can
    be read straight down the page across precisions
  * per-kernel attribution inside the layer, bucketed into qkv / core / proj / quantize / norm, which
    is what localizes the projection regression to a kernel rather than a stage label

Reuses export_attention_aligned.py's merge/alignment machinery by overriding its MODES, so the trace
format and the slot layout are identical to the 08-03 traces and the two can be compared directly.
"""

import argparse
import collections
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
HERE = "docs/attn_modiff_profile_2026-08-04"
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers"),
                os.path.join(ROOT, "integration/benchmarks/report"),
                os.path.join(ROOT, "docs/perfetto_traces_2026-08-03/scripts")]

import torch

import export_attention_aligned as EA
import layer_pipeline_bench as lb
from profile_tree import classify

# Five modes. The two MoDiff rows are the point of this run.
MODES = [("fp16", "FP16"), ("int8_baseline", "INT8"), ("int8", "INT8+MoDiff"),
         ("int4_baseline", "INT4"), ("int4", "INT4+MoDiff")]
EA.MODES = MODES
EA.OUT = f"{HERE}/traces"

#: Buckets for the per-kernel attribution. Order matters: first match wins, so the specific
#: projection/qkv GEMM names are tested before the generic GEMM catch-all.
BUCKETS = [
    ("qkv projection", ("fused_gn_qkv", "gemm_w8a8_kernel_awq_out_i8", "gemm_w4a4_kernel_awq_out_i8")),
    ("attn core (flash)", ("flash_attn",)),
    ("out projection", ("gemm_w8a8_kernel_awq", "gemm_w4a4_kernel_awq", "cutlass", "gemm_f16")),
    ("Q/K/V quantize", ("quantize_attn", "aq_qtok", "quantize_act", "scale_quantize")),
    ("GroupNorm", ("group_norm", "gn_")),
    ("softmax / bmm (fp16 SDPA)", ("softmax", "bmm", "gemv", "sm80_xmma", "sm86_xmma")),
    ("elementwise / copy", ("elementwise", "copy", "cat", "transpose", "permute", "contiguous")),
]


def bucket_of(name):
    n = name.lower()
    for label, keys in BUCKETS:
        if any(k.lower() in n for k in keys):
            return label
    return "other"


def kernel_table(trace, n_inst, iters):
    """{bucket: ms} and {kernel: ms} for ONE attention forward, from a session trace.

    session_events() already drops the profiler warm-up iterations and the harness syncs, so what is
    left is `iters` forwards of `n_inst`-instance-equivalent work; divide to get one forward.
    """
    anchor, end, evs = EA.session_events(trace)
    per_b, per_k = collections.defaultdict(float), collections.defaultdict(float)
    for e in evs:
        if e.get("cat") != "kernel":
            continue
        us = float(e.get("dur", 0.0))
        per_b[bucket_of(e["name"])] += us / iters
        per_k[e["name"][:60]] += us / iters
    return dict(per_b), dict(per_k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--out", default=f"{HERE}/traces/attention_modiff_5mode_aligned.json")
    a = ap.parse_args()
    os.makedirs(f"{HERE}/traces", exist_ok=True)
    os.makedirs(f"{HERE}/data", exist_ok=True)

    # Calibration: layer_pipeline_bench loads the un-suffixed (stub-derived) files. Left as is,
    # deliberately, and this is a TIMING run so it does not matter: the scale VALUES do not affect
    # kernel selection or duration, and the routes depend only on is_calibrated, which is True
    # either way. Any ACCURACY number would need the _realckpt files -- none is reported here.
    _orig = lb.collect_layers

    def collect(mode_key):
        os.environ["MODIFF_DELTA_MODE"] = "dynamic"
        os.environ["MODIFF_DELTA_REPORT"] = "0"     # diverges at W4A4; see FINDINGS 2026-08-04
        os.environ["MODIFF_LINEAR"] = "0"           # +25 ms/step at batch 128; off by default
        return _orig(mode_key)
    lb.collect_layers = collect
    EA.lb = lb                                      # EA.profile_attention calls lb.collect_layers

    bn = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    for _ in range(8):
        bn = bn @ bn
    torch.cuda.synchronize(); del bn; torch.cuda.empty_cache()

    sessions, shapes, buckets, kernels = {}, None, {}, {}
    for mode_key, label in MODES:
        print(f"\n=== {label} ({mode_key}) ===", flush=True)
        per_shape = EA.profile_attention(mode_key, a.iters)
        if shapes is None:
            shapes = sorted(per_shape, key=lambda s: -(s[2] * s[3]))
        for shape, v in per_shape.items():
            sessions[(mode_key, shape)] = v
            b, k = kernel_table(v[0], v[1], a.iters)
            buckets[(label, f"C{shape[1]}/T{shape[2] * shape[3]}")] = b
            kernels[(label, f"C{shape[1]}/T{shape[2] * shape[3]}")] = k

    slots, n_events = EA.merge(sessions, shapes, a.out)
    print(f"\nwrote {a.out}  ({n_events} events, "
          f"{os.path.getsize(a.out) / 2**20:.2f} MiB)")

    out = {"buckets": {f"{m}|{s}": v for (m, s), v in buckets.items()},
           "kernels": {f"{m}|{s}": v for (m, s), v in kernels.items()},
           "modes": [l for _, l in MODES],
           "shapes": [f"C{s[1]}/T{s[2] * s[3]}" for s in shapes]}
    with open(f"{HERE}/data/attn_modiff_buckets.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {HERE}/data/attn_modiff_buckets.json")

    # console summary: total per (mode, shape) and the projection buckets specifically
    print(f"\n{'=' * 100}\nAttention layer, us per forward, by bucket\n{'=' * 100}")
    for s in out["shapes"]:
        print(f"\n  {s}")
        allb = sorted({b for m in out["modes"] for b in buckets.get((m, s), {})})
        print("    " + f"{'bucket':<28}" + "".join(f"{m:>14}" for m in out["modes"]))
        for b in allb:
            row = "".join(f"{buckets.get((m, s), {}).get(b, 0.0):14.1f}" for m in out["modes"])
            print(f"    {b:<28}{row}")
        tot = "".join(f"{sum(buckets.get((m, s), {}).values()):14.1f}" for m in out["modes"])
        print(f"    {'TOTAL':<28}{tot}")


if __name__ == "__main__":
    main()

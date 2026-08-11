"""Method C, part 2: bucket the traces offline, and reconcile them with the profiler-free clock.

Reads `traces/*.json.gz` written by `trace_configs.py` and produces, per configuration:

  * GPU kernel self-time per step, bucketed by component
  * the per-kernel table behind each bucket
  * for every configuration that names a `delta_from`, the per-kernel DIFF against it -- which is
    how the components get separated at all. Name-based bucketing cannot tell a conv's
    `delta_absmax_fp16_kernel` from a projection's (this is exactly why `fusion_profile.py` could
    not do the component split), but the difference between two traces that differ in one component
    is that component, whoever launched the kernels.
  * the alignment ratio against `data/differential_timing.json`, which is the honest total.

Nothing here touches the GPU, so the bucket rules below can be edited and re-run for free.

## On `kernel_suites_bench.suite_of()`

That function is the starting point and is reused for the `suite` column, but it is NOT correct on
raw CUDA kernel names and this script does not pretend otherwise. It was written for *entry-point*
names (the Python-level dispatch targets `kernel_suites_bench` captures), where "conv2d" appears
literally. The kernels a conv entry point actually launches are
`cutlass::Kernel<modiff::ImplicitGemmConvolutionEVT...>` and
`sm80_xmma_fprop_implicit_gemm_..._nhwckrsc_nhwc_...`; both contain "gemm" and neither contains
"conv2d", so `suite_of` files the two largest kernels in a quantized step (119 ms of 220 in the
int8 trace) under `linear`. `bucket_of` below fixes that by testing the conv patterns first, and
both columns are reported so the disagreement is visible rather than silently resolved.
"""
import argparse
import collections
import glob
import gzip
import json
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "integration/benchmarks/report")]

import kernel_suites_bench as ks                                                # noqa: E402

TRACES = "docs/component_attribution_2026-08-07/traces"
DIFFTIME = "docs/component_attribution_2026-08-07/data/differential_timing.json"

#: (bucket, regex) tested in order -- FIRST match wins, so the conv patterns must precede any
#: pattern containing "gemm". Every rule is a claim about which family launches that kernel; the
#: `unclassified` report at the end of the run is the check that no large kernel is being guessed.
RULES = [
    # ---- convolution. Both the CUTLASS int8 path and the cuDNN fp16 path are "implicit gemm", so
    #      these must be tested before any rule containing "gemm". `cudnn` is here because the fp16
    #      convs land in `cutlass__5x_cudnn::Kernel`; no fp16 Linear in this UNet reaches cuDNN
    #      (they go to cuBLAS `ampere_*gemm*`), so the pattern is unambiguous in this tree only. ----
    ("conv", r"implicitgemmconvolution|fprop_implicit_gemm|nhwckrsc|conv2d|nhwcaddpadding|cudnn"),
    # ---- attention score path: the fused int8/int4 flash kernels, and fp16 SDPA's own flash. ----
    ("attention", r"flash_attn|flash_fwd|pytorch_flash|scaled_dot_product|softmax|sdpa"),
    # ---- the attention path's own quantize/repack passes (csrc/kernels/attention/attn_quant_gemm.cu,
    #      all `aq_`-prefixed). Kept separate from `quantize` because these are exactly the passes a
    #      qout epilogue absorbs, so their appearing or vanishing IS the epilogue measurement. ----
    ("attn_quantize", r"^aq_|attn_quant|quantize_attn"),
    # ---- GN(+SiLU), with or without a fused quantize. The MoDiff variant (…delta_quantize…) also
    #      writes a_hat, and is the fusion the attention qkv gained on 2026-08-06. `gn_stats_*` are
    #      the chan-major two-pass statistics the MoDiff GN path computes separately. ----
    ("norm_quantize", r"group_norm|gn_accum|gn_finalize|gn_stats|rowwisemoments|layer_norm|"
                      r"computefusedparams"),
    # ---- the MoDiff delta machinery when it is NOT fused into GN: its own separate passes. ----
    ("delta_quantize", r"update_ahat|delta_absmax|step1_static_quantize|apply_delta|retire"),
    # ---- W8A8 / W4A4 projections and any remaining fp16 GEMM. Deliberately ahead of `quantize`:
    #      `gemm_w8a8_kernel_awq_out_i8` IS the projection GEMM, quantizing only in its epilogue,
    #      and filing it under quantize hid 5.25 ms/step of linear work in the int8 trace. ----
    ("linear_gemm", r"gemm|_gemv|cublas"),
    # ---- standalone activation quantize / dequantize / attention-output quantize / repacking. ----
    # `quant_act` is spelled without the "ize" (csrc/kernels/linear/gemm_wxax.cu), so a bare
    # `quantize` pattern silently left 1.35 ms/step of it in `other`.
    ("quantize", r"quantize|quant_act|dequant|int8_pack|qi8packed|reshuffle|awq_pack"),
    # ---- data movement the epilogues did not absorb. ----
    ("elementwise", r"elementwise|direct_copy|cat2_channels_last|upsample|avg_pool|indexselect|"
                    r"vectorized|fill_|copy_"),
]
_RULES = [(b, re.compile(p)) for b, p in RULES]


def normalize(name):
    """Collapse a kernel symbol to a stable, comparable label.

    Template arguments are dropped so `awq_out_i8<1>` and `<2>` merge -- they are the same kernel
    at two tile counts. CUTLASS names are already mangled and have no `<`, so they survive whole and
    stay distinguishable from each other.
    """
    n = name.replace("void ", "")
    n = re.sub(r"<.*", "", n)
    return n.strip()


def short(name, n=64):
    return name if len(name) <= n else name[: n - 3] + "..."


def bucket_of(name):
    low = name.lower()
    for b, rx in _RULES:
        if rx.search(low):
            return b
    return "other"


def load(path):
    with gzip.open(path) as f:
        return json.load(f)["traceEvents"]


def measure(path):
    """Per-step GPU kernel self time, per normalized kernel name."""
    ev = load(path)
    spans = sorted((e["ts"], e["ts"] + e["dur"], e["name"]) for e in ev
                   if e.get("cat") == "gpu_user_annotation" and e["name"].startswith("step/"))
    assert spans, f"{path}: no gpu_user_annotation step slices"

    per_kernel = collections.defaultdict(lambda: {"us": 0.0, "calls": 0})
    per_step = collections.Counter()
    outside = {"us": 0.0, "calls": 0}
    dev_cats = ("kernel", "gpu_memcpy", "gpu_memset")
    for e in ev:
        if e.get("cat") not in dev_cats:
            continue
        ts, dur = e["ts"], e["dur"]
        step = next((s[2] for s in spans if s[0] <= ts < s[1]), None)
        name = normalize(e["name"]) if e["cat"] == "kernel" else f"[{e['cat']}]"
        if step is None:
            outside["us"] += dur
            outside["calls"] += 1
            continue
        per_kernel[name]["us"] += dur
        per_kernel[name]["calls"] += 1
        per_step[step] += dur

    nsteps = len(spans)
    kern = {k: {"ms_per_step": v["us"] / 1e3 / nsteps, "calls_per_step": v["calls"] / nsteps,
                "bucket": bucket_of(k), "suite_of": ks.suite_of(k)}
            for k, v in per_kernel.items()}
    return {
        "steps": nsteps,
        "gpu_ms_per_step": sum(per_step.values()) / 1e3 / nsteps,
        "per_step_ms": {s: round(us / 1e3, 3) for s, us in sorted(per_step.items())},
        # Kernels outside every step slice are warm-up/teardown residue; reported so "the buckets
        # do not sum to the trace" can never be a silent discrepancy.
        "outside_steps_ms_total": outside["us"] / 1e3,
        "kernels": kern,
    }


def bucket_table(kern, key="bucket"):
    t = collections.defaultdict(lambda: {"ms_per_step": 0.0, "calls_per_step": 0.0, "kernels": 0})
    for k, v in kern.items():
        b = t[v[key]]
        b["ms_per_step"] += v["ms_per_step"]
        b["calls_per_step"] += v["calls_per_step"]
        b["kernels"] += 1
    return dict(sorted(t.items(), key=lambda kv: -kv[1]["ms_per_step"]))


def diff_kernels(a, b, floor=0.05):
    """b - a, per kernel. `floor` ms/step drops rows too small to reason about."""
    rows = []
    for k in set(a) | set(b):
        x = a.get(k, {}).get("ms_per_step", 0.0)
        y = b.get(k, {}).get("ms_per_step", 0.0)
        if abs(y - x) >= floor:
            rows.append({"kernel": k, "bucket": bucket_of(k), "from_ms": x, "to_ms": y,
                         "delta_ms": y - x})
    rows.sort(key=lambda r: -abs(r["delta_ms"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default=TRACES)
    ap.add_argument("--output",
                    default="docs/component_attribution_2026-08-07/data/trace_buckets.json")
    ap.add_argument("--top", type=int, default=14)
    a = ap.parse_args()

    man = json.load(open(os.path.join(a.traces, "manifest.json")))
    wall = {}
    if os.path.exists(DIFFTIME):
        wall = {k: v["ms_per_step"] for k, v in json.load(open(DIFFTIME))["arms"].items()}

    out = {"gpu": man.get("gpu"), "configs": {}, "diffs": {}}
    for tr in man["traces"]:
        path = os.path.join(a.traces, tr["file"])
        if not os.path.exists(path):
            print(f"  (missing {tr['file']})")
            continue
        m = measure(path)
        m["delta_from"] = tr["delta_from"]
        m["mode"] = tr["mode"]
        m["route_check"] = tr["route_check"]
        m["buckets"] = bucket_table(m["kernels"])
        m["suite_of_buckets"] = bucket_table(m["kernels"], key="suite_of")
        w = wall.get(tr["config"])
        m["wall_ms_per_step"] = w
        m["trace_over_wall"] = m["gpu_ms_per_step"] / w if w else None
        out["configs"][tr["config"]] = m

    for lab, m in out["configs"].items():
        b = m["delta_from"]
        if b and b in out["configs"]:
            out["diffs"][f"{b} -> {lab}"] = {
                "trace_delta_ms_per_step": m["gpu_ms_per_step"]
                - out["configs"][b]["gpu_ms_per_step"],
                "wall_delta_ms_per_step": (m["wall_ms_per_step"] - out["configs"][b]["wall_ms_per_step"]
                                           if m["wall_ms_per_step"] and
                                           out["configs"][b]["wall_ms_per_step"] else None),
                "by_bucket": {k: (m["buckets"].get(k, {}).get("ms_per_step", 0.0)
                                  - out["configs"][b]["buckets"].get(k, {}).get("ms_per_step", 0.0))
                              for k in set(m["buckets"]) | set(out["configs"][b]["buckets"])},
                "by_kernel": diff_kernels(out["configs"][b]["kernels"], m["kernels"]),
            }

    os.makedirs(os.path.dirname(a.output), exist_ok=True)
    with open(a.output, "w") as f:
        json.dump(out, f, indent=1)

    # ------------------------------------------------------------------ report
    print(f"\n{'config':<20}{'trace GPU':>11}{'wall':>9}{'trace/wall':>12}   steps")
    for lab, m in out["configs"].items():
        w = f"{m['wall_ms_per_step']:.2f}" if m["wall_ms_per_step"] else "—"
        r = f"{m['trace_over_wall']:.3f}" if m["trace_over_wall"] else "—"
        print(f"{lab:<20}{m['gpu_ms_per_step']:>11.2f}{w:>9}{r:>12}   {m['steps']}")

    labs = list(out["configs"])
    buckets = sorted({b for m in out["configs"].values() for b in m["buckets"]},
                     key=lambda b: -sum(m["buckets"].get(b, {}).get("ms_per_step", 0.0)
                                        for m in out["configs"].values()))
    print(f"\nGPU ms/step by component (trace, profiler attached)\n{'bucket':<16}"
          + "".join(f"{l[:13]:>14}" for l in labs))
    for b in buckets:
        print(f"{b:<16}" + "".join(
            f"{out['configs'][l]['buckets'].get(b, {}).get('ms_per_step', 0.0):>14.2f}"
            for l in labs))
    print(f"{'TOTAL':<16}" + "".join(f"{out['configs'][l]['gpu_ms_per_step']:>14.2f}" for l in labs))

    for name, d in out["diffs"].items():
        print(f"\n--- {name}   trace {d['trace_delta_ms_per_step']:+.2f} ms/step"
              + (f"   wall {d['wall_delta_ms_per_step']:+.2f}" if d["wall_delta_ms_per_step"]
                 is not None else "") + " ---")
        for b, v in sorted(d["by_bucket"].items(), key=lambda kv: -abs(kv[1])):
            if abs(v) >= 0.05:
                print(f"    {b:<16}{v:>+9.2f}")
        for r in d["by_kernel"][:a.top]:
            print(f"      {short(r['kernel'], 66):<68}{r['from_ms']:>7.2f} ->{r['to_ms']:>7.2f}"
                  f" {r['delta_ms']:>+8.2f}  [{r['bucket']}]")

    unc = collections.defaultdict(float)
    for m in out["configs"].values():
        for k, v in m["kernels"].items():
            if v["bucket"] == "other":
                unc[k] = max(unc[k], v["ms_per_step"])
    if unc:
        print("\nunclassified ('other'), max ms/step across configs:")
        for k, v in sorted(unc.items(), key=lambda kv: -kv[1])[:12]:
            print(f"    {v:>7.3f}  {short(k, 90)}")

    print(f"\nWROTE {a.output}")


if __name__ == "__main__":
    main()

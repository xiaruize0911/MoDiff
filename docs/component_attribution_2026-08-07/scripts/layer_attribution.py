"""Attribute the traced kernel time to the LAYER KIND that launched it: conv / attn / lin.

`bucket_traces.py` groups by what a kernel *does*. This groups by *who called it*, which is the
question `component_profile.py` was built for and failed at. Name-based bucketing cannot do it:
`gn_apply_delta_quantize_flat_vec2_kernel` serves 62 MoDiff convs AND the 21 attention qkv, and
`static_quantize_and_update_ahat_..._vec2` serves 8 convs AND the 21 proj. One kernel, two families.

This does it without instrumenting the measured region, using two facts the traces already carry:

**1. Launch counts identify the layer sets exactly**, and they agree with the fusion audit in
`docs/delta_clip_2026-08-06` (62 convs on `forward_gn_fused_modiff`, 8 on the unfused
`_forward_modulated`, 21 attention blocks, 42 projections = 21 qkv + 21 proj):

    gn_apply_delta_quantize_flat_vec2      62 -> 83   (+21 qkv)
    gn_delta_absmax_flat_vec2              62 -> 83   (+21 qkv)
    gn_stats_partials_chanmajor            62 -> 83   (+21 qkv)
    gn_stats_reduce_partials               62 -> 83   (+21 qkv)
    static_quantize_and_update_ahat_vec2    8 -> 29   (+21 proj)
    delta_absmax_fp16                       8 -> 29   (+21 proj)
    gemm_w8a8_kernel_awq                   21 -> 42   (qkv joins proj)

`_assert_inventory` checks every one of these, so an attribution built on a layer count that no
longer holds fails instead of silently re-weighting.

**2. The `modiff_conv_k1 -> modiff_full_k1` differential IS the projection contribution.** The two
configs differ in exactly one thing -- MoDiff on the 42 projections -- so for a shared kernel the
conv share is its value in `modiff_conv_k1` and the projection share is the difference. Splitting by
call count instead would be wrong: a conv at [128,C,H,W] and a qkv at [128*1024,192] do not cost the
same per call.

Boundary rules, stated because they are choices and not facts:

* **`lin` is the 42 attention projections.** They live INSIDE the 21 attention blocks (see
  docs/delta_clip_2026-08-06 "The 42 Linear layers are the attention projections"), so `attn` here
  means the score path only -- QK^T, softmax, AV, and the quantize/repack passes feeding it. The two
  are reported separately and also summed, because "attention" means both things in this project.
* **The attention block's GroupNorm follows the kernel it was fused into.** Under full MoDiff the
  qkv's GN is inside `gn_apply_delta_quantize_flat_vec2`, so it lands in `lin`; in int8 PTQ the same
  normalization is a separate `group_norm_silu_quantize_nhwc_vec2` call and lands in `attn`. That
  moves ~2 ms between the two columns and is flagged in the output rather than smoothed over.
* **Everything in the ResBlock/up-down path that is not attention or a projection is `conv`** --
  upsample, avgpool, the skip concat, the residual elementwise adds.

Residual uncertainty: the conv kernels are not bit-identical between the two configs (the two EVT
convs read 14.19/11.85 in conv_k1 and 14.01/11.73 in full_k1, ~1.2% apart), so a differential split
carries roughly that much. Reported as `conv_drift_pct`.

Writes data/layer_attribution.json. Offline; no GPU.
"""
import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(os.path.dirname(os.path.abspath(__file__)))]

from bucket_traces import short                                                 # noqa: E402

BUCKETS = "docs/component_attribution_2026-08-07/data/trace_buckets.json"

#: kernel-name substring -> family, for kernels launched by exactly one family.
#: `split` marks the ones both conv and the projections call; those go through the differential.
SOLE = [
    # ---- conv / ResBlock / up-down path ----
    ("ImplicitGemmConvolutionEVT", "conv"),      # the fused int8 MoDiff convs
    ("_ZN7cutlass6KernelINS_4conv", "conv"),     # generic CUTLASS conv
    ("cudnn", "conv"),
    ("fprop_implicit_gemm", "conv"),
    ("nhwcAddPadding", "conv"),
    ("upsample_nearest2d", "conv"),
    ("avg_pool2d", "conv"),
    ("cat2_channels_last", "conv"),
    ("CatArrayBatchedCopy", "conv"),
    ("unrolled_elementwise_kernel", "conv"),     # 130 calls, identical in both configs
    ("group_norm_silu_nhwc_kernel", "conv"),     # 8 calls, the unfused modulated convs
    ("group_norm_silu_delta_quantize_resize", "conv"),
    ("group_norm_silu_quantize_resize", "conv"),
    ("gn_accum", "conv"), ("gn_finalize", "conv"),
    ("scale_quantize_int8", "conv"), ("RowwiseMoments", "conv"),
    ("ComputeFusedParams", "conv"), ("indexSelect", "conv"),
    # ---- attention score path ----
    ("flash_attn", "attn"), ("flash_fwd", "attn"), ("pytorch_flash", "attn"),
    ("aq_", "attn"),                             # csrc/kernels/attention/attn_quant_gemm.cu
    ("quantize_attn", "attn"),
    # ---- the 42 attention projections ----
    ("gemm_w8a8_kernel_awq", "lin"), ("gemm_w4a4_kernel_awq", "lin"),
    ("quant_act_int8_kernel", "lin"),
    ("ampere_", "lin"), ("sm80_xmma_gemm", "lin"), ("sm86_xmma_gemm", "lin"),
]
#: shared by conv and the projections -> split by the conv_k1/full_k1 differential
SPLIT = ["gn_apply_delta_quantize_flat", "gn_delta_absmax_flat", "gn_stats_partials",
         "gn_stats_reduce_partials", "static_quantize_and_update_ahat", "delta_absmax_fp16",
         "elementwise_kernel", "vectorized_elementwise_kernel"]
#: in int8 PTQ / conv-only this one serves 62 convs + the 21 attention GNs; under full MoDiff the
#: attention half has moved into gn_apply_delta_quantize. Split by the per-call rate measured in
#: modiff_conv_k1, where its 21 calls are the attention GNs alone.
GN_SHARED = "group_norm_silu_quantize_nhwc_vec2"

#: config -> (differential base, family the difference belongs to).
#: A config absent here has no base, so every shared kernel is charged to conv (see the caveat in
#: the report). The base must differ from the treatment in EXACTLY the thing being attributed.
SPLIT_BASE = {
    "modiff_full_k1": ("modiff_conv_k1", "lin"),
    "modiff_full_k4": ("modiff_conv_k4", "lin"),
    # No projection MoDiff here -- the difference is the attention path losing its qout epilogue and
    # paying for the output add itself, so it belongs to attn, not lin.
    "ptq_no_projquant": ("int8_ptq", "attn"),
}
#: `base_no_conv_modiff` is deliberately excluded: it swaps the fused EVT convs for the generic
#: CUTLASS conv, so "the conv part is unchanged" -- the assumption every differential split rests
#: on -- is false by 25 ms. See the trap section of FINDINGS.
NO_SPLIT = {"base_no_conv_modiff"}

#: Kernels the treatment adds exactly 21 launches to (21 qkv, or 21 proj). Asserted rather than
#: assumed, per pair, because the split charges the whole time difference to those 21 layers.
#: Only the DELTA is fixed: the base count is K-dependent (at K=4 `gn_delta_absmax_flat` fires on
#: one step in four, so it reads 15.5 calls/step against 62 at K=1) and is recorded, not checked.
ADDS_21 = ["gn_apply_delta_quantize_flat", "gn_delta_absmax_flat", "gn_stats_partials",
           "gn_stats_reduce_partials", "static_quantize_and_update_ahat", "delta_absmax_fp16",
           "gemm_w8a8_kernel_awq("]


def family(name):
    for pat, fam in SOLE:
        if pat in name:
            return fam
    for pat in SPLIT:
        if pat in name:
            return "split"
    if GN_SHARED in name:
        return "gn_shared"
    return "unattributed"


def _find(kern, pat):
    return {n: v for n, v in kern.items() if pat in n}


def _assert_inventory(label, base, treat):
    """The treatment must add exactly 21 launches to each shared MoDiff kernel it touches."""
    seen = {}
    for pat in ADDS_21:
        a = sum(v["calls_per_step"] for v in _find(base, pat).values())
        b = sum(v["calls_per_step"] for v in _find(treat, pat).values())
        if a == 0 and b == 0:
            continue                            # kernel absent at this refresh setting
        seen[pat] = (round(a, 1), round(b, 1))
        assert abs((b - a) - 21) < 0.05, (
            f"{label}: {pat!r} launches went {a:.1f} -> {b:.1f}, a delta of {b-a:.1f} rather than "
            f"the 21 qkv/proj the split charges it to. The layer inventory has changed.")
    return seen


def _conv_drift(base, treat):
    """How far the conv kernels themselves move between the two configs -- the split's error bar."""
    a = sum(v["ms_per_step"] for n, v in base.items() if "ImplicitGemmConvolutionEVT" in n)
    b = sum(v["ms_per_step"] for n, v in treat.items() if "ImplicitGemmConvolutionEVT" in n)
    return (abs(b - a) / a * 100 if a else 0.0), a, b


def attribute(kern, base=None, target=None, gn_rate=None):
    """Return {family: ms/step} plus the per-kernel detail behind it.

    `base` is the differential reference and `target` the family its difference belongs to; with
    `base` None every shared kernel is charged to conv.
    """
    out = {"conv": 0.0, "attn": 0.0, "lin": 0.0, "unattributed": 0.0}
    detail = []
    for n, v in kern.items():
        ms, fam = v["ms_per_step"], family(n)
        if fam == "split":
            if base is None:                    # nothing to difference against -> all conv
                out["conv"] += ms
                detail.append((n, "conv", ms, "sole (no differential base)"))
                continue
            # Exact name, not a substring probe: `elementwise_kernel` is a substring of
            # `unrolled_elementwise_kernel` and `vectorized_elementwise_kernel`, so a substring
            # lookup summed all three into the base and drove the projection share to zero.
            # Kernel names are normalized identically in both configs, so `n` matches directly.
            b = base.get(n, {}).get("ms_per_step", 0.0)
            add = max(0.0, ms - b)
            out["conv"] += ms - add
            out[target] += add
            detail.append((n, "split", ms, f"conv {ms-add:.2f} / {target} {add:.2f}"))
        elif fam == "gn_shared":
            # 21 of its calls are the attention blocks' GroupNorm; the rest are convs.
            attn = min(ms, 21 * gn_rate) if gn_rate else 0.0
            out["attn"] += attn
            out["conv"] += ms - attn
            detail.append((n, "gn_shared", ms, f"conv {ms-attn:.2f} / attn {attn:.2f}"))
        else:
            out[fam if fam in out else "unattributed"] += ms
            detail.append((n, fam, ms, "sole"))
    return out, detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buckets", default=BUCKETS)
    ap.add_argument("--output",
                    default="docs/component_attribution_2026-08-07/data/layer_attribution.json")
    a = ap.parse_args()

    d = json.load(open(a.buckets))
    C = d["configs"]

    # per-call cost of the attention GroupNorm, measured where its 21 calls are alone in the kernel
    gn = _find(C["modiff_conv_k1"]["kernels"], GN_SHARED)
    gn_rate = sum(v["ms_per_step"] for v in gn.values()) / 21.0 if gn else 0.0

    out = {"gn_attn_rate_ms_per_call": gn_rate, "pairs": {}, "excluded": sorted(NO_SPLIT),
           "configs": {}}
    for lab, (bl, tgt) in SPLIT_BASE.items():
        if lab not in C or bl not in C:
            continue
        inv = _assert_inventory(lab, C[bl]["kernels"], C[lab]["kernels"])
        drift, x, y = _conv_drift(C[bl]["kernels"], C[lab]["kernels"])
        out["pairs"][f"{bl} -> {lab}"] = {"target_family": tgt, "conv_drift_pct": drift,
                                          "conv_evt_ms": [x, y], "launch_inventory": inv}

    for lab, m in C.items():
        if lab in NO_SPLIT:
            continue
        bl, tgt = SPLIT_BASE.get(lab, (None, None))
        fam, detail = attribute(m["kernels"], C[bl]["kernels"] if bl else None, tgt, gn_rate)
        tot = m["gpu_ms_per_step"]
        # The trace accounts for 95-98% of the wall clock (GPU idle between kernels, plus the DDIM
        # scheduler math the traced steps skip). Scaling each family by that config's own ratio
        # gives shares against the number actually reported, on the convention the layer-level
        # report already uses -- it assumes the gap is spread proportionally, which is an
        # assumption, so both columns are kept.
        sc = (m["wall_ms_per_step"] / tot) if (m["wall_ms_per_step"] and tot) else 1.0
        out["configs"][lab] = {
            "gpu_ms_per_step": tot, "wall_ms_per_step": m["wall_ms_per_step"],
            "families": fam, "families_wall_scaled": {k: v * sc for k, v in fam.items()},
            "wall_scale": sc, "attn_plus_lin": fam["attn"] + fam["lin"],
            "shares_pct": {k: (100 * v / tot if tot else 0.0) for k, v in fam.items()},
            "split_base": bl, "split_target": tgt,
            "sum_check": sum(fam.values()) - tot,
            "detail": sorted(detail, key=lambda r: -r[2])[:24],
        }

    with open(a.output, "w") as f:
        json.dump(out, f, indent=1)

    order = [l for l in ("fp16", "int8_ptq", "modiff_conv_k4", "modiff_full_k4",
                         "modiff_conv_k1", "modiff_full_k1", "ptq_no_projquant")
             if l in out["configs"]]
    print("GPU ms/step by LAYER KIND (trace; shared kernels split by the differential pairs "
          "listed below)\n")
    print(f"{'family':<22}" + "".join(f"{l[:13]:>15}" for l in order))
    for k in ("conv", "attn", "lin", "unattributed"):
        print(f"{k:<22}" + "".join(f"{out['configs'][l]['families'][k]:>15.2f}" for l in order))
    print(f"{'  attn+lin':<22}" + "".join(f"{out['configs'][l]['attn_plus_lin']:>15.2f}"
                                          for l in order))
    print(f"{'TOTAL (trace)':<22}" + "".join(f"{out['configs'][l]['gpu_ms_per_step']:>15.2f}"
                                             for l in order))
    print(f"{'wall (no profiler)':<22}" + "".join(
        f"{out['configs'][l]['wall_ms_per_step']:>15.2f}" for l in order))
    print(f"\n{'share %':<22}" + "".join(f"{l[:13]:>15}" for l in order))
    for k in ("conv", "attn", "lin"):
        print(f"{k:<22}" + "".join(f"{out['configs'][l]['shares_pct'][k]:>14.1f}%" for l in order))

    print(f"\nsame, scaled to the profiler-free wall clock\n{'family':<22}"
          + "".join(f"{l[:13]:>15}" for l in order))
    for k in ("conv", "attn", "lin", "unattributed"):
        print(f"{k:<22}" + "".join(
            f"{out['configs'][l]['families_wall_scaled'][k]:>15.2f}" for l in order))
    print(f"{'TOTAL':<22}" + "".join(f"{out['configs'][l]['wall_ms_per_step']:>15.2f}"
                                     for l in order))

    print(f"\nattention GroupNorm {gn_rate*1e3:.1f} us/call (from modiff_conv_k1, "
          f"where its 21 calls are the attention GNs alone)")
    print("differential pairs:")
    for name, p in out["pairs"].items():
        print(f"  {name:<40} -> {p['target_family']:<4}  conv drift {p['conv_drift_pct']:.2f}% "
              f"({p['conv_evt_ms'][0]:.2f} -> {p['conv_evt_ms'][1]:.2f} ms EVT conv)")
        print(f"      launches: " + ", ".join(f"{k.split('_kernel')[0][:26]} {v[0]}->{v[1]}"
                                              for k, v in p["launch_inventory"].items()))
    print(f"excluded (differential assumption invalid): {out['excluded']}")
    bad = {l: round(v["families"]["unattributed"], 3) for l, v in out["configs"].items()
           if v["families"]["unattributed"] > 0.05}
    print(f"unattributed: {bad if bad else 'none above 0.05 ms/step in any config'}")

    for cfg in ("modiff_full_k1", "modiff_full_k4"):
        if cfg not in out["configs"]:
            continue
        print(f"\n--- {cfg}, per kernel ---")
        print(f"  {'kernel':<52}{'ms':>8}  family / split")
        for n, fam, ms, how in out["configs"][cfg]["detail"]:
            if ms < 0.3:
                continue
            s = short(n.replace("at::native::", "").replace("(anonymous namespace)::", ""), 52)
            if "ImplicitGemmConvolutionEVT" in n:
                s = "cutlass modiff ImplicitGemmConvEVT [int8 conv]"
            print(f"  {s:<52}{ms:>8.2f}  {fam if how.startswith('sole') else how}")
    print(f"\nWROTE {a.output}")


if __name__ == "__main__":
    main()

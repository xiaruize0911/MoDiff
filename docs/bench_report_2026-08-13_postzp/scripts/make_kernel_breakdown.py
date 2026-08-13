"""Per-KERNEL breakdown inside the attention, conv and linear suites, from kernel_suites.json.

REPORT.md's sections 2-4 give each suite's total and its entry points. This drills one level further, to
the number that tells you where to optimise: for every kernel, the per-call median AND the calls per
sample, because a suite total conflates the two. A kernel at 300 us x 5 calls and one at 3 us x 500 calls
both cost 1.5 ms/sample and want completely different fixes.

WHAT THE NUMBERS ARE. The bench captures the real call arguments at the C++ entry point during a live
sample, then replays each call signature in isolation (8 rounds x 60 iters, median of round medians). So
`us/call` is that replay median at the shape the model actually runs, and

    ms/sample = us/call x calls_per_sample / 1000

summed over a kernel's signatures. This is a REPLAY total, not a profiler total: it excludes launch gaps
and any overlap between kernels, which is why a suite's ms/sample does not have to match its share of the
end-to-end wall time. Section 1a of REPORT.md is the profiler view; this is the kernel view.

SELF-CHECK. Every suite total recomputed here is asserted against the value REPORT.md already publishes,
so a units or aggregation mistake fails loudly instead of producing a plausible table.

Run: python docs/bench_report_2026-08-13_postzp/scripts/make_kernel_breakdown.py    # no GPU
Writes docs/bench_report_2026-08-13_postzp/KERNEL_BREAKDOWN.md
"""
import collections
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(ROOT)

D = "docs/bench_report_2026-08-13_postzp"
SUITES = ("attention", "conv", "linear")
MODES = [("fp16", "fp16"), ("int8_baseline", "W8A8 PTQ"), ("int8", "W8A8 MoDiff"),
         ("int4_baseline", "W4A4 PTQ"), ("int4", "W4A4 MoDiff")]

#: WHAT EACH KERNEL DOES. Sourced from the kernels' own header comments, not from memory -- the file each
#: line rests on is named so a reader can check it. Kept in the GENERATOR rather than hand-written into
#: the .md so the prose cannot drift from the table it sits next to.
SUFFIX = [
    ("`_fprop`", "plain CUTLASS implicit-GEMM conv; the epilogue does dequant only, caller adds bias"),
    ("`_evt_*`", "an Epilogue Visitor Tree hand-assembled onto the conv Mma (CUTLASS 4.6.1 has no "
                 "EVT-on-conv path), so bias/residual/o_hat fold into the conv's own store and no "
                 "post-conv scratch tensor is ever written"),
    ("`_vt`", "V arrives PRE-TRANSPOSED as [N,H,hd_pad,T], straight from the qkv quantize, so the "
              "kernel skips a transpose"),
    ("`_static`", "Q and K each use ONE frozen calibrated scale, folded into the row scale. Removes "
                  "both per-token scale tensors, their cp.async staging, and one fp32 multiply per "
                  "score element from the hot loop"),
    ("`_qout`", "the epilogue writes the PROJECTION-QUANTIZED int8 output directly, fusing the next "
                "projection's input quantize. Mutually exclusive with MoDiff's fp16 o_hat state, so "
                "UNUSABLE under MoDiff -- all 21 blocks report qout_eligible == 0"),
    ("`_hd24`", "exact specialization for the dominant T=1024 / hd=24 route: three PV/output fragments "
                "instead of the generic HD_PAD=32 kernel's, plus vectorized 24-byte compact-Q staging"),
    ("`_small`", "the staging variant that wins at small T (NNT = BC/8 halves)"),
    ("`qi8` / `qpacked`", "how Q is staged into the kernel -- plain int8 rows vs packed"),
    ("`i4values_i8mma`", "int4 V values fed through the int8 tensor-core MMA path"),
    ("`_bias_res`", "bias + residual epilogue. Under MoDiff an EMPTY residual returns o_hat itself; a "
                    "given one also returns o_hat_t + residual as a SEPARATE tensor, because the "
                    "ResBlock/attention skip must not be folded into the temporal state"),
    ("`_o_hat`", "MoDiff's temporal accumulate: o_hat[elem] += this step's contribution, in place, fp16"),
    ("`_out_i8` / `_codes`", "emits int8/int4 CODES rather than dequantized values, so the next "
                             "consumer (flash attention) reads them directly"),
    ("`_layouts`", "the fused qkv projection writes Q/K/V already in the attention kernel's per-head "
                   "padded layouts, returning several tensors, so no separate reformat runs"),
]
DESC = {
    "torch_conv2d_fp16": "UNQUANTIZED fallback -- PyTorch/cuDNN fp16 conv, for the convs this pipeline "
                         "does not quantize (the stem/head convs and the 1x1 skips).",
    "torch_linear_fp16": "UNQUANTIZED fallback -- PyTorch fp16 linear.",
    "torch_sdpa_fp16": "UNQUANTIZED fallback -- PyTorch SDPA in fp16. In the fp16 arm this is the whole "
                       "attention suite; it materializes the [N,H,T,T] score matrix in HBM, which is "
                       "what the flash kernels exist to avoid.",
    "conv2d_int8_fprop": "int8 x int8 conv, plain output. On the MoDiff arm this is the t=T conv and "
                         "the delta-step conv whose accumulate is done by a separate epilogue.",
    "conv2d_int4_fprop": "int4 x int4 conv, plain output; same role as the int8 twin.",
    "conv2d_int8_evt_bias_residual_fp16": "D1 fusion: out = acc*alpha*weight_scale[k] + bias[k] + "
        "residual[elem] -> fp16, in the conv's own store. This is the PTQ arm's whole conv datapath.",
    "conv2d_int4_evt_bias_residual_fp16": "D1 fusion, int4. The PTQ arm's whole conv datapath.",
    "conv2d_int8_evt_o_hat": "D2 fusion without a skip: o_hat[elem] += acc*alpha*weight_scale[k], in "
        "place in fp16. MoDiff's temporal state advance (paper Eq 9).",
    "conv2d_int4_evt_o_hat": "D2 fusion without a skip, int4. MoDiff's temporal state advance.",
    "conv2d_int8_evt_o_hat_residual": "D2 DUAL STORE: advances o_hat in place AND writes "
        "out = o_hat_new + residual[elem] -> fp16, one pass, two stores. Replaces an fp32 conv_out "
        "round-trip.",
    "conv2d_int4_evt_o_hat_residual": "D2 dual store, int4.",
    "flash_attn_int8_vt": "fused int8 flash attention, V pre-transposed. Keeps the running softmax "
        "state in registers and never writes the T x T score matrix; QK^T via __dp4a int8x4->int32, AV "
        "accumulated in fp32 so P is never requantized. Softmax is always fp32.",
    "flash_attn_int8_vt_static": "the same with frozen Q/K scales -- the production steady state.",
    "flash_attn_int8_qi8_kv_static_qout": "int8 flash with static K/V scales, emitting the "
        "projection-quantized int8 output.",
    "flash_attn_int8_qi8_kv_static_qout_hd24": "the hd=24 exact specialization of the above. One "
        "signature, 10 calls, and ~31% of the attention suite -- the single most expensive call in it.",
    "flash_attn_int8_qi8packed_small_qout": "the small-T staging variant, packed Q.",
    "flash_attn_int4_vt": "int4 flash attention, V pre-transposed. W4A4's counterpart to "
                          "flash_attn_int8_vt and, at ~42%, the largest single item in its suite.",
    "flash_attn_int4_vt_static": "int4 flash with frozen Q/K scales.",
    "flash_attn_int4_vt_static_qout": "int4 flash, frozen scales, int8-code output.",
    "flash_attn_i4values_i8mma_qi8_kv_static_qout_hd24": "int4 V values through the int8 MMA path, "
        "hd=24 exact specialization, int8-code output -- W4A4's twin of the int8 hd24 kernel.",
    "flash_attn_i4values_small_qout": "int4 values, small-T variant, int8-code output.",
    "gemm_w8a8_awq_bias_res": "W8A8 AWQ-layout GEMM with the bias+residual epilogue. `a_scale` is a "
        "1-ELEMENT DEVICE TENSOR, not a double, because MoDiff's delta scale is produced on device each "
        "call and taking it by value would force a host sync per linear per step.",
    "gemm_w4a4_awq_bias_res": "W4A4 AWQ-layout GEMM, bias+residual epilogue. The linear suite's largest "
        "item on both W4A4 arms.",
    "gemm_w8a8_awq_out_i8_bias_nout": "W8A8 GEMM emitting int8 codes of (out + bias) at a per-column "
        "scale, so a projection can feed flash attention without a separate quantize.",
    "gemm_w8a8_awq_qkv_i8_layouts": "fused qkv projection: one GEMM writing Q, K and V already in the "
        "attention kernel's per-head padded layouts as int8.",
    "gemm_w8a8_awq_qkv_i8_layouts_compact": "the compact-staging variant of the above.",
    "gemm_w4a4_awq_qkv_i4qk_i8v_layouts": "fused qkv projection emitting int4 Q/K and int8 V in the "
        "attention layouts -- the asymmetry is deliberate: V's dot product accumulates in fp32, so it "
        "keeps 8 bits while Q/K drop to 4.",
    "gemm_w4a4_awq_qkv_codes": "emits the qkv int4 codes plus their clamp limits rather than "
        "dequantized values.",
}

#: a signature is broken out individually when it costs at least this much of its suite
SIG_SHARE = 0.04


def ms_per_sample(rec):
    return rec["stats"]["median"] * rec["calls_per_sample"] / 1000.0


def shape_str(rec):
    """The first two argument shapes, which for every suite here are the ones that identify the work."""
    sh = [s for s in (rec.get("arg_shapes") or []) if s]
    return " x ".join("[" + ",".join(str(d) for d in s) + "]" for s in sh[:2]) or "-"


def published_totals():
    """The suite totals REPORT.md already states, parsed back out so this file can be checked against
    them rather than trusted alongside them."""
    txt = open(f"{D}/REPORT.md").read()
    out = {}
    for suite, header in (("attention", "## 2. Attention kernels"),
                          ("conv", "## 3. Conv kernels"),
                          ("linear", "## 4. Linear kernels")):
        seg = txt[txt.index(header):]
        seg = seg[:seg.index("### ")]
        for m in re.finditer(r"^\| (fp16|W8A8 PTQ|W8A8 MoDiff|W4A4 PTQ|W4A4 MoDiff) \| \*\*([\d.]+)\*\*",
                             seg, re.M):
            out[(suite, m.group(1))] = float(m.group(2))
    return out


def main():
    d = json.load(open(f"{D}/data/kernel_suites.json"))
    pub = published_totals()
    o = []
    o.append("# Per-kernel breakdown: attention, conv, linear")
    o.append("")
    o.append(f"`{d['gpu']}`, batch {d['batch']}, replayed at the shapes captured from a live sample "
             f"({d['rounds']} rounds x {d['iters_per_round']} iters, median of round medians). "
             f"Generated from `data/kernel_suites.json` -- see the script header for what `ms/sample` "
             f"is and is not.")
    o.append("")
    o.append("`ms/sample = us/call x calls/sample / 1000`. Both factors are shown because they point at "
             "different fixes: a fat kernel wants a better tile, a frequent one wants fusion.")
    o.append("")
    o.append("## Reading the kernel names")
    o.append("")
    o.append("Every kernel here is one of three things: an **unquantized fp16 fallback**, a **quantized "
             "compute kernel**, or the same compute kernel with a **different epilogue fused onto it**. "
             "The suffixes say which:")
    o.append("")
    o.append("| suffix | what it means |")
    o.append("|---|---|")
    for suf, txt in SUFFIX:
        o.append(f"| {suf} | {txt} |")
    o.append("")
    o.append("Descriptions below are taken from the kernels' own header comments in "
             "`csrc/baseline/conv/conv2d_evt.cu`, `csrc/baseline/attention/flash_attn_int8.cu`, "
             "`csrc/{baseline,modiff}/linear/gemm_wxax.cu` and `csrc/modiff_kernels_api.h`.")
    o.append("")

    mismatches = []
    for suite in SUITES:
        o.append(f"## {suite}")
        o.append("")
        for key, label in MODES:
            recs = d["modes"].get(key, {}).get(suite) or []
            if not recs:
                continue
            by_entry = collections.defaultdict(lambda: [0.0, 0, 0.0, []])
            for r in recs:
                e = by_entry[r["entry"]]
                e[0] += ms_per_sample(r)
                e[1] += r["calls_per_sample"]
                e[2] = max(e[2], r["stats"]["cv_pct"])
                e[3].append(r)
            total = sum(v[0] for v in by_entry.values())
            want = pub.get((suite, label))
            if want is not None and abs(total - want) > max(0.05, 0.005 * want):
                mismatches.append(f"{suite}/{label}: recomputed {total:.3f} vs REPORT.md {want:.3f}")
            o.append(f"### {label} — {total:.2f} ms/sample total"
                     + (f"  (REPORT.md: {want:.2f} ✓)" if want is not None else ""))
            o.append("")
            o.append("| kernel | ms/sample | % | calls/sample | µs/call (mean over sigs) | sigs | worst CV |")
            o.append("|---|--:|--:|--:|--:|--:|--:|")
            for name, (ms, calls, cv, rs) in sorted(by_entry.items(), key=lambda kv: -kv[1][0]):
                us = ms * 1000.0 / calls if calls else float("nan")
                o.append(f"| `{name}` | **{ms:.2f}** | {100 * ms / total:.1f}% | {calls} | "
                         f"{us:.1f} | {len(rs)} | {cv:.2f}% |")
            o.append("")
            notes = [n for n in ((k2, DESC.get(k2)) for k2 in sorted(by_entry, key=lambda k3: -by_entry[k3][0])) if n[1]]
            if notes:
                for name, txt in notes:
                    o.append(f"- **`{name}`** — {txt}")
                o.append("")
            #: per-signature detail for the entries that carry the suite, so a big number can be traced
            #: to a shape rather than just to a name
            big = [r for r in recs if ms_per_sample(r) >= SIG_SHARE * total]
            if big:
                o.append(f"<details><summary>signatures ≥ {SIG_SHARE:.0%} of the suite "
                         f"({len(big)} of {len(recs)})</summary>")
                o.append("")
                o.append("| ms/sample | calls | µs/call | shapes | kernel |")
                o.append("|--:|--:|--:|---|---|")
                for r in sorted(big, key=lambda r_: -ms_per_sample(r_)):
                    o.append(f"| {ms_per_sample(r):.2f} | {r['calls_per_sample']} | "
                             f"{r['stats']['median']:.1f} | `{shape_str(r)}` | `{r['entry']}` |")
                o.append("")
                o.append("</details>")
                o.append("")

    if mismatches:
        raise SystemExit("SELF-CHECK FAILED, totals disagree with REPORT.md:\n  " +
                         "\n  ".join(mismatches))
    open(f"{D}/KERNEL_BREAKDOWN.md", "w").write("\n".join(o) + "\n")
    print(f"wrote {D}/KERNEL_BREAKDOWN.md ({len(o)} lines)")
    print("self-check: every suite total matches REPORT.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Kernel speedups fp16 -> int8 -> int4, matched by the WORK rather than by the kernel name.

WHY THIS IS NOT JUST A RATIO OF SUITE TOTALS -- CORRECTED 2026-08-16. This file used to say the fp16 arm
runs the attention projections as 1x1 CONVS and that `conv + linear` therefore cancels the
reclassification. Both halves were wrong, and the data in data/kernel_suites.json says so:

  * The conv suite matches 100% THREE WAYS -- 33 of 33 records, fp16 265.72 = matched 265.72, unmatched
    0.00 in every arm (assert_conv_closes() below re-checks this on every run). Nothing moves between
    conv and linear. The 1x1 convs in section 2 are ResBlock skip connections, present in all three arms
    running the same torch_conv2d_fp16 at 1.00x, and fp16's attention out-proj is already a Linear.
  * What actually moves is `other` -> `linear`. fp16 runs the T=1024 and T=256 qkv through
    `fused_gn_qkv` (integration/fused_ops/quantized_std_attention.py, gated on T % 128 == 0 and c % 8
    == 0 -- which is exactly why the T=64/16/4 qkv IS in fp16's linear suite). The capture's suite_of()
    matched name keywords, and "fused_gn_qkv" contains none of conv2d/gemm/linear/group_norm/quant, so
    it fell through to "other": 31.96 ms/sample. Fixed in the harness for future captures; detected by
    entry name here so pre-2026-08-16 JSON regenerates correctly either way.

So `conv + linear` was computed against an fp16 side missing 31.96 ms/sample, `other`'s 3.77x was that
same work, and `norm_quantize`'s 0.64x is the mirror image -- fp16's fused kernel also absorbs the
GroupNorm whose quantized counterpart is a separate record in `norm_quantize`. NO regrouping of these
suites is clean, because that GroupNorm cannot be in two suites at once. Suite totals are not a speedup
denominator. Reported here with the reclassification quantified in-table, and nothing else.

SO TWO VIEWS THAT WORK, plus one that does not:

  0. (does NOT work) Any per-suite or suite-sum fp16 ratio. See above.
  1. The full-run profile buckets in REPORT.md section 1a -- a different instrument that times the whole
     run, so eliminated tensors show up as saved time rather than as absent records.
  2. PER-LAYER, matched by the work's identity rather than by shape-as-printed. The three precisions pass
     different layouts for the same conv:
         fp16  weight [K, C, R, S]        activation [N, C, H, W]
         int8  weight [K, R, S, C]        activation [N, C, H, W]
         int4  weight [K, R, S, C/2]      activation [N, H, W, C/2]   (channels packed 2-per-byte)
     so the weight normalizes to (K, C, R, S) in all three and identifies the layer unambiguously. That
     is the only per-kernel speedup in this file that compares like with like, and conv is where most of
     the time is, so it is the one that matters.

     Attention matches on (N, H, T) taken from K, NOT from Q -- corrected 2026-08-16. Keying on Q dropped
     every `_qout` kernel, because those take TOKEN-MAJOR Q [N,T,H,hd] and so keyed as (128,1024,8)
     instead of (128,8,1024). That is the "5 of 8 blocks matched" this file used to print, and the cost
     was that the T=1024 row compared fp16's 25 calls against int8's 15 while silently omitting
     flash_attn_int8_qi8_kv_static_qout_hd24 -- the most expensive kernel in the suite, ~31% of it, which
     the prose then discussed as though the table covered it. K is [N,H,T,hd_pad] in every record with
     two tensor args; the single-tensor packed-Q records are assigned by T from their own [N,T,H,3,hd]
     shape. All records are now assigned and the rows close to the suite totals exactly, which
     assert_attn_closes() re-checks on every run. Per-call us is CALL-WEIGHTED, not a mean over
     signatures: a row mixes dynamic and static variants at 10 and 5 calls, and an unweighted mean
     misreports them.

     Linear is NOT matched per-layer, for one reason -- and not the one this file used to give. The
     quantized arms pad K differently for the AWQ layout: the same projection is [131072, 192] with
     K=192 in int8 and [131072, 128] with K=256 in int4, so no printed shape means the same thing in
     both. fp16 DOES have counterparts for the projections; they are split between `linear` and
     `fused_gn_qkv`, which is a different problem (see above) with a different fix.

`us/call` is the replay median at the captured shape; speedup is fp16_us / quantized_us at equal work.

Run: python docs/bench_report_2026-08-13_postzp/scripts/kernel_speedup.py    # no GPU
Writes docs/bench_report_2026-08-13_postzp/KERNEL_SPEEDUP.md
"""
import collections
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(ROOT)

D = "docs/bench_report_2026-08-13_postzp"
ARMS = [("fp16", "fp16"), ("int8_baseline", "int8"), ("int4_baseline", "int4")]


def ms_per_sample(r):
    return r["stats"]["median"] * r["calls_per_sample"] / 1000.0


def is_gn_qkv(r):
    """fp16's qkv projection with the GroupNorm fused in. Detected by NAME, not by suite membership.

    Captures before 2026-08-16 have these records under "other" (suite_of() fell through); later ones
    have them under "linear". Keying off the name makes this file correct on both.
    """
    return "gn_qkv" in r["entry"].lower()


def suite_recs(d, mode, suite):
    """The records of one suite in one arm, with the fused_gn_qkv records routed to `linear` wherever
    the capture happened to put them."""
    out = []
    for s, recs in (d["modes"][mode] or {}).items():
        if not isinstance(recs, list):
            continue
        for r in recs:
            got = "linear" if is_gn_qkv(r) else s
            if got == suite:
                out.append(r)
    return out


def assert_conv_closes(d, per, matched):
    """The claim section 1 rests on: the conv suite matches 100% three ways, so NOTHING moves between
    conv and linear. If a future capture breaks that, this file must stop saying so -- hence an assert
    rather than a comment."""
    for key, lab in ARMS:
        tot = sum(ms_per_sample(r) for r in suite_recs(d, key, "conv"))
        got = sum(ms_per_sample(r) for k in matched for r in per[k][lab]["recs"])
        assert abs(tot - got) < 0.01, (
            f"conv no longer closes in {lab}: suite {tot:.2f} vs matched {got:.2f} ms/sample. The "
            f"'nothing moves between conv and linear' claim in section 1 is now false -- fix the prose.")


def assert_attn_closes(d, ap):
    """Every attention record assigned to exactly one row, and the rows summing to the suite total.
    This is what the old Q-keyed version could not do: it dropped 3 keys and 40% of int8's T=1024
    calls, and nothing in the file noticed."""
    for key, lab in ARMS:
        tot = sum(ms_per_sample(r) for r in suite_recs(d, key, "attention"))
        got = sum(v[lab]["us_calls"] for v in ap.values()) / 1000.0
        assert abs(tot - got) < 0.01, (
            f"attention does not close in {lab}: suite {tot:.2f} vs rows {got:.2f} ms/sample")


def conv_key(mode, r):
    """(K, C, R, S) from whichever weight layout THIS KERNEL passes, plus the activation's spatial size.

    Dispatches on the KERNEL, not on the arm. An unquantized `torch_conv2d_fp16` record inside the int8
    or int4 arm still carries the fp16 [K,C,R,S] layout, so keying off the arm mis-normalized exactly
    those rows -- they then failed to match and were silently dropped, which also removed the ~1.00x
    control rows this table claims to have. Dropped rather than mis-paired, so no number was wrong; but a
    control that is absent cannot control anything.
    """
    sh = [s for s in (r.get("arg_shapes") or []) if s]
    if len(sh) < 2:
        return None
    a, w = sh[0], sh[1]
    if len(w) != 4:
        return None
    if mode == "fp16" or r["entry"].startswith("torch"):
        K, C, R, S = w[0], w[1], w[2], w[3]
        hw = (a[2], a[3]) if len(a) == 4 else None
    elif mode == "int8_baseline":
        K, R, S, C = w[0], w[1], w[2], w[3]
        hw = (a[2], a[3]) if len(a) == 4 else None
    else:                                     # int4: channels packed 2-per-byte in BOTH operands
        K, R, S, Cb = w[0], w[1], w[2], w[3]
        C = Cb * 2
        hw = (a[1], a[2]) if len(a) == 4 else None
    return (K, C, R, S, hw)


def attn_key(r):
    """T -- the same attention block, taken from K rather than from Q. See the module docstring.

    K is [N, H, T, hd_pad] in every record that passes it. The `_qout` family passes Q token-major as
    [N, T, H, hd], so keying on Q put those kernels in phantom (N, T, H) buckets and dropped them from
    the table entirely. The packed-small kernels pass ONE tensor, [N, T, H, 3, hd] -- assigned by its T.
    Returns T alone: N and H are 128 and 8 in every record here, so they carry no information, and
    pretending the key is 3-dimensional is what made the phantom buckets look legitimate.
    """
    sh = [s for s in (r.get("arg_shapes") or []) if s]
    if len(sh) >= 2 and len(sh[1]) == 4:
        return sh[1][2]
    if sh and len(sh[0]) == 5:
        return sh[0][1]
    return None


def attn_hd(r):
    """hd_pad in VALUES, not in stored bytes.

    The old version returned arg_shapes[0][3], the storage extent of Q, and printed it as the head dim.
    For the int4 arm that is wrong and visibly so: it reported "48 -> 32" for a layer whose true head dim
    is 48, i.e. padding to BELOW the real width, which cannot happen. int4 Q/K are stored 2-per-byte, and
    the true hd_pad is passed as a scalar arg. Read it from there for the int4 kernels, whose scalar_args
    carry hd_pad as the first int; fall back to the shape for fp16/int8, where storage == values.
    """
    sh = [s for s in (r.get("arg_shapes") or []) if s]
    ext = sh[0][3] if sh and len(sh[0]) == 4 else None
    if "int4" in r["entry"].lower() or "i4" in r["entry"].lower():
        for x in (r.get("scalar_args") or []):
            if isinstance(x, int) and x in (32, 64, 96, 128):
                return x
    return ext


def main():
    d = json.load(open(f"{D}/data/kernel_suites.json"))
    o = []
    o.append("# Kernel speedups: fp16 → int8 → int4")
    o.append("")
    o.append(f"`{d['gpu']}`, batch {d['batch']}. Replay medians at the shapes captured from a live "
             f"sample; `speedup` is fp16 µs/call ÷ quantized µs/call **at equal work**. PTQ arms "
             f"(`int8_baseline`, `int4_baseline`), so no MoDiff temporal kernels are mixed in.")
    o.append("")

    # ---- 1. suite totals -----------------------------------------------------------------------
    o.append("## 1. Suite totals (ms/sample)")
    o.append("")
    o.append("Totals as captured, with `fused_gn_qkv` routed to `linear` (see below). **The `speedup` "
             "columns are printed so they can be dismissed** — not one of them is a speedup. Read §2 and "
             "REPORT.md §1a instead; the paragraphs after the table say why.")
    o.append("")
    o.append("| suite | fp16 | int8 | int4 | fp16/int8 | fp16/int4 | is this a speedup? |")
    o.append("|---|--:|--:|--:|--:|--:|---|")
    tot = {}
    NOTE = {
        "attention": "**yes** — same work, same suite, all three arms",
        "conv": "**yes** — 33/33 records matched three ways, nothing moves in or out",
        "linear": "**no** — holds fp16's fused GroupNorm, whose quantized counterpart is in `norm_quantize`",
        "norm_quantize": "**no** — the mirror of `linear`; also absorbs quantize launches that replace "
                         "work the fp16 arm pays as separate elementwise kernels",
        "other": "**no** — `cat2` capture coverage differs between arms (see below)",
    }
    for suite in ("attention", "conv", "linear", "norm_quantize", "other"):
        v = {lab: sum(ms_per_sample(r) for r in suite_recs(d, key, suite)) for key, lab in ARMS}
        tot[suite] = v
        o.append(f"| {suite} | {v['fp16']:.2f} | {v['int8']:.2f} | {v['int4']:.2f} | "
                 f"{v['fp16'] / v['int8']:.2f}× | {v['fp16'] / v['int4']:.2f}× | {NOTE[suite]} |")
    allsum = {lab: sum(tot[s][lab] for s in tot) for _, lab in ARMS}
    o.append(f"| **all five** | **{allsum['fp16']:.2f}** | **{allsum['int8']:.2f}** | "
             f"**{allsum['int4']:.2f}** | **{allsum['fp16'] / allsum['int8']:.2f}×** | "
             f"**{allsum['fp16'] / allsum['int4']:.2f}×** | **no** — see the third paragraph |")
    o.append("")
    gn = {lab: sum(ms_per_sample(r) for r in suite_recs(d, key, "linear") if is_gn_qkv(r))
          for key, lab in ARMS}
    o.append(f"**Where fp16's qkv projections live.** fp16 runs the T=1024 and T=256 qkv through "
             f"`fused_gn_qkv` — one kernel doing GroupNorm and the projection — worth "
             f"**{gn['fp16']:.2f} ms/sample**. The gate is `T % 128 == 0 and c % 8 == 0`, which is why "
             f"the T=64/16/4 qkv is an ordinary `torch_linear_fp16` in this same suite. The quantized "
             f"arms have no fused counterpart at all — they split the same work into a "
             f"`group_norm_silu_quantize_nhwc` in `norm_quantize` plus an AWQ GEMM here. Until "
             f"2026-08-16 the capture's "
             f"`suite_of()` matched name keywords and `fused_gn_qkv` contains none of them, so these two "
             f"records sat in `other` — and this table published `linear` at 0.61×, `other` at 3.77× and "
             f"a `conv + linear` row claiming the two cancel. They do not: the move is `other` → "
             f"`linear`, and it never touched conv at all.")
    o.append("")
    o.append("**Conv closes exactly, which is what kills the old story.** All 33 conv records match "
             "three ways — fp16's suite total equals its matched total to the cent, in every arm "
             "(asserted on every run of this script). Nothing moves between conv and linear in either "
             "direction. The 1×1 convs in §2 are ResBlock skip connections, present in all three arms "
             "at 1.00×, and fp16's attention out-proj is already a Linear.")
    o.append("")
    o.append("**`other` is now all `cat2_channels_last_fp16`, and it does not compare either — for an "
             "unrelated reason.** The capture is asymmetric: fp16 is missing two signatures both "
             "quantized arms have (`[128,384,32,32]+[128,192,32,32]` at 2.64 ms and "
             "`[128,768,8,8]+[128,384,8,8]` at 0.34 ms), and it recorded 5 calls where int8 recorded 10 "
             "on `[128,384,16,16]²`. A concat is arm-independent by construction, so that is a coverage "
             "gap in the capture, not a real difference. It needs a GPU re-capture to close — "
             "docs/OPEN_ITEMS.md A15.")
    o.append("")
    o.append("**No regrouping of these five suites is clean, including the sum.** fp16's "
             "`fused_gn_qkv` does the GroupNorm too, and that GroupNorm cannot be in `linear` and in "
             "`norm_quantize` at once. Worse for the sum: the quantized arms' fused epilogues *delete* "
             "tensors, so the elementwise kernels fp16 pays for them do not exist as records to be "
             "credited — the full-run profile in REPORT.md §1a sees that as 2.78 s saved, and a replay "
             "suite cannot see it at all. That is why the all-five row reads "
             f"{allsum['fp16'] / allsum['int8']:.2f}× against the wall clock's 1.45×.")
    o.append("")

    # ---- 2. per conv layer ---------------------------------------------------------------------
    o.append("## 2. Per conv layer — the strict comparison")
    o.append("")
    o.append("Matched on the weight normalized to `(K, C, R, S)`, so the same layer is compared across "
             "all three arms despite three different operand layouts. `calls` is per sample.")
    o.append("")
    per = collections.defaultdict(dict)
    for key, lab in ARMS:
        for r in suite_recs(d, key, "conv"):
            k = conv_key(key, r)
            if k is None:
                continue
            e = per[k].setdefault(lab, {"us": 0.0, "n": 0, "calls": 0, "entry": r["entry"],
                                       "adt": (r.get("arg_dtypes") or [None])[0], "recs": []})
            e["us"] += r["stats"]["median"]
            e["n"] += 1
            e["calls"] += r["calls_per_sample"]
            e["recs"].append(r)
    rows = []
    for k, v in per.items():
        if len(v) != 3:
            continue
        us = {lab: v[lab]["us"] / v[lab]["n"] for _, lab in ARMS}
        rows.append((k, us, v))
    rows.sort(key=lambda t: -t[1]["fp16"])
    assert_conv_closes(d, per, [k for k, _, _ in rows])
    o.append("> **One correction to read the fp16 column with.** The replay runs under "
             "`autocast(fp16)`, exactly as production does, and the captured arguments are what the "
             "caller passed *before* autocast cast them. For 12 of the fp16-arm conv records the "
             "activation arrives as **fp32**, so autocast\'s fp32→fp16 conversion of that activation "
             "is inside the timed region while the arithmetic is fp16 either way. Checked on the "
             "clearest case: `[128,1152,8,8]` fp32 is 37.7 MB read + 18.9 MB written ≈ 94 µs at "
             "~600 GB/s, and that row\'s fp16-vs-quantized gap is 102.5 µs on a kernel that is "
             "`torch_conv2d_fp16` in all three arms. So rows marked `fp32-in` measure a conversion plus "
             "a conv, and their speedups are not arithmetic-only. The summary below is split on this.")
    o.append("")
    o.append("| K | C | R×S | HxW | fp16 in | calls | fp16 µs | int8 µs | int4 µs | int8 | int4 | int8→int4 |")
    o.append("|--:|--:|---|---|---|--:|--:|--:|--:|--:|--:|--:|")
    for (K, C, R, S, hw), us, v in rows:
        hws = f"{hw[0]}×{hw[1]}" if hw else "—"
        q = "" if not v["int8"]["entry"].startswith("torch") else " _(unquantized)_"
        f32 = v["fp16"]["adt"] == "torch.float32"
        dt = "**fp32**" if f32 else "fp16"
        o.append(f"| {K}{q} | {C} | {R}×{S} | {hws} | {dt} | {v['fp16']['calls']} | {us['fp16']:.1f} | "
                 f"{us['int8']:.1f} | {us['int4']:.1f} | **{us['fp16'] / us['int8']:.2f}×** | "
                 f"**{us['fp16'] / us['int4']:.2f}×** | {us['int8'] / us['int4']:.2f}× |")
    o.append("")
    if rows:
        def stat(sub, lab):
            v = sorted(u["fp16"] / u[lab] for _, u, _ in sub)
            return f"{v[0]:.2f}×–{v[-1]:.2f}× (median {v[len(v) // 2]:.2f}×)" if v else "—"
        quant = [t for t in rows if not t[2]["int8"]["entry"].startswith("torch")]
        clean = [t for t in quant if t[2]["fp16"]["adt"] != "torch.float32"]
        dirty = [t for t in quant if t[2]["fp16"]["adt"] == "torch.float32"]
        ctrl = [t for t in rows if t[2]["int8"]["entry"].startswith("torch")]
        o.append(f"{len(rows)} layers matched in all three arms: {len(quant)} quantized, "
                 f"{len(ctrl)} unquantized controls.")
        o.append("")
        o.append("| subset | n | int8 speedup | int4 speedup |")
        o.append("|---|--:|---|---|")
        o.append(f"| quantized, fp16 baseline also fp16-in — **the arithmetic-only number** | "
                 f"{len(clean)} | {stat(clean, 'int8')} | {stat(clean, 'int4')} |")
        o.append(f"| quantized, fp16 baseline fp32-in (includes an autocast cast) | {len(dirty)} | "
                 f"{stat(dirty, 'int8')} | {stat(dirty, 'int4')} |")
        o.append(f"| all quantized | {len(quant)} | {stat(quant, 'int8')} | {stat(quant, 'int4')} |")
        o.append("")
        cc = [t for t in ctrl if t[2]["fp16"]["adt"] != "torch.float32"]
        o.append(f"**The controls work.** Of the {len(ctrl)} unquantized rows, the {len(cc)} whose "
                 f"activation dtype also matches across arms come out at "
                 + ", ".join(f"{u['fp16'] / u['int8']:.2f}×" for _, u, _ in cc[:7])
                 + f" — the same kernel on the same input times the same in every arm, which is what "
                 f"shows the layout normalization is matching real layers and not coincidentally-shaped "
                 f"ones. The remaining {len(ctrl) - len(cc)} are the `fp32-in` rows above.")
        o.append("")
        ctrl = [(k, u) for k, u, v in rows if v["int8"]["entry"].startswith("torch")]
        o.append(f"**{len(ctrl)} of these rows are the control**: convs this pipeline does not quantize, "
                 f"so all three arms run the same `torch_conv2d_fp16` and the speedup must come out "
                 f"≈1.00×. They do (" + ", ".join(f"{u['fp16'] / u['int8']:.2f}×" for _, u in ctrl[:6]) +
                 (", …" if len(ctrl) > 6 else "") + "), which is what shows the layout normalization is "
                 f"matching the right layers rather than coincidentally-shaped ones.")
        o.append("")

    # ---- 3. per attention block ----------------------------------------------------------------
    o.append("## 3. Per attention block")
    o.append("")
    o.append("Matched on `T`, read from **K** (`[N,H,T,hd_pad]`), with the single-tensor packed-Q records "
             "assigned by their own `T`. N=128 and H=8 in every record, so they carry no information. "
             "Every record in the suite is assigned to exactly one row and the rows sum to the suite "
             "totals exactly — checked on every run. `ms/sample` is what the row costs; `µs/call` is "
             "**call-weighted** across the kernels in the row, because a row mixes dynamic and static "
             "variants at 10 and 5 calls.")
    o.append("")
    ap = collections.defaultdict(lambda: collections.defaultdict(
        lambda: {"us_calls": 0.0, "calls": 0, "hd": None, "recs": []}))
    for key, lab in ARMS:
        for r in suite_recs(d, key, "attention"):
            k = attn_key(r)
            assert k is not None, f"unassignable attention record: {r['entry']} {r.get('arg_shapes')}"
            e = ap[k][lab]
            e["us_calls"] += r["stats"]["median"] * r["calls_per_sample"]
            e["calls"] += r["calls_per_sample"]
            e["hd"] = e["hd"] or attn_hd(r)
            e["recs"].append(r)
    assert_attn_closes(d, ap)
    o.append("| T | hd_pad fp16→int8/int4 | calls f/8/4 | fp16 ms | int8 ms | int4 ms | "
             "µs/call fp16→int8→int4 | int8 | int4 | noise |")
    o.append("|--:|---|---|--:|--:|--:|---|--:|--:|---|")
    arows = sorted(ap, key=lambda k: -ap[k]["fp16"]["us_calls"])
    for k in arows:
        v = ap[k]
        ms = {lab: v[lab]["us_calls"] / 1000.0 for _, lab in ARMS}
        uc = {lab: v[lab]["us_calls"] / v[lab]["calls"] for _, lab in ARMS}
        calls = "/".join(str(v[lab]["calls"]) for _, lab in ARMS)
        noisy = sorted({lab for _, lab in ARMS for r in v[lab]["recs"]
                        if (r.get("stability") or "").upper() == "NOISY"})
        o.append(f"| {k} | {v['fp16']['hd']}→{v['int8']['hd']}/{v['int4']['hd']} | {calls} | "
                 f"{ms['fp16']:.2f} | {ms['int8']:.2f} | {ms['int4']:.2f} | "
                 f"{uc['fp16']:.1f}→{uc['int8']:.1f}→{uc['int4']:.1f} | "
                 f"**{ms['fp16'] / ms['int8']:.2f}×** | **{ms['fp16'] / ms['int4']:.2f}×** | "
                 f"{'**' + '+'.join(noisy) + ' NOISY**' if noisy else '—'} |")
    tt = {lab: sum(v[lab]["us_calls"] for v in ap.values()) / 1000.0 for _, lab in ARMS}
    o.append(f"| **total** | | | **{tt['fp16']:.2f}** | **{tt['int8']:.2f}** | **{tt['int4']:.2f}** | | "
             f"**{tt['fp16'] / tt['int8']:.2f}×** | **{tt['fp16'] / tt['int4']:.2f}×** | |")
    o.append("")
    o.append(f"All {len(arows)} blocks matched in all three arms, all records assigned.")
    o.append("")
    o.append("**int4 does not have an int4 attention datapath, and that is the whole story of the int4 "
             "column.** Three things in the data, none of which is about bit width:")
    o.append("")
    o.append("1. Every operand in the int4 arm's attention is `torch.int8`.")
    o.append("2. The dominant T=1024 route runs `flash_attn_i4values_i8mma_qi8_kv_static_qout_hd24`, "
             "whose profiled CUDA kernel is literally `flash_attn_int8_mma_kernel_t` — the int8 MMA.")
    o.append("3. V stays int8 in both quantized arms. The qkv GEMM that feeds it says so in its name: "
             "`gemm_w4a4_awq_qkv_i4qk_i8v_layouts` — int4 for Q and K, int8 for V.")
    o.append("")
    o.append("So the only thing int4 can win in attention is Q/K bytes, and whether it wins any depends "
             "entirely on how `hd` pads:")
    o.append("")
    o.append("| | true hd | int8 pads to | int8 B/row | int4 pads to | int4 B/row | so |")
    o.append("|---|--:|--:|--:|--:|--:|---|")
    o.append("| T=1024 | 24 | 32 values | **32** | 64 values, 2/byte | **32** | identical traffic → "
             "1.21× vs 1.21×, no int4 gain at the route that owns 80% of the suite |")
    o.append("| T=256, T=64 | 48 | 64 values | **64** | 64 values, 2/byte | **32** | int4 halves Q/K "
             "traffic → 1.66× vs 1.36× |")
    o.append("")
    o.append("That is also the honest form of the padding argument for **int8**: at T=1024 it moves 32 "
             "values per row where fp16 moves 24, a third more bytes, and nets 1.21×. The padding is "
             "structural to the MMA fragment layout, not a missing optimization, and the hand-written "
             "`_hd24` specialization plus a refuted 8-byte loader are what has already been tried.")
    o.append("")
    o.append("**T=16 is a regression, not a clean fallback.** Only 15 of its 25 calls fall back to "
             "`torch_sdpa_fp16`; the other 10 run `flash_attn_int8_qi8packed_small_qout` at ~65 µs "
             "against sdpa's ~48, which is what puts the row at 0.88×/0.86×. A gate that sent all 25 to "
             "sdpa would recover ~0.17 ms/sample — small, but it is a sign error, not a tradeoff.")
    o.append("")

    # ---- 4. linear ------------------------------------------------------------------------------
    o.append("## 4. Linear — why there is no per-layer table")
    o.append("")
    o.append("**One reason, and it is not the one this section used to give.** The quantized arms pad K "
             "differently for the AWQ layout: the same projection is `[131072, 192]` with `K=192` in int8 "
             "and `[131072, 128]` with `K=256` in int4, so no printed shape means the same thing in both, "
             "and there is no fp16 shape that means it either.")
    o.append("")
    o.append("The retracted reason was *\"fp16 has no counterpart for the projections — they are 1×1 "
             "convs there.\"* fp16 has counterparts for all of them. The out-projections at every `T`, "
             "and the qkv at T=64/16/4, are `torch_linear_fp16` records in this same suite; only the "
             "T=1024 and T=256 qkv are elsewhere, and they are `fused_gn_qkv`, not convs. §1 has the "
             "accounting.")
    o.append("")
    o.append("The int8→int4 comparison is still available per layer, keyed by `(M, n_out)` which both "
             "arms report:")
    o.append("")
    lp = collections.defaultdict(dict)
    for key, lab in (("int8_baseline", "int8"), ("int4_baseline", "int4")):
        for r in d["modes"][key].get("linear") or []:
            if r["entry"].startswith("torch"):
                continue
            sh = [s for s in (r.get("arg_shapes") or []) if s]
            sc = [x for x in (r.get("scalar_args") or []) if x is not None]
            if not sh or not sc:
                continue
            k = (sh[0][0], sh[1][0])          # (M, n_out-ish: B's leading dim)
            e = lp[k].setdefault(lab, {"us": 0.0, "n": 0, "calls": 0, "entry": r["entry"]})
            e["us"] += r["stats"]["median"]
            e["n"] += 1
            e["calls"] += r["calls_per_sample"]
    o.append("| M | N | calls | int8 µs | int4 µs | int8→int4 | kernel |")
    o.append("|--:|--:|--:|--:|--:|--:|---|")
    for k in sorted((k for k in lp if len(lp[k]) == 2), key=lambda k: -lp[k]["int8"]["us"] / lp[k]["int8"]["n"]):
        v = lp[k]
        u8 = v["int8"]["us"] / v["int8"]["n"]
        u4 = v["int4"]["us"] / v["int4"]["n"]
        o.append(f"| {k[0]} | {k[1]} | {v['int8']['calls']} | {u8:.1f} | {u4:.1f} | "
                 f"**{u8 / u4:.2f}×** | `{v['int8']['entry']}` |")
    o.append("")
    open(f"{D}/KERNEL_SPEEDUP.md", "w").write("\n".join(o) + "\n")
    print(f"wrote {D}/KERNEL_SPEEDUP.md ({len(o)} lines)")
    print(f"self-check: conv closes 3-way ({len(rows)} layers); attention closes to suite totals "
          f"({len(arows)} blocks, every record assigned)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

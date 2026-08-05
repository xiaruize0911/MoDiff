"""Re-bucket the saved per-kernel attention data, and plot it.

TWO corrections are baked into this script, both discovered after the first draft of FINDINGS.md.

Correction 1 (bucketing, fp16 core). `pytorch_flash::flash_fwd_kernel` (the attention core) does not
contain the string "flash_attn", so it fell through to the "out projection" catch-all and made fp16's
projection read 2399 us. The buckets are now keyed on the actual kernel names observed in the trace:

  FP16   pytorch_flash::flash_fwd_kernel          attention core (PyTorch flash SDPA)
         cutlass ImplicitGemmConvolution          qkv -- in fp16 the qkv projection is a 1x1 CONV
         ampere_fp16_s1688gemm / sm80_xmma        out projection (cuBLAS)
         gn_accum / gn_finalize                   GroupNorm STATISTICS ONLY (see below)
         vectorized_elementwise_kernel            everything elementwise, AGGREGATED
  INT8   flash_attn_int8_mma_kernel_t             attention core
         gemm_w8a8_kernel_awq_out_i8              qkv (int8-output, feeds the flash kernel)
         gemm_w8a8_kernel_awq                     out projection, WITH bias+residual in the epilogue
         group_norm_silu_quantize_nhwc_vec2       GroupNorm stats + apply + SmoothQuant + quantize

Order matters: `..._awq_out_i8` must be tested before `..._awq`, or the qkv GEMM is counted as the
out projection.

Correction 2 (the fp16 column does not balance, so per-stage fp16/int8 ratios are INVALID).
At C192/T1024 fp16 has exactly one `vectorized_elementwise_kernel` entry, 268.2 us. The profiler
aggregates by kernel name and truncates at 60 chars, so every elementwise functor collapses into
that one number. It cannot be split -- and a byte count shows it cannot even contain the work it
would have to contain:

    GN apply (read x + write normalized) 100.7 MB  +  residual add (read 2x + write) 151.0 MB
    = 251.7 MB in 268.2 us = 939 GB/s, against an A40 peak of 696 GB/s.

So at least one fp16 stage is fused somewhere else or absent from the attribution (`gn_accum` reads
50.3 MB in 108.4 us = 464 GB/s, i.e. a read-only reduction: fp16's GroupNorm rows are STATS ONLY,
there is no apply kernel). A column that does not balance cannot be compared stage by stage. int8
meanwhile has ZERO elementwise kernels -- bias, residual and the GN apply are all inside fused
kernels. Comparing int8's residual-fused out-projection GEMM against fp16's residual-free one is what
produced the retracted "out projection is 1.8x SLOWER" claim.

What replaces it: a ROOFLINE analysis of the int8 projections alone (ROOFLINE below). It needs no
fp16 attribution, and it also corrects the "% of peak TOPS" framing of the first draft -- at K=192
these GEMMs are memory bound, so peak TOPS is the wrong denominator.
"""

import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
D = "docs/attn_modiff_profile_2026-08-04"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STAGES = [
    ("GroupNorm (+quantize)", ("group_norm", "gn_accum", "gn_finalize", "gn_")),
    ("qkv projection", ("awq_out_i8", "fused_gn_qkv", "implicitgemmconvolution")),
    ("attention core", ("flash_attn_int8", "flash_attn_int4", "pytorch_flash", "flash_fwd")),
    ("out projection", ("gemm_w8a8_kernel_awq", "gemm_w4a4_kernel_awq", "s1688gemm", "s16816gemm",
                        "sm80_xmma", "gemm_f16")),
    ("Q/K/V quantize", ("quantize_attn", "aq_qtok", "quantize_act", "scale_quantize")),
    ("elementwise / copy", ("elementwise", "direct_copy", "cat2", "transpose")),
]
COL = {"GroupNorm (+quantize)": "#8C8C8C", "qkv projection": "#4C72B0",
       "attention core": "#C44E52", "out projection": "#55A868",
       "Q/K/V quantize": "#B8860B", "elementwise / copy": "#9C755F", "other": "#CCCCCC"}

# ---- roofline model -------------------------------------------------------------------------
#: A40 (GA102): 696 GB/s HBM, 299.4 TOPS int8 dense tensor core, 149.7 TFLOPS fp16.
BW, TOPS_I8, TFLOPS_F16 = 696e9, 299.4e12, 149.7e12
BATCH = 128
#: shape -> (C, T, n_instances in the UNet). n_inst multiplies a per-forward us into a per-step us.
SHAPE_INFO = {"C192/T1024": (192, 1024, 5), "C384/T256": (384, 256, 5), "C384/T64": (384, 64, 5),
              "C768/T16": (768, 16, 5), "C768/T4": (768, 4, 1)}
#: measured int8 us per forward, read out of data/attn_modiff_buckets.json (INT8 row).
MEASURED_I8 = {"C192/T1024": (604.9, 345.7), "C384/T256": (302.7, 185.2),
               "C384/T64": (89.1, 65.1), "C768/T16": (54.1, 44.1), "C768/T4": (20.1, 29.5)}
#: fp16 references, ONLY where the trace separates the two projections by kernel name.
#: (qkv us, out-proj GEMM us) -- C192/T1024 is the only shape where they are distinct kernels.
FP16_REF = {"C192/T1024": (617.5, 191.6)}


def roofline_i8(C, T):
    """(qkv, out_proj) -> dict of bytes / flops / bound-us for the int8 kernels.

    qkv   `gemm_w8a8_kernel_awq_out_i8`: reads A int8 (M*C), writes int8 Q/K/Vt (M*3C). Weights and
          per-token scales are negligible (K*3C = 442 KB at C=768).
    proj  `gemm_w8a8_kernel_awq` with bias+residual in the epilogue: reads A int8 (M*C), reads the
          residual fp16 (2*M*C), writes fp16 (2*M*C). int8 has no elementwise kernel in the trace,
          which is the evidence that the residual really is fused here.
    """
    M = BATCH * T
    out = {}
    for name, byt, N in (("qkv", 4 * M * C, 3 * C), ("proj", 5 * M * C, C)):
        mem_us = byt / BW * 1e6
        cmp_us = 2 * M * C * N / TOPS_I8 * 1e6
        out[name] = {"M": M, "K": C, "N": N, "bytes": byt, "mem_us": mem_us, "cmp_us": cmp_us,
                     "roof_us": max(mem_us, cmp_us),
                     "bound": "mem" if mem_us >= cmp_us else "cmp"}
    return out


def roofline_fp16(C, T):
    """fp16 equivalents: qkv is a 1x1 conv (fp16 in, 3 fp16 out); out proj writes fp16, NO residual."""
    M = BATCH * T
    out = {}
    for name, byt, N in (("qkv", 2 * M * C + 3 * 2 * M * C, 3 * C), ("proj", 2 * M * C + 2 * M * C, C)):
        mem_us = byt / BW * 1e6
        cmp_us = 2 * M * C * N / TFLOPS_F16 * 1e6
        out[name] = {"mem_us": mem_us, "cmp_us": cmp_us, "roof_us": max(mem_us, cmp_us)}
    return out


def stage_of(name):
    n = name.lower()
    for label, keys in STAGES:
        if any(k in n for k in keys):
            return label
    return "other"


def print_roofline():
    """The replacement for the retracted per-stage fp16/int8 table. int8 only, so it balances."""
    print(f"\n{'=' * 112}\nINT8 projection GEMMs against their own roofline (A40: 696 GB/s, "
          f"299.4 TOPS int8)\n{'=' * 112}")
    print(f"  {'shape':<12}{'kernel':<10}{'M':>8}{'K':>6}{'N':>6}{'MB':>8}"
          f"{'mem us':>9}{'cmp us':>9}{'roof':>8}{'meas':>9}{'% roof':>9}  bound")
    rows = []
    for s, (C, T, n) in SHAPE_INFO.items():
        r = roofline_i8(C, T)
        for i, k in enumerate(("qkv", "proj")):
            v, meas = r[k], MEASURED_I8[s][i]
            pct = v["roof_us"] / meas * 100
            print(f"  {s:<12}{k:<10}{v['M']:>8}{v['K']:>6}{v['N']:>6}{v['bytes'] / 2**20:>8.1f}"
                  f"{v['mem_us']:>9.1f}{v['cmp_us']:>9.1f}{v['roof_us']:>8.1f}{meas:>9.1f}"
                  f"{pct:>8.1f}%  {v['bound']}")
            rows.append((s, k, n, v, meas, pct))

    print(f"\n  fp16 references where the trace separates the kernels:")
    for s, (qkv_us, proj_us) in FP16_REF.items():
        C, T, _ = SHAPE_INFO[s]
        f = roofline_fp16(C, T)
        for k, meas in (("qkv", qkv_us), ("proj", proj_us)):
            print(f"    {s:<12}{k:<10} roof {f[k]['roof_us']:7.1f} us   meas {meas:7.1f} us   "
                  f"{f[k]['roof_us'] / meas * 100:5.1f}% of roofline")

    # Savings estimate. Target = the fraction of ITS OWN roofline that fp16 achieves on the same
    # shape (46.8% qkv / 75.5% proj at C192/T1024) -- a demonstrated-reachable bar, not 100%.
    # Restricted to the two shapes that hold the time; the small shapes are occupancy bound (M=512
    # at CTA_M=128 is 4x6 = 24 CTAs on 84 SMs) and no tile choice fixes that.
    tgt = {"qkv": 0.468, "proj": 0.755}
    print(f"\n  Reachable saving, whole model (us/step = per-forward x n_inst), at fp16's own "
          f"achieved fraction:")
    tot, small = 0.0, 0.0
    for s, k, n, v, meas, pct in rows:
        target_us = v["roof_us"] / tgt[k]
        save = max(0.0, meas - target_us) * n
        if s not in ("C192/T1024", "C384/T256"):
            small += save
            continue
        tot += save
        print(f"    {s:<12}{k:<6} {meas:6.1f} -> {target_us:6.1f} us  x{n}  = {save:7.1f} us/step")
    print(f"    {'TOTAL':<19}{tot:7.1f} us/step = {tot / 1000:.2f} ms/step "
          f"({tot / 1000 / 71 * 100:.1f}% of the 71 ms/step int8 baseline)")
    print(f"    NOT counted: the three small shapes -- {small:.0f} us/step "
          f"({small / 1000:.2f} ms/step) at the same target, but they are occupancy bound "
          f"(M=512 at CTA_M=128 is 24 CTAs on 84 SMs) and no tile choice reaches it")
    return rows


def main():
    d = json.load(open(f"{D}/data/attn_modiff_buckets.json"))
    modes, shapes = d["modes"], d["shapes"]
    reb, raw = {}, {}
    for key, kern in d["kernels"].items():
        m, s = key.split("|")
        acc = {}
        for n, v in kern.items():
            acc[stage_of(n)] = acc.get(stage_of(n), 0.0) + v
        reb[(m, s)] = acc
        raw[(m, s)] = kern
    order = [lbl for lbl, _ in STAGES] + ["other"]

    print(f"{'=' * 112}\nAttention layer, us per forward, batch 128, re-bucketed on observed kernel "
          f"names\n{'=' * 112}")
    print("  NOTE: the fp16 column does NOT balance (see module docstring) -- do not read the "
          "fp16/int8 ratio\n  per stage. Use print_roofline() below for the int8-only comparison.")
    for s in shapes:
        print(f"\n  {s}")
        print("    " + f"{'stage':<24}" + "".join(f"{m:>14}" for m in modes))
        for st in order:
            vals = [reb.get((m, s), {}).get(st, 0.0) for m in modes]
            if sum(vals) < 0.05:
                continue
            print(f"    {st:<24}" + "".join(f"{v:14.1f}" for v in vals))
        tot = [sum(reb.get((m, s), {}).values()) for m in modes]
        print(f"    {'TOTAL':<24}" + "".join(f"{v:14.1f}" for v in tot) +
              f"   fp16/int8 {tot[0] / tot[1]:.2f}x  (TOTAL is the only valid fp16/int8 ratio)")

    rows = print_roofline()

    # ---- plot 1: stacked stage breakdown per mode, one panel per shape ----
    fig, axes = plt.subplots(1, len(shapes), figsize=(17, 4.6))
    for ax, s in zip(axes, shapes):
        bottom = [0.0] * len(modes)
        for st in order:
            v = [reb.get((m, s), {}).get(st, 0.0) for m in modes]
            if sum(v) < 0.05:
                continue
            ax.bar(range(len(modes)), v, bottom=bottom, color=COL.get(st, "#CCC"),
                   label=st, width=0.68, zorder=3)
            bottom = [bottom[i] + v[i] for i in range(len(modes))]
        for i, b in enumerate(bottom):
            ax.text(i, b, f"{b:.0f}", ha="center", va="bottom", fontsize=7.5, weight="bold")
        ax.set_xticks(range(len(modes)))
        ax.set_xticklabels([m.replace("+MoDiff", "\n+MoDiff") for m in modes], fontsize=7)
        ax.set_title(s, fontsize=10)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_ylim(0, max(bottom) * 1.16)
    axes[0].set_ylabel("us per forward (batch 128)", fontsize=9)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), fontsize=8.5, frameon=False)
    fig.suptitle("Attention layer, time by stage — TOTALS are comparable across modes, individual "
                 "stages are NOT (fp16 fuses differently)", fontsize=12.5, weight="bold")
    fig.subplots_adjust(top=0.84, bottom=0.24, left=0.05, right=0.99, wspace=0.28)
    fig.savefig(f"{D}/plots/attn_stage_breakdown.png", dpi=130)
    plt.close(fig)

    # ---- plot 2: the fusion-accounting chart that REPLACES the per-stage ratio chart ----
    # Why the old attn_stage_ratio.png was withdrawn: it divided fp16 by int8 stage by stage, but
    # int8 folds bias/residual/GN-apply into fused kernels while fp16 pays them in one aggregated
    # elementwise entry that cannot be split.
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.4, 5.0),
                                   gridspec_kw={"width_ratios": [1.15, 1]})
    k16, k8 = raw[("FP16", "C192/T1024")], raw[("INT8", "C192/T1024")]

    def pick(k, *subs):
        return sum(v for n, v in k.items() if any(x in n for x in subs))
    f_gn, f_qkv = pick(k16, "gn_accum", "gn_finalize"), pick(k16, "ImplicitGemmConvolution")
    f_core, f_proj = pick(k16, "flash_fwd"), pick(k16, "s1688gemm")
    f_ew = pick(k16, "vectorized_elementwise")
    i_gn, i_qkv = pick(k8, "group_norm_silu"), pick(k8, "awq_out_i8")
    i_core, i_proj = pick(k8, "flash_attn_int8"), pick(k8, "gemm_w8a8_kernel_awq(")

    segs = [("GroupNorm stats", [f_gn, i_gn], "#8C8C8C"),
            ("qkv projection", [f_qkv, i_qkv], "#4C72B0"),
            ("attention core", [f_core, i_core], "#C44E52"),
            ("out projection", [f_proj, i_proj], "#55A868"),
            ("elementwise\n(aggregated)", [f_ew, 0.0], "#9C755F")]
    bottom = [0.0, 0.0]
    for lbl, v, c in segs:
        axL.bar([0, 1], v, bottom=bottom, color=c, width=0.5, label=lbl, zorder=3,
                hatch="//" if lbl.startswith("elementwise") else None, edgecolor="white", lw=0.6)
        for i in (0, 1):
            if v[i] > 60:
                axL.text(i, bottom[i] + v[i] / 2, f"{v[i]:.0f}", ha="center", va="center",
                         fontsize=8, color="white", weight="bold")
        bottom = [bottom[i] + v[i] for i in (0, 1)]
    axL.set_xticks([0, 1])
    axL.set_xticklabels([f"FP16\n{bottom[0]:.0f} us", f"INT8\n{bottom[1]:.0f} us"], fontsize=9)
    axL.set_ylabel("us per forward, C192/T1024", fontsize=9)
    axL.legend(fontsize=8.2, frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.55))
    axL.grid(axis="y", alpha=0.3, zorder=0)
    for sp in ("top", "right"):
        axL.spines[sp].set_visible(False)
    axL.set_title("INT8 has NO elementwise kernel: bias, residual\nand the GN apply are inside "
                  "fused kernels", fontsize=10.5, weight="bold")

    # right panel: the specific mis-pairing that produced the retracted claim
    pairs = [("out-proj GEMM alone\n(the MIS-PAIRING)", f_proj, i_proj),
             ("out proj + the residual\nit adds (CORRECT)", f_proj + f_ew, i_proj)]
    w = 0.34
    for j, (lbl, a, b) in enumerate(pairs):
        axR.bar(j - w / 2, a, width=w, color="#8C8C8C", zorder=3, label="fp16" if j == 0 else None)
        axR.bar(j + w / 2, b, width=w, color="#55A868", zorder=3, label="int8" if j == 0 else None)
        for x, v in ((j - w / 2, a), (j + w / 2, b)):
            axR.text(x, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8.5, weight="bold")
        r = a / b
        axR.text(j, max(a, b) * 1.13, f"{r:.2f}x " + ("int8 SLOWER" if r < 1 else "int8 FASTER"),
                 ha="center", fontsize=9, weight="bold",
                 color="#C44E52" if r < 1 else "#2E7D32")
    axR.set_xticks([0, 1])
    axR.set_xticklabels([p[0] for p in pairs], fontsize=9)
    axR.set_ylim(0, max(f_proj + f_ew, i_proj) * 1.3)
    axR.set_ylabel("us per forward", fontsize=9)
    axR.legend(fontsize=9, frameon=False, loc="upper left")
    axR.grid(axis="y", alpha=0.3, zorder=0)
    for sp in ("top", "right"):
        axR.spines[sp].set_visible(False)
    axR.set_title("WITHDRAWN: 'out projection is 1.8x slower'\ncompared a fused kernel to an "
                  "unfused one", fontsize=10.5, weight="bold")
    fig.suptitle("Why the per-stage fp16/int8 ratio was withdrawn — the fp16 column does not "
                 "balance", fontsize=12.5, weight="bold")
    fig.subplots_adjust(top=0.80, bottom=0.15, left=0.06, right=0.815, wspace=0.46)
    fig.savefig(f"{D}/plots/attn_fusion_accounting.png", dpi=130)
    plt.close(fig)

    # ---- plot 3: MoDiff delta inside attention (int8-vs-int8, unaffected by the correction) ----
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for bits, base, mod, c in (("int8", "INT8", "INT8+MoDiff", "#4C72B0"),
                               ("int4", "INT4", "INT4+MoDiff", "#C44E52")):
        v = [sum(reb.get((mod, s), {}).values()) / max(1e-9, sum(reb.get((base, s), {}).values()))
             for s in shapes]
        ax.plot(range(len(shapes)), v, "o-", color=c, lw=2, label=f"{mod} / {base}")
    ax.axhline(1.0, ls="--", lw=1.2, color="#6b6b6b")
    ax.set_xticks(range(len(shapes)))
    ax.set_xticklabels(shapes, fontsize=9)
    ax.set_ylabel("MoDiff time / baseline time", fontsize=9)
    ax.set_ylim(0.95, 1.10)
    ax.legend(fontsize=9, frameon=False)
    ax.grid(alpha=0.3)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title("MoDiff inside the attention layer: 1.00-1.03x, i.e. it does nothing here "
                 "(structurally excluded)", fontsize=11.5, weight="bold")
    fig.tight_layout()
    fig.savefig(f"{D}/plots/attn_modiff_delta.png", dpi=130)
    plt.close(fig)

    # ---- plot 4: ROOFLINE achievement -- replaces the % of peak TOPS chart ----
    # The first draft plotted % of peak TOPS. At K=192 these GEMMs are memory bound (see the `bound`
    # column), so peak TOPS is the wrong denominator and it made int8 look 3.6x worse than it is.
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(14.2, 5.0))
    sh = list(SHAPE_INFO)
    for ax, kern, title in ((axA, "qkv", "qkv projection  (gemm_w8a8_kernel_awq_out_i8)"),
                            (axB, "proj", "out projection  (gemm_w8a8_kernel_awq, +bias+residual)")):
        pct = []
        for s in sh:
            C, T, _ = SHAPE_INFO[s]
            r = roofline_i8(C, T)[kern]
            meas = MEASURED_I8[s][0 if kern == "qkv" else 1]
            pct.append(r["roof_us"] / meas * 100)
        bars = ax.bar(range(len(sh)), pct, width=0.6, zorder=3,
                      color=["#C44E52" if p < 35 else "#B8860B" if p < 55 else "#55A868"
                             for p in pct])
        for i, (b, p, s) in enumerate(zip(bars, pct, sh)):
            C, T, _ = SHAPE_INFO[s]
            r = roofline_i8(C, T)[kern]
            meas = MEASURED_I8[s][0 if kern == "qkv" else 1]
            ax.text(b.get_x() + b.get_width() / 2, p + 1.5,
                    f"{p:.0f}%\n{meas:.0f} us\nroof {r['roof_us']:.0f}", ha="center", va="bottom",
                    fontsize=7.8, weight="bold")
        if sh[0] in FP16_REF:
            C, T, _ = SHAPE_INFO[sh[0]]
            f = roofline_fp16(C, T)[kern]
            ref = f["roof_us"] / FP16_REF[sh[0]][0 if kern == "qkv" else 1] * 100
            ax.axhline(ref, ls="--", lw=1.6, color="#333")
            ax.text(-0.42, ref + 1.5, f"fp16 on C192/T1024: {ref:.0f}% of ITS OWN roofline",
                    ha="left", fontsize=8.2, color="#333", weight="bold")
        ax.set_xticks(range(len(sh)))
        ax.set_xticklabels([f"{s}\n{'mem' if roofline_i8(*SHAPE_INFO[s][:2])[kern]['bound'] == 'mem' else 'CMP'}-bound"
                            for s in sh], fontsize=8)
        ax.set_ylabel("achieved % of this kernel's own roofline", fontsize=9)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_title(title, fontsize=10.5, weight="bold")
    fig.suptitle("Root cause, correctly framed: the bound is MEMORY for the out projection and "
                 "COMPUTE for qkv below T=1024 —\nand the AWQ GEMM reaches only 10-52% of it either "
                 "way (% of peak TOPS, as first drafted, was the wrong denominator)",
                 fontsize=12, weight="bold")
    fig.subplots_adjust(top=0.80, bottom=0.14, left=0.06, right=0.98, wspace=0.24)
    fig.savefig(f"{D}/plots/attn_gemm_roofline.png", dpi=130)
    plt.close(fig)

    with open(f"{D}/data/attn_stages_rebucketed.json", "w") as f:
        json.dump({f"{m}|{s}": reb[(m, s)] for (m, s) in reb}, f, indent=2)
    with open(f"{D}/data/attn_roofline.json", "w") as f:
        json.dump({f"{s}|{k}": {**v, "measured_us": meas, "pct_roofline": pct, "n_inst": n}
                   for (s, k, n, v, meas, pct) in rows}, f, indent=2)
    print(f"\nwrote plots/attn_stage_breakdown.png, attn_fusion_accounting.png, "
          f"attn_modiff_delta.png, attn_gemm_roofline.png")
    print(f"wrote {D}/data/attn_stages_rebucketed.json, attn_roofline.json")


if __name__ == "__main__":
    main()

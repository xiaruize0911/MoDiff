"""Emit FINDINGS.md from the committed measurements.

Every table below is generated from data/*.json and the conv-block ablation's committed CSV.
Hand-written prose names the handful of figures that are NOT from here (the register counts, the
ULP gate, and one-line derivations); see the "What is not generated" section of the output.
"""
import csv, json, os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = "/workspace/MoDiff"
PEAK, N, STEP_W8A8, STEP_W4A4 = 696.0, 128, 77.0005, 68.2706

i8 = json.load(open(f"{HERE}/data/gn_write_elision_int8.json"))
i4 = json.load(open(f"{HERE}/data/gn_write_elision_int4.json"))
cv = json.load(open(f"{HERE}/data/conv_charge.json"))
abl = list(csv.DictReader(open(f"{ROOT}/docs/conv_block_ablation_2026-08-26/data/combined_w8a8_w4a4.csv")))

# probe shape label -> ablation row key (Cin, H, W). The probe names a shape by the GN's own
# tensor (Cin at H x W); the ablation names it Cin->Cout.
ABL = {(r["Cin"], r["H"], r["W"], r["precision"]): r for r in abl}
PROBE2ABL = {"192,32x32": ("192", "32", "32"), "384,16x16": ("384", "16", "16"),
             "384,32x32": ("384", "32", "32"), "576,32x32": ("576", "32", "32"),
             "768,16x16": ("768", "16", "16"), "768,2x2": ("768", "2", "2"),
             "384,8x8": ("384", "8", "8")}
CONV2PROBE = {"192->192,32x32": "192,32x32", "384->384,16x16": "384,16x16",
              "384->192,32x32": "384,32x32", "576->192,32x32": "576,32x32",
              "768->384,16x16": "768,16x16"}
FIVE = list(CONV2PROBE)
L = []
w = L.append

w("# Can a_hat be overlapped after all? The GN side says yes; the conv side says no")
w("")
w("**Date** 2026-08-26 · **GPU** NVIDIA A40 (GA102, 84 SMs, 696 GB/s) · **Batch** 128 · "
  "**Delta mode** static (the shipped default, `MODIFF_DELTA_MODE`)")
w("")
w("[perf_report_2026-08-26](../perf_report_2026-08-26/REPORT.md) closed batch-split 2-stream")
w("pipelining as a route to hiding a_hat ([OPEN_ITEMS C11](../OPEN_ITEMS.md)) — inter-kernel")
w("concurrency, refuted because the conv holds every SM. This is the other reading of \"overlap\":")
w("**intra-kernel**, putting a_hat's traffic on SMs the conv already owns, which is the mechanism")
w("that makes o_hat cheap. It is refuted too, and the measurement says why.")
w("")
w("| Question | Answer | Confidence |")
w("|---|---|---|")

s8 = sum(r["saved_ms"] * r["freq"] for r in i8)
s4 = sum(r["saved_ms"] * r["freq"] for r in i4)
s8f = sum(r["saved_ms"] * r["freq"] for r in i8 if r["shape"] in [CONV2PROBE[k] for k in FIVE])
paid = sum(r["conv_cost_ms"] * r["freq"] for r in cv)
w(f"| 1. What does the a_hat WRITE cost the GN kernel? | **{s8:.3f} ms/step** (W8A8) / "
  f"**{s4:.3f}** (W4A4) — {100*s8/STEP_W8A8:.2f}% / {100*s4/STEP_W4A4:.2f}% of a step, and "
  f"{100*s8/3.551:.0f}% of a_hat's entire measured cost | high — reproduces the committed ablation to 0.2–0.9% |")
w("| 2. Does the register budget survive an a_hat RMW inside the conv? | **Yes.** 240 regs, "
  "0 stack, 0 local — unchanged. The flagged risk did not materialise | high — `cuobjdump -res-usage` |")
w(f"| 3. Does the conv absorb that traffic cheaply? | **No.** It charges **{paid:.3f} ms/step** for "
  f"what the GN gives back at {s8f:.3f}. Net **{s8f-paid:+.3f} ms/step = "
  f"{100*(s8f-paid)/STEP_W8A8:+.2f}%** of a step | high — 5 trials, order rotated, sd < 3% |")
w("| 4. Is o_hat cheap because it lives inside the conv? | **No.** Placement before the mainloop "
  "vs after the epilogue differs by < 1%. o_hat is cheap for a different reason | high — direct A/B |")
w("")
w("---")
w("")
w("## 1. The GN side: eliding the a_hat write is worth 31% of the apply kernel")
w("")
w("Instrument: [`probe.cu`](scripts/probe.cu), a verbatim copy of")
w("[`gn_apply_delta_quantize_flat_vec2_kernel`](../../csrc/modiff/norm/group_norm_silu.cu:1701)")
w("with the a_hat store and the code store behind template flags, and `gn_report_delta_absmax`")
w("dropped (production passes `absmax_buf = nullptr` in static mode and the helper's first")
w("statement is `if (absmax_buf == nullptr) return;`). Launch geometry is production's: block 256,")
w("`grid = ceil(numel/2/256)`, 256-float dynamic shared kept so occupancy is identical.")
w("")
w("**Validity.** The probe also calls the shipped path. Against the committed")
w("[conv-block ablation](../conv_block_ablation_2026-08-26/data/combined_w8a8_w4a4.csv):")
w("")
w("| shape | freq | probe `prod` W8A8 | ablation | Δ | probe `prod` W4A4 | ablation | Δ |")
w("|---|--:|--:|--:|--:|--:|--:|--:|")
for r8 in i8:
    k = PROBE2ABL[r8["shape"]]
    a8 = float(ABL[(k[0], k[1], k[2], "W8A8")]["gn_modiff"])
    r4 = next(x for x in i4 if x["shape"] == r8["shape"])
    a4 = float(ABL[(k[0], k[1], k[2], "W4A4")]["gn_modiff"])
    w(f"| `{r8['shape']}` | {r8['freq']} | {r8['med']['prod']:.4f} | {a8:.4f} | "
      f"{100*(r8['med']['prod']-a8)/a8:+.2f}% | {r4['med']['prod']:.4f} | {a4:.4f} | "
      f"{100*(r4['med']['prod']-a4)/a4:+.2f}% |")
w("")
w("The five dominant shapes agree to 0.2–0.9%. `768,2x2` is 0.018 ms and launch-noise dominated;")
w("it is carried for completeness, not for its number.")
w("")
w("**Result.** `w1c1` = today (read x, read a_hat, write a_hat, write code). `w0c1` = the write elided.")
w("")
w("| shape | freq | W8A8 w1c1 | w0c1 | saved | % of apply | W4A4 w1c1 | w0c1 | saved | % of apply |")
w("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
for r8 in i8:
    r4 = next(x for x in i4 if x["shape"] == r8["shape"])
    w(f"| `{r8['shape']}` | {r8['freq']} | {r8['med']['w1c1']:.4f} | {r8['med']['w0c1']:.4f} | "
      f"{r8['saved_ms']:.4f} | **{r8['saved_pct_of_apply']:.1f}%** | {r4['med']['w1c1']:.4f} | "
      f"{r4['med']['w0c1']:.4f} | {r4['saved_ms']:.4f} | **{r4['saved_pct_apply']:.1f}%** |")
w(f"| **freq-weighted** | | | | **{s8:.3f} ms/step** | | | | **{s4:.3f} ms/step** | |")
w("")
w(f"As a share of the steady step: **{100*s8/STEP_W8A8:.2f}%** (W8A8, 77.00 ms) and "
  f"**{100*s4/STEP_W4A4:.2f}%** (W4A4, 68.27 ms).")
w("")
w("**It beats its own byte model, and that is informative.** The apply kernel moves 7 B/elem with")
w("the write and 5 without, so a pure-bandwidth prediction is 2/7 = 28.6%. Measured 29–32%, and the")
w("achieved bandwidth *rises*:")
w("")
w("| shape | W8A8 GB/s w1c1 → w0c1 | % of peak |")
w("|---|--:|--:|")
for r in i8:
    w(f"| `{r['shape']}` | {r['gbs_w1c1']:.0f} → {r['gbs_w0c1']:.0f} | "
      f"{100*r['gbs_w1c1']/PEAK:.0f}% → {100*r['gbs_w0c1']/PEAK:.0f}% |")
w("")
w("The surplus is the write-allocate traffic the store drags in on top of its own 2 B.")
w("")
w("> **This number is the ceiling for any scheme that removes the a_hat write**, not just for")
w("> moving it into the conv — an int8/fixed-point a_hat cache inherits the same numerator.")
w("")
w("---")
w("")
w("## 2. The register risk did not materialise, and the arithmetic is exact")
w("")
w("The conv-side arm adds a CTA-partitioned `a_hat += code/scale` inside")
w("`ImplicitGemmConvolutionEVT::operator()` ([the patch](scripts/conv_ahat_rmw.patch)). The")
w("partition is over the flat tensor and independent of the tile the CTA computes, so **each")
w("element is visited exactly once** — no ownership predicate, and none of the R×S multiple-visit")
w("problem that an iterator-level fusion would have. It runs before the swizzle bounds check so")
w("CTAs that return early still take a share.")
w("")
w("| check | result |")
w("|---|---|")
w("| registers (`cuobjdump -res-usage`) | **REG:240 STACK:0 LOCAL:0** — identical to the shipped kernel |")
w("| a_hat vs an fp32-accumulate / `__float2half_rn` reference | **0 ULP** |")
w("| o_hat vs the shipped `conv2d_int8_evt_o_hat` | **bit-identical** |")
w("| negative control (scale off by 1%) | **37391 ULP** — the gate fires |")
w("")
w("240 of the SM's 65,536 registers per thread × 256 threads = 61,440, i.e. the same one-block-per-SM")
w("occupancy the perf report documents. The design failed for a different reason than the one flagged.")
w("")
w("---")
w("")
w("## 3. The conv side: it charges 3x what the GN gives back")
w("")
w("| shape | freq | conv | conv+a_hat (pre-mainloop) | (post-epilogue) | charge | |")
w("|---|--:|--:|--:|--:|--:|--:|")
for r in cv:
    w(f"| `{r['shape']}` | {r['freq']} | {r['med']['conv']:.4f} | {r['med']['conv+ahat']:.4f} | "
      f"{r['med']['conv+ahat_post']:.4f} | {r['conv_cost_ms']:+.4f} ms | **{r['cost_pct']:+.1f}%** |")
w("")
w("```")
w(f"GN gives back : {s8f:+.3f} ms/step")
w(f"conv charges  : {paid:+.3f} ms/step")
w(f"NET           : {s8f-paid:+.3f} ms/step = {100*(s8f-paid)/STEP_W8A8:+.2f}% of the W8A8 step")
w("```")
w("")
w("**Placement is not the variable.** Before the mainloop and after the epilogue differ by under 1%")
w("on every shape. So the hypothesis this experiment was built on — *o_hat is cheap because it sits")
w("inside a compute-bound kernel* — is directly refuted.")
w("")
w("**The mechanism.** Compare how fast each kernel moves the bytes in question:")
w("")
w("| shape | conv, added bytes | GN, the same bytes given back |")
w("|---|--:|--:|")
for r in cv:
    p = next(x for x in i8 if x["shape"] == CONV2PROBE[r["shape"]])
    Cin = int(r["shape"].split("->")[0]); H = int(r["shape"].split(",")[1].split("x")[0])
    ne = N * Cin * H * H
    cg = ne * 5 / (r["conv_cost_ms"] * 1e-3) / 1e9
    gg = ne * 2 / (p["saved_ms"] * 1e-3) / 1e9
    w(f"| `{r['shape']}` | {cg:.0f} GB/s (**{100*cg/PEAK:.0f}%** of peak) | "
      f"{gg:.0f} GB/s ({100*gg/PEAK:.0f}%) |")
w("")
w("**The conv is a worse place to move bytes than the GN kernel, not a better one.** Its 23–25% of")
w("peak bandwidth (measured in the perf report's §4) is not headroom that can be claimed: adding a")
w("low-MLP streaming loop to a kernel that holds every SM runs it at 57–69% of peak, against the GN")
w("kernel's 73–81%.")
w("")
w("### Why o_hat really is cheap — a correction to perf_report §2")
w("")
w("The perf report prices o_hat's incremental bytes at **2.35× / 4.06× cheaper** than a_hat's and")
w("attributes it to intra-kernel latency hiding. That attribution is wrong, and this experiment is")
w("what shows it: putting a_hat's bytes in the same kernel, in either position, buys nothing.")
w("")
w("o_hat is cheap because **its store replaces a store the baseline already performs**, and its load")
w("rides the write-allocate that store needs anyway — the cache line is fetched to be written")
w("regardless. Its true incremental DRAM transaction count is near zero. a_hat has no such twin: it")
w("is a separate tensor no other kernel touches, and its 4 B/elem are irreducible DRAM traffic.")
w("**The o_hat per-byte advantage is a property of o_hat, not of the conv, and does not transfer.**")
w("")
w("### The remaining variant, priced")
w("")
w("Fusing into the activation iterator instead of a separate CTA-partitioned loop would avoid")
w("re-reading the codes, taking the added traffic from 5 B/elem to 4 — about a 20% reduction, so a")
w(f"charge near {paid*0.8:.1f} ms/step against the {s8f:.3f} available. Still deeply negative, and it")
w("costs the exactness the CTA partition gets for free (ownership predication plus halo handling).")
w("Not worth building.")
w("")
w("---")
w("")
w("## What this leaves standing")
w("")
w("1. **\"Overlap a_hat\" is now closed on both readings** — inter-kernel (C11) and intra-kernel")
w("   (this doc), each with a measured mechanism rather than an argument.")
w(f"2. **The {s8:.3f} / {s4:.3f} ms/step ceiling is the durable result.** It bounds every scheme")
w("   that removes the a_hat *write*, including an int8/fixed-point a_hat cache — that idea now has")
w("   a measured numerator without anyone building it.")
w("3. **Selective per-layer MoDiff is not bounded by it**, because dropping MoDiff on a layer")
w("   removes the a_hat read, the a_hat write and o_hat together. On the five dominant shapes that")
w("   is the full 3.551 ms/step (W8A8), not the write's share.")
w("")
w("## Scope and limitations")
w("")
w("- **The probe ran with `mod_scale` / `smooth_inv` null.** Both add per-element ALU but no")
w("  tensor-sized traffic, and both arms carry them equally, so the *difference* should be")
w("  unaffected — untested.")
w("- **Five of twenty shapes**, the ones carrying 63% of MoDiff's conv-block overhead. `768,2x2`")
w("  and `384,8x8` are carried as references and behave differently (launch-bound, 63–78% of peak).")
w("- **The conv-side arm is not in the tree.** It was reverted after measurement at the owner's")
w("  request; [`conv_ahat_rmw.patch`](scripts/conv_ahat_rmw.patch) reproduces it against the commit")
w("  this doc lands on. The GN-side probe is standalone and needs no patch.")
w("- **No `ncu`** (`ERR_NVGPUCTRPERM`), so the bandwidth figures are derived from measured")
w("  durations and analytic byte counts, as in the perf report.")
w("")
w("## What is not generated by [`make_findings.py`](scripts/make_findings.py)")
w("")
w("The register counts (`cuobjdump`), the four rows of the numerics gate (ULP / bit-identity /")
w("negative control), the byte-per-element models (read from kernel source), and the inline")
w("derivations: 2/7 = 28.6%, the 61,440 register product, the 20% iterator-fusion estimate, and")
w("a_hat's 3.551 ms/step on these five shapes (from the ablation CSV).")
w("")

open(f"{HERE}/FINDINGS.md", "w").write("\n".join(L) + "\n")
print(f"wrote FINDINGS.md, {len(L)} lines")
print(f"GN side W8A8 {s8:.3f} / W4A4 {s4:.3f} ms/step; conv charges {paid:.3f}; net {s8f-paid:+.3f}")

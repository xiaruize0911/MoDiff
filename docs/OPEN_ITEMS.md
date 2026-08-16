# Open items, 2026-08-16

Standing list, three sections: **values that are not what a committed doc says they are**, **questions
with no answer yet**, and **headroom with a known lever**. Every row cites the file it comes from, and
every number is read from committed data rather than restated from memory.

Provenance: sections B and C are carried forward from the `Open` sections of
[`profile_kernels_layers_2026-08-11`](profile_kernels_layers_2026-08-11/FINDINGS.md),
[`aq_fusion_2026-08-12`](aq_fusion_2026-08-12/FINDINGS.md),
[`fid_2026-08-05`](fid_2026-08-05/FINDINGS.md),
[`static_qdiff_2026-08-12`](static_qdiff_2026-08-12/FINDINGS.md) and
[`zp_coverage_2026-08-13`](zp_coverage_2026-08-13/README.md). Section A rows marked **[2026-08-16]** were
found by re-reading `bench_report_2026-08-13_postzp/data/kernel_suites.json` against the prose it
generated; A1–A6 are **fixed** as of this doc and the corrected numbers are in
[`KERNEL_SPEEDUP.md`](bench_report_2026-08-13_postzp/KERNEL_SPEEDUP.md) and
[`SUMMARY.md`](bench_report_2026-08-13_postzp/SUMMARY.md).

---

## A. Not the expected value

Measured numbers that contradicted a claim a committed doc made about them. Ordered by how much of the
published story they move.

| # | claim as published | what the data says | state |
|---|---|---|---|
| **A15** **[2026-08-16]** | `other` reads fp16 6.10 against int8's 10.11 — as if quantization made concatenation 1.66× *more* expensive | **It did not. fp16's concats were invisible to the instrument.** `openaimodel._skip_concat` takes `cat2_channels_last_fp16` only when both halves are fp16 **and** channels_last, and falls back to `th.cat` otherwise; ~3 skip-concats/step take that fallback in the fp16 arm, and the capture wrapped `mc.*` plus three `F.*` functions but **not `torch.cat`**. A fresh capture reproduced the asymmetry *exactly* — same 4 signatures, same counts, two absent from fp16 entirely — which is what ruled out a transient gap and pointed at the wrapper list instead | **fixed** — `torch.cat` is now wrapped; existing JSON is unaffected, the next capture closes the account |
| **A7** **[revised 2026-08-16]** | MoDiff's temporal machinery costs 2.8% (W8A8) / 1.1% (W4A4) — "roughly free" | **No longer true, and by an order of magnitude: 12.4% and 15.1%.** Not because MoDiff got slower — because C1 made the PTQ arms 5.7–6.7 ms/step faster and the MoDiff arms **did not move** (73.67 vs 73.19; 58.93 vs 58.50). In MoDiff mode the ResBlock takes the GN→delta-quantize path, which C1's swap does not reach. The arm-order caveat still holds on top of this and is now the smaller effect | **open** → C10 |
| **A9** | MoDiff is the quality answer | True at W8A8 (97.3% of the quantization error removed, FID 7.802 vs fp16's 7.803). At W4A4 it removes **31.3%** and FID is 200.1 vs PTQ's 278.0 — the dominant error is in the **weights**, which an activation method cannot reach ([fid_2026-08-05](fid_2026-08-05/FINDINGS.md)) | **diagnosed**, not fixed → B5 |
| **A11** | fix #4 (weight zero point / AdaRound) was deprioritised | Deprioritised on ‖W−Q(W)‖, the one metric AdaRound is willing to lose. On conv output error it wins 1.35×; end to end, weight-only, **1.58×** ([FINDINGS_WEIGHT_ZP](zp_coverage_2026-08-13/FINDINGS_WEIGHT_ZP.md)) | **open** — reprioritise, → C6 |

---

## A0. RESOLVED: the invariant holds; every non-zero reading was a measurement artifact

Opened 2026-08-16 on the belief that MoDiff's fused GN→delta-quantize path violated its bit-identity
invariant (`max_code_diff` 27–38, later 221). **It does not. Both paths are correct to fp16 rounding, and
each successive non-zero number was an artifact of how it was being measured** — including two artifacts
of my own making.

### The final verdict

[`integration/tests/gn_modiff_gate.py`](../integration/tests/gn_modiff_gate.py), scoring each
implementation against an **fp64 reconstruction** rather than against the other:

| | max \|Δ\| vs fp64 truth |
|---|--:|
| reference (`group_norm_silu_nhwc` + `step1_static_quantize_fprop_silu`) | **1.0** |
| fused (`group_norm_silu_delta_quantize_nhwc`) | **1.0** |

1.0 is fp16 rounding. Identical under both block-size policies. 18 of 40 cases are scored; the 22 with
modulation or smoothing are reported **UNSCORED**, because the reconstruction only matches to ≤1 code
without them and the order in which the kernels apply mod/affine/SiLU is not pinned down by any comment —
guessing it produced a confident 254. A partial verdict that is trustworthy beats a total one that is not.

### Four artifacts, in the order they were peeled off

1. **The gate was unrunnable** — its wrapper missed the `x2=` the cat2 fold added on 2026-08-13, and it
   passed 11 arguments to a kernel grown to 18. So "it failed" and "nobody could run it" were the same
   observable. Repaired; the trailing args now come from the conv's own accessor.
2. **It was non-deterministic** — a max over the first 40 calls of a *live* sample, and fp16 sampling
   varies ~4–6e-3 between processes, so the statistic ranged **23–81 unchanged**. Replaced by
   capture-once / replay-forever.
3. **My replay had an aliasing bug** — `torch.load(map_location="cuda")` made `c["a_hat"].cuda()` a no-op
   and `_cl()` on an already-channels_last tensor a no-op too, so every "fresh clone" was **the same
   memory**. The fused kernel updates a_hat in place, so the reference call mutated the input the fused
   call then read. That produced **221**, and an order-dependence ("whoever runs first is correct") that I
   briefly wrote up as a kernel state bug.
4. **Arm-to-arm comparison cannot decide anything** — it cannot separate "one is wrong" from "they encode
   differently", and it is sensitive to both (2) and (3). Scoring against fp64 is immune to all of it.

### What it cost, and the one lesson worth keeping

Along the way I asserted, and then had to retract: *"the invariant does not hold"*; *"C10 makes it worse,
35 → 81"* (n=1 noise); *"the fused kernel is wrong at small spatial sizes"* (my aliasing); *"the two paths
mutually pollute"* (same). Each retraction came from one more control, and the control that ended it was
the cheapest one available from the start: **compare each implementation to a reference, never to the other
implementation.** That is the identical correction I had to make to `quality_gn_fast_paired.py` earlier the
same day, when it reported an arm-to-arm relL2 that could not distinguish "worse" from "differently
rounded". Learning it once did not transfer.

### Consequences

- **The invariant holds.** `csrc/modiff/norm/group_norm_silu.cu`'s two comments citing this gate as
  evidence of fragility should be read as historical.
- **C10's correctness objection is withdrawn, this time with evidence.** Both block-size policies score
  1.0 against fp64. The earlier "221 = 221 exonerates it" was worthless because the policy was inert
  (dead code behind chanmajor); this is not.
- **C10 remains unmeasured and remains a no-op on the MoDiff arms** — moving them still requires changing
  chanmajor's `BLK`, which is now unblocked on correctness but still needs the headroom re-derived.

---

---

## B. Unsolved

1. ~~**W4A4: per-step vs constant delta.**~~ **RESOLVED 2026-08-16 — flat at W4A4 too, so the
   hypothesis is refuted at both bit widths.** The arm was confounded, not unmeasurable: the native
   per-step table had been fitted with SmoothQuant **on** (via `dynamic_delta_ab`'s CALIB) and was being
   applied on top of the qdiff file, which has smoothing **off** — so the delta distribution it was fitted
   to was not the one it saw, and it read 2.93× worse than the constant. Rebuilt in the matching
   configuration (`AB_CALIB4=…int4_calibration_qdiff.pt python …/int4_delta_table.py`) and re-run:

   | axis | qdiff constant | native per-step | dynamic | PTQ |
   |---|--:|--:|--:|--:|
   | W8A8 | 0.0495 | 0.0533 | 0.0611 | 0.1138 |
   | **W4A4** | **0.3122** | **0.3237** | 0.3577 | 0.8642 |

   A genuinely per-step table is **3.7% worse** than the paper's single scalar at W4A4, and both beat
   dynamic. The hypothesis — that at 4 bits the constancy is the cost, because 15 levels cannot carry a
   3–6× step-to-step swing — is dead at both widths. **Consequence: there is no point exporting a
   per-step qdiff delta table**, which had been a candidate work item. Staticness, not constancy, is what
   costs, and there is no better static table to reach for.
2. ~~**FID for W8A4 + MoDiff**~~ **MEASURED 2026-08-16 — mechanism reproduces, headline claim does not.**
   10k samples, DDIM 50, the committed protocol: **W8A4 PTQ 311.47 → W8A4 MoDiff 35.30**, an 8.8×
   improvement recovering 90.9% of the distance to fp16 (7.803). The unmodulated end matches the paper's
   355.85 in order and verdict.

   But the paper's A4 claim is not "MoDiff helps" — it is that **dropping a bit becomes free**: its 3.97
   beats its own A8 baseline of 4.24, a ratio of 0.94. The same *internal* comparison on our tree gives
   **35.30 / 16.37 = 2.16** — dropping the activation bit costs 2.16× where the paper reports it saving
   6%. Both ratios are within-protocol, so the 10k-vs-50k bias largely cancels and cannot explain it.

   **35.30 must not be divided by 3.97.** Ours is 10k, the paper's is 50k and 10k is biased upward — our
   fp16 reads 7.803 where LDM's published churches figure is ~4, and the paper's A8 baseline (4.24) is
   *better than our fp16*, which is the tell. That division is the same class of unit error as A16.
   → [FINDINGS_W8A4_FID](gn_fast_reduce_2026-08-16/FINDINGS_W8A4_FID.md)
3. ~~**FID for W4A4 + MoDiff with the new weight scale**~~ **MEASURED 2026-08-16 — the sign is the
   opposite one. relL2 and FID disagree.** The MSE weight scale (`_int4_weight_scale`, default since
   2026-08-05, adopted on a paired −7.5% latent relL2 with 4/4 seeds improving) is **5.70% WORSE on FID**:

   | arm (10k, DDIM 50, same seeds, same real reference) | FID |
   |---|--:|
   | `MODIFF_INT4_WSCALE=absmax` — paired arm, same tree | **170.854** |
   | MSE — the current default | 180.593 |
   | 08-05's recorded absmax number, 11 days and one tree ago | 200.139 |

   **And the naive comparison gets the sign wrong.** 180.593 against the committed 200.139 reads −9.77%
   and looks like a win; the paired arm on the same tree reads **+5.70%** and is a loss. The entire −9.8%
   belongs to eleven days of other changes (zero point, cat2 fold, the delta-clip constants) — absmax
   itself moved 200.139 → 170.854. This is why the paired arm was run instead of quoting the committed
   number.

   **Second time this session that a proxy metric pointed the wrong way**, and the parallel is exact:
   C7's warm-up count was raised 3 → 5 on the per-round *activation reconstruction* while the *latent*
   says 1 round is 26.5% better; the MSE scale was adopted on *latent relL2* while *FID* says absmax is
   better. A11 is the same shape from the other direction — fix #4 was deprioritised on ‖W−Q(W)‖, the one
   metric AdaRound is willing to lose. **Three items, one failure mode: a decision made on the cheapest
   available metric rather than the one that defines the goal.**

   **Actionable:** `MODIFF_INT4_WSCALE` should be re-evaluated as a default, and `docs/fid_2026-08-05`'s
   adoption rationale annotated. Caveat: single 10k FID per arm, so the pairing removes seed noise but
   there are no error bars on FID itself; 9.7 points on a 170 baseline is the same order as the relL2
   effect it contradicts.
4. **FID at 50k**, for publication-grade absolute numbers. ~1.5 h/mode, and 68–78% of that is not the
   UNet. [FID_50K_ESTIMATE](bench_report_2026-08-13_postzp/FID_50K_ESTIMATE.md)
5. **A weight-side method for W4A4.** No candidate has landed. This is what gates W4A4 being usable at
   all — the kernels are already there (3.83× median per conv). Follows from A9.
6. **Quality of the qkv-i8 fusion (route b).** **MEASURED 2026-08-16 for the first time — the item was
   misfiled as "needs more seeds" when the truth was "never measured".**
   `quality_route_b_paired.py` reported BIT-IDENTICAL at 3 and 8 seeds and it was **vacuous**: route (b)
   fires **0 times in both arms** there (`probe_route_b_fires.py`). That harness builds an arm whose
   attention is PTQ — the `_qout` family, `flash_attn_int8_qi8_kv_static_qout[_hd24]` — and `_qout` is
   mutually exclusive with MoDiff's fp16 o_hat state by construction, so route (b) has no o_hat to
   advance. The difference from the timing A/B, whose counters prove 10 calls/step, is one env var:
   `MODIFF_LINEAR=1`.
   Re-measured in a config where it fires (3920 calls ON, 0 OFF), 8 seeds, relL2 vs a per-seed fp16
   reference: **−1.51% ± 1.24% (SEM)** — not resolved, and the point estimate *favours* route (b)
   (0.0303 vs 0.0308). So there is **no quality argument against** the +0.79 ms/step.
   **Still open, but only as a decision:** flipping the default on an unresolved measurement would
   contradict the bar this project set for itself in `aq_fusion_2026-08-12`. Resolving it needs ~30
   seeds (stdev 3.5%, so 2×SEM < 1.5% wants n ≈ 22+). Left flag-gated pending that.
7. ~~**Part 3, the a_hat-aware flash qout epilogue: the first gate is a numerics decision.**~~
   **ANSWERED 2026-08-16, and the answer is NO.** The gate was never unmeasurable — `MODIFF_DELTA_REPORT`
   (default 0) already selects exactly the scheme in question: report-on quantizes with the pair currently
   in force while publishing the next, so a step uses a scale measured one step earlier; report-off spends
   a separate reduction pass for a fresh one. That pass is what the epilogue needs to remove, so B7's two
   framings ("from a previous step's `report_next`" and "accepted one step stale") are the same branch.

   Measured, W8A8, 8 seeds, relL2 vs a per-seed fp16 reference:

   | arm | mean relL2 | paired |
   |---|--:|---|
   | OFF — fresh scale, extra pass | 0.0346 | — |
   | **ON — one step stale, safety 1.15** | **0.1788** | **+460.9% ± 88.9% (SEM), RESOLVED** |

   **5.2× the latent error.** Not marginal, and resolved on 8 seeds. Non-vacuity passed by tallying the
   `report_next` argument itself (a kernel-name counter is blind to this flag): OFF False×27440 / True×0,
   ON True×7280 / False×20160 — so even with only ~26% of calls on the stale path the damage is 5.2×.

   **The safety margin does not rescue it, and why it does not is the diagnosis.** Sweeping
   `MODIFF_DELTA_SAFETY`: 1.15 → +460.9%, 1.5 → +708.4%, 2.5 → +544.0%, all resolved, non-monotone. So
   the failure is **not clipping** — clipping is the one thing safety exists to prevent, and more headroom
   does not help. The mechanism is the opposite: the MoDiff residual **shrinks** as *t* decreases, so a
   scale measured one step earlier is systematically **too coarse** for the delta it is applied to, and a
   safety multiplier makes it coarser still. Headroom is the wrong medicine for a grid that is already too
   wide.

   **So the 6.7 ms ceiling cannot be bought this way, and B7's design brief changes:** Part 3 needs a
   different way to get a *fresh* scale into the epilogue, not a cheaper way to tolerate a stale one.
   Closed negative.
8. ~~**The dual-output GEMM's signature.**~~ **ANSWERED 2026-08-16, and the question was already moot.**
   `gemm_w8a8_awq_o_hat_out_i8` takes `inv_out_scale` as a separate `float*` — per column — and the
   scalar `TORCH_CHECK` is on `a_scale`, the delta's activation scale, which is necessarily scalar. The
   kernel indexes it per column throughout (`s00 = a_scale * w_scale[gc0] * inv_out_scale[gc0]`,
   `gemm_wxax.cu:328`). It is also not the blocker it was filed as: this item was carried from
   2026-08-11's "zero call sites", and the 2026-08-12 `aq_fusion` session wired it — see C2.
9. ~~**Per-layer profile coverage gap, 12–36%.**~~ **RESOLVED 2026-08-16 — it is two different things,
   and the 36% end is a MISLABELLING, not a hole.** Measured on the post-C1 profile
   (`bench_report_2026-08-16_gnfast/data/profile_layers.json`):

   | config | wall | kinds sum | gap |
   |---|--:|--:|--:|
   | fp16 | 103.58 | 0.00 | **100%** |
   | W8A8 PTQ | 65.31 | 46.63 | 28.6% |
   | W8A8 conv-only | 79.20 | 66.99 | 15.4% |
   | the five `conv+proj` configs | — | — | **10.8–12.4%** |

   * **fp16's 100% is scope, not a gap.** The instrument wraps quantized/MoDiff-wrapped modules; the fp16
     arm has none (`n_convs = n_attn = n_proj = 0`), so there is nothing to attribute and nothing missing.
   * **The PTQ arms' 28.6% is a mislabelling.** `n_proj = 42` in every arm, yet `proj (42 linears)` reads
     **0.00** in the PTQ and conv-only configs — because `_flash_proj_qout` issues the projection as
     `_mc.gemm_w8a8_awq_bias_res(xq, proj.qweight, …)` and never calls `QuantLinearWxAx.forward`, which is
     what the profiler wraps. The time is not lost: `kinds` computes
     `attn = max(attn_gross − proj_total, 0)`, so with `proj_total = 0` nothing is subtracted and the
     projections stay **inside** `attn (score path)`. That bucket's 20.00 ms/step against B10's 18.12 for
     the whole attention block (score + proj) confirms it. So the label is wrong, not the total.
   * **The real unattributed remainder is 10.8–12.4%**, uniform across every config the instrument covers,
     and *that* is what the filed fix (time the ResBlock's own `forward` as a residual bucket) addresses.

   Same shape as A1 and A5: the measurement was right and its name was wrong. The one-line fix is to count
   the projection where it is issued (the `mc.gemm_*` entry point) rather than where it is declared.
10. ~~**The PTQ attn/proj split** — needs `_flash_proj_qout` instrumented.~~ **ANSWERED 2026-08-16 with
    no instrumentation at all.** `_flash_proj_qout` already issues its two halves as separate kernel
    launches with distinct names (the flash `_qout` kernel and `gemm_wXaX_awq_bias_res`), and the suite
    capture already timed both. What was missing was arithmetic: deciding which `linear` records are
    attention projections. ms/step at batch 128
    ([attn_proj_split.py](bench_report_2026-08-16_gnfast/scripts/attn_proj_split.py)):

    | arm | score | proj | block | split |
    |---|--:|--:|--:|---|
    | fp16 | 12.67 | 9.96 | 22.63 | 56% / 44% |
    | W8A8 PTQ | 10.22 | 7.90 | 18.12 | 56% / 44% |
    | W4A4 PTQ | 10.00 | 7.11 | 17.11 | 58% / 42% |

    vs fp16: score 1.24×/1.27×, projections 1.26×/1.40×, whole block 1.25×/1.32×. **The split barely
    moves with precision and the two halves gain almost equally** — the attention block is not a fast
    score path dragged down by slow projections or the reverse. It is uniformly ~1.25×, so C3 and C4
    between them address only the score half of a 56/44 problem.

    Two classifier traps produced a wrong table first, both understating fp16 and so overstating the
    speedup: fp16 passes the activation 3-D as `[b,T,c]` where the quantized arms flatten to `[b*T,c]`,
    and fp16's two largest projections are `fused_gn_qkv`, filed under `other` (A1). Third time in one
    session that suite membership, not the measurement, was the error.
11. ~~**Whether to build the per-output-pixel windowed reduction epilogue.**~~ **NOT A QUESTION —
    merged into C6.** As filed this item contained its own answer: the measurements already say build it
    for fix #4 (1.58× end-to-end, weight-only) and not for fix #2 (1.06× against a 1.15× bar). There is
    nothing left to decide, only to build, and that is C6. Filing a settled decision as an open question
    is how C2 and C8 stayed on a list for days after they closed.

---

## C. Headroom with a known lever

| # | target | size | lever | blocked on |
|---|---|--:|---|---|
| ~~**C1**~~ | ~~GroupNorm+SiLU family~~ | ~~32.2% at 1.13×~~ | **DONE 2026-08-16.** Not a roofline bound and the design was in the tree: `..._fast` (`fast_reduce=true`) was reachable from attention but not from `fused_resblock.py`, which owns 62 of 83 GN calls/step. **+6.65 ms/step W8A8, +7.24 W4A4** | closed → [gn_fast_reduce_2026-08-16](gn_fast_reduce_2026-08-16/FINDINGS.md) |
| ~~**C2**~~ | ~~the `aq_*` trio~~ | ~~4.60 ms~~ | **STALE WHEN FILED — this landed 2026-08-12.** Route (b) is wired (`quantized_std_attention.py:1050`) behind `MODIFF_FUSE_QKV_I8`, worth **+0.79 ms/step** on the 10 hd=48 blocks. The 4.60 ms was never all available: 1.47 ms was the hd=48 share (taken) and 3.13 ms the hd=24/T=1024 share, **refuted** (A14). Remaining: the flag is opt-in, which is B6, not a kernel problem | closed → B6 |
| **C3** | attention T=1024 / hd=24 | 15.6 ms/sample, ~31% of the attention suite, at 1.21× | **The search space is documented and mostly exhausted — "no design" understated what has been tried.** `docs/final_report_2026-07-28/INT8_QKV_EPILOGUE_RECHECK_2026-07-30.md` lists three rejected approaches with reasons: whole-head **persistent** Packed fusion at T256/T1024 (*"insufficient CTA occupancy and too much serial query work"* — the kernel exists as `flash_attn_int8_packed_persistent_qout` and is gated off because it loses), **compact global K+V at hd24** (producer got faster, 8-byte K staging made flash slower), and **compact V-only** (−0.5%). A14 adds the 8-byte loader. Its own "next high-leverage implementation" — a GEMM epilogue emitting head-major padded Q/K and transposed V, worth the 311 µs repack — **has since landed** as `gemm_w8a8_awq_qkv_i8_layouts[_compact]`, and both fire in the current capture. That doc's remaining T=1024 budget was Flash 1469 µs / qkv GEMM 359 / proj GEMM 351 / repack 311 / GN 268; repack is gone and C1 took the GN. **What is left is the flash kernel's own inner loop**, i.e. the hd 24→32 fragment padding — structural, not a missing optimization | new kernel. Four approaches tried; the next one has to attack the MMA fragment layout itself, and B10 caps the prize: attention is 56/44 score/projection, so the score half alone cannot take the block past ~1.4× |
| **C4** | int4 attention | zero gain today | V is int8 and the MMA is the int8 path in every arm (A3). Nothing to win until there is a real int4 datapath | design |
| **C5** | GN-stats epilogue, Stage C | — | **The bar moved against it on 2026-08-16 and it is now decisively negative.** Stage C folds the GN stats into the conv epilogue to remove a separate launch, and was measured at **0.96×** — "mechanism viable, margin is not". That ratio was against the GN kernel as shipped. C1 then made that kernel **1.91–2.03× faster**, including **2.23× on the `768×4×4` shape this item names as its gate**. The launch the epilogue would remove is now worth about half what it was, so the same mechanism scores ~0.5×. Nothing needs measuring to reject it | **closed** as refuted — reopen only if the epilogue's own reduction can beat `fast_reduce`, not the generic path |
| **C6** | weight zero point (fix #4) | **1.58×** end-to-end, weight-only | **RE-SCOPED 2026-08-16 — it is not a missing capability, and it is not an epilogue reduction.** The term is `z_w[k] · Σ_i a[i]`, and `i` runs over (c_in, r, s) — so `Σ_i a[i]` sums over **all input channels** as well as the window, making it **one scalar per output pixel**, a `[N, H_out, W_out]` field. It does not depend on `k`, which the source doc already noted ("one reduction serves all K"). Two consequences: **(1)** it need not be computed in the epilogue at all — a `Σ` over C then a **separable** R×S box filter costs O(N·C·H·W), i.e. about one activation read, not O(·R·S); **(2)** the epilogue term is then a rank-1 outer product `− z_w[k] · S[p]`, per-channel × per-pixel. Verified that `conv2d_evt.cu` already includes `cutlass/epilogue/threadblock/fusion/visitors.hpp`, and that header already provides **`VisitorRowBroadcast`** and **`VisitorColBroadcast`** — the existing tree uses `RowVec` for `weight_scale[k]`/`bias[k]` and `AuxLd` for the per-element residual, so the missing piece is a broadcast node that is already in the header being used | **no longer research.** Remaining work: a box-filter pre-pass, plus `Sm80EVT<Add, <tree>, Sm80EVT<Mul, RowVec(z_w), ColBroadcast(S)>>`. Risks: the EVT is hand-assembled onto the conv Mma (CUTLASS 4.6.1 has no EVT-on-conv path), so adding a node is manual; and whether row/col maps to channel/pixel under this tile's layout must be checked, not assumed. **Not** the "windowed-reduction epilogue that does not exist" it was filed as |
| ~~**C10**~~ | ~~MoDiff GN→delta-quantize~~ | ~~7.80/8.16 ms/step~~ | **REFUTED 2026-08-16 — the premise compared the wrong two things.** C1's 1.91× was on the PTQ family, which uses a **group-major** stats decomposition where `fast_reduce` genuinely wins. The MoDiff path uses **chanmajor** (`BLK = C/K`) and never reaches the group-major code, so the policy is inert there (cross-arm check: 0/40 cases differ). Forcing group-major to measure what it would buy: **chanmajor 1050.4 µs vs group-major+fast 1060.0 µs at batch 128** — chanmajor already wins. The ranking **inverts with batch** (at batch 8 group-major+fast leads 179.7 vs 223.6) because chanmajor's block size is batch-independent while group-major's grid is `N×G` — 256 blocks on 84 SMs at batch 8, 4096 at 128. End-to-end confirms: MoDiff arm **73.9 vs 73.2 ms/step, 0.7 slower**, 3 of 3 runs | **closed, negative.** The shared-policy header stays (it makes the bit-identity invariant structural); all three decompositions score 1.0 vs fp64 |
| ~~**C7**~~ | ~~MoDiff warm-up~~ | ~~+663/+615 ms per cold sample~~ | **MEASURED 2026-08-16, and the answer inverts the reason the current value was chosen.** `warmup_steps` is 5 for both precisions; at 8 seeds, **1 round beats 5 by 26.5% ± 1.27 on 8/8 seeds at W4A4** (relL2 0.4009 vs 0.5466) and is unresolved at W8A8. Rounds 2/3/5 are indistinguishable. The 3→5 change was justified on the per-round *activation* reconstruction (0.4006 → 0.00001 at A4); the latent goes the other way, and the disagreement is largest exactly where that metric improves most. **Not applied**: W8A8 is the shipping arm and unresolved there, and the committed FID numbers were measured at 5, so replacing the default needs the FID row rather than the relL2 row | **answered** → [FINDINGS_WARMUP](gn_fast_reduce_2026-08-16/FINDINGS_WARMUP.md); remaining work is one FID run at 1 round |
| ~~**C8**~~ | ~~70 orphaned int4 wrappers~~ | ~~114 MiB~~ | **STALE WHEN FILED — already reclaimed.** The object-identity dedup landed 2026-08-13 (`int4_optimized.py:1885`, "DEDUPLICATED BY OBJECT IDENTITY (`_memo`), added 2026-08-13 to fix a 114 MiB leak"). The 114 MiB is what the fix *saved*, measured after the fact, not a debt outstanding | closed |
| ~~**C9**~~ | ~~`_qout` under MoDiff~~ | — | **NOT HEADROOM — a structural fact, and it is the reason route (b) exists.** `_qout` writes the projection-quantized int8 output, which cannot coexist with MoDiff's fp16 o_hat state; `qout_eligible == 0` on all 21 blocks is the correct answer, not a missed one. Route (b) (`gemm_w8a8_awq_o_hat_out_i8`) is the way to get both, and it landed. Nothing to recover | closed → B6 |

---

## What is settled and should not be re-litigated

So the list above is read against a fixed background, not an open one:

- **W8A8 + MoDiff is the result.** FID 7.802 against fp16's 7.803 at 1.41×, 1.95/255 mean paired pixel
  distance, 97.3% of the quantization error removed.
- **Nearly a third of the end-to-end gain is fusion, not low precision.** The elementwise/copy bucket
  falls 3.42× and is *identical* at W8A8 and W4A4.
- **Going 8→4 bits buys exactly one bucket**: 2643 of the 2674 ms saved is GEMM/conv, 98.8%.
- **Per-conv-layer int4 is 3.83× median** (14 layers, arithmetic-only subset), with 7 unquantized
  controls at 1.00× proving the layout normalization matches real layers.
- **The paper reproduces**, and two swept calibration constants (`DELTA_CLIP_RATIO = 8`,
  `ACT_CLIP_RATIO = 4.5`) closed most of the W4A4 gap without a kernel change.
- **Suite totals are not a valid speedup denominator** (A1/A2). The load-bearing tables are the
  per-layer matched conv table and the full-run profile buckets.

---

## Archive: closed since this list was opened

Kept because the *pattern* is the reusable part, not the conclusion. Nothing here is still open; each
row names the file that holds the evidence.

| # | what it was | how it closed |
|---|---|---|
| **A1** **[2026-08-16]** | linear reads 0.61×/0.67× because "in fp16 the attention projections are 1×1 convs" counted in the conv suite, and `conv + linear` cancels it | **fixed** |
| **A2** **[2026-08-16]** | `other` 3.77×, `norm_quantize` 0.64× | **fixed** |
| **A3** **[2026-08-16]** | attention T=1024 nets 1.19× (int8) / 1.16× (int4) because "the flash kernels take a padded head dim" | **fixed** |
| **A4** **[2026-08-16]** | T≤16 "falls back to `torch_sdpa_fp16` in every arm — correctly" at 1.00×/0.99× | **fixed, then partly retracted.** The published 1.00× was wrong (it is 0.87×: only 15 of 25 calls fall back at 49.4 µs, the other 10 take `qi8packed_small_qout` at 65.8 µs) — that stands. But I then called it "a sign error, not a tradeoff", and that was wrong: `quantized_std_attention.py:484` documents the loss with its own measurements and names what it buys — one uniform dataflow and no separate `quant_attn_out_int4_pack` pass on those blocks. So the ~0.16 ms/step is not free to reclaim. Corrected in both reports |
| **A5** **[2026-08-16]** | "5 of 8 blocks matched in all three arms" | **fixed** |
| **A6** **[2026-08-16]** | int4 attention's one real win, T=256 at 1.66× | **fixed** (flagged in the table) |
| **A8** **[2026-08-16]** | the GroupNorm+SiLU family is 32.2% of the W4A4 run at 1.13×, and C1 said "no design has landed" | **fixed** → [gn_fast_reduce_2026-08-16](gn_fast_reduce_2026-08-16/FINDINGS.md) |
| **A16** **[2026-08-16]** | `ms/sample` in the suite tables, next to an e2e table using the same label | **fixed** — KERNEL_SPEEDUP §1 now carries a `ms/step` column (attention 12.67, all-five 97.66 against the wall clock's 103.00) and states the collision in the table's own preamble |
| **A10** | fix #2 (activation zero point) is a quality lever | **closed** |
| **A12** | the csrc/ split bought −1.3 ms | **closed** as artifact; same-session references reproduce to 0.21 ms |
| **A13** | the `aq_*` fusion is blocked because "the non-qout siblings were deleted" | **closed**; blocker was stale → C2 |
| **A14** | hd=24 wants an 8-byte loader | **closed** → C3 |

Two patterns run through the closed rows, and both are about process rather than about kernels.

**1. Items were carried forward on the strength of their own prose rather than re-checked against the
code.** A13's blocker had been deleted. C2 had already landed. C8 had already been fixed. A1's stated
cause was contradicted by the very JSON that produced the table. A8's "no design has landed" was one
`getattr` away from false — and that one was worth 6.6–7.2 ms/step, the largest item on the list, sitting
behind a sentence nobody re-derived. **Re-derive before prioritising**; the cost of not doing so was
measured here at roughly a 10% end-to-end speedup left on the floor for three days.

**2. A gate that does not assert it fired will eventually report a believable nothing.** Four instances
now, three of them found the same day:

| harness | reported | actually |
|---|---|---|
| `quality_route_b_paired.py` | BIT-IDENTICAL at 3 and 8 seeds | route (b) fired **0 times in both arms** — attention was PTQ, where it cannot apply |
| `quality_gn_fast_paired.py` (first draft) | BIT-IDENTICAL | ran MoDiff mode, where the flag it toggled is inert |
| `quality_gn_fast_paired.py` (second draft) | arm-to-arm relL2 7.5e-3 | cannot distinguish "worse" from "differently rounded, same distance from fp16" |
| the three in `b2525d5` | pass | vacuous |

Every timing and quality harness written on 2026-08-16 counts the kernel it claims to be switching and
**fails loudly** if the two arms ran the same code. That is the cheapest available defence and it caught
two of its own authors' mistakes within the hour.

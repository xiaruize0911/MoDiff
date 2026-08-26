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
| **A9** | MoDiff is the quality answer | True at W8A8 (97.3% of the quantization error removed, FID 7.802 vs fp16's 7.803). At W4A4 it removes **31.3%** and FID is 200.1 vs PTQ's 278.0 — the dominant error is in the **weights**, which an activation method cannot reach ([fid_2026-08-05](fid_2026-08-05/FINDINGS.md)) | **diagnosed**, not fixed → B5. **W4A4 REVISED 2026-08-17: FID 55.49, not 181–200** — two mode flags (`int4_l1` + `DELTA_STATIC=1`), no kernel change → [w4a4_quality_2026-08-17](w4a4_quality_2026-08-17/FINDINGS.md) |
| **A11** | fix #4 (weight zero point / AdaRound) was deprioritised | Deprioritised on ‖W−Q(W)‖, the one metric AdaRound is willing to lose. On conv output error it wins 1.35×; end to end, weight-only, **1.58×** ([FINDINGS_WEIGHT_ZP](zp_coverage_2026-08-13/FINDINGS_WEIGHT_ZP.md)) | **open** — reprioritise, → C6. **RE-PRICED 2026-08-17: the 1.35× is not fix #4's prize.** It conflates AdaRound's *rounding* (no kernel; already runnable, and measured end to end at FID 181.5 → 140.2) with its per-channel *zero point* (needs the windowed reduction). Isolated on the same instrument: with `z_w` 0.0504 vs on our symmetric grid 0.0602 — **fix #4 is worth 1.20×, on 60/70 convs**, on top of what already runs |
| **A20** **[new 2026-08-17]** | C6 needs a windowed-reduction capability the tree lacks | **Step 1 exists now.** `modiff_cutlass.int4_window_sum` (`csrc/modiff/conv/zpw_window_sum.cu`) computes `S[p]`, gated exact at **max\|Δ\| = 0.0** on 8 real shapes with a sign-extension negative control ([test_zpw_window_sum.py](../integration/tests/test_zpw_window_sum.py)). The decomposition it serves is verified to 9.2e-16 and is **padding-clean** — unlike fix #2, whose `−z_a·Σ(missing w_q)` term is what killed its cheap route. Two corrections to C6's own scoping: `VisitorColBroadcast` **is** the node needed (M is the pixel axis, N the channel axis), and the surface is **four** entry points, one of which (`conv2d_int4_fprop`) has no EVT at all | **open, but the remaining work is PYTHON, not CUTLASS.** The additive post-pass is now measured at **1.65e-4** relative on real kernels ([test_zpw_additive.py](../integration/tests/test_zpw_additive.py)) — fp16 epilogue precision, three orders below W4A4's own 0.31 relL2 — against **6.0–6.8e-1** for the uncorrected kernel. So fix #2's −1.6%-vs-−7.1% penalty does **not** transfer, and the fused EVT is a performance question, not a correctness one. It runs on the existing signed-int4 kernel via `x_q − z_w = (x_q − 8) + (8 − z_w)`. Remaining: wire the correction at every int4 conv call site **at once** (`x_packed` is in hand at all of them), taking care that the o_hat entries mutate the cache in place, and add a `MODIFF_ZP_STRICT`-style guard that RAISES on any uncovered site first — partial coverage is what produced fix #2's false "answered NEGATIVELY" |
| **A25** **[new 2026-08-18]** | fix #4 is worth 1.20x on conv output error, so wiring it is the remaining path for W4A4 (A11/A20) | **WIRED, GATED, AND IT DIVERGES -- because the prize was priced against the WRONG NETWORK.** All 14 int4 conv sites are now corrected (10) or refuse (4), and all three wiring patterns pass independent float64 gates with negative controls: o_hat 4.67e-04, direct-output 4.95e-08, dual-store 4.28e-04 on both the cache and the return. End to end it reads **FID 309.689** against a 52.584 baseline and **latent relL2 2.55** -- and zeroing the weights entirely gives 1.0, so that is a different function, not a degraded one. The cause is not the arithmetic: **AdaRound's weights are the EMA network's**, whose fp32 weights differ from the deployed non-EMA ones by a median **0.0860** (paper_repro section 9). The 1.20x was measured by scoring AdaRound against **its own** fp32 reference, which never tested whether those weights suit our network. Corroborated three ways independent of the wiring, all measured today: AdaRound hurts at 20.922 -> 22.422 on the screen, 222.515 -> 240.524 at 10k FID, and 52.584 -> 309.689 with the zero point. It helped ONLY when the baseline was 181.5 | **open, and re-scoped: the lever is not "import the paper's AdaRound".** The gated machinery is correct and reusable; what it needs is a weight reconstruction run **on our own network** (the `qdiff/` package and `cali_data/church.pt` are both on disk) instead of the paper's checkpoint |
| **A24** **[new 2026-08-18]** | a constant that works on the conv delta is a reasonable seed for the projections' delta | **NO -- 4x.** The projections' static delta table at the conv path's `DELTA_CLIP_RATIO=8` reads **FID 222.515** against **54.300** for no table at all. Swept: **1.0 -> 52.584**, 2.0 -> 54.555, 8.0 -> 222.515. The conv 8 is for a heavy-tailed delta; the projections' is not, and under-sizing 8x clips. At the swept 1.0 the table wins on BOTH axes -- **-8.81 ms/step AND -1.72 FID** -- and is the new best W4A4. Two defects came with it, both closed: the table auto-loaded from disk (30-minute window where L1+static measured 222), and changing the default silently reinterpreted the baked artifact (the ratio now travels in the file as `__clip_ratio__`) | **closed** -> [w4a4_quality_2026-08-17](w4a4_quality_2026-08-17/FINDINGS.md) section 6.4 |
| **A23** **[new 2026-08-17]** | W8A4 + MoDiff is the paper's configuration and it **does NOT beat W8A8 PTQ** — FID 35.303 vs 16.366, so the paper's "dropping a bit is free" claim does not hold here (commit e9d756d) | **REVERSED. W8A4 + MoDiff = FID 7.835**, i.e. fp16 parity (7.803) and W8A8+MoDiff parity (7.802), beating W8A8 PTQ's 16.366 decisively. The 35.303 was measured on **L0 + dynamic**; two mode flags (`w8a4_l1` + `DELTA_STATIC=1`) move it 4.55×. Matched baseline in the same conditions: **w8a4_l0 + dynamic = 35.693**, next to the committed 35.303, so the comparison is sound. **The paper's headline claim reproduces on this tree.** Consequence for W4A4: 4-bit activations cost essentially NOTHING, so the entire 54.30 → 7.835 gap is the WEIGHTS — A9/B5's diagnosis, now quantified at 46.5 FID | **the A4 axis is closed at parity.** What remains for W4A4 is fix #4 (A20) and AdaRound, i.e. the weight side only |
| **A22** **[new 2026-08-17]** | `ACT_CLIP_RATIO = 4.5` closed most of the W4A4 gap and is the shipped default ([paper_repro](paper_repro_2026-08-12/FINDINGS.md) §3, worth 1.84×/1.98× on relL2) | **It never reached the deployed arm.** The constant is applied ONCE, in `end_calibration` ([int4_optimized.py:1761](../integration/kernels/int4_optimized.py:1761)), and baked into whatever `export_int4_static_scales` writes; the load path only fills the value (`set_static_scale`, :1892). `int4_calibration_realckpt.pt` is dated **2026-08-04** and the constant landed **08-12**, so the file carries no 4.5. Verified by calibrating twice on the same protocol and matching the shipped file against both: it matches **ratio 1.0 (0.9398)**, not 4.5 (**4.3022**). **Every committed W4A4 FID number, including 2026-08-17's 55.49, ran without it.** Found by a non-vacuity check whose 4.30× "difference" was the constant, not the effect under test — the same unit-error class as qdiff_bridge §9.1's 18.1× | **MEASURED, real but small: 1.124 FID (2.0%).** Both arms freshly calibrated so vintage is not a second variable: L1+static at ratio **1.0 → 55.424**, at **4.5 → 54.300**. Against the 0.416 FID repeat spread that is 2.7×, so resolved. The 1.0 arm landing on 55.424 next to the independently measured 55.490/55.074 validates the setup. **But paper_repro measured this same constant at 1.84×/1.98× on relL2** — on FID it is worth 2%. That is the FOURTH time relL2 has overstated against FID here (B3, C7, the n=16 pixel screen, this). **Fix: re-export the calibration file** — free at runtime, and 54.300 is the new best W4A4 |
| **A21** **[new 2026-08-17]** | `DELTA_CLIP_RATIO = 8` was fitted on L0 + dynamic and is protocol-dependent, so it should be re-swept per arm | **Re-swept on L1 + static with FID (the 16-image screen cannot resolve it, A19): 4 → 60.861, 8 → 55.28, 16 → 56.945.** The docstring's warning is real — both neighbours are clearly worse — but the optimum did NOT move when 42 layers joined the delta path and the scale went from per-call absmax to a fixed table. A parabola in log2(ratio) puts the vertex at ≈9.7, ~0.5 FID below 8, i.e. **inside 1.3× the 0.416 FID repeat spread**. `MODIFF_DELTA_TABLE_RATIO` (new) makes the re-sweep an env var by scaling the loaded table by `r/8`, arithmetically identical to re-exporting | **closed.** The axis is exhausted at the resolution of the only instrument that resolves |
| **A19** **[new 2026-08-17]** | 16-image `mean \|Δ\|` screens rank arms | **They cannot: the CROSS-PROCESS floor is 8.7/255**, and every arm comparison is cross-process (one `generate_fid_samples` invocation each). Larger than every effect in [w4a4_quality](w4a4_quality_2026-08-17/FINDINGS.md) §1, whose largest gap is 6.17. Intra-process, the same L0 config is **bit-exact (0.0000)**. `torch.backends.cudnn.benchmark = True` ([benchmark_ldm.py:59](../integration/benchmarks/benchmark_ldm.py:59), re-set at :546) accounts for ~28% — disabling it gives 6.316 — and **the remaining 6.3 is unidentified**. Very likely the mechanism behind P5's unexplained *"arm order moves W4A4 MoDiff by 28%"* | **open.** Consequence: pixel screens at n=16 are for spotting order-of-magnitude breakage, not for ranking. FID at 10k has a spread of **0.416** on the same arm and is the instrument that resolves |
| **A18** **[new 2026-08-17]** | every arm reproduces run to run | **`MODIFF_LINEAR=1` (L1) does not.** 3 draws, one process, same seed, state reset between: **4.5–6.2/255**. Every other arm is bit-exact at **0.0000** (W4A4 L0+dynamic, W4A4 L0+static, W8A8 L0+dynamic). Present in both delta modes, consistent with `wxax_linear.py` always taking the per-call `delta_absmax_fp16` — there is no static delta table for the linears. Two candidates ruled out: the ambient-seeded wxax calibration (now seeded; spread did not move, 4.11 → 4.37) and atomics/split-K in `gemm_w*_awq_o_hat` (neither is present) | **open, root cause unknown.** It sits in the arm A9 now recommends, and it puts a 4.5–6.2/255 floor under any 16-image screen containing an L1 arm |

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

## A17. Dependency pre-flight: six failures, always mid-run

**[new 2026-08-16]** This container has lost its Python environment at least once since 2026-08-13, and
every rediscovery this session happened *inside* a job rather than before it:

| # | missing | discovered when |
|---|---|---|
| 1 | `matplotlib` | regenerating report plots |
| 2 | `markdown`, `weasyprint` + the `libpango` system libs | rendering SUMMARY.pdf |
| 3 | `omegaconf`, `einops`, `pytorch-lightning`, `torchmetrics`, `tqdm` | first model build of a paired A/B |
| 4 | `pytorch-fid` | the **last step** of a 25-minute B2 pipeline |
| 5 | `ninja` | silently — distutils then rebuilt **nothing** on a header-only change, so a `.so` without the change would have been validated had the log not been read |
| 6 | `lmdb` | starting B4's 50k reference export |
| 7 | `omegaconf`, `einops`, `pytorch-lightning`, `torchmetrics`, `tqdm`, `matplotlib`, `pytorch-fid`, `scipy` | **[2026-08-17]** the environment was gone again — but the pre-flight named all eight in 3 s, before any GPU second |
| 8 | `ninja` | **[2026-08-17]** #5 recurred. Installed before any rebuild, because a fix #4 build would otherwise have been silently invalidated |

**A17 is working.** #7 is the first of these eight discovered *before* a job rather than inside one, which
is the whole point of the check. #8 was found by asking for it rather than by a wrong result.

`aq_fusion_2026-08-12`'s provisioning note has now been stale three times. **#5 is the dangerous one**: it
did not fail, it silently produced a stale artifact, and only reading the build log caught it.

**The fix is cheap and has been paid for six times over:** an import/`ldconfig` pre-flight at the entry
points that own multi-stage work (`run_all.sh`, `generate_fid_samples.py`, the A/B harnesses), asserting
the set each one needs *before* the first GPU second is spent. Install under the existing
`torch==2.4.1` constraint file so nothing swaps torch out from under the built extension.

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

   **RESOLVED by a paired bootstrap** ([fid_bootstrap.py](gn_fast_reduce_2026-08-16/scripts/fid_bootstrap.py),
   40 replicates, one index set applied to both arms because they share a seed sequence):

   | | FID | bootstrap ± |
   |---|--:|--:|
   | absmax | 171.739 | ± 0.458 |
   | MSE | 181.523 | ± 0.486 |
   | **difference** | **+9.783** | **± 0.283**, 95% CI **[+9.218, +10.250]** |

   0 is outside the interval. The difference's bootstrap mean (+9.783) matches the point estimate
   (+9.746), which is the number that matters — both arms' absolute values sit slightly high because
   resampling with replacement biases a covariance estimate, but that bias is common to the pair and
   cancels in the difference.

   **What this still does not cover:** the bootstrap captures sampling variance of the generated set only.
   The pairing removes noise-draw variance *from the difference* by construction, but a different seed
   *sequence* has not been tried. So the claim is "on this seed sequence, MSE is worse by 9.78 ± 0.28",
   not "MSE is worse in expectation over seeds".

   **Actionable, and left as a decision rather than taken:** `MODIFF_INT4_WSCALE` should be re-evaluated
   as a default and `docs/fid_2026-08-05`'s adoption rationale annotated. I did not flip it, for two
   reasons: flipping it makes latent relL2 **worse** by 7.5% (also paired, also resolved), so this needs
   an explicit ruling on which metric is authoritative — that is exactly the choice this item exists to
   expose; and W4A4 is unusable at either setting (FID ~171–181 against fp16's 7.8), so the value of the
   finding is what it implies for metric selection elsewhere, not for this arm's shipping default.
4. ~~**FID at 50k**~~ **DONE 2026-08-16 for what it was needed for: the headline claim now holds outside
   the biased N, and the remaining 3 modes are unnecessary.**

   | protocol | fp16 | W8A8+MoDiff | Δ |
   |---|--:|--:|--:|
   | 10000 gen vs 10000 real | 7.803 | 7.802 | **−0.001** |
   | **14730 gen vs 50000 real** | **6.450** | **6.449** | **−0.001** |
   | 50000 gen vs 50000 real | 5.804 | *(not generated — see below)* | — |

   **The parity is not a 10k artifact.** Absolute values move with protocol (7.80 → 6.45 → 5.80) while the
   gap between the arms is −0.001 in every one. That also demonstrates the bias factor is **common across
   modes**, which was the premise the session's 10k relative conclusions rested on — B2's 2.16-vs-0.94 and
   B3's +5.70% ± 0.28 both survive, and **3 more modes at ~1.5 h each are not needed**.

   **The 10k bias factor is 1.344×** (fp16 7.803 → 5.804). Separately, **50k is still 1.44× LDM's published
   ~4.02**, which sample count cannot explain: `paper_repro_2026-08-12` already lists **EMA weights** as an
   unfixed deviation ("integration's loader never swaps EMA"), and it degrades FID in this direction. DDIM
   50 steps versus the published step count is the other candidate. This is the **first external anchor**
   this pipeline has ever had, which matters because four conclusions were retracted this session and all
   of them rested on it.

   **Why W8A8+MoDiff stopped at 14731 — and a new constraint to record:** `Errno 122 Disk quota exceeded`.
   `df` reports 213 TB free on `/workspace`, so this is a **MooseFS quota, not capacity**, and reading `df`
   is not sufficient to plan a multi-GB run. Two further details worth carrying: the quota failure left a
   **truncated PNG** that only surfaced as `PIL.UnidentifiedImageError` at read time (so a generated
   directory needs a completeness check, not just a file count), and the equal-N comparison was salvaged
   with **hard links**, which cost no quota. FID artifacts now on disk: ~23 GB across `/workspace/fid*`.

5. **A weight-side method for W4A4** — **there is a candidate, its inputs are on disk, and it does not
   need C6.** Filed as "no candidate has landed"; the pieces are all present:

   * **`/workspace/quant_models/church_w4a8_ckpt.pth` (2.36 GB) is on disk**, as is
     `/workspace/cali_data/church.pt`. `paper_repro_2026-08-12` recorded both as *"turned out to be
     obtainable"* and downloaded them — after the run that used `--skip_weight_recon` (RTN) *because the
     checkpoint was not on disk*. Nothing since has used them.
   * **The 1.58× is AdaRound *plus* a weight zero point**, and the zero point is what needs C6 — the table
     in `FINDINGS_WEIGHT_ZP` is headed "AdaRound + weight zero point", and `paper_repro` §5 says importing
     those weights *"needs a per-output-channel weight zero point"*.
   * **But the same document measured the symmetric fallback**: "AdaRound re-quantised on **our** grid"
     scores 0.1581 / 0.3235 against qdiff's native 0.1506 / 0.3110 — **a 4–5% loss** to drop the zero
     point entirely. Against a 1.58× end-to-end prize, that trade looks strongly favourable, and it needs
     **no CUDA work at all**: load the checkpoint, re-quantise onto the existing symmetric
     `_int4_weight_scale`, measure.

   **The metric rule to measure it against, now settled by evidence rather than preference:** B4 showed the
   FID *difference* between two arms is −0.001 across three protocols (10k/10k, 14730/50k, and the 50k
   anchor) — so relative FID is a stable judge. B3 showed latent relL2 can point the **opposite way** to
   FID (MSE weight scale: −7.5% relL2, **+5.7% FID**). A11 is itself an instance (deprioritised on
   ‖W−Q(W)‖). **So: FID differences decide; relL2 screens.** Screening on relL2 is still worth doing first
   because it is minutes rather than ~25 min/arm — but it cannot be the verdict.

   **A risk that changes the first step, found 2026-08-16.** The 1.58× comes from
   `zp_coverage_2026-08-13/scripts/weight_zp_end_to_end.py`, whose own first line says it measures
   *"with no kernel at all"* — and it does `import act_fake_quant as A`, i.e. it is built on the harness
   **P4 RETIRED**. P4's record: it *"failed a third self-check"* and *"would have said 'implement' where
   the truth is negative"*. The script argues its case is different (a weight fake-quant is static where an
   activation one is not, which is a reasonable argument) but that argument has not been independently
   checked, and P4's failure mode is exactly the direction this item points.

   So **the first step is not wiring, it is verifying 1.58× without that harness** — which is now possible
   precisely because the checkpoint is on disk. Load it, re-quantise onto the symmetric grid, and measure
   the real path. If 1.58× survives, wire it; if it does not, this item closes negative and C6 loses its
   main motivation with it.

   **The verification is fully specified — the rounding formula is confirmed from source, not inferred.**
   `qdiff/adaptive_rounding.py:49-61`, with `soft_targets=False` (the inference path):

   ```python
   x_floor = torch.floor(W / delta)
   x_int   = x_floor + (alpha >= 0).float()        # the learned rounding, hardened
   x_quant = torch.clamp(x_int + zero_point, 0, n_levels - 1)   # W4 -> n_levels = 16
   W_q     = (x_quant - zero_point) * delta
   ```

   The checkpoint carries `weight_quantizer.{alpha, delta, zero_point}` per layer alongside the fp
   `weight`, so `W_q` is **exactly reconstructible with no fake-quant of any kind**. That is what makes the
   verification independent of the retired harness.

   **Steps 1–2 DONE 2026-08-16.** 89 convs, median ‖Q(W)−W‖/‖W‖:

   | | median |
   |---|--:|
   | AdaRound native (asymmetric, needs C6) | 0.1410 |
   | ours, RTN + MSE symmetric | **0.1256** *(the function's own docstring records 0.1254)* |
   | **AdaRound re-quantised onto our symmetric grid** | **0.1506** |

   **Method double-validated**: the reconstruction gives exactly 16 discrete values per output channel
   (impossible by chance if the formula were wrong), and our baseline reproduces the docstring to 0.2%.
   An earlier attempt read +35.7% for dropping the zero point — that used an **absmax** grid, which the
   same docstring records at 0.1825 against my measured 0.1821, so the discrepancy was the grid, not the
   method.

   **The premise for "does not need C6" holds:** dropping the zero point costs **+6.8%** (0.1410 → 0.1506),
   close to `paper_repro`'s recorded +5.0%.

   **And a trap worth naming, because it is the same one A11 fell into.** The re-quantised weights are
   **20% worse than our own baseline on ‖W−Q(W)‖** (0.1506 vs 0.1256). That is *not* evidence against
   AdaRound — ‖W−Q(W)‖ is the one metric AdaRound is willing to lose, which is exactly why A11 was
   wrongly deprioritised. Killing it here on this number would repeat that error verbatim. The verdict
   must come from step 3.

   Remaining steps:
   1. ~~Reconstruct `W_q` per conv from the checkpoint with the formula above.~~ done
   2. ~~Re-quantise `W_q` onto the existing **symmetric** `_int4_weight_scale`.~~ done — costs +6.8%.
   3. Load both that and the current RTN+MSE weights into the real `OptimizedInt4Conv2d` and measure
      end-to-end latent relL2 against fp16 — **this is the number that either confirms or kills 1.58×**.
      **Its main unknown is now removed (2026-08-16): the layer-name mapping is a clean bijection.**
      89 conv layers on each side, `model.X` → `model.diffusion_model.X` hits **89/89**, and **0 shape
      mismatches**. So the substitution is a rename, not a matching problem — which is where this kind of
      cross-checkpoint work usually goes wrong silently.
      **DONE 2026-08-16 — AdaRound wins on the real kernels, but by 1.208×, not 1.58×.**

      | arm | mean relL2 vs fp16 (4 seeds) |
      |---|--:|
      | ours (RTN + MSE scale) | 0.4699 |
      | **AdaRound → our symmetric grid** | **0.3889** |
      | paired per-seed diff | **−17.15% ± 1.77% (SEM)** — resolved |
      | ratio ours/AdaRound | **1.208×** |

      Non-vacuity **PASS**: 89 convs substituted in the AdaRound arm, 0 in the baseline. **So the candidate
      is real, it does not need C6** (this is measured *after* dropping the zero point onto our symmetric
      grid), and the retired fake-quant harness **overstated the prize by 31%** — 1.58× against a true
      1.208×. P4's warning about that instrument was directionally right without being right about the sign.

      **A correction I have to make about my own instrument.** Two earlier runs were written up as
      artifacts caught by the counter. Only the first was: it substituted 2/89 by matching post-conversion
      module names. **The second was a real result that my counter wrongly rejected** — the VAE's
      first-stage checkpoint is *also* named `model.ckpt`, so `tally["n"] = n` let its 0 overwrite the
      UNet's 89, reporting 0/89 for a run that had substituted all 89. Its −17.57% / 1.214× reproduces as
      −17.15% / 1.208× here. **The counter written to catch false positives produced a false negative
      instead**, and the fix is `+=` rather than `=`. The lesson is narrower than "counters are good": a
      counter is itself an instrument and needs the same scepticism as the thing it guards.

   4. **DONE 2026-08-16 — the FID verdict agrees with the screen and is LARGER than it.**

      | arm | FID vs LSUN-churches 10k |
      |---|--:|
      | ours (RTN + MSE scale) | 181.514 |
      | **AdaRound → our symmetric grid** | **140.187** |
      | difference | **−41.327 (−22.77%)** |
      | 40× paired bootstrap, n=10000 | −41.317 ± 0.283, 95% CI **[−41.769, −40.800]** |

      **RESOLVED** — 0 is nowhere near the interval. Both arms regenerated from the current tree (37 min,
      serial), 10000 images each, arm B's non-vacuity assertion passing *before* the first image
      (89 convs). `eigh` agrees with pytorch_fid to 1.28e-12. Our 181.5 matches B3's independently
      measured MSE arm at 180.593 to ~1 FID, which is an unplanned cross-check of both runs.

      **B5 IS RESOLVED: implement.** AdaRound's rounding, re-quantised onto our symmetric MSE grid,
      takes W4A4 from FID 181.5 to 140.2. It does not need C6 — the zero point is dropped, costing the
      +6.8% on ‖W−Q(W)‖ that got A11 deprioritised, and it still wins by 22.77% on the metric that decides.

      **What this says about the retired instrument.** The fake-quant harness claimed 1.58× on latent
      relL2 where the truth on real kernels is 1.208× — it overstated the quantity it measured by 31%.
      But the quantity it measured was the wrong one: on FID the prize is −22.77%, *larger* than the
      screen suggested. So P4 was right to retire it and I was right not to trust its number, but the
      reason the number was wrong is not the reason I expected. It was not overselling a small effect;
      it was mis-measuring a large one.

      **A9 closes with this.** A9's diagnosis was that W4A4's dominant error lives in the weights, which
      MoDiff (activation-only) cannot reach. Changing nothing but the weight rounding moves FID 22.77%.
      That is the diagnosis confirmed, and it is also the answer to "why does W4A4 lag" — not the
      activation path, and not MoDiff.

      **The caveat that bounds this.** Measured at `MODIFF_LINEAR=0` (conv-only MoDiff), because that is
      the legacy `int4` key's default. The tree's default since 2026-08-06 is `LINEAR=1`. Weight rounding
      is orthogonal to whether the 42 attention projections carry a_hat/o_hat, so the direction should
      hold; the magnitude at L1 is untested and I am not quoting −41.3 for it.


   Eighth item this session filed as missing that was already present.
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
   **CLOSED 2026-08-16 at 32 seeds — and the sample-size estimate above was wrong by 16×.**
   32 seeds: **−0.683% ± 1.125% (SEM), stdev 6.365%** — still not resolved, non-vacuity PASS
   (`gemm_w8a8_awq_o_hat_out_i8` 15680 ON / 0 OFF). Two things changed with the extra seeds: the point
   estimate **halved** (−1.51% → −0.683%) and the stdev came in at **6.365% against the 3.5% the n ≈ 22+
   estimate was built on**. Resolving a 0.68% effect at that spread needs SEM < 0.34%, i.e. **n ≈ 350**,
   not 22. So this was never "needs more seeds" either: the effect is smaller than the scale this harness
   can measure at any seed count worth spending.
   **Verdict: stays flag-gated, and the reason is now the right one.** There is no quality argument
   against the +0.79 ms/step and none for it — the honest statement is that route (b) is
   quality-neutral to within ±1.1%. Chasing 350 seeds to decide a sub-1% latent effect would also be
   deciding on the metric C7 and B3 both caught pointing the wrong way.
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
| **C4** | a real int4 attention datapath | **~2 ms/step (~4%)**, newly sized | **CHARACTERISED 2026-08-16, and it is a deliberate design rather than an oversight.** The production T=1024 route is `flash_attn_i4values_i8mma_qi8_kv_static_qout_hd24`, whose own comment says it carries *"exactly the same signed-int4 Q/K values and scales, but stores them **UNPACKED** and executes QK ... through **m16n8k32.s8**"*, with `TORCH_CHECK(... hd_pad=32)`. So the values are 4-bit but the storage and the MMA are int8 — which is exactly why A3 measured 1.21× against int8's 1.21×: **zero bytes saved**. (Correction: I first read A3's `hd_pad=64` as 2× over-padding. That 64 belongs to the generic `flash_attn_int4_vt`, not the production hd24 route, which is 32. There is no padding waste to reclaim.) **What a real datapath would buy:** packed storage (32 B → 16 B per row for Q/K, same for V) executed on sm86's `s4` tensor cores, which the hardware does support. The 48 KB smem constraint behind `FA_TILE_HD 64` gets *looser*, not tighter, since a packed tile is half the bytes. **Sizing it honestly:** attention is 17.1% of the W4A4 run at 1749 ms/batch, B10 splits it 56/44 score/projection, and softmax + the fp32 O accumulate do not shrink — so halving Q/K/V bytes is worth roughly **1.7–2.3 ms/step, 3–4%**, not a 2× | **new kernel, and the prize is now quantified rather than assumed.** Needs: `s4` MMA inner loop, packed smem layout, and V moved to int4 (currently `i8v` — the AV accumulation's precision cost is unmeasured). Not "nothing to win", not a large win either |
| **C5** | GN-stats epilogue, Stage C | — | **The bar moved against it on 2026-08-16 and it is now decisively negative.** Stage C folds the GN stats into the conv epilogue to remove a separate launch, and was measured at **0.96×** — "mechanism viable, margin is not". That ratio was against the GN kernel as shipped. C1 then made that kernel **1.91–2.03× faster**, including **2.23× on the `768×4×4` shape this item names as its gate**. The launch the epilogue would remove is now worth about half what it was, so the same mechanism scores ~0.5×. Nothing needs measuring to reject it | **closed** as refuted — reopen only if the epilogue's own reduction can beat `fast_reduce`, not the generic path |
| **C6** | weight zero point (fix #4) | **1.58×** end-to-end, weight-only | **RE-SCOPED 2026-08-16 — it is not a missing capability, and it is not an epilogue reduction.** The term is `z_w[k] · Σ_i a[i]`, and `i` runs over (c_in, r, s) — so `Σ_i a[i]` sums over **all input channels** as well as the window, making it **one scalar per output pixel**, a `[N, H_out, W_out]` field. It does not depend on `k`, which the source doc already noted ("one reduction serves all K"). Two consequences: **(1)** it need not be computed in the epilogue at all — a `Σ` over C then a **separable** R×S box filter costs O(N·C·H·W), i.e. about one activation read, not O(·R·S); **(2)** the epilogue term is then a rank-1 outer product `− z_w[k] · S[p]`, per-channel × per-pixel. Verified that `conv2d_evt.cu` already includes `cutlass/epilogue/threadblock/fusion/visitors.hpp`, and that header already provides **`VisitorRowBroadcast`** and **`VisitorColBroadcast`** — the existing tree uses `RowVec` for `weight_scale[k]`/`bias[k]` and `AuxLd` for the per-element residual, so the missing piece is a broadcast node that is already in the header being used | **no longer research.** Remaining work: a box-filter pre-pass, plus `Sm80EVT<Add, <tree>, Sm80EVT<Mul, RowVec(z_w), ColBroadcast(S)>>`. Risks: the EVT is hand-assembled onto the conv Mma (CUTLASS 4.6.1 has no EVT-on-conv path), so adding a node is manual; and whether row/col maps to channel/pixel under this tile's layout must be checked, not assumed. **Not** the "windowed-reduction epilogue that does not exist" it was filed as. **DEPRIORITISED 2026-08-16 by B5's verdict:** the zero point's whole value was the 1.58% it buys on ‖W−Q(W)‖, and B5 measured the AdaRound arm **with the zero point dropped** onto our symmetric grid winning by 22.77% FID. So C6 is no longer on the path to W4A4 quality — it is an independent 4–5% kernel optimisation, correctly scoped but no longer urgent |
| ~~**C10**~~ | ~~MoDiff GN→delta-quantize~~ | ~~7.80/8.16 ms/step~~ | **REFUTED 2026-08-16 — the premise compared the wrong two things.** C1's 1.91× was on the PTQ family, which uses a **group-major** stats decomposition where `fast_reduce` genuinely wins. The MoDiff path uses **chanmajor** (`BLK = C/K`) and never reaches the group-major code, so the policy is inert there (cross-arm check: 0/40 cases differ). Forcing group-major to measure what it would buy: **chanmajor 1050.4 µs vs group-major+fast 1060.0 µs at batch 128** — chanmajor already wins. The ranking **inverts with batch** (at batch 8 group-major+fast leads 179.7 vs 223.6) because chanmajor's block size is batch-independent while group-major's grid is `N×G` — 256 blocks on 84 SMs at batch 8, 4096 at 128. End-to-end confirms: MoDiff arm **73.9 vs 73.2 ms/step, 0.7 slower**, 3 of 3 runs | **closed, negative.** The shared-policy header stays (it makes the bit-identity invariant structural); all three decompositions score 1.0 vs fp64 |
| **C14** **[new 2026-08-26]** | the GN stats pass costs 36% more in W4A4 than in W8A8 | **0.96 ms/step**, W4A4 only | **Same kernel, same shapes, same call counts, 36% apart.** After C13 both arms run `gn_stats_partials_chanmajor_vec2_kernel` on every call; the W4A4 arm reads **3.624 ms/step** against W8A8's **2.665**. Verified per block size in the trace: the two arms launch the identical mix (19.0 calls at C=384, 22.8 at C=768, 8.5 at C=192, 4.8 at C=1536, 1.9 each at C=576/1152), and the GN input is fp16 in BOTH arms, so the kernel moves the same bytes doing the same work. It is not a regression — the pre-C13 W4A4 trace shows the same gap against the pre-C13 W8A8 one — and it is not the delta mode, which is static in both. Candidates not yet separated: clock/power state (W4A4's whole step is 61.7 ms of GPU busy against 74.5, so the arms sit at different sustained memory-throughput points), L2 residency differences from the surrounding int4 kernels, or something about how the int4 path allocates the stats partials | **open.** Cheap to attack: run the two arms' stats kernels back to back in ONE process on the same tensors. If the gap survives that, it is the kernel's environment and not the arm; if it vanishes, it is clocks and the W4A4 in-model numbers throughout this repo carry a scaling factor nobody has priced |
| ~~**C13**~~ | ~~GN stats + resize a_hat: scalar memory access~~ | ~~1.55 ms/step of bandwidth headroom~~ | **DONE 2026-08-26, −1.29 ms/step, BIT-IDENTICAL.** Two pure memory-access fixes, both inside the conv block. **(a)** `gn_stats_partials_chanmajor` was LATENCY-bound, not bandwidth-bound — 49.8% of peak in-model against the apply kernel's 84.5% on the same tensors — because each thread issued one 2-byte load per `hw` iteration (a warp requests 64 B, half a sector) into a serial accumulate. New `_vec2` kernel: a thread owns two ADJACENT channels loaded as one `__half2`, and the `hw` loop runs four positions with the loads hoisted. **3.435 → 2.625 ms/step**, 49.8% → 71.6% of peak; it also removes the need for the `K>1` channel split (block is C/2, so C ≤ 2048 fits the 1024-thread cap). **(b)** the resize kernel's a_hat RMW was two scalar `__half` loads + two `__float2half_rn` stores at `ci`/`ci+1` while `compute_pair` around it was already vec2 — now one `gn_load2`/`gn_store2`. **3.184 → 2.719 ms/step.** GN family total **14.751 → 13.459 (−8.8%)**; GPU busy 75.964 → 74.443. Bit-identity is structural in both (same values, same order, same rounding mode) and **gated, not asserted** | **closed, positive** → [gn_vec2_2026-08-26](gn_vec2_2026-08-26/FINDINGS.md), numerically verified against fp64 in [MSE_VERIFICATION.md](gn_vec2_2026-08-26/MSE_VERIFICATION.md) (stats MSE ~1e-18, all residual code disagreements are +-1 at a rounding-boundary, identical between scalar and vec2). Final W8A8 numbers: stats 3.435 → 2.665, resize 3.184 → 2.713, GN family **14.751 → 13.487 (−8.6%, 1.61% of a step)**. Gates: [test_gn_stats_vec2.py](../integration/tests/test_gn_stats_vec2.py) (16 shapes, forked processes, and it checks **the profiler saw two different kernels** — without that it would pass just as happily if the flag were never read, cf. A22) and [test_gn_resize_ahat_vec2.py](../integration/tests/test_gn_resize_ahat_vec2.py) (two-build capture/compare, 32 cases, 198.9M nonzero codes). **Two measured negatives worth keeping:** the apply kernel gains nothing from ILP (U=1 as shipped beats U=2/4/8 — it is genuinely at the bandwidth roof, consistent with C12); and an **end-to-end image A/B cannot gate any of this** — HEAD against ITSELF gave different sha256 over 64 images, which is A18/A19 showing up as a would-be false alarm |
| ~~**C12**~~ | ~~overlap a_hat INSIDE the conv kernel~~ | ~~the a_hat write is 2.024 ms/step (W8A8) / 1.742 (W4A4)~~ | **REFUTED 2026-08-26, measured on both sides.** C11 closed the *inter-kernel* reading of "overlap"; this is the *intra-kernel* one — put a_hat's traffic on SMs the conv already owns, the mechanism that makes o_hat cheap. **GN side (the prize):** eliding the a_hat write gives back **2.024 ms/step W8A8 / 1.742 W4A4** = 2.63%/2.55% of a step, 31%/30% of the apply kernel — it beats its own 2/7 byte model because the store drags write-allocate traffic with it. **Register risk did NOT materialise:** REG:240 STACK:0 LOCAL:0, unchanged; a_hat exact to **0 ULP**, o_hat bit-identical, negative control 37391 ULP. **Conv side (what kills it):** the conv charges **5.792 ms/step** for what the GN gives back at 1.926 — net **−3.866 ms/step, −5.02% of a step**. Placement before the mainloop vs after the epilogue differs by **< 1%**, so "o_hat is cheap because it is inside the conv" is directly refuted. Mechanism: the conv moves the added bytes at **57–69% of peak**, the GN kernel at **73–81%** — a kernel holding every SM is a WORSE place for a low-MLP streaming loop, and its 23–25% peak-bandwidth idle is not claimable headroom | **closed, negative** → [ahat_overlap_2026-08-26](ahat_overlap_2026-08-26/FINDINGS.md). Two durable by-products: **(a)** the 2.024/1.742 ceiling bounds every scheme that removes the a_hat write, including an int8/fixed-point cache — that idea now has a measured numerator with nothing built; **(b)** a **correction to perf_report §2** — o_hat's 2.35×/4.06× per-byte advantage is NOT intra-kernel latency hiding, it is that o_hat's store replaces one the baseline already does and its load rides that store's write-allocate. It is a property of o_hat, not of the conv, and does not transfer. Selective per-layer MoDiff is not bounded by the ceiling, because it drops the a_hat read, the write and o_hat together |
| ~~**C11**~~ | ~~hide a_hat behind the conv~~ | ~~a_hat = 14.8%/20.4% of the conv block, 6.4%/6.6% of a step~~ | **REFUTED 2026-08-26, and the evidence has been deleted at the owner's request — this row is all that is left.** Split batch 128 across two streams offset by one GN so `conv(A)` runs alongside `GN(B)`; numerically exact (the GN stats grid is `nblocks = min(HW,32)` × N, so a batch split changes `gridDim.y` only and the per-`(n,g)` fp32 accumulation order is bit-identical). With the **baseline arm** pipelined as a control — the control the first attempt lacked — freq-weighted absolute MoDiff-specific overhead **rose 11.1% (+1.95 ms, paired *t*, df = 4, p = 0.0004)**. The one shape that gained 28% (`768,4×4`) gave the baseline **30.5%** on the same run, i.e. 89.2% of it was arm-independent occupancy. Mechanism: the binding constraint is **SM occupancy, not bandwidth** — `GN(B)`'s blocks need SM slots to issue loads from, and `conv(A)` holds them (61,440 of 65,536 registers per block ⇒ 1 block/SM). And a compute-bound *co-runner* is not the same thing as fusion into a compute-bound *kernel*, which is why o_hat is cheap and a_hat is not. **Nothing was ever wired into the production path.** Measured on A40, batch 128, 5 shapes × 6 configs × 5 trials | **closed, negative.** Deleted: `bench_pipeline{,2}.py`, `pipeline_result2.json`, `pipeline_summary.csv`, `3_pipeline.png` and §3 of [perf_report_2026-08-26](perf_report_2026-08-26/REPORT.md), which now carries a tombstone and a revision-8 note. With C5 (0.5×) and C10 (regression) all three "restructure the kernels" routes are measured negative; the remaining lever is **fewer a_hat bytes**, not hidden ones |
| ~~**C7**~~ | ~~MoDiff warm-up~~ | ~~+663/+615 ms per cold sample~~ | **MEASURED 2026-08-16, and the answer inverts the reason the current value was chosen.** `warmup_steps` is 5 for both precisions; at 8 seeds, **1 round beats 5 by 26.5% ± 1.27 on 8/8 seeds at W4A4** (relL2 0.4009 vs 0.5466) and is unresolved at W8A8. Rounds 2/3/5 are indistinguishable. The 3→5 change was justified on the per-round *activation* reconstruction (0.4006 → 0.00001 at A4); the latent goes the other way, and the disagreement is largest exactly where that metric improves most. **Not applied**: W8A8 is the shipping arm and unresolved there, and the committed FID numbers were measured at 5, so replacing the default needs the FID row rather than the relL2 row | **CLOSED 2026-08-16 — the FID run is done and it REVERSES the relL2 result.** warm-up=5 FID **181.514**, warm-up=1 **205.263**, difference **+23.749 (+13.08%)**, 40× paired bootstrap +23.718 ± 0.431, 95% CI **[+22.967, +24.348]** — RESOLVED. Non-vacuity by image comparison: 3 sampled PNGs all differ from the warm-up=5 arm, so the env var took effect. **VERDICT: keep 5.** relL2 said 1 round was 26.5% ± 1.27 better on 8/8 seeds; FID says it is 13.08% worse, and both are resolved. This is the SECOND time relL2 and FID point opposite ways (B3 was the first), and it retro-justifies the 3→5 change: that was decided on per-round *activation* reconstruction, the metric I treated as the suspect one, and it agrees with FID while the latent does not → [FINDINGS_WARMUP](gn_fast_reduce_2026-08-16/FINDINGS_WARMUP.md) |
| ~~**C8**~~ | ~~70 orphaned int4 wrappers~~ | ~~114 MiB~~ | **STALE WHEN FILED — already reclaimed.** The object-identity dedup landed 2026-08-13 (`int4_optimized.py:1885`, "DEDUPLICATED BY OBJECT IDENTITY (`_memo`), added 2026-08-13 to fix a 114 MiB leak"). The 114 MiB is what the fix *saved*, measured after the fact, not a debt outstanding | closed |
| ~~**C9**~~ | ~~`_qout` under MoDiff~~ | — | **NOT HEADROOM — a structural fact, and it is the reason route (b) exists.** `_qout` writes the projection-quantized int8 output, which cannot coexist with MoDiff's fp16 o_hat state; `qout_eligible == 0` on all 21 blocks is the correct answer, not a missed one. Route (b) (`gemm_w8a8_awq_o_hat_out_i8`) is the way to get both, and it landed. Nothing to recover | closed → B6 |

---

| **C15** **[new 2026-08-26]** | int8/fixed-point a_hat cache | ~2.0 ms/step W8A8, 1.7 W4A4 (the a_hat-write ceiling from C12) | **RE-SCOPED, not built.** The idea was floated as "free": `a_hat += q/scale` is an exact integer step, so store a_hat as an integer at grid spacing `1/scale`. **That is only exact if `scale` is constant across all 200 DDIM steps — it is not.** `_delta_scale_args` reads a per-step table, and real calibration data already on disk ([delta_calibration.json](modiff_correctness_2026-08-03/data/delta_calibration.json), 70 layers) shows `step_gain_tail` (scale at the tail ÷ scale at step 0) has **median 12.45×, max 124.5×**. A fixed a_hat grid re-quantizes on every update, and by linearity that error accumulates in `conv(a_hat) − o_hat` — it does not cancel. A no-CUDA simulation using the real step0/tail scale values ([int8_ahat_cache_2026-08-26](int8_ahat_cache_2026-08-26/FINDINGS.md)) found the storage drift is a tolerable 2–3× the code's own quantum at step 0 but **14.5× to 7137×** the quantum by the tail — for the high-gain layer the drift never stops growing. A flat 8-bit grid for the whole schedule is very likely not viable on most layers | **open, and both proposed fixes are now closed too.** **fp8 tested WORSE, not better** — its floating exponent spends 4–5 of 8 bits on a range a_hat does not need, leaving an LSB 13–27× coarser than a dedicated fixed grid at a_hat's typical magnitude. **Periodic refresh does not help either** — a bug in the first attempt (which shrank a_hat's represented RANGE instead of its resolution, causing catastrophic clipping) turned up the real constraint once fixed: refreshing changes WHERE a grid sits, never how many levels it has, and levels are what is short. Data-grounded across all 70 real calibrated layers ([bits_needed_70layers.json](int8_ahat_cache_2026-08-26/data/bits_needed_70layers.json)): median layer needs **11.6 bits** to span a_hat's range at the tail's resolution; only **2 of 70 (2.9%)** fit in 8. This is a bit-count shortfall, not a rounding-tolerance one — no format tested closes an ~11.6-vs-8-bit gap. Further progress needs an actual FID measurement of the now-quantified ~12× resolution loss, not more proxy math. Untried: per-layer gating on `step_gain_tail` (capped by how few layers qualify) and non-uniform/companding quantization (needs real a_hat value histograms to size) |

| **C16** **[new 2026-08-26]** | warp-granularity zero-code skip on a_hat's write | exact (q==0 => no-op), zero numerics risk if viable | **MEASURED, REFUTED — real data, no kernel written.** Hooked `group_norm_silu_delta_quantize_nhwc`/`_pack_nhwc` at the Python level and captured every delta code from a real 20-step int8 generation ([ahat_zero_skip_2026-08-26](ahat_zero_skip_2026-08-26/FINDINGS.md)). **52.98% of individual codes are zero, but only 1.70% of 64-element warps (the apply kernel's vec2 grid unit) have EVERY code zero.** Zeros are scattered, not clustered — a predicated skip would fire on 1.7% of warps and pay a `__all_sync`-class check on the other 98.3%, a net loss on any reasonable cost model. (Caveat: this run had no delta calibration file, so it measured the full-activation-scale fallback, not the shipped per-step table — likely why the early/late gap here is far smaller than C15's calibrated 12–125× `step_gain_tail`. The scatter-vs-cluster structure is a property of activation correlation, not scale choice, so unlikely to reverse under calibration, but the exact 53%/1.7% numbers are for this regime specifically) | **closed, negative.** No further action — the measurement was the entire cost of ruling this out |

| **C17** **[new 2026-08-26]** | interleave a_hat (2B) + code (1B) into one struct, so a sector carries more useful bytes | assumed some fraction of today's 10 sectors/warp | **REFUTED BY ARITHMETIC, no kernel written.** The premise was backwards: today's SEPARATE arrays are already sector-optimal — 2 and 1 both divide the 32 B sector evenly, so a_hat's read (4 sectors/warp), a_hat's write (4) and code's write (2) are each an EXACT multiple with zero waste, 10 sectors/warp total. An unpadded 3 B/elem interleaved struct is WORSE: 3 does not divide 32, so every sector straddles element boundaries and every write needs a read-modify-write. A 4 B/elem padded struct avoids misalignment but moves **16 sectors/warp — +60% more bytes for the same information** (only 2 of 4 read bytes and 3 of 4 written bytes are useful). The apply kernel is measured bandwidth-bound at 84.5% of peak ([C12](../ahat_overlap_2026-08-26/FINDINGS.md)/ILP test in [gn_vec2_2026-08-26](../gn_vec2_2026-08-26/FINDINGS.md)), so moving 60% more bytes cannot be recovered by any locality effect at that regime | **closed, negative.** Homogeneous-width, separate arrays is the correct layout; heterogeneous-width interleaving can only add alignment or padding overhead, never remove it, when every component width already divides the sector size |

| **C18** **[new 2026-08-26]** | contiguous vec4 load/store in the apply kernel | measured 0.045 ms/step, freq-weighted | **MEASURED, POSITIVE, DELIBERATELY NOT LANDED.** The earlier "ILP" test (C18's own U∈{1,2,4,8} check in [gn_vec2_2026-08-26](gn_vec2_2026-08-26/FINDINGS.md)) used `idx[u]=tid+u*stride` — grid-strided, not adjacent — so it never tested a genuinely wider CONTIGUOUS instruction. This does: one 64-bit load/store over 4 adjacent elements instead of vec2's 32-bit, restricted to shapes where `CPG=C/32` is a multiple of 4 (C ∈ {384,768,1152,1536}; C=192/576 are unsafe and need a vec2 fallback not built here). Bit-identical to vec2 on every checked shape. **Freq-weighted over the 14 eligible real shapes (50 of ~62 calls/step): 4.662 → 4.617 ms/step, −1.0%, saving 0.045 ms/step (≈0.06% of the 77 ms step).** Clean pattern: small/high-frequency shapes gain 15–32% (launch/occupancy-bound), large shapes gain nothing (already bandwidth-bound at the known 84.5%-of-peak ceiling, and they dominate the freq-weighting) — [ahat_vec4_apply_2026-08-26](ahat_vec4_apply_2026-08-26/FINDINGS.md) | **deferred at the project owner's request, not abandoned for a technical reason.** The probe and bit-identity check are complete and reusable; landing it needs a C=192/576 fallback plus the usual gate/MSE discipline, judged disproportionate to a 0.045 ms/step return |

| **C19** **[new 2026-08-26]** | skip a_hat's WRITE for K-1 of every K steps (a_hat stays full fp16, only the update CADENCE changes) | up to `(K-1)/K × 2.024` ms/step W8A8 (K=4 ≈ 1.52 ms/step, reusing the already-measured [C12](../ahat_overlap_2026-08-26/FINDINGS.md) write-elision kernel) | **REFUTED, more decisively than the int8/fp8 storage line (C15).** Freezing a_hat for K-1 steps means each step's delta is measured against an increasingly stale reference, widening across the window — the same mechanism [C7](../OPEN_ITEMS.md) already measured to cost +13.08% FID in a related context (fewer warm-up rounds at t=T). A direct simulation using the real step0/tail delta-scale trajectory ([int8_ahat_cache_2026-08-26](int8_ahat_cache_2026-08-26/FINDINGS.md), 2nd follow-up) found: **K=2 (the mildest possible skip) raises the code clip rate from 0% to 7.5–20%; K=4 (the cadence already precedented by `MODIFF_DELTA_REFRESH`) clips 82.5–85.5% of all 200 steps.** Two bugs found and fixed in this sim, in order: (1) a constant-variance synthetic trajectory gave a false 32.5% K=1 baseline against the real calibration file's own `obs_clipped_frac: 0.0`; (2) after fixing that, a step-by-step trace exposed the deeper issue — **the scheme is structurally inconsistent, not just numerically risky.** Every code in a window is computed against the SAME frozen a_hat anchor, so K codes are K independent "distance from the anchor" re-measurements, not increments; `o_hat += conv(code)` sums them as if they were independent increments, overshooting the true trajectory by 3-4x within a single window before any rounding is even considered. Keeping `o_hat`'s accumulation correct requires each code to be incremental relative to the immediately preceding true value, which requires tracking that value somewhere every step — exactly what a_hat's per-step write already provides. Deferring the write removes the one piece of state the scheme needs; any substitute buffer would cost the same bandwidth a_hat already does | **closed, negative — refuted at the design level, before the numerics question is even reached.** Not recommended for an FID follow-up, and no lower-effort variant survives: the mechanism the project owner correctly guessed would be "more conservative" than skipping the whole conv turns out not to be realizable at all without paying for a same-sized substitute buffer |

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

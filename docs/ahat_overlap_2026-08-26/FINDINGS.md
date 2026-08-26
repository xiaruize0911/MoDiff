# Can a_hat be overlapped after all? The GN side says yes; the conv side says no

**Date** 2026-08-26 · **GPU** NVIDIA A40 (GA102, 84 SMs, 696 GB/s) · **Batch** 128 · **Delta mode** static (the shipped default, `MODIFF_DELTA_MODE`)

[perf_report_2026-08-26](../perf_report_2026-08-26/REPORT.md) closed batch-split 2-stream
pipelining as a route to hiding a_hat ([OPEN_ITEMS C11](../OPEN_ITEMS.md)) — inter-kernel
concurrency, refuted because the conv holds every SM. This is the other reading of "overlap":
**intra-kernel**, putting a_hat's traffic on SMs the conv already owns, which is the mechanism
that makes o_hat cheap. It is refuted too, and the measurement says why.

| Question | Answer | Confidence |
|---|---|---|
| 1. What does the a_hat WRITE cost the GN kernel? | **2.024 ms/step** (W8A8) / **1.742** (W4A4) — 2.63% / 2.55% of a step, and 57% of a_hat's entire measured cost | high — reproduces the committed ablation to 0.2–0.9% |
| 2. Does the register budget survive an a_hat RMW inside the conv? | **Yes.** 240 regs, 0 stack, 0 local — unchanged. The flagged risk did not materialise | high — `cuobjdump -res-usage` |
| 3. Does the conv absorb that traffic cheaply? | **No.** It charges **5.792 ms/step** for what the GN gives back at 1.926. Net **-3.866 ms/step = -5.02%** of a step | high — 5 trials, order rotated, sd < 3% |
| 4. Is o_hat cheap because it lives inside the conv? | **No.** Placement before the mainloop vs after the epilogue differs by < 1%. o_hat is cheap for a different reason | high — direct A/B |

---

## 1. The GN side: eliding the a_hat write is worth 31% of the apply kernel

Instrument: [`probe.cu`](scripts/probe.cu), a verbatim copy of
[`gn_apply_delta_quantize_flat_vec2_kernel`](../../csrc/modiff/norm/group_norm_silu.cu:1701)
with the a_hat store and the code store behind template flags, and `gn_report_delta_absmax`
dropped (production passes `absmax_buf = nullptr` in static mode and the helper's first
statement is `if (absmax_buf == nullptr) return;`). Launch geometry is production's: block 256,
`grid = ceil(numel/2/256)`, 256-float dynamic shared kept so occupancy is identical.

**Validity.** The probe also calls the shipped path. Against the committed
[conv-block ablation](../conv_block_ablation_2026-08-26/data/combined_w8a8_w4a4.csv):

| shape | freq | probe `prod` W8A8 | ablation | Δ | probe `prod` W4A4 | ablation | Δ |
|---|--:|--:|--:|--:|--:|--:|--:|
| `192,32x32` | 7 | 0.4238 | 0.4272 | -0.80% | 0.4081 | 0.4088 | -0.17% |
| `384,16x16` | 7 | 0.2258 | 0.2270 | -0.52% | 0.2158 | 0.2174 | -0.73% |
| `384,32x32` | 2 | 0.8238 | 0.8259 | -0.26% | 0.7898 | 0.7932 | -0.43% |
| `576,32x32` | 1 | 1.2753 | 1.2809 | -0.44% | 1.2262 | 1.2267 | -0.04% |
| `768,16x16` | 2 | 0.4438 | 0.4448 | -0.24% | 0.4237 | 0.4277 | -0.92% |
| `768,2x2` | 12 | 0.0180 | 0.0214 | -15.85% | 0.0188 | 0.0190 | -0.93% |
| `384,8x8` | 8 | 0.0752 | 0.0762 | -1.27% | 0.0732 | 0.0729 | +0.39% |

The five dominant shapes agree to 0.2–0.9%. `768,2x2` is 0.018 ms and launch-noise dominated;
it is carried for completeness, not for its number.

**Result.** `w1c1` = today (read x, read a_hat, write a_hat, write code). `w0c1` = the write elided.

| shape | freq | W8A8 w1c1 | w0c1 | saved | % of apply | W4A4 w1c1 | w0c1 | saved | % of apply |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `192,32x32` | 7 | 0.3118 | 0.2130 | 0.0988 | **31.7%** | 0.2882 | 0.2023 | 0.0858 | **29.8%** |
| `384,16x16` | 7 | 0.1578 | 0.1086 | 0.0492 | **31.2%** | 0.1458 | 0.1016 | 0.0442 | **30.3%** |
| `384,32x32` | 2 | 0.6216 | 0.4227 | 0.1989 | **32.0%** | 0.5737 | 0.4076 | 0.1660 | **28.9%** |
| `576,32x32` | 1 | 0.9320 | 0.6345 | 0.2975 | **31.9%** | 0.8630 | 0.6105 | 0.2524 | **29.3%** |
| `768,16x16` | 2 | 0.3119 | 0.2145 | 0.0973 | **31.2%** | 0.2887 | 0.2017 | 0.0871 | **30.1%** |
| `768,2x2` | 12 | 0.0061 | 0.0058 | 0.0003 | **5.4%** | 0.0059 | 0.0057 | 0.0001 | **2.4%** |
| `384,8x8` | 8 | 0.0418 | 0.0302 | 0.0117 | **27.9%** | 0.0378 | 0.0288 | 0.0090 | **23.7%** |
| **freq-weighted** | | | | **2.024 ms/step** | | | | **1.742 ms/step** | |

As a share of the steady step: **2.63%** (W8A8, 77.00 ms) and **2.55%** (W4A4, 68.27 ms).

**It beats its own byte model, and that is informative.** The apply kernel moves 7 B/elem with
the write and 5 without, so a pure-bandwidth prediction is 2/7 = 28.6%. Measured 29–32%, and the
achieved bandwidth *rises*:

| shape | W8A8 GB/s w1c1 → w0c1 | % of peak |
|---|--:|--:|
| `192,32x32` | 565 → 591 | 81% → 85% |
| `384,16x16` | 558 → 579 | 80% → 83% |
| `384,32x32` | 567 → 595 | 81% → 86% |
| `576,32x32` | 567 → 595 | 81% → 85% |
| `768,16x16` | 565 → 586 | 81% → 84% |
| `768,2x2` | 451 → 340 | 65% → 49% |
| `384,8x8` | 526 → 521 | 76% → 75% |

The surplus is the write-allocate traffic the store drags in on top of its own 2 B.

> **This number is the ceiling for any scheme that removes the a_hat write**, not just for
> moving it into the conv — an int8/fixed-point a_hat cache inherits the same numerator.

---

## 2. The register risk did not materialise, and the arithmetic is exact

The conv-side arm adds a CTA-partitioned `a_hat += code/scale` inside
`ImplicitGemmConvolutionEVT::operator()` ([the patch](scripts/conv_ahat_rmw.patch)). The
partition is over the flat tensor and independent of the tile the CTA computes, so **each
element is visited exactly once** — no ownership predicate, and none of the R×S multiple-visit
problem that an iterator-level fusion would have. It runs before the swizzle bounds check so
CTAs that return early still take a share.

| check | result |
|---|---|
| registers (`cuobjdump -res-usage`) | **REG:240 STACK:0 LOCAL:0** — identical to the shipped kernel |
| a_hat vs an fp32-accumulate / `__float2half_rn` reference | **0 ULP** |
| o_hat vs the shipped `conv2d_int8_evt_o_hat` | **bit-identical** |
| negative control (scale off by 1%) | **37391 ULP** — the gate fires |

240 of the SM's 65,536 registers per thread × 256 threads = 61,440, i.e. the same one-block-per-SM
occupancy the perf report documents. The design failed for a different reason than the one flagged.

---

## 3. The conv side: it charges 3x what the GN gives back

| shape | freq | conv | conv+a_hat (pre-mainloop) | (post-epilogue) | charge | |
|---|--:|--:|--:|--:|--:|--:|
| `192->192,32x32` | 7 | 0.7310 | 1.0394 | 1.0455 | +0.3084 ms | **+42.2%** |
| `384->384,16x16` | 7 | 0.4273 | 0.5859 | 0.5935 | +0.1585 ms | **+37.1%** |
| `384->192,32x32` | 2 | 1.0530 | 1.6191 | 1.6119 | +0.5589 ms | **+53.1%** |
| `576->192,32x32` | 1 | 1.6630 | 2.4713 | 2.4507 | +0.7876 ms | **+47.4%** |
| `768->384,16x16` | 2 | 0.7875 | 1.0990 | 1.0968 | +0.3093 ms | **+39.3%** |

```
GN gives back : +1.926 ms/step
conv charges  : +5.792 ms/step
NET           : -3.866 ms/step = -5.02% of the W8A8 step
```

**Placement is not the variable.** Before the mainloop and after the epilogue differ by under 1%
on every shape. So the hypothesis this experiment was built on — *o_hat is cheap because it sits
inside a compute-bound kernel* — is directly refuted.

**The mechanism.** Compare how fast each kernel moves the bytes in question:

| shape | conv, added bytes | GN, the same bytes given back |
|---|--:|--:|
| `192->192,32x32` | 408 GB/s (**59%** of peak) | 509 GB/s (73%) |
| `384->384,16x16` | 397 GB/s (**57%** of peak) | 511 GB/s (73%) |
| `384->192,32x32` | 450 GB/s (**65%** of peak) | 506 GB/s (73%) |
| `576->192,32x32` | 479 GB/s (**69%** of peak) | 508 GB/s (73%) |
| `768->384,16x16` | 407 GB/s (**58%** of peak) | 517 GB/s (74%) |

**The conv is a worse place to move bytes than the GN kernel, not a better one.** Its 23–25% of
peak bandwidth (measured in the perf report's §4) is not headroom that can be claimed: adding a
low-MLP streaming loop to a kernel that holds every SM runs it at 57–69% of peak, against the GN
kernel's 73–81%.

### Why o_hat really is cheap — a correction to perf_report §2

The perf report prices o_hat's incremental bytes at **2.35× / 4.06× cheaper** than a_hat's and
attributes it to intra-kernel latency hiding. That attribution is wrong, and this experiment is
what shows it: putting a_hat's bytes in the same kernel, in either position, buys nothing.

o_hat is cheap because **its store replaces a store the baseline already performs**, and its load
rides the write-allocate that store needs anyway — the cache line is fetched to be written
regardless. Its true incremental DRAM transaction count is near zero. a_hat has no such twin: it
is a separate tensor no other kernel touches, and its 4 B/elem are irreducible DRAM traffic.
**The o_hat per-byte advantage is a property of o_hat, not of the conv, and does not transfer.**

### The remaining variant, priced

Fusing into the activation iterator instead of a separate CTA-partitioned loop would avoid
re-reading the codes, taking the added traffic from 5 B/elem to 4 — about a 20% reduction, so a
charge near 4.6 ms/step against the 1.926 available. Still deeply negative, and it
costs the exactness the CTA partition gets for free (ownership predication plus halo handling).
Not worth building.

---

## What this leaves standing

1. **"Overlap a_hat" is now closed on both readings** — inter-kernel (C11) and intra-kernel
   (this doc), each with a measured mechanism rather than an argument.
2. **The 2.024 / 1.742 ms/step ceiling is the durable result.** It bounds every scheme
   that removes the a_hat *write*, including an int8/fixed-point a_hat cache — that idea now has
   a measured numerator without anyone building it.
3. **Selective per-layer MoDiff is not bounded by it**, because dropping MoDiff on a layer
   removes the a_hat read, the a_hat write and o_hat together. On the five dominant shapes that
   is the full 3.551 ms/step (W8A8), not the write's share.

## Scope and limitations

- **The probe ran with `mod_scale` / `smooth_inv` null.** Both add per-element ALU but no
  tensor-sized traffic, and both arms carry them equally, so the *difference* should be
  unaffected — untested.
- **Five of twenty shapes**, the ones carrying 63% of MoDiff's conv-block overhead. `768,2x2`
  and `384,8x8` are carried as references and behave differently (launch-bound, 63–78% of peak).
- **The conv-side arm is not in the tree.** It was reverted after measurement at the owner's
  request; [`conv_ahat_rmw.patch`](scripts/conv_ahat_rmw.patch) reproduces it against the commit
  this doc lands on. The GN-side probe is standalone and needs no patch.
- **No `ncu`** (`ERR_NVGPUCTRPERM`), so the bandwidth figures are derived from measured
  durations and analytic byte counts, as in the perf report.

## What is not generated by [`make_findings.py`](scripts/make_findings.py)

The register counts (`cuobjdump`), the four rows of the numerics gate (ULP / bit-identity /
negative control), the byte-per-element models (read from kernel source), and the inline
derivations: 2/7 = 28.6%, the 61,440 register product, the 20% iterator-fusion estimate, and
a_hat's 3.551 ms/step on these five shapes (from the ablation CSV).


# Two bit-identical vectorisations: GroupNorm stats and the resize path's a_hat, -1.29 ms/step

**Date** 2026-08-26 · **GPU** NVIDIA A40 · **Batch** 128 · **Arm** W8A8 + MoDiff, static delta

Both changes are pure memory-access fixes inside the conv block. Neither moves a value or the
order values are added in, and both are gated on **bit-identity** rather than a tolerance,
because mean/inv_std feed the delta quantizer whose a_hat cache accumulates across 200 steps —
one ULP and the two arms are different functions, not one function measured twice.

| | before | after | delta |
|---|--:|--:|--:|
| GN stats (`gn_stats_partials_chanmajor` → `_vec2`) | 3.435 | 2.665 | **-0.770** |
| GN+resize+delta (`..._resize_nhwc_kernel`, a_hat vec2) | 3.184 | 2.713 | **-0.471** |
| GN apply (unchanged — see §3) | 7.791 | 7.778 | -0.013 |
| stats reduce (unchanged) | 0.341 | 0.330 | -0.010 |
| **GN family total** | **14.751** | **13.487** | **-1.264** (-8.6%) |
| GPU busy, whole step | 75.964 | 74.512 | -1.452 |

Attributed conservatively: the two targeted kernels account for **-1.241 ms/step**
= **1.61% of the 77.00 ms steady step**. The larger GPU-busy delta
includes a few hundredths spread across untouched kernels, which is run-to-run drift, not effect.

---

## 1. GroupNorm statistics: 49.8% → 71.6% of peak bandwidth

`gn_stats_partials_chanmajor_kernel` was **latency-bound, not bandwidth-bound** — nsys put it at
49.8% of peak against the apply kernel's 84.5% on the same tensors. Two causes:

1. one 2-byte load per thread per `hw` iteration, so a warp requests 64 B — half a sector;
2. the `hw` loop is not unrolled and the accumulate is serial, so exactly one load per thread is
   in flight.

Fix: a thread owns **two adjacent channels** loaded as one `__half2`, and the `hw` loop processes
four positions with the loads hoisted ahead of the first dependent add.

**Why it is bit-identical.** Each channel keeps its own fp32 accumulator; that accumulator still
walks `hw` ascending (unrolling reorders no adds, it only issues loads earlier); shared memory is
still indexed by channel and the group combine still sums CPG channels in ascending index. Only
the address and the load instruction change.

Microbenchmark over all 18 real GN shapes, batch 128, freq-weighted:

| variant | ms/step |
|---|--:|
| v0 shipped | 3.305 |
| v1 hw-unroll | 2.821 |
| v2 vec2-chan | 2.698 |
| v3 both | 2.603 |

Per shape (all bit-identical to v0):

| shape | freq | v0 | v3 | speedup | % peak v0 → v3 |
|---|--:|--:|--:|--:|--:|
| `768,2x2` | 12 | 0.0071 | 0.0052 | 1.37× | 16% → 22% |
| `384,8x8` | 8 | 0.0240 | 0.0166 | 1.45× | 38% → 55% |
| `384,16x16` | 7 | 0.0582 | 0.0464 | 1.25× | 62% → 78% |
| `768,4x4` | 7 | 0.0204 | 0.0126 | 1.62× | 22% → 36% |
| `192,32x32` | 7 | 0.1091 | 0.0917 | 1.19× | 66% → 79% |
| `1536,2x2` | 3 | 0.0134 | 0.0137 | 0.98× | 17% → 17% |
| `768,8x8` | 2 | 0.0545 | 0.0320 | 1.70× | 33% → 56% |
| `1536,4x4` | 2 | 0.0455 | 0.0456 | 1.00× | 20% → 20% |
| `384,32x32` | 2 | 0.2023 | 0.1739 | 1.16× | 71% → 83% |
| `768,16x16` | 2 | 0.1211 | 0.0874 | 1.39× | 60% → 83% |
| `384,4x4` | 1 | 0.0086 | 0.0063 | 1.35× | 26% → 36% |
| `1152,8x8` | 1 | 0.0531 | 0.0522 | 1.02× | 51% → 52% |
| `1152,4x4` | 1 | 0.0188 | 0.0192 | 0.98× | 36% → 35% |
| `192,16x16` | 1 | 0.0329 | 0.0264 | 1.25× | 55% → 69% |
| `576,32x32` | 1 | 0.3519 | 0.2583 | 1.36× | 62% → 84% |
| `768,8x8` | 1 | 0.0544 | 0.0320 | 1.70× | 33% → 57% |
| `576,16x16` | 1 | 0.1051 | 0.0670 | 1.57× | 52% → 81% |
| `384,32x32` | 1 | 0.2020 | 0.1735 | 1.16× | 72% → 83% |

It also removes the need for the `K > 1` channel split on this UNet: the block is `C/2` threads,
so every `C <= 2048` fits under the 1024-thread cap where the scalar kernel needed `K = 2` from
C = 1152 up.

Gate: [`test_gn_stats_vec2.py`](../../integration/tests/test_gn_stats_vec2.py) — 16 shapes,
driven through the public `group_norm_silu_delta_quantize_nhwc` entry point with
`MODIFF_GN_STATS_VEC2` on and off in **forked processes** (the flag is a function-local static,
so an in-process A/B silently measures one variant twice). It asserts three things, not one:
codes and a_hat bit-identical, codes not all zero, **and the profiler saw two different stats
kernels** — without that last check the test would pass just as happily if the flag were never
read, which is the failure mode OPEN_ITEMS A22 records.

## 2. The resize path's a_hat: two scalar accesses → one vec2

`group_norm_silu_delta_quantize_resize_nhwc_kernel` is the second-largest GN kernel and it also
carries a_hat. `compute_pair` already vectorised x, gamma, beta and the modulation, but the a_hat
read-modify-write did not: two scalar `__half` loads at `ci` and `ci + 1`, two scalar
`__float2half_rn` stores. Both the UP and DOWN paths, now one `gn_load2` / `gn_store2` each.

Bit-identity is structural — the same two 16-bit values, and `__float22half2_rn` rounds each
component to nearest even exactly as the two `__float2half_rn` calls did. The one real
precondition is that `ci` be even, which holds because `c_start = g·CPG` with CPG even; that is
now a `TORCH_CHECK` in the launcher rather than a comment, since a misaligned `reinterpret_cast`
is undefined behaviour that can appear to work.

In-model: **3.184 → 2.713 ms/step**.

Gate: [`test_gn_resize_ahat_vec2.py`](../../integration/tests/test_gn_resize_ahat_vec2.py), a
two-build capture/compare over 4 shapes × {upsample, downsample} × {int8, packed int4} ×
{mod, no mod} = 32 cases. All 32 bit-identical, 198,870,120 nonzero codes for non-vacuity.

### The control that mattered

The first attempt at this gate compared **end-to-end images** between the two builds. They
differed — and that is meaningless: two runs of the *same* build also differ. Measured here,
HEAD against itself over 64 images gave `sha256 4f15fc19…` and `44dbe9b2…`. That is
[OPEN_ITEMS A18/A19](../OPEN_ITEMS.md) (cross-process floor 8.7/255) showing up as a
would-be false alarm. Without the self-comparison it would have read as *the change broke
correctness*. Kernel-level capture under a fixed seed is the instrument that resolves.

## 3. The apply kernel is already at the roof — measured, not assumed

`gn_apply_delta_quantize_flat_vec2_kernel`, where a_hat's read-modify-write lives, is 84.5% of
peak. Its grid covers every element in one pass, so there is no loop to unroll; the equivalent
lever is to shrink the grid and give each thread U independent pairs. Elements are independent,
so any assignment is bit-identical by construction (asserted for every U).

| U | ms/step, freq-weighted over 18 shapes |
|---|--:|
| 1 (as shipped) | 8.090 |
| 2 | 8.199 |
| 4 | 8.459 |
| 8 | 8.733 |

**Negative: U = 1 as shipped is best.** Only the tiny shapes gain (`768,2x2` 1.07×, `1536,2x2`
1.06×) and together they are worth about 0.009 ms/step — not worth a dispatch. The kernel is
genuinely bandwidth-bound, which is consistent with
[ahat_overlap_2026-08-26](../ahat_overlap_2026-08-26/FINDINGS.md): a_hat's 4 B/elem are
irreducible without changing what is stored.

## 4. Two honest nulls from finishing the job

**The decoder skip-concat fold was still on the scalar kernel** — 14.2 of 58.9 stats calls per
step in the W4A4 arm (the W8A8 arm does not take that path). It was extended to vec2 too, with
the boundary argument that a pair never straddles C1 because every C1 here is a multiple of 32,
and it is now covered by phase 2 of the gate. **The gain is 0.049 ms/step: neutral.** Recorded
because the conversion is real, correct and measured — not because it bought anything.

**Unexplained, and orthogonal to this change: the same stats kernel costs 36% more in the W4A4
arm than in W8A8** — 3.624 vs 2.665 ms/step, on identical shapes with
identical call counts (19.0 at C=384, 22.8 at C=768, 8.5 at C=192, …, verified per block size in
the trace). The GN input is fp16 in both arms, so the kernel does the same work on the same bytes.
It predates this change — the pre-cat2 W4A4 trace shows the same gap — so it is not a regression,
but it is 0.96 ms/step sitting in the W4A4 arm with no explanation, and worth its own look.

## Scope and limitations

- **The cat2 / skip-concat variant of the stats kernel is untouched.** It also writes a
  concatenation, and folding a vectorised store into it is a separate change with its own gate.
- **Odd channels-per-group** falls back to the scalar stats kernel and is rejected outright by
  the resize launcher. Every C in this UNet is a multiple of 32 with G = 32, so CPG is even.
- **C1 + C2 > 2048 declines vec2** (block would exceed 1024 threads) and correctly falls back to
  the scalar K-split kernel; the 1536+768 fold is the one such shape and the gate asserts the
  fallback rather than treating it as a pass by accident.
- **In-model numbers come from one nsys trace per arm**, windowed to the second sample batch
  (20 DDIM steps). Kernel durations are CUPTI activity records and are not inflated by the
  profiler; wall time is, which is why GPU-busy rather than wall is quoted throughout.
- **`ncu` is unavailable** (`ERR_NVGPUCTRPERM`), so `% of peak` is derived from measured
  durations and analytic byte counts, as everywhere else in this project.


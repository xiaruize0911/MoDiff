# Truncated-SVD a_hat storage: adaptive basis still breaks the accumulator pair; the residual does not fit in 8 bits

**Status: measured, refuted. No CUDA code written.** Prompted by storing only the
largest singular components of `a_hat` (and a cheap residual of that
approximation). This is the first `a_hat` idea that is *not* a fixed operator
(int8 grid, `avg_pool`, skip-K). It still dies on the same two facts that
closed C15 and C20: the accumulator-pair invariant, and the bit budget of
whatever you actually store.

W8A8 · LDM-8 LSUN-Churches · A40 · batch 2 capture, 30 DDIM steps, seed 777.
Layers: `input_blocks.1.0.in_conv` (192×32×32), `input_blocks.4.0.in_conv`
(192×16×16, the C20 layer), `input_blocks.4.0.out_conv` (384×16×16).
Protocol cloned from
[`ahat_ds_recursion.py`](../ahat_downsample_2026-08-27/scripts/ahat_ds_recursion.py):
hook `forward_gn_fused_modiff`, capture `silu(gn(x))` + the live delta scale,
replay offline.

## The three encodings

Each sample treats `a_hat` as a `C × HW` matrix. `A_k = U_k Σ_k V_k^T`.

| id | what is stored | what `o_hat` accumulates | prior |
|---|---|---|---|
| **S1** | `SVD_k(a_hat)` only | `conv(dequant(code))` | C20 with an adaptive projector |
| **S2** | rank-k factors; encode only the projected delta | `conv(inc_k)` (pair stays in the subspace) | the only scheme that is invariant-safe *a priori* |
| **S3** | `A_k + R` | either codes (C20-style) or `Δrecon` (pair-sync) | C15, but quantize the SVD residual instead of `a_hat` itself |

k ∈ {4, 8, 16, 32}. S3 residual variants: fp16, dynamic int8, drop, keep-largest-10%.

## 1. The spectrum does not license a small k

Median truncated energy `Σ_{i≤k} σ_i² / Σ σ²` over 30 steps, `C × HW` SVD:

| layer | tensor | k=4 | k=8 | k=16 | k=32 |
|---|---|--:|--:|--:|--:|
| 192×32×32 | `a_hat` | 0.34 | 0.51 | **0.69** | 0.87 |
| 192×32×32 | delta | 0.35 | 0.49 | 0.67 | 0.81 |
| 192×16×16 (C20) | `a_hat` | 0.35 | 0.48 | **0.63** | 0.78 |
| 192×16×16 (C20) | delta | 0.28 | 0.38 | 0.52 | 0.69 |
| 384×16×16 | `a_hat` | 0.37 | 0.49 | **0.62** | 0.76 |
| 384×16×16 | delta | 0.29 | 0.40 | 0.54 | 0.70 |

Channel-PCA of `a_hat` is *worse*, not better (C20 layer k=16: **0.53**).

**S2's kill criterion was ~95% at k=16.** The median layer is at 63%. Delta is
not a low-rank shortcut either — it is *less* concentrated than `a_hat`. The
orthogonal complement is most of the tensor, every step.

## 2. Recursion: S1 is C20 with a fancier coarsener

Invariant residual `max |stored_recon − (a_hat_0 + Σ increment)|` — what
`conv(a_hat) − o_hat` sees. Production stays at fp16 noise. Downsample f=2 on
this same C20 layer grew to **44 by step 30**.

C20 layer, k=8 unless noted:

| step | prod | **S1** | S2 k=16 | S3 int8, codes | S3 int8, Δrecon | S3 drop R |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 0.001 | 0.001 | 0 | 0.007 | 0 | 1.64 |
| 2 | 0.002 | **2.06** | 0.0009 | 0.013 | 0 | 3.22 |
| 5 | 0.003 | **8.17** | 0.002 | 0.029 | 0 | 7.71 |
| 15 | 0.006 | **15.7** | 0.003 | 0.073 | 0 | 14.2 |
| 30 | **0.008** | **24.7** | 0.004 | **0.109** | **0** | 23.6 |

S1 grows ~+2 per step for the first six steps, then ~+0.6. Linear, same sign,
same mechanism as `avg_pool`: the delta re-encodes the discarded tail, `o_hat`
keeps it, the next SVD throws it away again. An adaptive basis does **not**
make the operator unbiased. 192×32×32 is slower (final S1 k=8 = 3.58) but
still 700× production.

S2's bookkeeping is clean (resid 0.004, fp16). The reconstruction of the
activation is not: `|target − o_hat's view|` ends at **3.94 vs production 0.41**.
That is the 37–48% of energy S2 refused to encode, sitting in the conv output.
Not a storage format — it is a different, much worse, layer.

S3 with fp16 `R` and `o_hat ← Δrecon` is bit-identical to production on the
invariant (resid 0) and matches `|target−acc|`. It also stores a dense fp16
residual, so it saves nothing.

## 3. S3's only interesting window: int8 / sparse `R`

C15 needed **11.6 bits** (median layer) to span `a_hat`'s range at the tail
quantum. The hope was that `R = a_hat − A_k` would be small enough for 8 bits.

Bits needed = `log2(2 · range / quantum)`, same formula as C15, on the live
captured `a_hat` and the live per-step delta scale:

| layer | `a_hat` bits (median) | R k=8 | R k=16 | R k=32 |
|---|--:|--:|--:|--:|
| 192×32×32 | 13.04 | 13.02 | 12.67 | 12.42 |
| 192×16×16 | 10.62 | 10.05 | **9.62** | 9.13 |
| 384×16×16 | 10.48 | 9.87 | 9.56 | 8.79 |

**No k we tried puts R in 8 bits.** SVD is L2-optimal; the bit budget is a
range question. The orthogonal complement still has outliers of size ~1.4–2.5
against a tail quantum of ~10⁻³, so dropping 30–40% of the energy barely moves
`max|R|`. k=16 on the C20 layer saves **one bit** versus storing `a_hat` raw
(9.62 vs 10.62) — the same order of shortfall C15 already measured.

Two S3 int8 disciplines:

- **`o_hat` follows codes** (the real kernel): resid climbs 0.007 → 0.109 in 30
  steps, ~14× production. Slower than S1, still a running sum of storage
  snaps. C15's int8 `a_hat` drifted 0.05–0.15 in a 1-D proxy and landed FID
  182. This is that number.
- **`o_hat` follows `Δrecon`** (the C20 rescue): resid = 0 by construction,
  `|target−acc|` matches production at 30 steps. Restoring the pair this way
  is `conv(e_t)` under another name. C20 already priced that as a full extra
  conv per layer per step. It also still materializes a dense `C×HW`
  reconstruction to subtract from `silu(gn(x))`, so the apply kernel's 8/10
  `a_hat` sectors do not disappear.

Sparse-10% `R` + codes on the C20 layer: resid **9.9 at step 30**, same linear
class as dropping `R` entirely (23.6). Scattered outliers, same finding as
[`ahat_zero_skip`](../ahat_zero_skip_2026-08-26/FINDINGS.md).

## 4. Even a free SVD would miss the 2 ms ceiling

A40, batch 128, CUDA event around `torch.linalg.svd` / `svd_lowrank` /
`eigh(C×C)`:

| shape | full SVD | low-rank k=16, 128 samples (extrap.) | channel `eigh` | a_hat-write ceiling |
|---|--:|--:|--:|--:|
| 192×32×32 | **840 ms** | 160 ms | 3.9 ms | 2.0 ms |
| 192×16×16 | 788 ms | 161 ms | 4.1 ms | 2.0 ms |
| 384×16×16 | 1147 ms | 152 ms | 10.9 ms | 2.0 ms |

One layer's SVD is two orders of magnitude above the traffic it would save.
Channel PCA is the only cheap-looking cousin and is still ≥2× the ceiling
*per layer*, on a spectrum that is worse than spatial SVD.

A scheme that decoded `A_k` back to dense `C×HW` before the delta subtract
would also keep those 8 sectors. The only bandwidth win is to quantize in
the factor space (S2). S2 is the one the energy plot already killed.

## Verdict

| scheme | invariant | quality proxy | bits / time | ship? |
|---|---|---|---|---|
| S1 SVD cache | linear divergence, 24.7 vs 0.008 at step 30 | `|tgt−acc|` tracks the residual | n/a | **no — C20, adaptive basis** |
| S2 subspace MoDiff | holds | k=16 keeps 63% of energy; `|tgt−acc|` 3.9 vs 0.41 | SVD ≫ 2 ms | **no — energy** |
| S3 fp16 `R` | holds if `o_hat←Δrecon` | = production | 0 bytes saved | **no — tautology** |
| S3 int8 `R`, codes | slow climb to 0.11 | same hole as C15 | R needs 9.6–12.7 bits | **no** |
| S3 int8 `R`, Δrecon | holds | 30-step `|tgt−acc|` looks like prod | extra conv + dense decode; still >8 bits | **no — C20 rescue, still over budget** |
| S3 sparse-10% | linear | resid 9.9 | zeros not clustered | **no** |

C20's principle survives contact with an adaptive projector:

> `a_hat` is not a free-standing cache. Any lossy transform desynchronises
> the pair, and the error accumulates rather than cancelling.

The residual-of-SVD idea does not open a new bit-budget hole: `max|R|` is not
small. **Do not ship a_hat as factors.** The compute reading of the same algebra
is a different question — §5.

## 5. Follow-up: save `conv(delta)` FLOPs, not a_hat DRAM

The residual that costs money is `o_hat += conv(dequant(code))` (~38 ms of the
74 ms step), not `a − a_hat`. Linearity still gives

```
delta ≈ U_k Z    ⇒    conv(delta) = (W U_k) ∗ Z
```

i.e. an INT8 conv with `Cin = k` and weights folded once per step. That is
independent of how `a_hat` is stored. Pair stays in sync if `o_hat` only ever
adds `conv(projected delta)` — same bookkeeping as S2, but k is chosen for
the **conv**, not for DRAM.

### What k is even allowed

k for a given energy of the *delta* (`C×HW` SVD, median over 30 steps):

| layer | rank max | k@90% | k@95% | k/C at 95% |
|---|--:|--:|--:|--:|
| 192×32×32 | 192 | 58 | **89** | 0.46 |
| 192×16×16 | 192 | 79 | **104** | 0.54 |
| 384×16×16 | 256 | 84 | **121** | 0.32 |

k=16 (the storage experiment) is 52% of delta on the C20 layer. The conv-sized
k is ~C/2, not C/10. Channel-PCA needs still more. k@95% **rises** toward the
tail on 192×32×32 (53 → 99) — late steps are not cheaper.

### Frozen basis is still dead; per-step refresh is not

`|target − o_hat view|` after 30 steps, C20 layer, production = **0.414**:

| k | frozen U from step 1 | **refresh U from this step's delta** | top-k channels (no SVD) |
|--:|--:|--:|--:|
| 16 | 3.66 | 0.89 | 2.46 |
| 32 | 3.38 | 0.56 | 1.62 |
| **64** | 2.97 | **0.417** | 1.12 |
| 96 | 2.49 | 0.398 | 0.86 |

Refresh + k=64 matches production on this proxy (0.417 vs 0.414). Frozen U
does not, even at k=160. Coordinate top-k needs ~2× more channels than PCA.
The 384-ch layer is the same story (refresh k=64: 0.779 vs prod 0.750).

### CUTLASS actually moves at that k

PTQ INT8 3×3, batch 128, production autotune (Cin must be 16-aligned):

| shape | full Cin | k-Cin | ms full → k | wall-clock | FLOP ratio |
|---|--:|--:|---|--:|--:|
| 192→192 32×32 | 192 | 96 / 64 | 0.546 → 0.345 / 0.219 | **1.58× / 2.49×** | 0.50 / 0.33 |
| 192→192 16×16 | 192 | 96 / 64 | 0.146 → 0.096 / 0.066 | 1.53× / 2.21× | 0.50 / 0.33 |
| 384→384 16×16 | 384 | 128 / 64 | 0.502 → 0.177 / 0.110 | **2.84× / 4.55×** | 0.33 / 0.17 |

Not a FLOP fairy tale: 64-in on the 192×32 layer is 2.5× wall-clock. If the
~38 ms conv bucket moved ~2×, that is ~19 ms, A at ~55 ms — next to the 53 ms
2× line. That is an e2e **upper bound**, not a measurement: it assumes every
residual conv participates and ignores the basis tax.

### The basis tax

Per-step full SVD: 800 ms at batch 128. Dead.

Per-sample `svd_lowrank`: ~1 ms × 128. Dead.

**One U for the whole batch**, randomized range finder (`(C × N·HW) @ Ω` +
QR of `C×k` + fold `W U`):

| shape | GEMM+QR+fold |
|---|--:|
| 192×32×32 k=64 | **0.38 ms** |
| 384×16×16 k=64 | **0.36 ms** |
| 384×16×16 k=128 | 0.63 ms |

Per layer this is fine against a 0.2–0.5 ms conv. Across ~20 residual convs
it is 7–12 ms unless gated to the heavy shapes. Per-sample QR was 9–34 ms
and is not usable.

### What is actually live

Not a_hat storage. A **per-step, batch-shared rank-k residual conv**:

1. Cheap range-finder U of this step's delta (or of `silu(gn(x)) − a_hat`).
2. `Z = Uᵀ delta`, INT8 conv with folded `W U`, `o_hat +=` that.
3. `a_hat +=` the projected, dequantized increment (S2 bookkeeping).

k starts at **64** (aligned 64), quality-proxy-neutral on three layers.
Do not freeze U. Do not use channel top-k as a substitute. Do not SVD
inside the apply kernel.

The quality probe is §7. Folded `Cin=k` CUTLASS and the retain sweep are §8.

## 6. How to treat the SVD split of delta

Write `delta = delta_k + r` with `delta_k = U_k U_k^T delta` and `r ⊥ U_k`.
Three ways to feed that into the accumulator pair. Only the first two keep
`o_hat ≈ conv(a_hat)`.

```
d  = a - a_hat
U  = range_finder(d, k)          # one U for the batch, ~0.3 ms
Z  = U^T d                       # k × H × W
d_k = U Z
r  = d - d_k                     # dropped tail
```

**Drop `r` (default).** Quantize and conv only `Z`. Lift the dequantized
codes back with `U` when updating `a_hat`:

```
q    = Q(Z)                      # k-channel INT8, scale from absmax(Z)
a_hat += U @ dequant(q)
o_hat += conv_k(q;  W' = W U)    # Cin = k
```

`r` is not stored and not conv'd. It stays in `a - a_hat` and is the next
step's delta (ordinary MoDiff error feedback). This is the refresh-k=64
proxy that matched production `|tgt−acc|`. Do not also add `conv(r)` to
`o_hat` — that is the C20 extra-conv rescue.

**Do not** update `a_hat` with the full `d` while `o_hat` only gets
`conv(d_k)`. That desynchronises the pair on purpose.

**Do not** freeze `U` from t=T. Recompute it from this step's `d`.

Quantize in k-space, not on the lifted `d_k`. The static delta table was
calibrated on the full `d`; `Z` is smaller, so use absmax(`Z`) (or a new
table) or the codes sit on a grid that is too coarse.

k is 64, 16-aligned, gated to the heavy residual convs. The apply kernel
still writes a dense fp16 `a_hat` (`U @ dequant(q)` is a C-channel axpy,
cheap next to conv).

## 7. Live generation: k=64 holds; the first grid was a quantizer confound

Monkey-patch `forward_gn_fused_modiff` on the three TARGET layers. Python
GN+SiLU+smooth, `d = tgt − a_hat`, optional SVD project, **dynamic absmax**
on the tensor that is actually conv'd, `a_hat += deq`, existing C-channel
INT8 `_evt_ohat`. Hits 147 = 3 layers × 49 modulated steps. Protocol: n=4,
DDIM 50, seed `20260805`, scheme-A env.

A first k=64 vs A comparison (relL2 0.099 vs 0.284) looked like an SVD win.
It was not. The patch also replaced fused GN + the **static delta table**
with Python GN + **per-step absmax**. Control arm `py_full` (same Python
path, no SVD) isolates the projector:

| arm | SVD | quantizer | relL2 vs fp16 | vs A |
|---|---|---|--:|--:|
| **A** (fused production) | — | static delta table | **0.284** | 0 |
| **py_full** | no | dynamic absmax of `d` | **0.112** | 0.240 |
| **k=64** | yes | absmax of `d_k` | **0.101** | 0.238 |
| **k=16** | yes | absmax of `d_k` | 0.165 | 0.264 |

`py_full` 0.112 matches the historical A vs fp16 (~0.110). This A run at
0.284 is the fused/static path on this seed, not a new baseline. **k=64
matches `py_full`** (0.101 vs 0.112; n=4, within sample noise, visually
the same churches). **k=16 does not** — visible softening, +0.05 relL2.

Drop-tail at the §5 operating point therefore holds on real samples, on
three layers, through the live accumulator pair. It does **not** explain
A→fp16 0.284→0.10; that gap is the quantizer/GN swap. Do not claim an
SVD quality win. Do not claim speedup: codes are still lifted to C
channels and go through the production INT8 conv.

Grid: [`plots/delta_svd_grid.png`](plots/delta_svd_grid.png). JSON:
[`data/delta_svd_gen.json`](data/delta_svd_gen.json). Script:
[`scripts/delta_svd_gen.py`](scripts/delta_svd_gen.py).

## 8. Folded k-conv: quality holds on 3 layers; e2e is a net loss

Monkey-patch still Python GN+absmax on the patched layers (same path as §7).
The conv is now real: range-finder U, `Z = Uᵀ d`, absmax-Q(`Z`), fold
`W_k = Q(W U)`, CUTLASS `Cin=k`. Pair stays in sync (`a_hat += U dequant(q)`).
k = align16(retain × Cin). Protocol: n=4 quality, batch 128 timing (median of
2 after 1 warmup), seed `20260805`, A40, DDIM 50. 70 INT8 3×3 convs; 62 hit
the fused GN path (3038 hits = 62 × 49).

### Isolated conv vs the basis tax (batch 128, EVT o_hat)

This EVT sweep sees a **tile cliff**, not the older PTQ 2.49× curve. Cin
32–128 on 192→192 32×32 are all **0.46 ms**; full Cin=192 is **0.73 ms**
(1.59×). Tax (range-finder GEMM + QR + fold, k=64) is **1.34 ms** on that
shape — **5× the 0.27 ms the conv saves**. 192×16: tax 0.57 vs save ~0.07.
384×16: tax 0.82 vs save ~0.11. The previous PTQ microbench (0.546 → 0.219
at k=64) is the optimistic kernel; even that save loses to 1.34 ms of basis.

### Same-run e2e (A = 73.7 ms/step, relL2 vs fp16 **0.056**)

n=4 relL2 moves between runs; compare arms inside this table, not to §7.

| arm | layers | retain | ms/step | vs A | vs fp16 | vs A relL2 |
|---|---|--:|--:|--:|--:|--:|
| **A** fused production | — | — | **73.7** | 1.00× | 0.056 | 0 |
| 3-layer py_full | 3 | 1.00 | 84.5 | 0.87× | 0.084 | 0.039 |
| 3-layer k-conv | 3 | 0.25 | 83.6 | 0.88× | **0.083** | 0.079 |
| 3-layer k-conv | 3 | 0.33 | 84.5 | 0.87× | 0.125 | 0.090 |
| 3-layer k-conv | 3 | 0.50 | 86.6 | 0.85× | 0.105 | 0.064 |
| 3-layer k-conv | 3 | 0.67 | 88.3 | 0.83× | 0.113 | 0.076 |
| all-3×3 py_full | 62 | 1.00 | 238 | 0.31× | 0.063 | 0.015 |
| all-3×3 k-conv | 62 | 0.25 | 283 | 0.26× | 0.225 | 0.211 |
| all-3×3 k-conv | 62 | 0.33 | 314 | 0.24× | 0.184 | 0.166 |
| all-3×3 k-conv | 62 | 0.50 | 367 | 0.20× | 0.141 | 0.116 |
| all-3×3 k-conv | 62 | 0.67 | 416 | 0.18× | 0.098 | 0.064 |

k on the three layers at retain 0.33 is 64 / 64 / 128 (192, 192, 384). At
0.25: 48 / 48 / 96.

**3 layers.** Drop-tail k-conv matches `py_full` at retain 0.25 (0.083 vs
0.084). Churches look like A. **No e2e speedup**: 0.83–0.88× vs A. The
Python GN swap already costs ~11 ms (84.5 vs 73.7); range-finder+fold on
top does not win it back. Higher retain is slower (larger QR/fold) with
no quality payoff at n=4.

**All 62 fused 3×3.** `py_full` already 0.31× vs A (~164 ms of Python GN).
k-conv adds more tax as retain grows (QR of a bigger `C×k`) and **quality
falls as retain falls**. retain 0.67 is the only all-layer arm near A
(0.098 vs 0.056) and it is the slowest (416 ms). retain 0.25 is 0.225
relL2 and visibly soft. Do not put this on the whole UNet.

### Verdict

| claim | measured |
|---|---|
| Folded `Cin=k` INT8 conv runs | yes, 62 layers, 3038 hits |
| 3-layer quality vs py_full at retain ≥ 0.25 | holds |
| Isolated conv faster at k < C | 1.59× on 192@32 (tile cliff below 128) |
| e2e vs production A | **0.83–0.88× (3 layers), 0.18–0.31× (all)** |
| Ship? | **no — basis tax > conv save; Python GN tax dominates** |

A fused range-finder that is ≪ 0.27 ms/layer on the heavy shapes would
reopen the speed question. This Python + `torch.linalg.qr` path does not.
Do not freeze U. Do not CUDA-SVD. Do not roll this into the apply kernel.

Grid: [`plots/delta_svd_kconv_grid.png`](plots/delta_svd_kconv_grid.png).
JSON: [`data/delta_svd_kconv.json`](data/delta_svd_kconv.json). Script:
[`scripts/delta_svd_kconv.py`](scripts/delta_svd_kconv.py).

## 9. Fused `delta_lowrank_fprop`: native GN helps; the basis is still the tax

One C++ op (`csrc/modiff/norm/delta_lowrank.cu`) wraps fp16 GEMM range-finder,
fp32 QR, `Z = m Q`, absmax-Q(`Z`), fold `W U`, and `a_hat += Z Qᵀ`. Patched
layers use native `group_norm_silu_nhwc` instead of Python GN. Not wired into
production. Same protocol as §8 (n=4, batch 128, seed `20260805`). A this run
is **73.5 ms/step**, relL2 **0.064**.

This is not a custom kernel. It is ATen `mm` + `linalg_qr` behind pybind. The
Python interpreter is gone; the two `P×C×k` GEMMs and the QR are not.

### Isolated tax vs conv save (batch 128, k=64)

| shape | fused tax | full conv | k-conv | save | net |
|---|--:|--:|--:|--:|--:|
| 192@32 | **2.11** | 0.715 | 0.457 | 0.26 | **+1.85** |
| 192@16 | 0.79 | 0.210 | 0.135 | 0.07 | +0.71 |
| 384@16 | 1.06 | 0.410 | 0.192 | 0.22 | +0.85 |

Python tax on 192@32 was **1.34 ms**. The C++ wrap is *slower* (fold + copies
live in the timed op). Still ~8× the 0.26 ms the conv saves. Dispatch was
never the bottleneck.

### Same-run e2e

| arm | layers | retain | ms/step | vs A | vs fp16 |
|---|---|--:|--:|--:|--:|
| **A** fused production | — | — | **73.5** | 1.00× | 0.064 |
| fused k-conv | 3 | 0.25 | 77.0 | **0.95×** | 0.107 |
| fused k-conv | 3 | 0.33 | 77.5 | 0.95× | 0.118 |
| fused k-conv | 3 | 0.50 | 79.4 | 0.93× | 0.101 |
| fused k-conv | 62 | 0.33 | 203 | 0.36× | 0.181 |

Hits: 147 = 3 × 49; 3038 = 62 × 49. 3-layer churches match A by eye. 62-layer
retain 33% is softer (small maps clip `k` to `P`, down to k=16).

**Native GN is the real win vs §8.** 3-layer Python k-conv was 83.6 ms (0.88×);
fused 77.0 (0.95×). 62-layer Python 314 ms (0.24×) → fused 203 (0.36×). The
~11 ms / ~164 ms Python GN tax mostly vanished. The leftover vs A is the
range-finder itself (~3.5 ms on three layers, ~130 ms on 62).

### Verdict

| claim | measured |
|---|---|
| C++ op + native GN + Cin=k CUTLASS runs | yes |
| Isolated tax ≪ 0.27 ms | **no — 2.11 ms on 192@32** |
| 3-layer e2e vs A | **0.93–0.95×** (closer, still a loss) |
| 62-layer e2e vs A | **0.36×** |
| Ship? | **no** |

A kernel that is actually fused (one launch, no QR of `C×k` every step) and
≪ 0.27 ms would reopen speed. Wrapping `torch.linalg.qr` in pybind does not.
Do not freeze U. Do not CUDA-SVD. Production stays scheme A.

Grid: [`plots/delta_svd_kconv_fused_grid.png`](plots/delta_svd_kconv_fused_grid.png).
JSON: [`data/delta_svd_kconv_fused.json`](data/delta_svd_kconv_fused.json).
Script: [`scripts/delta_svd_kconv_fused.py`](scripts/delta_svd_kconv_fused.py).

## Scope

- Three layers, 30 steps, batch 2. The S1 slope on the C20 layer is the same
  order as downsample's +1.5/step; more layers would tighten the number.
- Frozen S2 basis from step-1 SVD. Refreshing U from **this step's delta**
  is the compute scheme in §5; it does not help the storage formats in §1–4.
- Dynamic per-tensor int8 on `R` (best-case range). A static grid would be
  worse, as C15 already showed for `a_hat`.
- `svd_lowrank` cost is extrapolated from 8 samples × 16. A fused kernel
  would move the constant, not the two-order gap vs 2 ms.

## Files

- [`scripts/capture.py`](scripts/capture.py) — real W8A8 generation, three layers
- [`scripts/analyze.py`](scripts/analyze.py) — spectrum, S1/S2/S3 replay, bits, SVD timing
- [`data/spectrum.json`](data/spectrum.json)
- [`data/recursion.json`](data/recursion.json)
- [`data/bit_budget.json`](data/bit_budget.json)
- [`data/svd_cost.json`](data/svd_cost.json)
- [`data/residual_compute.json`](data/residual_compute.json) — k for 90/95% of delta
- [`data/kconv_microbench.json`](data/kconv_microbench.json) — INT8 Cin sweep
- [`data/subspace_k_quality.json`](data/subspace_k_quality.json) — frozen-U proxy
- [`scripts/delta_svd_gen.py`](scripts/delta_svd_gen.py) — live n=4 DDIM, three layers
- [`data/delta_svd_gen.json`](data/delta_svd_gen.json)
- [`plots/delta_svd_grid.png`](plots/delta_svd_grid.png)
- [`scripts/delta_svd_kconv.py`](scripts/delta_svd_kconv.py) — folded Cin=k CUTLASS, retain sweep
- [`data/delta_svd_kconv.json`](data/delta_svd_kconv.json)
- [`plots/delta_svd_kconv_grid.png`](plots/delta_svd_kconv_grid.png)
- [`scripts/delta_svd_kconv_fused.py`](scripts/delta_svd_kconv_fused.py) — native GN + `delta_lowrank_fprop`
- [`data/delta_svd_kconv_fused.json`](data/delta_svd_kconv_fused.json)
- [`plots/delta_svd_kconv_fused_grid.png`](plots/delta_svd_kconv_fused_grid.png)
- `csrc/modiff/norm/delta_lowrank.cu` — experimental; not in production forward

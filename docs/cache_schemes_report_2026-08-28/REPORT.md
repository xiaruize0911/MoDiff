# Skip / Replay / Quant cache schemes

NVIDIA A40 · batch 128 · 50 DDIM · UNet only (attention / linear out of scope).
Quality: churches n=6, seed `20260805`. Speed: seed `20260827`.
Speedup is versus **W8A8 MoDiff full** with fp16 `a_hat`, unless noted.

Skip and replay are never enabled together.

| Knob | Default | Meaning |
|---|---|---|
| `MODIFF_CACHE_SKIP_K` | `1` | Still compute every step; skip in-place `a_hat`/`o_hat` stores on K−1 of K steps |
| `MODIFF_REPLAY_K` | `1` | Skip GN+quantize+conv; `out = o_hat` [+ live ResBlock skip] |
| `MODIFF_REPLAY_BLOCK` | `1` / `out` | Skip emb+out-GN+out_conv on replay. `full`/`in` also skip in-GN+in_conv (same path: out-GN has no input without in_conv) |
| `MODIFF_AHAT_BITS` | `16` | `IMODE=0`: `16` = fp16, `8`/`4` = held dequant. `IMODE=1`: integer storage width (int16 / int8 / unpacked qmax=7) |
| `MODIFF_AHAT_REFRESH` | `0` | `1` = unpack int8→fp16 on commit, pack with a fresh absmax |
| `MODIFF_IMODE` | `0` | `1` = I-MoDiff: integer `a_hat`, frozen `s*` = step0 δ, no dequant. Default stays fp16 + per-step table |
| `MODIFF_DELTA_FREEZE` | `0` | `1` = freeze scale[0] without integer math (control arm for I-MoDiff) |

---

## 1. Skip — recompute every step, freeze cache stores

Still `code = Q(x − a_hat)`, `out = o_hat + conv(code)`. On skip steps the kernels do not store `a_hat`/`o_hat`, so the next residual is against a frozen checkpoint. `t = T` always commits.

### Speed

Isolated `a_hat` store is **1.10 ms/step**. Residual `o_hat` double-write is ~0.8 ms. That is the whole budget.

| Arm | ms/step | vs W8A8 full fp16 |
|---|--:|--:|
| W8A8 full fp16 | 93.44 | 1.00× |
| W8A8 skip-K=4 fp16 | 92.19 | **1.01×** |
| W4A4 full fp16 | 88.59 | 1.05× |
| W4A4 skip-K=4 fp16 | 86.92 | 1.08× |

A dedicated skip-K write-elision bench was 93.34 → 91.20 ms (**2.1 ms, 2.3%**) at K=4. K=2 and K=8 were not re-timed on this CUDA path: they cannot exceed that ~2% ceiling.

### Quality (relL2 vs fp16)

| Arm | relL2 | Notes |
|---|--:|---|
| W8A8 skip-K=1 | 0.118 | Same as W8A8 full |
| W8A8 skip-K=2 | 0.146 | Nearly identical |
| W8A8 skip-K=4 | 0.156 CUDA / 0.194 rollback | Slight softening |
| W8A8 skip-K=8 | 0.328 | Haze, lost texture |
| W8A8 skip-K=16 | 0.429 | Washed-out silhouettes |
| W8A8 skip-K=32 | 0.614 | Unusable |
| W4A4 skip-K=1 | 0.320 | Softer than W8A8 |
| W4A4 skip-K=2 | 0.337 | |
| W4A4 skip-K=4 | 0.369 CUDA / 0.388 rollback | Painterly blur |

K=1/2/4 from `docs/ahat_fake_quant_2026-08-27/data/ahat_skip_k_real.json` (Python cache rollback, same freeze semantics). K=8/16/32 from the large-K grid. CUDA `MODIFF_CACHE_SKIP_K=4` quality is `ahat_bits_gen.json`.

### Samples

W8A8 and W4A4, K=1 / 2 / 4 (fp16 reference on row 1):

![skip K=1,2,4](../ahat_fake_quant_2026-08-27/plots/ahat_skip_k_real_grid.png)

W8A8 large K (fp16, K=1, 8, 16, 32):

![skip large K](../ahat_fake_quant_2026-08-27/plots/ahat_skip_k_real_w8a8_largek.png)

---

## 2. Replay — store `o_hat`, skip residual compute

Skip steps do not launch GN, quantize, or conv. They return stored `o_hat` plus the live ResBlock skip. Attention and time-embed still run.

### Speed (W8A8)

| Arm | ms/step | vs K=1 |
|---|--:|--:|
| replay-K=1 (full) | 93.36 | 1.00× |
| replay-K=2 | 74.77 | **1.25×** |
| replay-K=4 | 65.95 | **1.42×** |
| replay-K=8 | 61.47 | **1.52×** |
| INT8 baseline (no MoDiff) | 73.08 | 1.28× |

W4A4 replay-K=4: **64.75 ms, 1.44×** vs W8A8 full. Replay-K=2 already matches the non-MoDiff INT8 baseline; K=4 beats it.

Source: `docs/residual_replay_2026-08-27/data/replay_bench.json`.

### Quality (relL2 vs fp16)

| Arm | relL2 | compute / replay layer-calls |
|---|--:|--:|
| W8A8 replay-K=1 | 0.121 | 3430 / 0 |
| W8A8 replay-K=2 | 0.186 | 1680 / 1750 |
| W8A8 replay-K=4 | 0.286 | 840 / 2590 |
| W8A8 replay-K=8 | 0.402 | 420 / 3010 |
| W4A4 replay-K=1 | 0.321 | 3430 / 0 |
| W4A4 replay-K=2 | 0.346 | 1680 / 1750 |
| W4A4 replay-K=4 | 0.422 | 840 / 2590 |
| W4A4 replay-K=8 | 0.551 | 420 / 3010 |

### Samples

fp16 reference, then W8A8 and W4A4 replay K=1 / 2 / 4 / 8:

![replay K sweep](../residual_replay_2026-08-27/plots/replay_grid.png)

Replay does **not** zero the conv branch. It freezes the increment: `conv(Q(delta)) = 0`, so `out = o_hat_frozen + skip(x_now)`. Current `x` is unused on the conv path; `a_hat`/`o_hat` stay put. Attention and the skip 1×1 still run.

### `reuse_o_hat` kernel (conv-layer)

CUDA primitive: copy stored `o_hat`, or `out = o_hat + residual`. Wired into `OptimizedInt8Conv2d._replay_out` / int4 twin as `reuse_o_hat_add` when a live skip is present. No-residual replay still returns the `o_hat` **view** (a copy kernel would only add bandwidth).

Same protocol as `one_layer_200.py` / `conv_layer_microbench.py`. This-run full: 1.123 ms one-layer, 32.17 ms conv-set.

| Primitive | one-layer | vs full | conv-set | vs full |
|---|--:|--:|--:|--:|
| full GN+quant+conv | 1.123 | 1.00× | 32.17 | 1.00× |
| `reuse_o_hat` copy | 0.179 | **6.29×** | 3.71 | **8.67×** |
| `reuse_o_hat_add` | 0.263 | 4.27× | 5.44 | 5.91× |
| `torch.add` (old Python) | 0.266 | 4.23× | 5.48 | 5.87× |
| K=4 mix copy / add | 0.415 / 0.478 | 2.71 / 2.35× | 10.83 / 12.12 | 2.97 / **2.65×** |

K=4 add matches the previous `torch.add` replay mix (12.24 ms, 2.65×). The kernel is not faster than aten; skip-add is bandwidth-bound. Python replay-K=4 at **3.97×** on one layer was a view with no store — do not replace that with a copy.

Samples after wiring (n=6, seed `20260805`, 50 DDIM): K=2 stays close to fp16; K=4 smears. Arithmetic matches the old add.

![reuse_o_hat pipeline samples](plots/fig_reuse_o_hat_samples.png)

### Whole ResBlock (`MODIFF_REPLAY_BLOCK=full`)

Tried skipping in-GN+in_conv as well. Same-process W8A8, batch 128, 50 DDIM, vs K=1 = 93.33 ms:

| Arm | ms/step | vs K=1 | relL2 vs fp16 | FID vs fp16 (N=2048) |
|---|--:|--:|--:|--:|
| K=2 `out` | 74.13 | 1.26× | 0.183 | 5.40 (prior folder) |
| K=2 `full` | 74.30 | 1.26× | 0.183 | **5.21** |
| K=4 `out` | 65.19 | 1.43× | 0.286 | 16.3 (prior folder) |
| K=4 `full` | 65.28 | 1.43× | 0.285 | **16.0** |

K=2 full vs out relL2 = **0**. `in_conv` already self-replays on those steps; the early-out is not faster. Drop `full` as a separate arm. Keep shipping `BLOCK=out`.

Source: `data/replay_block_full.json`, `data/reuse_o_hat_microbench.json`.

---

## 3. Quant — quantize the `a_hat` cache

In-kernel storage, not Python fake-quant. `bits=8`: int8 NHWC + fp32 `[scale, qmax=127]`. `bits=4`: qmax=7 in unpacked int8 bytes (same footprint as int8, not nibble-packed). Held scale is t=T absmax. Refresh unpacks to fp16 on commit and packs with a fresh absmax.

### Speed — full step, no skip, no replay

| Arm | ms/step | vs W8A8 full fp16 |
|---|--:|--:|
| W8A8 a_hat fp16 | 93.44 | 1.00× |
| W8A8 a_hat int8 held | 94.35 | 0.99× |
| W8A8 a_hat int8 refresh | 186.13 | 0.50× |
| W8A8 a_hat int4 held | 95.11 | 0.98× |
| W8A8 a_hat int4 refresh | 186.67 | 0.50× |
| W4A4 a_hat fp16 | 88.59 | 1.05× |
| W4A4 a_hat int4 held | 89.87 | 1.04× |
| W4A4 a_hat int4 refresh | 176.25 | 0.53× |

Quant is **not** a speed lever. Held is the same or slightly slower. Refresh is ~2× slower.

### Quality — full step

| Arm | relL2 vs fp16 | Visual |
|---|--:|---|
| W8A8 a_hat fp16 | 0.120 | Sharp |
| W8A8 a_hat int8 held | **0.692** | Watercolor / muddy |
| W8A8 a_hat int8 refresh | 0.792 | Noise |
| W8A8 a_hat int4 held | **2.42** | Broken |
| W8A8 a_hat int4 refresh | 4.84 | Broken |
| W4A4 a_hat fp16 | 0.320 | Soft |
| W4A4 a_hat int4 held | 1.03 | Broken |
| W4A4 a_hat int4 refresh | 1.18 | Broken |

Held t=T scale wrecks full-step quality. Refresh does not recover it and costs 2×. Samples for these rows are in the combo grid below (full int8/int4 rows).

Source: `docs/ahat_bits_2026-08-27/data/ahat_bits_bench.json`, `ahat_bits_gen.json`.

---

## 4. Combinations — (skip or replay) + quant

Held t=T scale. Refresh is omitted from the speed summary: it is ~2× slower and worse quality than held on these arms.

Skip/replay only quantize `a_hat` on commit, so a bad scale is not re-snapped every step. That is why **int8 + skip/replay looks usable** while **int8 full does not**.

### W8A8

| Arm | ms/step | speedup | relL2 |
|---|--:|--:|--:|
| full fp16 | 93.44 | 1.00× | 0.120 |
| full int8 held | 94.35 | 0.99× | 0.692 |
| full int4 held | 95.11 | 0.98× | 2.42 |
| skip-K=4 fp16 | 92.19 | 1.01× | 0.156 |
| skip-K=4 int8 held | 93.86 | 1.00× | **0.255** |
| skip-K=4 int4 held | 94.07 | 0.99× | 0.893 |
| replay-K=4 fp16 | 66.43 | **1.41×** | 0.286 |
| replay-K=4 int8 held | 67.75 | **1.38×** | 0.337 |
| replay-K=4 int4 held | 67.84 | 1.38× | 0.797 |

### W4A4

| Arm | ms/step | vs W8A8 full | relL2 |
|---|--:|--:|--:|
| full fp16 a_hat | 88.59 | 1.05× | 0.320 |
| full int4 held | 89.87 | 1.04× | 1.03 |
| skip-K=4 fp16 | 86.92 | 1.08× | 0.369 |
| skip-K=4 int4 held | 88.47 | 1.06× | 0.657 |
| replay-K=4 fp16 | 64.75 | **1.44×** | 0.422 |
| replay-K=4 int4 held | 65.88 | 1.42× | 0.676 |

Quant on top of skip/replay adds ~1–1.4 ms and a bit of relL2. Int4 a_hat stays blurry even with skip or replay.

### Samples

Same six seeds across full / skip-K=4 / replay-K=4 × fp16 / int8 / int4 (held and refresh):

![all arms](../ahat_bits_2026-08-27/plots/ahat_bits_grid.png)

---

## 5. I-MoDiff — integer `a_hat` (16 / 8 / 4)

Held int8 (`MODIFF_AHAT_BITS=8`, IMODE off) still does `code * s → float`, subtracts from fp16 `x`, then snaps on a finer per-step δ table. That is the FID-121 path. I-MoDiff changes the formula, not the conv:

```
s*     = frozen static_delta_scale[0] reciprocal (step0 α)
x_i    = sat(round(x / s*))          # only float→int
q      = sat_i8(x_i − a_hat)         # integer sub, W8A8 ±127
a_hat += q                           # sat to ±qmax(bits)
o_hat += conv(q), α = s*
```

`bits=16` stores int16 NHWC (same DRAM as fp16). `bits=8/4` use int8 / unpacked int4-grid (`qmax=127/7`). Default remains fp16 `a_hat` + per-step table. Control arm `frozen_s` (`MODIFF_DELTA_FREEZE=1`) freezes scale[0] without integer math. Replay-K was not mixed with I-mode.

Source: `data/imode.json`. Invariants: `integration/tests/test_imode.py`.

### Invariants (synthetic layer)

I2 is `o_hat ≈ conv(a_hat.float() * s*)` with SmoothQuant weights. Increment identity: `Δa_hat` equals `q` except at saturation.

| Arm | I2 | max\|Δa\| | sat |
|---|--:|--:|--:|
| imode16 | 0.0011 | 1 | 0.001 |
| imode8 | 0.0011 | 1 | 0.244 |
| imode4 | 0.5892 | 1 | 1.000 |
| increment | 0.0004 | 2 | 0.001 |

Synthetic I2 is good because calibration seeds an activation-sized `s*`, so `x/s*` fits. Real-table overflow is below.

### Quality (n=6 relL2, then FID N=2048 vs fp16)

Same seed `20260805`, `MODIFF_LINEAR=0`, 50 DDIM.

| Arm | relL2 | max \|a_hat\|/qmax | n_over | FID vs fp16 | vs W8A8-full |
|---|--:|--:|--:|--:|--:|
| W8A8 full fp16 | 0.121 | — | 0 | **0.92** | 0 |
| frozen_s (fp16, scale[0] only) | 0.119 | — | 0 | **1.51** | 0.96 |
| I-MoDiff int16 | 0.409 | 0.194 | 0/70 | **28.8** | 28.5 |
| I-MoDiff int8 | 0.446 | 1.0 | 66/70 | **76.3** | 75.9 |
| I-MoDiff int4 | 0.755 | 1.0 | 70/70 | **344** | 344 |
| int8 held (prior) | 0.692 | — | — | **121** | 120 |

Freeze-the-table is free: `frozen_s` ≈ full. Integer math is not. `s*` is the step0 **delta** scale (residual-sized), so `|x|/s*` ≫ 127. t=T `q0 = sat_i8(round(x/s*))` clips; later `q` saturates ±127/step trying to catch up. imode16 peak `|a_hat|` ≈ 0.194×32767 ≈ 6360 ≈ 50×127. imode16 is better than held-121 but far from frozen_s (replay-K=4 territory, FID 16, is already a drop). 8/4 overflow as predicted.

### Speed

int16 matches fp16 bandwidth. ALU saved is not DRAM. 8/4 would be a storage win if quality held; it does not.

| Arm | one-layer ms | conv-set ms | e2e (warmup=5) | e2e (warmup=1) |
|---|--:|--:|--:|--:|
| W8A8 full fp16 | 1.052 | 32.35 | **93.42** | 82.53 |
| I-MoDiff int16 | 1.064 | 33.11 (0.977×) | 84.63 | 84.19 |
| I-MoDiff int8 | 0.989 | 32.38 | 82.72 | 82.60 |
| I-MoDiff int4 | 0.979 | 32.31 | 82.55 | 82.43 |

e2e vs 93.4 looks faster because I-mode skips the 5-round residual warmup at t=T (no-op on this grid). Fair same-work comparison is `MODIFF_WARMUP_STEPS=1`: imode16 is **+2%**. Conv-set imode16 is 2% slower. 8/4 are not faster at e2e.

imode16 + replay-K=2 was **not** timed: quality does not hold.

### Judgment

- **Do not ship I-mode.** I2 holds, but imode16 FID 28.8 is not close to `frozen_s` 1.51, and it is not a DRAM win.
- **Drop 8/4.** Overflow 66/70 and 70/70; FID 76 / 344.
- **Do not drop because freeze wrecked FID.** `frozen_s` is fine. The failure is `sat_i8(x/s*)` when `s*` is residual-sized. A second `s*` strategy is out of this round's scope.

---

## Takeaways

1. **Replay** is the only real speed lever (K=2 ≈ 1.25× with FID 5.4; K=4 ≈ 1.41–1.42× with FID 16). Quality cost at K=2 is the shippable point.
2. **Skip** is ~1–2% e2e. Quality stays high through K=4 on W8A8; K≥8 hazes out.
3. **Quant** (in-kernel int8/int4 `a_hat`) is not faster. Held t=T scale breaks full-step images (int8 0.69, int4 2.42). Refresh is a 2× tax and worse.
4. **Skip or replay + int8 held** keeps most of replay's speed (1.38×) and restores structure that full int8 loses (0.255 / 0.337 vs 0.692). **Int4 a_hat** is not viable even in combination.
5. **`reuse_o_hat_add`** is now the ResBlock replay epilogue. It matches `torch.add`. Do not copy `o_hat` when a view suffices.
6. **`BLOCK=full`** (skip in_conv too) is not faster than `out` and is bit-identical at K=2. Drop it.
7. **I-MoDiff** (integer `a_hat`, frozen `s*`) keeps I2 and freeze quality, but integer `sat_i8(x/s*)` with residual-sized `s*` yields FID 28.8 / 76 / 344. Not a speed lever. Do not ship; do not mix with replay this round.
8. Leftover e2e time is skip 1×1 (never quantized) and attention (`MODIFF_LINEAR=0`). Per-layer K is the remaining knob inside the 32 ms conv bucket.

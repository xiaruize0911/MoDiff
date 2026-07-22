# MoDiff 5-mode benchmark — per-step kernel breakdown (measured)

**GPU:** NVIDIA A40 (48 GB, SM 8.6) · **PyTorch:** 2.4.1+cu124 · **CUDA:** 12.4 · driver 580
**Model:** LSUN-Churches LDM-8 UNet (unconditional, 256×256 → 32×32 latent) · **Batch:** 128 · **Sampler:** DDIM
**Date:** 2026-07-21 · all timing/shape/count data measured at the current code state (`git HEAD`).

**5 modes:** `fp16`, `int8_baseline`, `int4_baseline`, `int8_modiff`, `int4_modiff`.
int8/int4 use the **latest fused kernels**: deep-fuse CUTLASS int8/int4 conv (weight-scale folded into the
epilogue, bias/residual folded into the `from_half` store), fused-flash quantized attention
(`flash_attn_int8_vt` / `flash_attn_int4_vt`), and W8A8/W4A4 AWQ GEMM (`gemm_wxax`) for qkv/proj.
`_modiff` adds the temporal-delta conv cache (o_hat conv + accumulate every step). **Both attention
activation-quantize fusions (§7) are ON by default** for int8 modes — qkv-quant folded into GroupNorm
(`MODIFF_FUSE_GN_QKV_I8`) and proj-quant folded into the attention-output transpose
(`MODIFF_FUSE_PROJ_I8`) — so §1–§3 measure the most-fused pipeline (the quant attention block does **zero**
standalone activation-quantize kernels); set either flag to `0` to disable. autocast fp16 on for all modes.

**What is new in this redo:** every kernel is benchmarked on **every shape it actually runs in the model**,
and each shape carries its **per-step call count**, so we report the **time it contributes to one DDIM step**
(`count/step × µs/call`). Shapes and counts are not hand-listed — they come from instrumenting the real
model forward (`scripts/enumerate_shapes.py`, hooks + fused-path method wraps → `data/kernel_shapes.csv`).

**Method (measured only):** e2e speed = wall time, GPU clock burn-in → 30 warmup → 5×200 timed steps with
`torch.cuda.synchronize()`; e2e profile = `torch.profiler` CUDA self-time bucketed by kernel, ms/step.
Per-kernel speed = CUDA-event median, 50 warm + 200×5 (conv) / 30 warm + 100×5 (attn) / 50 warm + 200×5
(linear), GPU clock burn-in. Checkpoint is loaded (dispatch/shapes faithful → **speed faithful**; generation
quality not evaluated here). Raw data in `data/*.csv` + `data/dataset.json`; figures in `figs/*.png`; the
exact shape/count ground truth in `data/kernel_shapes.csv`.

---

## 0. Kernel shape inventory (ground truth — what runs, and how often per step)

Per DDIM step the UNet runs **one forward**; MoDiff's temporal cache **skips no convolution** — every conv,
linear and attention block fires on every step (verified: identical counts on steady-state steps in all modes).

| family | distinct shapes | calls / step | notes |
|---|--:|--:|---|
| **conv** | 33 | **89** | quant modes: 20 geometries (70 calls) run int8/int4; 13 geometries (19 calls: skip / 1×1 pointwise / Cin<32 / final-out) stay fp16 cuDNN |
| **linear** | 14 | **79** | qkv/proj: 10 shapes / **42 calls** (W8A8/W4A4 AWQ GEMM); time-embed: 4 shapes / 37 calls (K<2048 → fp16-gated in all modes) |
| **attention** | 5 | **21** | 3 blocks (T=1024/256/64, hd≤48) run fused-flash int8/int4; 2 blocks (hd=96, T=16/4) stay fp16 SDPA |

fp16 mode has 12 distinct linear shapes / 69 calls (10 qkv calls are folded into a fused GN→qkv kernel).
Full per-shape × per-mode table: `data/kernel_shapes.csv`.

---

## 1. E2E DDIM step speed · `data/e2e_speed.csv` · `figs/fig_e2e_speed.png`

| mode | ms/step | min ms | vs fp16 |
|---|--:|--:|--:|
| fp16 | 188.1 | 187.3 | 1.00× |
| int8_baseline | 121.0 | 120.8 | 1.55× |
| **int4_baseline** | **114.9** | **114.8** | **1.64×** |
| int8_modiff | 136.8 | 136.7 | 1.38× |
| int4_modiff | 141.5 | 138.1 | 1.33× |

![e2e speed](figs/fig_e2e_speed.png)

Baseline (cache-free static quant) is fastest; MoDiff's temporal-delta cache adds sub/accumulate overhead
per step (see §3 conv) that costs back ~0.17–0.24× — MoDiff buys accuracy, not e2e speed.

## 2. E2E per-component timing profile (GPU self-time, ms/step) · `data/e2e_timing_profile.csv` · `figs/fig_e2e_timing_profile.png`

Measured on the fully fused pipeline (both attention quantize fusions ON, §7).

| bucket | fp16 | int8_base | int4_base | int8_modiff | int4_modiff |
|---|--:|--:|--:|--:|--:|
| attention (flash / softmax) | 44.2 | 34.3 | 33.5 | 33.7 | 33.4 |
| attn bmm fp16 (QKᵀ/AV) | 42.0 | 0.2 | 0.2 | 0.2 | 0.2 |
| conv (int GEMM) | 44.7 | 24.4 | 14.0 | 28.2 | 15.6 |
| qkv/proj int GEMM | 0.0 | 7.5 | 6.9 | 7.4 | 6.9 |
| GroupNorm | 21.0 | 23.2 | 22.2 | 21.9 | 21.8 |
| quantize/dequant | 0.0 | 18.1 | 16.8 | 27.5 | 25.2 |
| modiff cache (o_hat) | 0.0 | 0.0 | 0.0 | 11.0 | 11.0 |
| elementwise/copy | 32.4 | 10.8 | 18.1 | 5.7 | 11.3 |
| upsample/concat + other | 6.0 | 6.5 | 6.8 | 6.4 | 6.2 |
| **gpu_busy** | 196.6 | 125.3 | 118.6 | 142.1 | 131.7 |
| **wall** | 188.1 | 119.9 | 114.4 | 136.2 | 147.0 |

![e2e timing profile](figs/fig_e2e_timing_profile.png)

The **attention quantize fusions (§7)** show up in `gpu_busy`, not in one bucket: int8_baseline gpu_busy drops
**129.3 → 125.3 ms** vs the fully-unfused run (the 42 standalone `quantize_act_int8` kernels/step → 0). For
qkv the quantize folds into GroupNorm (near-free on the memory-bound GN write); for proj the fused
`quantize_attn_out_int8` also absorbs the output transpose, so `elementwise/copy` drops 12.7 → **10.8 ms**
while `quantize/dequant` nets flat (that kernel is bucketed under "quant"). MoDiff's extra cost is still
visible: **+11 ms modiff-cache** and a larger `quantize/dequant` bucket (27.5 ms) — the temporal-delta path
re-quantizes conv deltas every step. `wall` is essentially unchanged vs unfused (the quantizes were
latency-hidden); the `int4_modiff` profile wall (147) is a noisier secondary measurement — see §1 for the
clean 5×200 wall (141.5).

## 3. Per-kernel-family time in one step · `data/perstep_summary.csv` · `figs/fig_perstep_summary.png`

Standalone kernel time attributed to one DDIM step (`Σ count/step × µs/call`). These are back-to-back
micro-measurements — they show **where the per-step kernel work goes**, not a sum equal to the e2e wall.

| family (calls/step) | fp16 | int8_base | int4_base | int8_modiff | int4_modiff |
|---|--:|--:|--:|--:|--:|
| conv (89) | 46.11 ms | 33.16 ms | **23.09 ms** | 46.05 ms | 33.63 ms |
| linear qkv/proj+temb (79) | 8.14 ms | 9.15 ms | 10.83 ms | 9.15 ms | 10.83 ms |
| attention incl GN+quant (21) | 93.87 ms | 56.24 ms | 55.38 ms | 56.24 ms | 55.38 ms |

![per-step summary](figs/fig_perstep_summary.png)

Headlines: **attention is the biggest lever** (int8/int4 ≈ 1.67×/1.70× on the block, driven by the T=1024
block); **conv** gets 1.39×/2.00× in baseline but MoDiff's delta path erases the int8 win (46.1 ≈ fp16 46.1);
**qkv/proj linear is a slight net loss** as a standalone kernel (0.89×/0.75×: low-K IMMA + a separate
activation-quantize; GEMM-only alone would be 1.21×/1.47×, see §5). Note this row is the *standalone* kernel
cost — in the actual pipeline the qkv activation-quantize is **fused into GroupNorm** (§7, default on), which
recovers the qkv front-end to 1.11× vs fp16; the net e2e speedup is carried by conv and attention.

---

## 4. Conv kernel — all 33 geometries, 5 modes · `data/conv_kernel_speed.csv` · `figs/fig_conv_kernel.png` · `figs/fig_conv_perstep.png`

Per-call kernel time (µs) of the fused conv-layer op (deep-fuse store); `×N` = calls/step. `quant_eligible`
geometries run int8/int4; the rest stay fp16 cuDNN in every mode. Top geometries by per-step weight:

| Cin→Cout | HW | ×/step | fp16 | int8_base | int4_base | int8_modiff | int4_modiff | i8b × | i4b × |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 768→768 | 2² | 13 | 76 | 64 | 64 | 87 | 58 | 1.19× | 1.20× |
| 384→384 | 8² | 9 | 233 | 156 | 98 | 218 | 157 | 1.50× | 2.36× |
| 384→384 | 16² | 8 | 847 | 594 | 371 | 754 | 565 | 1.43× | 2.28× |
| 768→768 | 4² | 8 | 239 | 145 | 83 | 193 | 120 | 1.64× | 2.88× |
| 192→192 | 32² | 7 | 1018 | 771 | 582 | 1431 | 997 | 1.32× | 1.75× |
| 384→384 | 32² | 2 | 3272 | 2216 | 1354 | 2894 | 2096 | 1.48× | 2.42× |
| 768→384 | 16² | 2 | 1715 | 1023 | 595 | 1243 | 859 | 1.68× | 2.88× |
| **conv total / step** | | 89 | **46.11 ms** | **33.16 ms** | **23.09 ms** | **46.05 ms** | **33.63 ms** | **1.39×** | **2.00×** |

![conv kernel](figs/fig_conv_kernel.png)
![conv per-step](figs/fig_conv_perstep.png)

int8 conv 1.19–1.68×, int4 conv 1.20–2.96× per shape (largest wins on the big-channel mid-resolution convs).
**MoDiff conv is ≈ fp16** (46.05 vs 46.11 ms/step for int8): the o_hat delta-quantize + conv + accumulate
overhead cancels the int8 kernel speedup — consistent with the e2e result that baseline > modiff.

## 5. Linear (qkv/proj + time-embed) — all 14 shapes · `data/linear_kernel_speed.csv` · `figs/fig_linear_kernel.png`

No modiff variant (static W/A quant in all modes → int8_baseline ≡ int8_modiff). **The activation quantize is
NOT fused** — it is a separate `quantize_act_int8`/`quantize_act_int4_pack` kernel before the AWQ GEMM, and it
is not folded into the upstream GroupNorm either (`MODIFF_FUSE_GN_QKV=0` in quant modes; only bias/residual/
dequant are fused into the GEMM epilogue). "full" = quantize + GEMM (**what the model runs**); "gemm-only" =
raw AWQ GEMM on a pre-quantized input (the counterfactual: if the quantize were fused away).

| role | K→N | M | ×/step | fp16 | int8 full | int8 gemm-only | int4 full | int4 gemm-only | i8 × | i4 × |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| qkv | 192→576 | 131072 | 5 | 412 | 528 | 398 | 779 | 341 | 0.78× | 0.53× |
| proj | 192→192 | 131072 | 5 | 193 | 297 | 156 | 584 | 135 | 0.65× | 0.33× |
| qkv | 384→1152 | 32768 | 5 | 415 | 331 | 262 | 244 | 191 | 1.25× | 1.70× |
| qkv | 768→2304 | 2048 | 5 | 80 | 76 | 62 | 51 | 42 | 1.06× | 1.56× |
| proj | 768→768 | 2048 | 5 | 31 | 40 | 28 | 29 | 18 | 0.78× | 1.08× |
| **linear total / step** | | | 79 | **8.14 ms** | **9.15 ms** | 6.73 ms | **10.83 ms** | 5.53 ms | **0.89×** | **0.75×** |
| **vs fp16** | | | | 1.00× | **0.89×** | *1.21×* | **0.75×** | *1.47×* | | |

![linear kernel](figs/fig_linear_kernel.png)

The quantize is the swing factor: **GEMM-only would be 1.21× (int8) / 1.47× (int4)**, but the unfused
activation quantize (~140 µs/call on the M=131072 projections) plus int4's K-padding pass turns the total into
**0.89× / 0.75×**. Two low-K, large-M projections dominate: `192→192` int8 GEMM-only is 1.23× (157 vs 193 µs)
yet 0.65× once the ~141 µs quantize is added; `192→576` (K=192) is too small for IMMA to even beat fp16 at the
GEMM. High-K shapes (768→2304) win. Net qkv/proj is a slight loss at this shape mix; conv and attention carry
the e2e speedup. (Fusing the quantize into the GN→qkv path is implemented in §7 — it recovers GPU work but
is e2e-neutral because the quantize was already latency-hidden.)
Time-embed linears (K∈{192,768}<2048) K-gate to fp16 in every mode, so int8/int4 == fp16 there.

## 6. Attention (WITH GroupNorm, fair) — all 5 blocks · `data/attn_kernel_speed.csv` · `figs/fig_attn_kernel.png`

GroupNorm + Q/K/V quantize + attention. Only hd≤48 & T%64==0 blocks run fused-flash int8/int4; hd=96 blocks
stay fp16 SDPA (baseline ≡ modiff — attention has no modiff variant).

| block (C/hd/T) | ×/step | GN µs | fp16 tot | int8 tot | int4 tot | i8 × | i4 × | rel-L2 (i8/i4) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 192/24/1024 | 5 | 480 | 16700 | 8247 | 8099 | 2.03× | 2.06× | 0.025 / 0.144 |
| 384/48/256 | 5 | 275 | 1627 | 2152 | 2159 | 0.76× | 0.75× | 0.018 / 0.150 |
| 384/48/64 | 5 | 162 | 319 | 721 | 689 | 0.44× | 0.46× | 0.015 / 0.142 |
| 768/96/16 | 5 | 48 | 113 | (fp16) | (fp16) | 1.00× | 1.00× | — |
| 768/96/4 | 1 | 12 | 77 | (fp16) | (fp16) | 1.00× | 1.00× | — |
| **attention total / step** | 21 | | **93.87 ms** | **56.24 ms** | **55.38 ms** | **1.67×** | **1.70×** | — |

![attention kernel](figs/fig_attn_kernel.png)

The **T=1024 block dominates** (16.7 ms fp16 → 8.2 ms int8, 2×) and carries the weighted win; the smaller
flash blocks (T=256/64) are actually slower quantized (flash setup + quantize > benefit at small T), but they
are cheap. This is why attention is the single largest e2e speedup contributor.

---

## 7. Optimization: fuse the attention activation-quantizes (qkv→GroupNorm, proj→attention-output) · `data/fuse_gn_qkv_quant.csv` · `data/fuse_gn_qkv_e2e.csv` · `figs/fig_fuse_gn_qkv.png`

§5 showed the quantized attention linears lose to fp16 partly because each pays a *separate, unfused*
activation-quantize kernel. Both are now folded into the kernel that produces their input (int8 modes,
default on; each has an `=0` escape hatch). With both on, the quant attention block runs **zero** standalone
`quantize_act_int8` kernels/step (was 42/step = 21 qkv + 21 proj).

**(a) qkv → GroupNorm** (`MODIFF_FUSE_GN_QKV_I8`): the conv path already fuses GN+quantize
(`group_norm_silu_quantize_nhwc` → `forward_from_int8`); the qkv path never reused it. Emit the qkv GEMM's
int8 input straight out of GroupNorm (all attention C∈{192,384,768} are multiples of 64 → no K-pad), then AWQ
GEMM via a new `QuantLinearWxAx.forward_from_int8`. (A more aggressive `fused_gn_qkv_int8` kernel was
prototyped earlier but its companion `quantize_attn_qkv_from_i8` was never compiled, so that path is dead.)

**(b) proj → attention output** (`MODIFF_FUSE_PROJ_I8`): the proj reads the *attention output* (not GN), so
its quantize can't fold into GN — instead fold it into the output layout transform. `quantize_attn_out_int8`
does the head-major→token-major transpose+reshape **and** the int8 quantize in one kernel (replacing a
standalone transpose copy + `quantize_act_int8`), feeding `proj.forward_from_int8` with the residual fused in
the GEMM epilogue. (The `quantize_attn_out_int8` kernel + an `_apply_proj` helper already existed in
`token_major_attention.py` but the quant block's overridden `forward` never called them — now wired in.)
In-model OFF vs ON: output **bit-identical** (rel-L2 = 0.000000), 21 proj `quantize_act_int8`/step → 0.

**Isolated microbench** — qkv front-end (GN + quantize + qkv GEMM), per step, b128:

| | fp16 | int8 non-fused (today) | int8 GN→quant fused |
|---|--:|--:|--:|
| qkv front-end / step | 9.73 ms | 9.91 ms (0.98×) | **8.76 ms (1.11×)** |

Fusing removes the standalone quantize (**1.19 ms/step**) → **1.13×** vs non-fused. Correctness: fused vs
non-fused rel-L2 ≤ 0.0035; vs fp32 ≈ 0.02 (same as non-fused).

**In-model, both fusions** (int8_baseline): output **bit-identical** to the unfused path (rel-L2 = 0.000000);
`quantize_act_int8` calls drop 42 → 0 per step. In the §2 profile the effect is the shrinking
`quantize/dequant` bucket (int8_baseline 18.8 unfused → 17.1 qkv-only → the both-on value in §2) and a few ms
lower gpu_busy. **The e2e wall barely moves** (~120 ms/step): both quantizes were already latency-hidden — not
on the critical path — so folding them cuts GPU work but not wall. The churches-UNet wall is bound by
attention and the low-K GEMMs, which these quantize-fusions don't touch; the value is a correct,
lower-GPU-work pipeline that would help on a launch-bound / smaller-batch regime. Both default-on (int8);
int4 is not fused (K=192 qkv needs a pad-to-256 + nibble-pack pass — deferred). The microbench above isolates
the qkv fusion (removes 1.19 ms/step of standalone quantize, a 1.13× qkv front-end); proj is analogous.

![attention activation-quantize fusion](figs/fig_fuse_gn_qkv.png)

---

## Reproduce

```bash
source setup_cuda_env.sh
cd /workspace/MoDiff
python docs/benchmark_5mode_2026-07-21/scripts/enumerate_shapes.py   # -> data/kernel_shapes.csv (shape+count ground truth)
python docs/benchmark_5mode_2026-07-21/scripts/conv_kernel.py        # -> data/conv_kernel_speed.csv
python docs/benchmark_5mode_2026-07-21/scripts/linear_kernel.py      # -> data/linear_kernel_speed.csv
python docs/benchmark_5mode_2026-07-21/scripts/attn_kernel.py        # -> data/attn_kernel_speed.csv
python docs/benchmark_5mode_2026-07-21/scripts/e2e_speed.py          # -> data/e2e_speed.csv
python docs/benchmark_5mode_2026-07-21/scripts/e2e_timing_profile.py # -> data/e2e_timing_profile.csv
python docs/benchmark_5mode_2026-07-21/scripts/perstep_summary.py    # -> data/perstep_summary.csv
python docs/benchmark_5mode_2026-07-21/scripts/make_plots.py         # -> figs/*.png
# §7 GN->qkv quantize fusion (requires MODIFF_FUSE_GN_QKV_I8=1 in-model):
python docs/benchmark_5mode_2026-07-21/scripts/fuse_gn_qkv_quant.py     # -> data/fuse_gn_qkv_quant.csv (microbench)
python docs/benchmark_5mode_2026-07-21/scripts/verify_fuse_gn_qkv_e2e.py# -> data/fuse_gn_qkv_e2e.csv (in-model correctness+speed)
```

Environment note: this box's system Python (torch 2.4.1+cu124, matching the prebuilt `modiff_cutlass.so`) was
missing the LDM support stack; it was restored with `omegaconf einops tqdm pytorch-lightning==1.4.2
torchmetrics==0.6.0 kornia test-tube tensorboard pandas imageio matplotlib`, a pyDeprecate-compatible shim for
the renamed `deprecate` package, and a stub `taming.modules.vqvae.quantize` (the KL autoencoder never uses it).
torch itself was left untouched so the CUDA extension stays ABI-compatible. See `data/README.md` for the
column meaning of every CSV.

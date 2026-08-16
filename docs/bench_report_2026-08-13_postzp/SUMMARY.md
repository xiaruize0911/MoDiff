# Where we are: speed, block-level attribution, and what comes out

**A40 · LSUN-churches LDM, real 2.7 GB checkpoint · batch 128 · activation zero point 0 everywhere (`MODIFF_ZP_STRICT=1`), padding zero-fill**

One page over the whole measurement surface. Everything here is measured; the derivations are arithmetic
on those measurements and are shown. Sources: [REPORT.md](REPORT.md) (raw data),
[KERNEL_BREAKDOWN.md](KERNEL_BREAKDOWN.md) (what each kernel does),
[KERNEL_SPEEDUP.md](KERNEL_SPEEDUP.md) (per-kernel fp16→int8→int4), [data/](data/) (JSON).

**Two configurations appear below and they are not interchangeable.** Timing is DDIM **200** steps,
static delta, `MODIFF_LINEAR=0`. The samples in §5 are the 2026-08-05 FID run: DDIM **50**, dynamic
delta, `MODIFF_LINEAR=0`. Speed and quality were measured on different protocols, so no row combines
them.

---

## 1. End-to-end

3 timed repeats after 2 discarded warm-up samples; CV ≤ 0.24% on every row.

| mode | ms/step | ms/sample | ms/batch of 128 | **vs fp16** |
|---|--:|--:|--:|--:|
| fp16 | 103.00 | 160.9 | 20599.7 | 1.000× |
| W8A8 PTQ | 71.23 | 111.3 | 14245.2 | **1.446×** |
| W8A8 MoDiff | 73.19 | 114.4 | 14637.1 | 1.407× |
| W4A4 PTQ | 57.85 | 90.4 | 11569.9 | **1.780×** |
| W4A4 MoDiff | 58.50 | 91.4 | 11699.8 | 1.761× |

![e2e](plots/01_e2e.png)

MoDiff's temporal machinery costs **2.8% at W8A8 and 1.1% at W4A4** over the corresponding PTQ arm. That
is the price of the quality in §5. It is a small enough delta that run order inside one process can
perturb it (the arm-order effect documented in
[zp_coverage_2026-08-13/FINDINGS_NOISE_FLOOR.md](../zp_coverage_2026-08-13/FINDINGS_NOISE_FLOOR.md)), so
read 1–3% as "roughly free", not as a precise figure.

## 2. Where the time goes, and where the speedup comes from

GPU time by kernel bucket over the profiled window (ms of a 128×200 batch), from REPORT.md §1a. `saved`
and `% of gain` decompose the **9.03 s** that fp16 → W4A4 PTQ removes.

| bucket | fp16 | share | W8A8 PTQ | ×  | W4A4 PTQ | × | saved | **% of gain** | share of W4A4 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| GEMM / conv | 9389 | 45.6% | 7258 | 1.29× | 4615 | **2.03×** | 4774 | **52.9%** | 39.9% |
| GroupNorm+SiLU family | 4230 | 20.5% | 3671 | 1.15× | 3730 | 1.13× | 500 | 5.5% | **32.2%** |
| elementwise / copy | 3931 | 19.1% | 1151 | 3.42× | 1151 | **3.42×** | 2780 | **30.8%** | 9.9% |
| attention | 2295 | 11.1% | 1768 | 1.30× | 1680 | 1.37× | 615 | 6.8% | 14.5% |
| other | 754 | 3.7% | 397 | 1.90× | 395 | 1.91× | 359 | 4.0% | 3.4% |
| **total** | **20599** | 100% | **14245** | 1.45× | **11571** | **1.78×** | **9028** | 100% | 100% |

Three things this says that the headline number does not:

**Nearly a third of the gain is fusion, not low precision.** The elementwise/copy bucket falls 3.42× and
is *identical* at W8A8 and W4A4 — bit width has nothing to do with it. It is the quantize folded into the
GroupNorm+SiLU epilogue and the bias/residual/o_hat folded into the conv's EVT epilogue, so the
intermediate tensors that fp16 writes and re-reads are never materialized. 2.78 s of the 9.03 s.

**Going 8→4 bits buys exactly one bucket.** W8A8 PTQ → W4A4 PTQ saves 2674 ms, of which **2643 ms is
GEMM/conv** — 98.8%. Every other bucket is flat to within noise. So the int4 datapath is worth having
only in proportion to how matmul-bound the model is.

**The ceiling has moved.** At fp16 the matmuls were 45.6% of the run; at W4A4 they are 39.9% and the
GroupNorm+SiLU family is 32.2% and barely faster than fp16 (1.13×). Even a *free* conv would only take
W4A4 PTQ from 57.85 to 34.8 ms/step — a further 1.66×. The next real lever is the normalization family,
not more bits off the GEMM.

## 3. Per block

Real call arguments captured at the C++ entry point during a live sample, then replayed in isolation
(8 rounds × 60 iters, median of round medians). `ms/sample` = median µs/call × calls/sample.

| suite | fp16 | W8A8 PTQ | W4A4 PTQ | int8 × | int4 × | a speedup? |
|---|--:|--:|--:|--:|--:|---|
| attention | 63.34 | 51.08 | 50.01 | **1.24×** | **1.27×** | **yes** |
| conv | 265.72 | 149.09 | 85.59 | **1.78×** | **3.10×** | **yes** — 33/33 records matched three ways |
| linear | 60.92 | 47.15 | 43.45 | 1.29× | 1.40× | **no** — see below |
| norm_quantize | 92.23 | 143.35 | 144.44 | 0.64× | 0.64× | **no** |
| other | 6.10 | 10.11 | 10.11 | 0.60× | 0.60× | **no** |

**Only the first two rows are speedups, and there is no grouping of the rest that fixes it.** Corrected
2026-08-16 — the earlier version of this table said fp16 counts the attention projections as 1×1 convs
and that `conv + linear` therefore cancels the reclassification. Both halves were wrong. Conv closes
*exactly* three ways (fp16's suite total equals its matched total to the cent, in every arm), so nothing
moves between conv and linear at all; the 1×1 convs in §3a are ResBlock skip connections, present in all
three arms at 1.00×. What actually happens is that fp16 runs the T=1024 and T=256 qkv through one fused
`fused_gn_qkv` kernel — **31.96 ms/sample** — which the capture's name-matching classifier dropped into
`other`. Its GroupNorm half has no home: the quantized arms pay that in `norm_quantize`, so no
regrouping can put it in one place. And the sum is worse than either, because the quantized arms' fused
epilogues *delete* tensors, so the elementwise kernels fp16 pays for them do not exist as records to be
credited — the full-run profile in §2 sees that as 2.78 s saved and a replay suite cannot see it at all.

**So read §2 and §3a, not suite ratios.** Details and the arithmetic in
[KERNEL_SPEEDUP.md](KERNEL_SPEEDUP.md) §1; the standing consequences in [OPEN_ITEMS.md](../OPEN_ITEMS.md)
A1/A2.

![conv](plots/04_conv.png) ![attention](plots/03_attention.png)

### 3a. Conv, matched layer by layer

33 layers matched across all three arms by normalizing the weight to `(K, C, R, S)` — 20 quantized, 13
unquantized controls. Full table in [KERNEL_SPEEDUP.md](KERNEL_SPEEDUP.md) §2.

| subset | n | int8 | int4 |
|---|--:|---|---|
| quantized, fp16 baseline also fp16-in — **the arithmetic-only number** | 14 | 1.34–2.35× (median **1.97×**) | 2.72–4.04× (median **3.83×**) |
| quantized, fp16 baseline fp32-in (its autocast cast is inside the timed region) | 6 | median 2.24× | median 4.62× |
| **unquantized controls with matching input dtype** | 7 | 1.00, 1.00, 1.00, 1.01, 0.99, 1.01, 1.00× | 1.00, 1.00, 1.00, 1.02, 0.99, 1.01, 0.99× |

The control row is the load-bearing one: the same `torch_conv2d_fp16` on the same input times identically
in all three arms, which is what shows the layout normalization matched real layers rather than
coincidentally-shaped ones. Best single layer: `K=768 C=768 3×3 @ 8×8`, **4.74×** at int4.

**Why 3.83× per kernel becomes 1.78× end to end.** The chain is explicit: 3.83× on the 20 quantized conv
layers → 3.10× on the whole conv suite (13 unquantized convs dilute it) → 2.03× on the profiled matmul
bucket (which also carries the projection GEMMs and the fp16 fallbacks) → 1.78× on the wall clock,
because that bucket is only 45.6% of fp16's run. No step in that chain is a loss of efficiency; it is
Amdahl's law applied four times.

### 3b. Attention, matched by T

Corrected 2026-08-16. The previous version keyed on Q, and the `_qout` kernels take **token-major** Q, so
they landed in phantom buckets and were dropped — the T=1024 row was comparing fp16's 25 calls against
int8's 15, silently omitting `..._qout_hd24`, the most expensive kernel in the suite. Keyed on K, every
record is now assigned and the rows sum to the suite totals exactly. µs/call is call-weighted.

| T | hd_pad fp16→int8/int4 | calls f/8/4 | fp16 ms | int8 ms | int4 ms | µs/call | int8 | int4 | noise |
|--:|---|---|--:|--:|--:|---|--:|--:|---|
| 1024 | 24→32/64 | 25/25/25 | 50.92 | 42.05 | 42.16 | 2036.9→1682.0→1686.4 | **1.21×** | **1.21×** | — |
| 256 | 48→64/64 | 25/25/25 | 8.72 | 6.41 | 5.25 | 348.7→256.3→210.1 | **1.36×** | **1.66×** | **int4 NOISY** |
| 64 | 48→64/64 | 25/25/25 | 2.27 | 1.08 | 1.03 | 90.7→43.1→41.3 | **2.10×** | **2.20×** | — |
| 16 | 96→96/96 | 25/25/25 | 1.19 | 1.36 | 1.38 | 47.8→54.6→55.4 | **0.88×** | **0.86×** | — |
| 4 | 96→96/96 | 5/5/5 | 0.24 | 0.18 | 0.18 | 47.8→36.9→36.5 | 1.30× | 1.31× | **fp16 NOISY** |
| **total** | | | **63.34** | **51.08** | **50.01** | | **1.24×** | **1.27×** | |

Attention is the weakest block in the pipeline, and two separate things make it so.

**int8: padding.** The flash kernels take a padded head dim, so at the dominant T=1024 route they move 32
values per row where fp16 moves 24 — a third more bytes — and net 1.21×. That route is 50.9 of the
suite's 63.3 fp16 ms, so it sets the suite number. It already has a hand-written `_hd24` specialization
and an 8-byte loader that was built and refuted; the padding is structural to the MMA fragment layout, not
a missing optimization.

**int4: there is no int4 attention datapath.** Every operand in the int4 arm is `torch.int8`, the hd24
route's profiled kernel is literally `flash_attn_int8_mma_kernel_t`, and V stays int8 in both quantized
arms (`gemm_w4a4_awq_qkv_i4qk_i8v_layouts` — i4 for Q/K, i8 for V). So the only thing int4 can win is Q/K
bytes, and at T=1024 it wins none: hd=24 pads to 64 int4 = the same 32 B/row as int8's pad-to-32. It only
shows up at hd=48, where int8 pads to 64 B and int4 to 32 B — the 1.66× at T=256, whose two contributing
records are both flagged NOISY.

**T=16 is a sign error, not a fallback.** Only 15 of its 25 calls fall back to `torch_sdpa_fp16`; the
other 10 run `flash_attn_int8_qi8packed_small_qout` at ~65 µs against sdpa's ~48. Sending all 25 to sdpa
recovers ~0.17 ms/sample.

Per-kernel descriptions for all 28 entry points, including what each `_vt` / `_static` / `_qout` /
`_evt_*` suffix changes, are in [KERNEL_BREAKDOWN.md](KERNEL_BREAKDOWN.md).

## 4. Warm-up

MoDiff's t=T warm-up runs `_forward_first_step` on every modulated conv: one quantize on the calibrated
grid plus 4 residual rounds, so 5 convs where a steady step runs 1. Per-UNet-forward CUDA-event timing,
after one discarded sample so autotune and the attention scale freeze are already settled.

| mode | step 0 | steady median | excess | of a 200-step sample |
|---|--:|--:|--:|--:|
| fp16 | 103.7 | 101.75 | +1.9 | — |
| W8A8 PTQ | 69.9 | 70.67 | −0.8 | — (no MoDiff warm-up by construction) |
| **W8A8 MoDiff** | **735.5** | 72.61 | **+662.9** | **4.37%** |
| W4A4 PTQ | 56.7 | 57.44 | −0.8 | — |
| **W4A4 MoDiff** | **673.6** | 58.11 | **+615.4** | **5.04%** |

The PTQ arms are the control and they come out at −0.8 ms, i.e. step 0 is *not* special for anything
except MoDiff. The cost is **per cold sample** and scales as 1/steps: 4–5% at 200 steps, ~17–20% at 50.

**§1's ms/step does not contain it.** `_forward_first_step` fires only when the `a_hat` cache is absent
or misshapen, so it is paid once per cold sample and whether a sample is cold depends on the caller —
measured by counting calls: the e2e bench (repeated `sample()`, no reset) pays it 70 / 0 / 0 times; every
quality harness (which must reset, since a stale cache produces NaN latents) pays it 70 / 70 / 70.

## 5. What actually comes out

Paired samples: `generate_fid_samples.py` drives every mode with the same seed sequence, so each column
is one noise draw rendered five ways and any difference down a column is the mode. The indices are the
first six in sorted order — with a paired set there is nothing to select for.

![samples](plots/06_samples.png)

Same four latents, 128×128 center crop, to make the texture visible:

![zoom](plots/07_samples_zoom.png)

| mode | FID vs real (10k) | FID vs fp16 | mean \|Δ\| vs fp16, 500 paired images | quantization error removed |
|---|--:|--:|--:|--:|
| fp16 | 7.803 | 0.000 | 0.00/255 *(control)* | — |
| W8A8 PTQ | 16.366 | 6.394 | 15.05/255 | — |
| **W8A8 MoDiff** | **7.802** | **0.175** | **1.95/255** | **97.3%** |
| W4A4 PTQ | 277.963 | 277.981 | 40.74/255 | — |
| W4A4 MoDiff | 200.139 | 191.092 | 29.94/255 | 31.3% |

**W8A8 + MoDiff is the result.** FID 7.802 against the fp16 model's 7.803, at 1.41× the speed. The grid
shows why the FID parity is real and not an averaging artifact: W8A8 PTQ visibly *changes the image* —
seed #1's tower count, seed #2's roofline, seed #5's whole façade — while W8A8 MoDiff reproduces fp16's
composition down to individual windows, at 1.95/255 mean pixel distance versus PTQ's 15.05.

**W4A4 is not usable at either setting, and the grid says so plainly.** W4A4 PTQ produces smooth brown
fields with no structure at all; MoDiff recovers coarse church-shaped layout and 31% of the FID gap but
nothing at texture scale. The diagnosis is in
[fid_2026-08-05/FINDINGS.md](../fid_2026-08-05/FINDINGS.md): at W4A4 the dominant error is in the
**weights**, and MoDiff is an activation method — it cannot address what is broken there. The speed
column in §1 is therefore the honest answer to "why keep W4A4": it is where the kernel work pays off
(3.83× per conv), and it is waiting on a weight-side method.

Caveat carried from that report: FID at 10k is biased upward relative to the standard 50k, so these
absolute values compare to each other and not to published numbers. What 50k would cost is in
[FID_50K_ESTIMATE.md](FID_50K_ESTIMATE.md) (~1.5 h/mode, and 68–78% of that is not the UNet).

## 6. Reproduce

```bash
python integration/benchmarks/report/e2e_three_mode_bench.py --steps 200 --batch 128
```
```bash
python docs/bench_report_2026-08-13_postzp/scripts/kernel_speedup.py
```
```bash
python docs/bench_report_2026-08-13_postzp/scripts/warmup_cost.py
```
```bash
python docs/bench_report_2026-08-13_postzp/scripts/sample_grid.py
```

[SUMMARY.pdf](SUMMARY.pdf) is this file rendered for circulation, figures inline:

```bash
python docs/bench_report_2026-08-13_postzp/scripts/md_to_pdf.py
```

The first three want an idle GPU (~6–10 min each). `sample_grid.py` is CPU-only and reads the existing
`/workspace/fid` images; it asserts the columns are present in all five folders and that the fp16 control
comes out at exactly 0, which is what makes the pairing claim checkable rather than assumed.

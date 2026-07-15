# MoDiff modes on ResNet-50 — efficiency & a divergence finding

Companion to `REPORT.md` / `KERNEL_BENCHMARK.md`. We built MoDiff (temporal error-compensated
modulation — `o_hat`/`a_hat` delta caching) into the ResNet-50 pipeline as `int8_modiff` /
`int4_modiff` and measured **efficiency (speed + memory)** on a denoising-style correlated input
sequence. Harness: `integration/benchmarks/benchmark_resnet50_modiff.py`; build via
`build_quantized(..., modiff=True)` (bare torchvision model, stock `forward()` — MoDiff is
incompatible with the fullchain wrappers).

## Headline finding: MoDiff **diverges** on ResNet-50

MoDiff is designed for diffusion's sampler-renormalized trajectory. Driven through the stock ResNet
forward, the per-conv `o_hat` accumulation **explodes geometrically across the sequence** — and this is
**structural, not correlation-dependent**:

| input frame-to-frame diff | int8_modiff final-frame rel-vs-fp16 |
|---|--:|
| 5.0% (σ 0.6→0.02) | diverges → NaN |
| 0.45% | diverges → NaN |
| 0.05% | diverges → NaN |
| **0.02% (≈ static)** | **still diverges → NaN** |

A *single* MoDiff conv is numerically near-exact — the unit gates now pass on accuracy, **int8 rel 0.012
and int4 rel 0.223** vs the fp32 conv (at the int8/int4 quantization floor) — yet the **full network still
diverges to NaN**. That gap is the whole point: the divergence is **purely structural**, not a
first-step/calibration artifact. ~50 stateful convs coupled through the Bottleneck residual-add + ReLU
make each conv's accumulating `o_hat` compound across both layers and timesteps. MoDiff's telescoping
`o_hat_t = A(Q(a_t − a_hat_{t+1})) + o_hat_{t+1}` only stays bounded under diffusion's renormalized
sampling loop; a feedforward residual CNN has no such renormalization. **Conclusion: MoDiff modes are not
numerically usable on ResNet-50.**

(The earlier "int4 first-step ~13× / rel ~1.0" symptom was **not** a kernel bug — `_int4_conv` matches the
baseline `_conv_from_int4`/`_int8_conv` convention exactly. It was a **calibration-state inconsistency**
in the test harness `_calib_conv`: a SmoothQuant-derived static scale left desynced from the cached dequant
alpha, which int8's 8-bit range masked and int4's 4-bit range did not. Fixed in `d77c516`; the single-conv
int4 gate now passes on numerics. Note the ResNet `build_quantized` MoDiff path still shows a
mis-calibrated first frame in `--validate` — a separate builder-calibration issue — but it is moot here
because the network diverges regardless.)

## Efficiency (speed + memory) — kernel-path cost

The MoDiff kernels execute at a fixed cost regardless of the divergent values, so the efficiency numbers
are still meaningful as a characterization of the MoDiff path (they are **not** a usable-mode result):

![modiff efficiency](09_modiff_efficiency.png)

ResNet-50, A40, batch 64, T=24-frame sequence, per-frame latency:

| mode | ms / frame | vs fp16 | extra cache | first-step ms |
|---|--:|--:|--:|--:|
| fp16 | 21.85 | 1.00× | 0 | — |
| int8_fullchain | 15.26 | 1.43× | 0 | — |
| int4_fullchain | 14.13 | 1.55× | 0 | — |
| **int8_modiff** | **36.15** | **0.60×** | **2.48 GB** | 315.8 |
| **int4_modiff** | **33.07** | **0.66×** | **2.48 GB** | 306.7 |

**Reading this:**
- MoDiff modes are **~2.4× slower than fp16 and ~4× slower than the fullchain**. The `o_hat` path (per
  conv: `step1_static_quantize` delta-quantize → `conv2d_int{8,4}_fprop_o_hat` accumulate, plus fp32
  intermediate caches) does *more* work than either fp16 (cuDNN conv+ReLU) or the deep-fused/chained
  fullchain — and it skips no convolutions. This matches the codebase's own note that MoDiff is slower
  than the baseline; its payoff is accuracy, not speed.
- **~2.5 GB of cache** (batch 64) — the `a_hat` (input-sized) + `o_hat` (output-sized) fp16 buffers per
  converted conv, ≈ 2× the activation footprint. Confirmed against the analytical sum.
- The **first step is ~315 ms** (one-time warm-up: 3 successive-approximation refinement iterations across
  all convs), ~9× a steady-state frame.

## What we changed
- `integration/benchmarks/benchmark_resnet50.py`: `build_quantized(..., modiff=False)` param (leaves
  `enable_modiff(modiff)`; existing modes unchanged, default False).
- `integration/benchmarks/benchmark_resnet50_modiff.py` (new): denoising-sequence generator, stateless vs
  sequence-driven timing, cache-memory measurement, and a `--validate` path (dispatch counts, per-frame
  fidelity curve, reset isolation, memory).
- `integration/tests/test_kernel_correctness.py`: `test_int8_modiff_conv` and `test_int4_modiff_conv` —
  both now gate on the state machine **and** numerics (int8 rel 0.012, int4 rel 0.223) after the
  `_calib_conv` calibration-consistency fix (`d77c516`). Gate is ALL PASS.

## Recommendation
Do **not** ship MoDiff modes for ResNet/CNN inference — they diverge and are slower + memory-heavy. MoDiff
remains the right tool for its intended target (the diffusion UNet, where it is validated). If temporal
quantization amortization is wanted for CNN/video streaming, it needs a bounded formulation (e.g. periodic
re-anchoring / renormalization of `o_hat`, or applying the delta only where a residual path doesn't
re-accumulate it) — a research change, not a benchmark.

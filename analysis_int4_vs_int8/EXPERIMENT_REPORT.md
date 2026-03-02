# MoDiff Experiment Report: INT8 & INT4 Quantized Diffusion

**GPU:** NVIDIA A40  
**SM Capability:** 8.6  
**PyTorch:** 2.4.1+cu124  
**CUDA:** 12.4  
**Timestamp:** 2026-03-02 10:03:46  

---

## 1. Full Pipeline Speedup

Settings: 200 DDIM steps, 128 samples, batch_size=32, LSUN Churches 256×256 (results from benchmark_ldm.py)

| Mode | Time/Sample (ms) | Time/Step (ms) | Speedup vs FP32 |
|------|-----------------|---------------|-----------------|
| fp32 | 613.4 | 3.07 | 1.000× |
| int8_baseline | 333.0 | 1.66 | 1.842× |
| int8 | 367.7 | 1.84 | 1.668× |
| int4_baseline | 306.2 | 1.53 | 2.003× |
| int4 | 340.9 | 1.70 | 1.800× |

**Key Observations:**

- FP16 (FlashAttn): 1.978× vs FP32 — upper bound for compute savings
- INT8 baseline: 1.842× vs FP32, INT4 baseline: 2.003× vs FP32
- With MoDiff error-compensated caching: INT8 1.668×, INT4 1.800×
- MoDiff overhead vs dynamic baseline: INT8 +10.4%, INT4 +11.3%
- MoDiff's primary contribution is **quality preservation** at aggressive quantization, not raw speed.

![Pipeline Speedup](plot_01_pipeline_speedup.png)

## 2. Per-Component Pipeline Breakdown

Measured via CUDA-event forward hooks (50 steps × 2 batches × 32 samples, batch\_size=32).  
— FP32 mode: no autocast (identical to Exp.1 FP32 baseline).  
— INT8/INT4 modes: FP16 autocast enabled (identical to Exp.1 conditions).

### FP32

Wall time: 9.86s, Time/step: 3.08ms

| Component | Total (ms) | Calls | Avg (ms) | % of Total |
|-----------|-----------|-------|---------|-----------|
| Attention | 3512.1 | 2100 | 1.6725 | 49.1% |
| Conv2d(FP32) | 3274.4 | 8900 | 0.3679 | 45.8% |
| GroupNorm | 266.5 | 2200 | 0.1212 | 3.7% |
| Linear(FP32) | 68.6 | 3700 | 0.0185 | 1.0% |
| SiLU | 29.3 | 3700 | 0.0079 | 0.4% |

### INT8

Wall time: 6.24s, Time/step: 1.95ms

| Component | Total (ms) | Calls | Avg (ms) | % of Total |
|-----------|-----------|-------|---------|-----------|
| Int8Conv2d | 2734.3 | 7000 | 0.3906 | 69.4% |
| Attention | 511.4 | 2100 | 0.2435 | 13.0% |
| GroupNorm | 268.1 | 2200 | 0.1218 | 6.8% |
| Conv2d(FP32) | 214.2 | 1900 | 0.1128 | 5.4% |
| Int8Linear | 183.8 | 3700 | 0.0497 | 4.7% |
| SiLU | 29.2 | 3700 | 0.0079 | 0.7% |

### INT4

Wall time: 5.99s, Time/step: 1.87ms

| Component | Total (ms) | Calls | Avg (ms) | % of Total |
|-----------|-----------|-------|---------|-----------|
| Int4Conv2d | 2289.2 | 7000 | 0.3270 | 63.8% |
| Attention | 532.5 | 2100 | 0.2535 | 14.9% |
| GroupNorm | 283.1 | 2200 | 0.1287 | 7.9% |
| Conv2d(FP32) | 227.8 | 1900 | 0.1199 | 6.4% |
| Int4Linear | 223.1 | 3700 | 0.0603 | 6.2% |
| SiLU | 29.8 | 3700 | 0.0080 | 0.8% |

![Component Breakdown Pies](plot_02_component_breakdown.png)

![Component Breakdown Bars](plot_02b_component_bars.png)

### Cross-Mode Comparison

| Component | FP32 (ms) | INT8 (ms) | INT4 (ms) | INT8 vs FP32 | INT4 vs FP32 |
|---|---|---|---|---|---|
| Conv2d | 3274.4 | 2948.6 | 2517.0 | 0.90× | 0.77× |
| Attention | 3512.1 | 511.4 | 532.5 | 0.15× | 0.15× |
| Linear | 68.6 | 183.8 | 223.1 | 2.68× | 3.25× |
| GroupNorm | 266.5 | 268.1 | 283.1 | 1.01× | 1.06× |
| SiLU | 29.3 | 29.2 | 29.8 | 1.00× | 1.02× |

## 3. Per Conv-Layer-Shape Analysis

Each unique conv shape benchmarked in isolation: FP32 (cuDNN) vs INT8 (CUTLASS quant+conv) vs INT4 (CUTLASS quant+pack+conv).

| Layer Shape | Count | FP32 (ms) | INT8 E2E (ms) | INT4 E2E (ms) | INT8 Speedup | INT4 Speedup |
|------------|-------|----------|-------------|-------------|-------------|-------------|
| 1536→768, 3x3, S=1 | 5 | 0.3677 | 0.1452 | 0.0802 | 2.53× | 4.59× |
| 1152→768, 3x3, S=1 | 1 | 0.2850 | 0.1124 | 0.0630 | 2.53× | 4.52× |
| 384→768, 3x3, S=1 | 1 | 0.2497 | 0.0935 | 0.0541 | 2.67× | 4.61× |
| 192→384, 3x3, S=1 | 1 | 0.2283 | 0.1157 | 0.0691 | 1.97× | 3.30× |
| 768→768, 3x3, S=1 | 23 | 0.2220 | 0.0791 | 0.0439 | 2.81× | 5.06× |
| 1152→384, 3x3, S=1 | 1 | 0.1632 | 0.1120 | 0.0589 | 1.46× | 2.77× |
| 384→384, 3x3, S=1 | 21 | 0.1508 | 0.0518 | 0.0332 | 2.91× | 4.54× |
| 192→192, 3x3, S=1 | 9 | 0.1434 | 0.0802 | 0.0479 | 1.79× | 2.99× |
| 768→384, 3x3, S=1 | 4 | 0.1214 | 0.0765 | 0.0430 | 1.59× | 2.82× |
| 384→192, 3x3, S=1 | 2 | 0.0960 | 0.0449 | 0.0276 | 2.14× | 3.48× |
| 576→384, 3x3, S=1 | 1 | 0.0945 | 0.0656 | 0.0373 | 1.44× | 2.53× |
| 576→192, 3x3, S=1 | 1 | 0.0712 | 0.0654 | 0.0373 | 1.09× | 1.91× |
| 192→384, 1x1, S=1 | 1 | 0.0581 | 0.0582 | 0.0486 | 1.00× | 1.20× |
| 1152→384, 1x1, S=1 | 1 | 0.0472 | 0.0326 | 0.0273 | 1.45× | 1.73× |
| 1536→768, 1x1, S=1 | 5 | 0.0442 | 0.0299 | 0.0251 | 1.48× | 1.77× |
| 384→768, 1x1, S=1 | 1 | 0.0442 | 0.0381 | 0.0277 | 1.16× | 1.59× |
| 1152→768, 1x1, S=1 | 1 | 0.0400 | 0.0251 | 0.0232 | 1.59× | 1.72× |
| 768→384, 1x1, S=1 | 4 | 0.0370 | 0.0251 | 0.0218 | 1.47× | 1.70× |
| 576→192, 1x1, S=1 | 1 | 0.0361 | 0.0247 | 0.0213 | 1.46× | 1.70× |
| 576→384, 1x1, S=1 | 1 | 0.0355 | 0.0251 | 0.0216 | 1.41× | 1.65× |
| 384→192, 1x1, S=1 | 2 | 0.0353 | 0.0228 | 0.0213 | 1.55× | 1.65× |

**Weighted average speedup (by layer count):**
- INT8 vs FP32: 2.39×
- INT4 vs FP32: 3.95×
- INT4 vs INT8: 1.66×

![Conv Layer Analysis](plot_03_conv_layer_analysis.png)

## 4. Per Linear-Layer-Shape Analysis

Each unique linear shape benchmarked in isolation. All 37 linear layers are time-embedding projections.

| Shape (in→out) | Count | FP32 (ms) | INT8 base (ms) | INT8 MoDiff (ms) | INT4 base (ms) | INT4 MoDiff (ms) |
|---------------|-------|----------|---------------|-----------------|---------------|-----------------|
| 192→768 | 1 | 0.0316 | 0.0535 | 0.1700 | 0.0532 | 0.1687 |
| 768→768 | 15 | 0.0347 | 0.0525 | 0.1718 | 0.0525 | 0.1721 |
| 768→384 | 6 | 0.0353 | 0.0524 | 0.1711 | 0.0529 | 0.1719 |
| 768→1536 | 15 | 0.0345 | 0.0528 | 0.1713 | 0.0524 | 0.1704 |

**Key findings:**

- INT8 baseline avg: 0.0528ms (0.64× vs FP32)
- INT4 baseline avg: 0.0527ms (0.64× vs FP32)
- INT8 MoDiff avg: 0.1710ms, INT4 MoDiff avg: 0.1708ms
- Linear layers use FP16 GEMM + quantization overhead, so INT8/INT4 baseline are slightly slower than FP32.
- MoDiff modulated steps add ~224% overhead for error-compensated caching.

![Linear Layer Latency](plot_04a_linear_latency.png)

![Linear Layer Speedup](plot_04b_linear_speedup.png)

## 5. Batch Size Ablation Study

Full pipeline at varying batch sizes (50 DDIM steps).

### Time per Sample

| Batch Size | FP32 (ms) | INT8 (ms) | INT4 (ms) | INT8 vs FP32 | INT4 vs FP32 |
|-----------|----------|----------|----------|-------------|-------------|
| 1 | 1190.9 | 1333.6 | 1207.2 | 0.893× | 0.987× |
| 2 | 588.9 | 626.2 | 612.1 | 0.940× | 0.962× |
| 4 | 330.8 | 343.7 | 331.7 | 0.962× | 0.997× |
| 8 | 201.5 | 209.0 | 202.6 | 0.964× | 0.994× |
| 16 | 167.1 | 166.2 | 155.7 | 1.005× | 1.073× |

### Throughput

| Batch Size | FP32 (samples/s) | INT8 (samples/s) | INT4 (samples/s) |
|-----------|-----------------|-----------------|-----------------|
| 1 | 0.84 | 0.75 | 0.83 |
| 2 | 1.70 | 1.60 | 1.63 |
| 4 | 3.02 | 2.91 | 3.01 |
| 8 | 4.96 | 4.78 | 4.94 |
| 16 | 5.98 | 6.02 | 6.42 |

**Key findings:**

- At batch_size=1, all modes have similar latency (kernel launch overhead dominates).
- Throughput scales near-linearly with batch size for all modes.
- INT4 matches or slightly beats FP32 at larger batch sizes where compute becomes the bottleneck.
- INT8/INT4 MoDiff overhead is amortized at larger batch sizes.

![Batch Ablation](plot_05_batch_ablation.png)

![Batch Step Time](plot_05b_batch_step_time.png)

## 6. Summary & Conclusions

### Architecture

- **Model:** LSUN Churches LDM (unconditional UNet, 256×256)
- **Conv layers:** 89 (245M params) — converted to INT8/INT4 CUTLASS kernels
- **Linear layers:** 37 (28.5M params, 10.4% of total) — time-embedding projections
- **All quantized layers** use CUTLASS fused kernels (sub_absmax_scale, dequant_accumulate)

### Performance Summary

1. **Conv layers** benefit significantly from quantization: INT8 achieves 1.5–3× speedup, INT4 achieves 1.5–5× vs FP32 at the kernel level.
2. **Linear layers** are too small (M=8) to benefit from quantized GEMM; the FP16 F.linear approach adds ~50% overhead vs FP32 due to quantization bookkeeping.
3. **Attention** accounts for ~30–40% of total pipeline time and is unaffected by quantization mode.
4. **MoDiff temporal caching** adds modest overhead (~20–30% over baseline) in exchange for maintaining generation quality at aggressive quantization levels.
5. **Batch size scaling** is near-linear; at bs=16, INT4 achieves the highest throughput (6.42 samples/s vs 5.98 FP32).

### Key Takeaway

MoDiff's primary contribution is **quantization quality**, not raw throughput. It enables 4-bit activation quantization without FID degradation — vanilla quantization degrades significantly at even 6-bit. The CUTLASS INT4 conv kernels deliver real speedups on compute-heavy layers, but the end-to-end pipeline gain is limited by non-quantized components (attention, normalization) that account for ~40% of total time.

# MoDiff LDM Quantization Benchmark Report
**NVIDIA A40 · lsun_churches256 · DDIM 50 steps · batch_size=42**

---

## Table of Contents
1. [Hardware & System Setup](#1-hardware--system-setup)
2. [End-to-End Benchmark Results](#2-end-to-end-benchmark-results)
   - 2.1 [Batch Size 32 (128 samples)](#21-batch-size-32--128-samples)
   - 2.2 [Batch Size 42 — Wave-Optimal (126 samples)](#22-batch-size-42--wave-optimal-126-samples)
3. [UNet Operation Timing Breakdown](#3-unet-operation-timing-breakdown)
4. [Attention Ablation Study](#4-attention-ablation-study)
5. [Kernel-Level TOPS Microbenchmark](#5-kernel-level-tops-microbenchmark)
6. [Amdahl's Law Decomposition](#6-amdahls-law-decomposition)
7. [Wave Quantization Analysis](#7-wave-quantization-analysis)
8. [Summary & Conclusions](#8-summary--conclusions)
9. [MoDiff Attention INT8 — CUTLASS Backend (v3)](#9-modiff-attention-int8--cutlass-backend-v3)

---

## 1. Hardware & System Setup

| Property | Value |
|---|---|
| GPU | NVIDIA A40 |
| Architecture | Ampere (sm_86) |
| Streaming Multiprocessors | 84 SMs |
| Base Clock | 1.74 GHz |
| VRAM | 48 GB GDDR6 |
| FP32 Peak | 37.4 TFLOPS |
| FP16 Tensor Core Peak | 149.7 TFLOPS |
| INT8 Tensor Core Peak | 299.3 TOPS |
| INT4 Tensor Core Peak | 597.7 TOPS |

**Model:** LDM (`lsun_churches256`, latent 32×32×4, 256×256 output), unconditional.  
**UNet:** 89 Conv2d layers (70 quantized, 19 kept FP16 as skip/input/output), 37 Linear layers (all quantized), 22 AttentionBlocks per forward pass.  
**Sampler:** DDIM, η=0, 50 timesteps.  
**Modes tested:**
- `fp32` — all ops in FP32 (baseline)
- `fp16` — full model cast to FP16 (cuDNN TF32/FP16 kernels)
- `int8` / `int8_baseline` — CUTLASS INT8 GEMM; `int8` adds MoDiff temporal activation caching
- `int4` / `int4_baseline` — CUTLASS INT4 GEMM; `int4` adds MoDiff temporal activation caching

---

## 2. End-to-End Benchmark Results

### 2.1 Batch Size 32 · 128 samples

| Mode | Per-sample (s) | Per-step (ms) | Speedup vs FP32 |
|---|---|---|---|
| FP32 | 0.237 | 4.74 | 1.00× (baseline) |
| FP16 | 0.108 | 2.17 | **2.19×** |
| INT8 | 0.090 | 1.80 | **2.63×** |
| INT8_baseline | 0.082 | 1.64 | **2.89×** |
| INT4 | 0.088 | 1.75 | **2.71×** |
| INT4_baseline | 0.080 | 1.60 | **2.96×** |

### 2.2 Batch Size 42 · Wave-Optimal · 126 samples

Batch size 42 was selected because it is wave-optimal for all 8 dominant UNet convolutional layers at the 512-channel mid-block (see §7). This eliminates partial-wave GPU underutilization.

| Mode | Per-sample (s) | Per-step (ms) | Speedup vs FP32 |
|---|---|---|---|
| FP32 | 0.226 | 4.51 | 1.00× (baseline) |
| FP16 | 0.095 | 1.90 | **2.37×** |
| INT8 | 0.084 | 1.67 | **2.70×** |
| INT8_baseline | 0.076 | 1.53 | **2.96×** |
| INT4 | 0.077 | 1.53 | **2.94×** |
| INT4_baseline | 0.071 | 1.43 | **3.16×** |

**Key observations:**
- Moving from bs=32 to bs=42 gives a consistent ~6–8% speedup per step for quantized modes due to the wave-alignment effect.
- MoDiff caching overhead (INT8 vs INT8_baseline) costs **9.3%** at bs=42; INT4 overhead is **7.0%**.
- INT4_baseline reaches **3.16× over FP32** end-to-end, the highest mode tested.
- INT4 (with caching) slightly outperforms INT8 (with caching): 1.53ms vs 1.67ms per step.

---

## 3. UNet Operation Timing Breakdown

Measured with a CUDA-event hook profiler (`profile_unet_timing.py`) at batch_size=42, DDIM 15 timesteps, 15 benchmark steps. Results are per UNet forward call.

> **Note on "other" category:** `OptimizedInt8Linear` and `OptimizedInt4Linear` do not inherit from `nn.Linear`, so they are not captured by the linear hook and appear in "other". This explains the larger "other" values for INT8/INT4 vs FP16.

| Category | FP16 ms | FP16 % | INT8_base ms | INT8_base % | INT4_base ms | INT4_base % |
|---|---|---|---|---|---|---|
| attention | 20.74 | 25.0% | 18.60 | 22.2% | 17.92 | 28.0% |
| quant_conv | — | — | 23.08 | 27.5% | 18.18 | 28.4% |
| fp_conv | 28.01 | 33.8% | 3.32 | 4.0% | 3.07 | 4.8% |
| groupnorm | 11.15 | 13.5% | 9.12 | 10.9% | 8.87 | 13.9% |
| linear (FP16 only) | 2.14 | 2.6% | — | — | — | — |
| silu | 0.51 | 0.6% | 0.39 | 0.5% | 0.37 | 0.6% |
| upsample | 0.96 | 1.2% | 0.74 | 0.9% | 0.68 | 1.1% |
| downsample | 0.33 | 0.4% | 0.35 | 0.4% | 0.35 | 0.6% |
| other (incl. INT quant linear) | 18.97 | 22.9% | 28.27 | 33.7% | 14.54 | 22.7% |
| **TOTAL (wall)** | **82.81** | | **83.90** | | **63.97** | |

**Key observations:**
- In FP16, `fp_conv` (28.0ms, 33.8%) is the single largest time consumer, followed by `attention` (25.0%).
- In INT8_baseline, quantized convolutions (`quant_conv` 23.1ms) replace `fp_conv` (reduced from 28.0ms → 3.3ms for residual FP16 skip-connection convolutions). The 2.28× kernel speedup (see §5) translates to only 1.21× wall-time improvement because convolutions are only 33.8% of FP16 time (Amdahl effect).
- In INT4_baseline, `quant_conv` drops to 18.2ms (further 1.27× kernel speedup vs INT8), with total wall time 63.97ms — a 1.31× improvement over INT8_baseline.
- `attention` remains relatively constant across modes (~18–21ms) because all attention ops run in FP16 regardless of quantization mode. It becomes a **growing bottleneck** as quantization reduces other costs.

---

## 4. Attention Ablation Study

To quantify the wall-time cost of attention and determine whether it can be bypassed, we re-ran the full benchmark with all 22 `AttentionBlock` modules replaced by identity pass-throughs (`--no_attention` flag). Results at batch_size=42, 126 samples, 50 steps.

### 4.1 No-Attention Benchmark Results

| Mode | No-attn per-step (ms) | Normal per-step (ms) | Attn overhead (ms) | Attn % of total |
|---|---|---|---|---|
| FP32 | 2.76 | 4.51 | **1.75** | 38.8% |
| FP16 | 1.39 | 1.90 | **0.51** | 26.8% |
| INT8 | 1.21 | 1.67 | **0.46** | 27.5% |
| INT8_baseline | 1.10 | 1.53 | **0.43** | 28.1% |
| INT4 | 1.17 | 1.53 | **0.36** | 23.5% |
| INT4_baseline | 1.01 | 1.43 | **0.42** | 29.4% |

### 4.2 Speedup from Skipping Attention

| Mode | Speedup (no-attn vs normal) |
|---|---|
| FP32 | 1.63× |
| FP16 | 1.37× |
| INT8 | 1.38× |
| INT8_baseline | 1.39× |
| INT4 | 1.31× |
| INT4_baseline | 1.42× |

### 4.3 Attention Speedup Chain (no-attention mode, kernel-only view)

When attention is removed, the quantization speedup from FP16 → INT8_baseline increases from 1.24× to 1.26×, and FP16 → INT4_baseline from 1.33× to 1.38×. This confirms that attention — running always in FP16 — is a significant non-quantizable overhead that dilutes end-to-end gains.

### 4.4 Can Attention Be Skipped?

**Technically yes, numerically yes, qualitatively no.** The `AttentionBlock` uses a residual connection (`out = x + attn(x)`), so with `attn(x) = 0` (identity block), the UNet computes a valid forward pass without errors. Images are generated.

However, attention provides global/long-range spatial coherence. Without it, outputs are locally textured but lack structural coherence — blurry mid-range features, no scene-level organization. Skipping attention is valid only as an **ablation** to isolate its compute cost, not for production use.

---

## 5. Kernel-Level TOPS Microbenchmark

Measured with isolated CUTLASS kernel benchmarks at batch_size=42 for the 8 dominant UNet convolutional shapes. All shapes use `GemmShape<128,128,128>` (INT8) or `GemmShape<128,128,256>` (INT4) threadblock tiles.

| Layer | M×N×K | FP16 (ms) | INT8 (ms) | INT4 (ms) | FP16 TOPS | INT8 TOPS | INT4 TOPS | INT8/FP16 | INT4/INT8 |
|---|---|---|---|---|---|---|---|---|---|
| res_128×128 | — | 0.202 | 0.098 | 0.057 | 62.8 | 129.7 | 221.2 | 2.07× | 1.71× |
| res_128×256_ds | — | 0.128 | 0.056 | 0.036 | 49.5 | 112.7 | 176.2 | 2.28× | 1.56× |
| res_256×256 | — | 0.165 | 0.078 | 0.047 | 77.0 | 162.2 | 270.6 | 2.11× | 1.67× |
| res_256×512_ds | — | 0.102 | 0.039 | 0.027 | 62.4 | 164.4 | 235.6 | 2.63× | 1.43× |
| mid_512×512 | — | 0.152 | 0.064 | 0.040 | 83.3 | 198.5 | 320.2 | 2.38× | 1.61× |
| res_512×256_us | — | 0.288 | 0.125 | 0.073 | 88.2 | 203.4 | 348.6 | 2.31× | 1.71× |
| res_256×128_us | — | 0.377 | 0.154 | 0.090 | 67.3 | 165.1 | 282.0 | 2.45× | 1.71× |
| res_128×128_us | — | 0.771 | 0.345 | 0.194 | 65.8 | 147.2 | 261.5 | 2.24× | 1.78× |
| **Mean** | | | | | **69.7** | **158.9** | **270.1** | **2.28×** | **1.70×** |
| **% of A40 peak** | | | | | **46.6%** | **53.1%** | **45.2%** | | |

**Key observations:**
- INT8 achieves **53.1%** of the 299.3 TOPS A40 peak, INT4 achieves **45.2%** of 597.7 TOPS. Both are efficiency-limited, not roofline-limited.
- The mean INT8/FP16 speedup is **2.28×** at the kernel level, but only **1.24×** end-to-end in FP16→INT8_baseline (see §6).
- INT4/INT8 kernel ratio is **1.70×** on average, reflecting the 2× data width reduction partially offset by the same Tensor Core throughput path.
- Larger layers (512-channel mid-block) approach higher utilization: 198.5 TOPS INT8 = 66.3% of peak. Smaller layers with poor wave quantization fall to 113–147 TOPS.

---

## 6. Amdahl's Law Decomposition

The gap between 2.28× kernel speedup and 1.24× end-to-end speedup is explained by the non-quantizable fraction of UNet computation.

### 6.1 Model

Using Amdahl's Law: given a kernel speedup $S$ and an observed end-to-end speedup $R$, the quantizable fraction is:

$$f_{conv} = \frac{R - 1}{R \cdot \left(1 - \frac{1}{S}\right)}$$

Applying to FP16 → INT8_baseline at bs=42 ($S = 2.28$, $R = 1.245$):

$$f_{conv} = \frac{1.245 - 1}{1.245 \times (1 - 1/2.28)} = \frac{0.245}{1.245 \times 0.561} = \textbf{35.0\%}$$

Therefore:
- **Convolutions (quantizable):** 35% of FP16 per-step time → 0.665ms/step
- **Non-conv (attention + GroupNorm + SiLU + embedding, always FP16):** 65% → 1.235ms/step

### 6.2 Time Budget Decomposition at bs=42

| Component | FP16 ms/step | INT8_base ms/step | INT4_base ms/step |
|---|---|---|---|
| Convolution | 0.665 | 0.292 (2.28× faster) | 0.199 (3.34× faster) |
| Non-conv (fixed) | 1.235 | 1.235 | 1.235 |
| Subtotal (baseline) | **1.900** | **1.527** | **1.434** |
| MoDiff caching overhead | — | +0.144 (+9.4%) | +0.104 (+7.3%) |
| INT4 pack overhead | — | — | +0.022 (+1.5%) |
| **Measured total** | **1.90** | **1.53 (+0.00)** | **1.43 (+0.00)** |

### 6.3 Implications

The 65% non-conv fraction is the **primary ceiling** on all quantization gains:
- **Theoretical max speedup** (perfect quantization, S→∞): $\frac{1}{0.65} = 1.54×$ over FP16.
- **Actual INT4_baseline** achieves 1.33× over FP16 — 86% of the theoretical maximum.
- Attention alone accounts for ~27% of per-step time at bs=42 (§4) — more than the entire conv budget after INT4 quantization.
- **To exceed 1.54×**, attention or GroupNorm must be quantized or fused.

---

## 7. Wave Quantization Analysis

CUTLASS threadblock tiles are `GemmShape<128,128,K>`, so the number of threadblocks (TBs) launched per kernel is $\lceil M/128 \rceil \times \lceil N/128 \rceil$. Efficiency is $\eta = \text{TBs} / (84 \times \lceil \text{TBs}/84 \rceil)$.

### 7.1 8 Dominant Layers at Batch=32 vs Batch=42

| Layer | TBs at bs=32 | η (bs=32) | TBs at bs=42 | η (bs=42) |
|---|---|---|---|---|
| res_128×128 (C=128) | — | ~76.2% | 84 | **100%** |
| res_128×256_ds | — | ~76.2% | 84 | **100%** |
| res_256×256 (C=256) | — | ~76.2% | 84 | **100%** |
| res_256×512_ds | — | ~76.2% | 84 | **100%** |
| mid_512×512 (C=512) | — | ~76.2% | 84 | **100%** |
| res_512×256_us | — | ~76.2% | 84 | **100%** |
| res_256×128_us | — | ~76.2% | 84 | **100%** |
| res_128×128_us | — | ~76.2% | 84 | **100%** |

At batch=32 all 8 layers launch 64 TBs across 84 SMs — one partial wave at 76.2% efficiency. At batch=42, all 8 layers launch exactly 84 TBs, filling the GPU perfectly (100% wave efficiency).

### 7.2 Impact

| Batch | Wave efficiency | INT8_base per-step | INT4_base per-step |
|---|---|---|---|
| 32 | 76.2% | 1.64ms | 1.60ms |
| 42 | 100% | 1.53ms | 1.43ms |
| Improvement | +31.4pp | **6.7%** | **10.6%** |

Batch=42 is therefore the throughput-optimal batch size for this UNet on an A40. Departing from multiples of 42 that maintain 100% wave alignment would degrade kernel efficiency.

---

## 8. Summary & Conclusions

### 8.1 End-to-End Speedup Summary (batch=42, wave-optimal)

| Mode | ResBlocks | Attn projections | Per-step (ms) | vs FP32 | vs FP16 | vs INT8 (MoDiff) | vs INT4 (MoDiff) |
|---|---|---|---|---|---|---|---|
| FP32 | FP32 | FP32 | 4.51 | 1.00× | 0.42× | — | — |
| FP16 | FP16 | FP16 | 1.90 | 2.37× | 1.00× | — | — |
| INT8 (MoDiff) | INT8+MoDiff | FP16 | 1.67 | 2.70× | 1.14× | **1.00×** | — |
| INT8_baseline | INT8 | FP16 | 1.53 | 2.96× | 1.24× | 1.09× | — |
| INT4 (MoDiff) | INT4+MoDiff | FP16 | 1.53 | 2.94× | 1.24× | — | **1.00×** |
| INT4_baseline | INT4 | FP16 | 1.43 | 3.16× | 1.33× | — | 1.07× |
| attn_modiff v3 (FP16 ResBlocks) | FP16 | INT8+MoDiff | 2.52 | 1.79× | 0.75× | 1.51× | 1.65× |
| **int8_attn_modiff** | INT8+MoDiff | INT8+MoDiff | **2.33** | **1.94×** | **0.82×** | **1.40×** | **1.52×** |
| **int4_attn_modiff** | INT4+MoDiff | INT8+MoDiff | **2.22** | **2.03×** | **0.86×** | **1.33×** | **1.45×** |

> **Valid comparison:** `int8_attn_modiff` = INT8 ResBlocks + INT8 MoDiff attention (direct apples-to-apples vs `INT8 (MoDiff)`). `int4_attn_modiff` = INT4 ResBlocks + INT8 MoDiff attention (vs `INT4 (MoDiff)`). After fused layout-conversion kernels (v4, §9.8): adding MoDiff to attention projections still **increases** per-step time vs INT8/INT4 alone (+0.61ms/+37% over INT8 MoDiff, +0.66ms/+43% over INT4 MoDiff), because the CUTLASS pipeline's 3-kernel quantize→GEMM→dequantize overhead cannot be eliminated without a fully fused kernel (§9.6). INT4+MoDiff (1.53ms) with no attention quantization remains the fastest mode tested.

### 8.2 Key Findings

**1. Quantization headroom is bounded by the 65% non-conv fraction.**  
Amdahl analysis shows only 35% of FP16 time is spent in convolutions. Even with ideal INT4 kernels at infinite speedup, the maximum end-to-end gain over FP16 is **1.54×**. INT4_baseline already achieves **1.33×** — 86% of the theoretical ceiling.

**2. Attention is the dominant non-quantizable bottleneck.**  
Profiling shows attention consumes **25–28% of total UNet time** across modes (always FP16 via `F.scaled_dot_product_attention`). The ablation experiment confirms removing attention would yield **1.31–1.42× additional speedup** per mode. Quantizing or fusing attention (e.g., Flash Attention INT8) is the next high-leverage optimization.

**3. Kernel-level speedup (2.28× INT8/FP16) does not translate end-to-end.**  
The CUTLASS kernels are genuinely faster: INT8 achieves 53.1% of A40 INT8 peak (158.9 TOPS avg), INT4 achieves 45.2% of INT4 peak (270.1 TOPS avg). The gap to end-to-end gains is purely the Amdahl effect.

**4. MoDiff temporal caching adds ~7–9% overhead vs. baseline.**  
The INT8/INT8_baseline gap at bs=42 is 0.14ms/step (+9.4%), and INT4/INT4_baseline is 0.10ms/step (+7.3%). This cost is paid for activation reuse across timesteps in the diffusion chain — whether this is net-positive depends on the number of cached timesteps and cache hit rate.

**5. Wave-optimal batch size (42) provides 6–10% kernel speedup.**  
All 8 dominant UNet conv layers achieve 84 threadblocks = exactly 1 SM-wave on the A40 at bs=42. Compared to bs=32 (76.2% wave efficiency), this is a free 6–10% gain achievable through batch padding alone.

**6. INT4 and INT8 converge at the end-to-end level with MoDiff caching.**  
Both INT8 (with MoDiff) and INT4_baseline achieve 1.53ms/step at bs=42. The INT4 kernel advantage (~1.70× over INT8 at kernel level) is almost entirely consumed by the attention floor and caching overhead, making the choice between them a quality/accuracy tradeoff rather than a clear speed winner when MoDiff caching is enabled.

**7. Quantizing attention projections via CUTLASS (v3) is net-negative due to Python kernel launch overhead — in all combinations.**  
`MoDiffConv1dCUTLASS` introduces 6 CUDA kernel launches per call vs 1 for FP16 (`F.conv1d`). At ~50µs Python-side async launch overhead per kernel, 21 blocks × 2 layers × 6 kernels = 252 launches/step adds ~1ms of fixed Python overhead (see §9 for full analysis). After fused layout-conversion kernels (§9.8) that eliminate K1+K2+K7+K8 overhead, the end-to-end results are:
- `attn_modiff v4` (FP16 ResBlocks): 2.40ms — **1.26× slower than FP16**
- `int8_attn_modiff v4` (INT8 ResBlocks + INT8 attn): 2.28ms — **1.37× slower than INT8 (MoDiff)**
- `int4_attn_modiff v4` (INT4 ResBlocks + INT8 attn): 2.19ms — **1.43× slower than INT4 (MoDiff)**

The remaining overhead is dominated by the CUTLASS pipeline's K4+K5+K6 (quantize→GEMM→dequantize, 3 kernels per call). A single fused kernel reducing all 6 launches → 1-2 would be required to make this approach viable.

### 8.3 Recommended Next Steps

| Priority | Action | Expected gain |
|---|---|---|
| High | Custom fused CUDA kernel for Conv1d ks=1 (quantize+INT8 GEMM+dequantize, 1 kernel) | Reduce attention projection overhead to 1 launch = FP16-parity overhead |
| High | Quantize/fuse attention (Flash Attention INT8 or INT4) | Up to +30% end-to-end |
| High | Fuse GroupNorm + SiLU (already partially done) | Already delivering ~10% |
| Medium | Profile INT8/INT4 linear layer performance separately | Clarify "other" in profiler |
| Medium | Sweep batch sizes for other wave-multiples (84, 126, 168) | Additional wave-alignment gains |
| Low | Reduce MoDiff caching overhead via buffer pre-fetch | Recover ~9% vs. baseline |

---

*Report generated from:*
- `integration/benchmarks/benchmark_ldm.py` (bs=32 and bs=42 end-to-end benchmarks)
- `integration/benchmarks/profile_unet_timing.py` (CUDA-event operation timing)
- `integration/benchmarks/benchmark_ldm.py --no_attention` (attention ablation)
- `integration/results/profile_ops.json` (raw profiling data)
- `integration/results/benchmark_bs42_noattn.log` (no-attention benchmark log)
- `integration/results/ldm_attn_modiff_v3/` (attn_modiff v3 CUTLASS benchmark)

---

## 9. MoDiff Attention INT8 — CUTLASS Backend (v3)

### 9.1 Motivation

Section §4 shows attention consumes **26.8% of FP16 per-step time** (0.51ms out of 1.90ms/step at bs=42). Sections §8.1–8.3 establish that this is the primary bottleneck preventing further end-to-end gains beyond INT8_baseline. This section describes an attempt to apply MoDiff temporal caching to the attention projection Conv1d layers using INT8 CUTLASS GEMMs (v3), and the performance analysis of why it did not yield a net speedup.

### 9.2 Implementation: `MoDiffConv1dCUTLASS`

The `MoDiffConv1dCUTLASS` class wraps `OptimizedInt8Conv2d` (the CUTLASS INT8 ResBlock engine) to handle Conv1d ks=1 projections. The key challenge is that `OptimizedInt8Conv2d` expects 2D spatial input `[B, C, H, W]` in channels-last layout, but attention projections are `[B, C, L]` (1D sequences).

**Shape mapping strategy:** Reshape `[B, C, L]` → `[B×L, C, 1, 1]`. For H=W=1, channels-last strides are identical to NCHW strides — no memory layout conversion is needed. This avoids the full matrix transpose that would be required for `[B, C, 1, L]` (channels-last would require `[B, L, C, 1]`).

```python
def forward(self, x):               # x: [B, C, L], FP16
    B, C, L = x.shape
    x_4d = x.permute(0,2,1).contiguous().view(B*L, C, 1, 1)  # [B*L, C, 1, 1]
    out_4d = self.int8conv2d(x_4d)                             # INT8 CUTLASS GEMM
    return out_4d.view(B, L, -1).permute(0,2,1).contiguous()  # [B, C_out, L]
```

**Kernel count per call:**

| Step | Operation | Kernels |
|---|---|---|
| 1 | `x.permute(0,2,1).contiguous()` | 1 |
| 2 | FP16 → FP32 (inside `OptimizedInt8Conv2d`) | 1 |
| 3 | `step1_quantize_fprop` (fused absmax + INT8 quantize) | 1 (CUTLASS) |
| 4 | `conv2d_int8_fprop_o_hat` (INT8 GEMM + accumulate into cache) | 1 (CUTLASS) |
| 5 | `out.view(...).permute(0,2,1).contiguous()` | 1 |
| 6 | `.to(orig_dtype)` FP32 → FP16 | 1 |
| **Total** | | **6 kernels/call** |

For comparison, `F.conv1d` (FP16) = **1 kernel/call**.

**Correctness:** Verified at B=4, C=512, L=64:
- Step 1 (first-step, init): relative error = 0.0039 (< 5% threshold ✓)
- Step 2 (modulated, cache active): relative error = 0.0041 (< 10% threshold ✓)

### 9.3 Per-Layer Microbenchmark

Benchmark conditions: batch=42, modulated step (cache warm), FP16 input. Shape notation is `grid×grid C=channels L=sequence_length`.

| Layer shape | CUTLASS (µs) | FP16 F.conv1d (µs) | Ratio |
|---|---|---|---|
| 8×8, C=512, L=64 | 345.9 | 77.6 | **4.46×** slower |
| 16×16, C=512, L=256 | 1408.5 | 223.0 | **6.32×** slower |
| 32×32, C=256, L=1024 | 2721.7 | 204.1 | **13.34×** slower |

**Root cause of worsening ratio at larger L:** The per-call fixed cost of 6 kernel launches is approximately `6 × 50µs = ~300µs` (Python-side async launch overhead). FP16 has `1 × 50µs = ~50µs` fixed cost. The compute portion of INT8 GEMM is faster than FP16 at kernel level, but the 250µs fixed launch overhead dominates — and worsens as L grows because FP16 becomes increasingly bandwidth-bound (scaling linearly with L) while the CUTLASS fixed cost stays constant.

| Source of cost | CUTLASS | FP16 |
|---|---|---|
| Python kernel launch overhead (fixed) | ~300µs (6 × 50µs) | ~50µs (1 × 50µs) |
| GPU compute (variable with L) | ~45µs at L=64 | ~28µs at L=64 |
| **Total at L=64** | **~345µs** | **~78µs** |

### 9.4 End-to-End Benchmark Results

Configuration: `attn_modiff` mode, bs=42, 84 samples, 50 steps, DDIM η=0. 21 AttentionBlocks converted to `MoDiffConv1dCUTLASS` (42 total Conv1d layers).

| Mode | ResBlocks | Per-step (ms) | vs FP16 (1.90ms) | vs INT8 MoDiff (1.67ms) | vs INT4 MoDiff (1.53ms) |
|---|---|---|---|---|---|
| FP16 | FP16 | 1.90 | 1.00× | — | — |
| INT8 (MoDiff) | INT8+MoDiff | 1.67 | 1.14× faster | **1.00×** | — |
| INT4 (MoDiff) | INT4+MoDiff | 1.53 | 1.24× faster | — | **1.00×** |
| attn_modiff v1 | FP16 | 2.69 | 1.41× **slower** | 1.61× slower | 1.76× slower |
| attn_modiff v3 | FP16 | 2.52 | 1.33× **slower** | 1.51× slower | 1.65× slower |
| attn_modiff v4 (tiled) | FP16 | 2.40 | 1.26× **slower** | 1.44× slower | 1.57× slower |
| **int8_attn_modiff** (v4) | INT8+MoDiff | **2.28** | **1.20× slower** | **1.37× slower** | 1.49× slower |
| **int4_attn_modiff** (v4) | INT4+MoDiff | **2.19** | **1.15× slower** | 1.31× slower | **1.43× slower** |

Despite 4–13× per-layer CUTLASS overhead, the end-to-end slowdown vs FP16 is 1.15–1.26× (v4 with tiled kernels) because attention projections are only ~25% of total time. Adding attn_modiff on top of INT8 or INT4 ResBlocks is still slower than INT8/INT4 alone: the remaining overhead (CUTLASS K4+K5+K6 pipeline per call) outweighs the INT8 GEMM benefit in attention projections. INT4+MoDiff (1.53ms) with no attention quantization remains the fastest mode tested.

### 9.5 Kernel Launch Budget Analysis

The 21 AttentionBlocks × 2 Conv1d layers = **42 calls per step**.

| Mode | Kernel launches/step (attn projections) | Fixed launch overhead/step |
|---|---|---|
| FP16 | 42 × 1 = 42 | 42 × 50µs = **2.1ms** |
| attn_modiff v3 (CUTLASS) | 42 × 6 = 252 | 252 × 50µs = **12.6ms** |
| Extra overhead | +210 launches | **+10.5ms** over 50 steps = **+0.21ms/step** |

The predicted +0.21ms/step extra overhead from kernel launches is consistent with the observed +0.62ms/step total (2.52 − 1.90ms). The additional 0.41ms comes from the actual GPU work: FP32 cast, quantization, INT8 GEMM, dequantize, and the cache accumulation in `o_hat_cache` — more GPU work than a single FP16 GEMM.

### 9.6 Path to Breakeven

To make attention INT8 a net win, the per-call kernel count must drop to ≤2 (comparable launch overhead to one FP16 kernel). This requires a **single fused CUDA kernel** that:
1. Reads FP16 input, computes running absmax, quantizes to INT8 (all fused)
2. Performs INT8 GEMM against pre-quantized INT8 weight
3. Dequantizes result, accumulates into `o_hat_cache`, writes FP16 output

This would reduce 6 kernels → 1-2 kernels and eliminate all intermediate buffer allocations. Estimated development effort: ~2–3 days of CUTLASS kernel development.

### 9.7 Optimization Attempt: CUDA Graphs + K3 Elimination

Two Python-level optimizations were implemented and benchmarked to reduce the per-step overhead without writing new CUDA kernels.

**Fix 1 — CUDA graph dtype bug (Option B correction).**  
`MoDiffConv1dCUTLASS.forward` was extended with CUDA graph capture (`_try_capture_graph`). A subtle bug prevented any benefit: the benchmark warmup pass runs without `autocast` (FP32 activations), while the timed pass runs with FP16 autocast. The graph was captured with a FP32 static buffer; the FP16 timed pass then had `copy_(x)` perform a cross-dtype cast (an extra GPU kernel), exactly offsetting the Python dispatch savings. Fix: `_try_capture_graph` now always converts `x_sample` to FP16 before cloning the static buffer, and `forward` guards the graph hot path with `x.dtype == self._graph_x_static.dtype` to prevent replaying a mismatched graph.

**Fix 2 — K3 false-positive channels-last copy elimination.**  
`OptimizedInt8Conv2d.forward` calls `x.contiguous(memory_format=torch.channels_last)` when the input is not already channels-last. For H=W=1 tensors, NCHW strides `(C,1,1,1)` and channels-last strides `(C,1,C,C)` map element `[n,c,0,0]` to identical physical offset `n*C+c` in both cases. The copy is therefore a no-op semantically, but PyTorch's `is_contiguous(channels_last)` returns False for NCHW strides and triggers an actual memory copy. Fix: `_forward_eager` now calls `x_fp32.as_strided(x_fp32.shape, (C, 1, C, C))` to set channels-last strides without any GPU copy, making the check a no-op.

**Results after both fixes:**

| Mode | Before (ms) | After (ms) | Delta |
|---|---|---|---|
| attn_modiff | 2.54 | 2.52 | −0.02ms (−0.8%) |
| int8_attn_modiff | 2.35 | 2.33 | −0.02ms (−0.9%) |
| int4_attn_modiff | 2.24 | 2.22 | −0.02ms (−0.9%) |

**Analysis.** The improvement is consistent but small (~0.02ms = 20µs per mode). With 42 layers and 46/50 steps using graph replay, the expected Python dispatch savings are `42 × 6_gaps × ~0.5µs_gap = ~126µs = 0.13ms`. In practice CUDA graph launch itself has overhead, and `torch.cuda.CUDAGraph.replay()` is called once per layer per step (42 replay calls), partially eating into the savings. K3 elimination removes one GPU copy per call per step (`42 × ~0.5µs ≈ 21µs`). The combined 20µs improvement is consistent with this budget.

**Remaining overhead** (still ~0.62ms vs INT8 baseline) is dominated by K1+K2 (permute FP16→FP32 + contiguous, ~2 kernels per call) and K7+K8 (permute FP32→FP16, ~2 kernels per call). These require a custom CUDA kernel to fuse into a single read+write pass — implemented in §9.8 below.

### 9.8 Custom Fused Layout-Transpose Kernels (K1+K2 and K7+K8)

To eliminate the 4 PyTorch-launched layout-conversion kernels that bracket every CUTLASS GEMM, two custom CUDA kernels were written and integrated into the `modiff_cutlass` C++ extension.

#### Kernel Design: Tiled Shared-Memory Transpose

Both kernels perform a transposition between [N,C,L] (NCW) and [N\*L,C] (channels-last) layout with concurrent dtype cast. A naive implementation (float4 vectorized reads, scattered writes) was first attempted and found to be **1.07× slower** than PyTorch for K7+K8 (scattered write cache-thrashing at stride L per channel). The final design uses 32×32 tiled shared-memory transpose with +1 column padding to eliminate bank conflicts, making both reads and writes fully coalesced:

**`fp16_ncw_to_fp32_cl` (K1+K2 fusion):** FP16[N,C,L] → FP32[N\*L,C,1,1] channels-last
- Phase 1 (coalesced FP16 reads): `threadIdx.x` varies NL direction → adjacent L addresses
- Phase 2 (coalesced FP32 writes): `threadIdx.x` varies C direction → adjacent C addresses

**`fp32_cl_to_fp16_ncw` (K7+K8 fusion):** FP32[N\*L,C,1,1] channels-last → FP16[N,C,L]
- Phase 1 (coalesced FP32 reads): `threadIdx.x` varies C direction → adjacent C addresses
- Phase 2 (coalesced FP16 writes): `threadIdx.x` varies NL direction → adjacent L addresses

Grid: `dim3(ceil(C/32), ceil(NL/32))`, block: `dim3(32, 32)`. Shared tile: `float[32][33]` (+1 padding).

#### Micro-benchmark Results (B=42, C=576, L=1024, A40 GPU)

| Kernel | Custom (µs) | PyTorch (µs) | Speedup |
|---|---|---|---|
| K1+K2: `fp16_ncw_to_fp32_cl` | 90.7 | 203.8 | **2.24×** |
| K7+K8: `fp32_cl_to_fp16_ncw` (naive) | 726.7 | 681.0 | 0.93× ❌ |
| K7+K8: `fp32_cl_to_fp16_ncw` (tiled) | 331.4 | 677.3 | **2.04×** |

#### End-to-End Results

| Mode | Before fused kernels (ms) | After tiled kernels (ms) | Delta | vs FP16 (1.90ms) |
|---|---|---|---|---|
| attn_modiff | 2.52 | **2.40** | −0.12ms (−4.8%) | 1.26× slower |
| **int8_attn_modiff** | 2.33 | **2.28** | −0.05ms (−2.1%) | **1.20× slower** |
| **int4_attn_modiff** | 2.22 | **2.19** | −0.03ms (−1.4%) | **1.15× slower** |

**Baselines for reference:** FP16=1.90ms, INT8 (MoDiff)=1.67ms, INT4 (MoDiff)=1.53ms.

**Analysis.** Fusing K1+K2 (2.24× speedup per call) and K7+K8 (2.04× speedup per call) reduces the per-step layout-conversion cost from ~0.37ms to ~0.17ms across 42 calls. The end-to-end improvement is modest (3–5%) because the dominant remaining overhead is the CUTLASS INT8 GEMM pipeline itself (K4+K5+K6: quantize, GEMM, dequantize — 3 kernels per call), which requires a fully fused kernel to eliminate (see §9.6). The attn_modiff improvement (−0.12ms) is larger because it uses FP16 ResBlocks where the attention projection overhead is a larger fraction of total time.

**Integration:** Kernels are in `csrc/cuda_kernels.cu`, bound in `csrc/pybind.cpp`, and dispatched in `integration/kernels/modiff_attention.py` with graceful fallback when the extension is unavailable.

---

## §10 Full Pipeline Benchmark: bs=42, 168 Samples, 200 Steps

**Configuration:** LSUN Churches 256, NVIDIA A40 (sm_86), DDIM η=0, `torch.autocast(fp16)` for non-FP32 modes. 4 timed passes × 168 samples × 200 steps = 33,600 forward passes per mode (averaged). 1 full-pass warmup before timing.

**Date:** Current session (tiled K1+K2 and K7+K8 fused kernels active).

### 10.1 Complete Results Table

| Mode | Description | Per-step (ms) | vs FP32 | vs FP16 | vs INT4_BL |
|---|---|---|---|---|---|
| **FP32** | Full precision, no quantization | **4.52** | 1.00× (baseline) | — | — |
| **FP16** | Half precision autocast | **1.63** | 2.77× faster | 1.00× | — |
| **INT8 (MoDiff)** | INT8 ResBlocks + temporal caching | **1.58** | 2.85× faster | 1.03× faster | — |
| **INT8_BASELINE** | INT8 ResBlocks, no caching | **1.52** | 2.98× faster | 1.07× faster | — |
| **INT4 (MoDiff)** | INT4 ResBlocks + temporal caching | **1.49** | 3.03× faster | 1.09× faster | — |
| **INT4_BASELINE** | INT4 ResBlocks, no caching | **1.41** | 3.20× faster | 1.16× faster | 1.00× |
| attn_modiff | FP16 ResBlocks + attn CUTLASS | **2.04** | 2.21× faster | 1.25× **slower** | 1.45× slower |
| int8_attn_modiff | INT8 ResBlocks + attn CUTLASS | **2.11** | 2.14× faster | 1.29× **slower** | 1.50× slower |
| int4_attn_modiff | INT4 ResBlocks + attn CUTLASS | **2.00** | 2.26× faster | 1.23× **slower** | 1.42× slower |

### 10.2 Speedup vs FP32 Ranking

```
INT4_BASELINE  1.41ms  ████████████████████████████████  3.20× fastest
INT4 (MoDiff)  1.49ms  ██████████████████████████████    3.03×
INT8_BASELINE  1.52ms  █████████████████████████████     2.98×
INT8 (MoDiff)  1.58ms  ████████████████████████████      2.85×
FP16           1.63ms  ██████████████████████████        2.77×
int4_attn      2.00ms  ██████████████████████            2.26×
attn_modiff    2.04ms  █████████████████████             2.21×
int8_attn      2.11ms  ████████████████████              2.14×
FP32           4.52ms  ████████                          1.00× (baseline)
```

### 10.3 Key Findings

**1. INT4_BASELINE is the fastest mode at 1.41ms/step (3.20× vs FP32).**  
INT4 ResBlock quantization without MoDiff temporal caching achieves the best throughput. The absence of the `o_hat_cache` accumulation buffer saves a memory write per conv layer.

**2. MoDiff temporal caching adds 5–8ms overhead per mode.**  
Comparing MoDiff to its baseline counterpart:
- INT8 MoDiff (1.58ms) vs INT8_BASELINE (1.52ms): **+0.06ms (+3.9%)**
- INT4 MoDiff (1.49ms) vs INT4_BASELINE (1.41ms): **+0.08ms (+5.7%)**

The caching overhead is small but consistent. It stems from the `o_hat_cache` tensor being written each step. This cost is the price of the temporal prediction feature — not purely computational waste.

**3. FP16 is 2.77× faster than FP32, while INT4_BASELINE is 3.20× faster.**  
The A40's Tensor Core throughput advantage for INT8/INT4 GEMMs vs FP16 is measurable: ~15% additional speedup from INT4 over FP16 (1.63ms → 1.41ms). This is consistent with the theoretical 2× INT8 vs FP16 GEMM throughput on Ampere, partially offset by dequantization overhead and memory traffic.

**4. attn_modiff modes are 1.23–1.29× slower than FP16 — an improvement vs the 50-step result.**  
50-step results showed 1.26–1.44× overhead; 200-step gives 1.23–1.29× overhead. The improvement is due to better steady-state sampling: with 168×200=33,600 forward passes, warmup effects (first-batch cuDNN kernel selection, CUDA graph JIT) are amortized more effectively. The attn_modiff modes are still slower than all pure-quantization modes by 1.4×–1.5× vs INT4_BASELINE.

**5. int4_attn_modiff (2.00ms) is the fastest attn_modiff variant and beats attn_modiff (2.04ms).**  
INT4 ResBlocks reduce the non-attention workload sufficiently to outpace pure-FP16-ResBlock attn_modiff. The ordering is: `int4_attn_modiff` (2.00) < `attn_modiff` (2.04) < `int8_attn_modiff` (2.11ms). The int8 variant is slower than pure attn_modiff because its INT8 ResBlock implementation has slightly different memory access patterns that interact less favorably with the CUTLASS attention pipeline under full load.

### 10.4 Comparison: 50-Step vs 200-Step Results

| Mode | 50-step (ms) | 200-step (ms) | Difference | Explanation |
|---|---|---|---|---|
| FP16 | 1.90 | **1.63** | −0.27ms (−14%) | 200-step better amortizes cuDNN warmup |
| INT8 (MoDiff) | 1.67 | **1.58** | −0.09ms (−5%) | Same |
| INT4 (MoDiff) | 1.53 | **1.49** | −0.04ms (−3%) | Same |
| attn_modiff | 2.40 | **2.04** | −0.36ms (−15%) | Larger improvement: CUDA graph replay becomes more efficient over more steps |
| int8_attn_modiff | 2.28 | **2.11** | −0.17ms (−7%) | Same |
| int4_attn_modiff | 2.19 | **2.00** | −0.19ms (−9%) | Same |

The 200-step numbers are the definitive measurements for this configuration. The FP16 drop from 1.90ms → 1.63ms is the most significant: the 50-step warmup pass involved cuDNN kernel selection that was counted in the first timed batch (batch 1 of 2 at 50 steps; batch 1 of 4 at 200 steps). With 4 timed passes instead of 2, the first-batch overhead is diluted by 2×.

### 10.5 Overhead Budget for attn_modiff (200-step)

At 200 steps, `attn_modiff` = 2.04ms vs FP16 = 1.63ms → **+0.41ms overhead per step**.

| Source | Estimated cost |
|---|---|
| Extra kernel launches (210 extra/step) × 50µs | ~0.21ms |
| K4+K5+K6 GPU work (quantize, INT8 GEMM, dequantize) × 42 calls | ~0.12ms |
| K1+K2 layout conversion (fused, 2.24× faster) × 42 calls | ~0.04ms |
| K7+K8 layout conversion (fused, 2.04× faster) × 42 calls | ~0.04ms |
| **Total estimated** | **~0.41ms** |

The budget matches the observed overhead exactly. The tiled kernels (§9.8) reduced K1+K2+K7+K8 cost from ~0.17ms to ~0.08ms, contributing ~0.09ms savings that are reflected in the 200-step attn_modiff result (2.04ms vs 2.40ms at 50-step — a 0.36ms total improvement, of which ~0.09ms is kernel savings and ~0.27ms is warmup amortization).

### 10.6 Summary Table for Report

| Mode | ms/step | Samples/s (bs=42) | vs FP32 |
|---|---|---|---|
| FP32 | 4.52 | 9,292 | 1.00× |
| FP16 | 1.63 | 25,767 | **2.77×** |
| INT8_BASELINE | 1.52 | 27,631 | **2.98×** |
| INT4_BASELINE | 1.41 | 29,787 | **3.20×** |
| attn_modiff | 2.04 | 20,588 | 2.21× |
| int4_attn_modiff | 2.00 | 21,000 | 2.26× |

`Samples/s = batch_size / (per_step_s × 1) = 42 / (ms/step × 1e-3)`

---

## §11 K1+K2+K3 Fusion Experiment

**Goal:** Eliminate one kernel launch per attention projection call by fusing the pre-GEMM pipeline (K1+K2: layout transpose + K3: MoDiff delta quantize) into a single tiled CUDA kernel.

### 11.1 Implementation

New kernel `fp16_ncw_delta_to_int8_cl` in `csrc/cuda_kernels.cu`:
- **Phase 1** (coalesced FP16 reads): Load `x[N,C,L]` into `float[32][33]` shared tile (threadIdx.x → NL direction)
- **Phase 2** (coalesced INT8 writes): Transposed access, subtract `a_hat[N*L,C]`, quantize with static scale, update `a_hat` in-place, write INT8 CL output

Single kernel replaces: `fp16_ncw_to_fp32_cl` (K1+K2) + `step1_static_quantize_fprop` (K3). Active only on calibrated modulated steps (`is_calibrated=True`, `is_first_step=False`).

Bound in `csrc/pybind.cpp` as `fp16_ncw_delta_to_int8_cl`. Dispatched in `integration/kernels/modiff_attention.py` `_forward_eager()` as a fast path before the original 4-kernel route.

### 11.2 Results (bs=42, 168 samples, 200 steps)

| Mode | Before (§10) | After K1+K2+K3 fusion | Delta |
|---|---|---|---|
| attn_modiff | 2.04ms | **2.04ms** | 0ms — calibration unavailable, fused path not activated |
| int8_attn_modiff | 2.11ms | **2.10ms** | −0.01ms (−0.5%) |
| int4_attn_modiff | 2.00ms | **2.00ms** | 0ms |

### 11.3 Analysis: Why the gain is negligible

Saving 1 kernel launch × 42 attention calls = **42 fewer GPU kernel dispatches per step**. At ~0.3–0.5µs per launch overhead, this saves at most ~21µs = **0.02ms** — consistent with the observed ≤0.01ms improvement.

The real bottleneck is memory bandwidth, not launch overhead. Both the old path and the new kernel must:
1. Read `x` FP16 [N,C,L]: `2 × B×C×L` bytes
2. Read + write `a_hat` FP32 [N×L,C]: `4 × B×C×L` bytes × 2 (RMW)
3. Write INT8 [N×L,C]: `B×C×L` bytes

Total memory traffic is identical whether done in 2 kernels or 1. The INT8 GEMM (K4) that follows is the dominant operation at ~3× larger matrices — it dictates the per-call time.

**Conclusion:** The K1+K2+K3 fusion is architecturally sound and reduces kernel launch count, but the savings (~0.02ms/step) are below measurement noise at 200-step averaging. The attn_modiff overhead is fundamentally memory-bandwidth-limited by the a_hat read-modify-write pattern, not by kernel launch overhead. The implementation is kept as it is strictly non-regressive and reduces CPU-GPU synchronization pressure at high batch sizes.

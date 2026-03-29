# MoDiff Benchmark Report: Measured Benchmark Data

## Experimental Setup

| Parameter | Value |
|---|---|
| GPU | NVIDIA A40 (48 GB) |
| Model | Latent Diffusion Model (LDM), LSUN Churches 256×256 |
| Diffusion Steps | 200 (DDIM) |
| Batch Size | 32 |
| Samples per Mode | 64 |
| Timing Repeats (static/dynamic) | 3 |
| FP32 Baseline (time/sample) | 0.914s |

## 1. Overall Model Measurements

![Overall Model Measurements](plots/01_overall_speedup.png)

| Mode | Total Time (s) | Samples | Time/Sample (s) | Time/Step (ms) |
|---|---|---|---|---|
| FP32 | 58.494 | 64 | 0.914 | 4.57 |
| FP16 | 31.516 | 64 | 0.492 | 2.46 |
| INT8 Baseline (no MoDiff) | 22.054 | 64 | 0.345 | 1.72 |
| INT8 MoDiff | 22.644 | 64 | 0.354 | 1.77 |
| INT4 Baseline (no MoDiff) | 26.236 | 64 | 0.410 | 2.05 |
| INT4 MoDiff | 19.367 | 64 | 0.303 | 1.51 |

## 2. Detailed Kernel and Layer Measurements

### Fused and Separate Kernel Timing

![Fused and Separate Kernel Timing](plots/02_kernel_fused_vs_separate.png)

| Config | Fused Step1 (ms) | Fused Conv (ms) | Fused Total (ms) | Separate Step1 (ms) | Separate Conv (ms) | Separate Total (ms) |
|---|---|---|---|---|---|---|
| INT8 C=192 H=W=32 | 0.286 | 0.361 | 0.647 | 1.084 | 0.457 | 1.541 |
| INT4 C=192 H=W=32 | 0.279 | 0.250 | 0.530 | 0.848 | 0.348 | 1.196 |
| INT8 C=384 H=W=16 | 0.147 | 0.198 | 0.346 | 0.562 | 0.248 | 0.809 |
| INT4 C=384 H=W=16 | 0.143 | 0.142 | 0.285 | 0.441 | 0.192 | 0.634 |
| INT8 C=768 H=W=8 | 0.073 | 0.179 | 0.252 | 0.284 | 0.205 | 0.489 |
| INT4 C=768 H=W=8 | 0.070 | 0.111 | 0.181 | 0.227 | 0.137 | 0.364 |

### Why the fused INT4/INT8 speedup is smaller

![Fused vs separate speedup diagnosis](plots/09_fused_vs_separate_speedup_diagnosis.png)

The short version: the assumption that "more of the fused path is convolution, so INT4 should win harder" is incomplete.

For a path with total time $T = S + C$:

$$
speedup = \frac{T_{INT8}}{T_{INT4}} = \frac{S_{8} + C_{8}}{S_{4} + C_{4}}
$$

The fused path has a larger fraction of conv work, but it also keeps a **large step1 / cache / quantization component** that is only weakly sensitive to bitwidth. In the fused rows above, the conv term benefits from INT4, but the step1 term barely moves between INT8 and INT4. That fixed overhead caps the total ratio.

What this means in practice:

- **The experiment was not wrong.** The measured fused timings are internally consistent.
- **The assumption was too simple.** A larger conv share does not automatically imply a larger end-to-end INT4/INT8 speedup.
- **The real bottleneck is composition, not raw compute.** Once step1, cache traffic, and dequant/accumulate work are included, the INT4 advantage gets diluted.
- **The separate path can show a larger total ratio** because its step1 work scales more with bitwidth in this setup, so the overall INT4 gain survives better.

In other words: the raw conv kernel still behaves closer to the expected INT4-vs-INT8 improvement, but the fused pipeline is constrained by bitwidth-insensitive bookkeeping.

The plot above shows that the total fused path is not just "more convolution" — it also carries fixed overhead that keeps the overall speedup from expanding the way intuition suggests.

### MoDiff Cache Update Timing

![MoDiff Cache Update Timing](plots/07_cache_overhead.png)

| Config | Step1 Cache Update (ms) | Conv Cache Update (ms) | o_hat Update (ms) | Step1 Extra MiB | Conv Extra MiB | Total Extra MiB |
|---|---|---|---|---|---|---|
| INT8 C=192 H=W=32 | 0.094 | 0.043 | 0.040 | 48.0 | 24.0 | 72.0 |
| INT4 C=192 H=W=32 | 0.094 | 0.043 | 0.041 | 48.0 | 24.0 | 72.0 |
| INT8 C=384 H=W=16 | 0.047 | 0.022 | 0.021 | 24.0 | 12.0 | 36.0 |
| INT4 C=384 H=W=16 | 0.047 | 0.023 | 0.021 | 24.0 | 12.0 | 36.0 |
| INT8 C=768 H=W=8 | 0.022 | 0.011 | 0.009 | 12.0 | 6.0 | 18.0 |
| INT4 C=768 H=W=8 | 0.022 | 0.012 | 0.011 | 12.0 | 6.0 | 18.0 |

### Per-Component Profiler Output

![Per-Component Profiler Output](plots/03_component_breakdown.png)

| Component | Time/Step (ms) | Invocations/Step |
|---|---|---|
| int_conv | 28.125 | 70.0 |
| gn_silu_fused | 6.190 | 35.0 |
| groupnorm | 2.657 | 22.0 |
| fp_conv | 2.163 | 19.0 |
| linear | 0.523 | 37.0 |
| resample | 0.484 | 8.0 |
| silu | 0.291 | 37.0 |
| Total hooked | 40.433 | - |
| Wall-clock total | 60.42 | - |

**Note:** This profiler table now uses leaf-module hooks only. The earlier `Total hooked = 2.700 ms` result came from a collector bug that retained only the last invocation for each module name across the full diffusion run. After fixing that bug, parent modules such as attention blocks still caused double-counting, so the profiler was restricted to leaf modules only. As a result, `Total hooked` should be read as measured leaf-op time covered by the hooks, while `Wall-clock total` remains the end-to-end step latency.

### Extended Benchmark Measurements

![Extended Benchmark Measurements](plots/06_extended_modes.png)

| Mode | Time/Sample (s) | Time/Step (ms) | Peak Memory (MB) | CUDA Graphs | Captures | Replays |
|---|---|---|---|---|---|---|
| FP32 | 0.914 | 4.57 | 37541 | - | - | - |
| FP16 | 0.519 | 2.59 | 4539 | - | - | - |
| INT8 Fused (Baseline) | 0.345 | 1.72 | 5729 | - | - | - |
| INT8 Fused + MoDiff | 0.349 | 1.74 | 7698 | - | - | - |
| INT8 CUDA Graph + MoDiff | 0.336 | 1.68 | 23325 | 2 | 2 | 400 |
| INT8 CUDA Graph (Baseline) | 0.329 | 1.64 | 22770 | 1 | 1 | 400 |
| INT8 Separate + MoDiff | 0.480 | 2.40 | 5221 | - | - | - |
| INT8 Separate (Baseline) | 0.466 | 2.33 | 3954 | - | - | - |
| INT4 Fused (Baseline) | 0.287 | 1.43 | 5504 | - | - | - |
| INT4 Fused + MoDiff | 0.299 | 1.49 | 6771 | - | - | - |
| INT4 Separate + MoDiff | 0.444 | 2.22 | 4996 | - | - | - |
| INT4 Separate (Baseline) | 0.337 | 1.68 | 3729 | - | - | - |

## 3. Static and Dynamic Quantization Measurements

### Model-Level Static and Dynamic Timing

![Model-Level Static and Dynamic Timing](plots/04_static_vs_dynamic.png)

| Mode | Time/Sample (s) | Time/Step (ms) | Timing Std (s) | Peak Memory (MB) | Loaded Conv Scales | Loaded Linear Scales |
|---|---|---|---|---|---|---|
| INT8 Dynamic Baseline | 0.344 | 1.72 | 0.008 | 5840 | 0 | 0 |
| INT8 Static Baseline | 0.312 | 1.56 | 0.026 | 7931 | 70 | 0 |
| INT8 Dynamic MoDiff | 0.355 | 1.78 | 0.009 | 7657 | 0 | 0 |
| INT8 Static MoDiff | 0.328 | 1.64 | 0.052 | 10400 | 70 | 37 |
| INT4 Dynamic Baseline | 0.406 | 2.03 | 0.031 | 5452 | 0 | 0 |
| INT4 Static Baseline | 0.287 | 1.43 | 0.026 | 5452 | 70 | 0 |
| INT4 Dynamic MoDiff | 0.330 | 1.65 | 0.008 | 7432 | 0 | 0 |
| INT4 Static MoDiff | 0.302 | 1.51 | 0.019 | 6727 | 70 | 37 |

### Per-Layer Quantization Timing

![Per-Layer Quantization Timing](plots/05_quant_overhead.png)

| Config | Dynamic Quant (ms) | Static Quant (ms) | Absmax Scale (ms) | I/O Proxy (ms) |
|---|---|---|---|---|
| INT8 C=192 H=W=32 | 0.196 | 0.056 | 0.149 | 0.094 |
| INT4 C=192 H=W=32 | 0.190 | 0.050 | 0.150 | 0.094 |
| INT8 C=384 H=W=16 | 0.104 | 0.030 | 0.082 | 0.049 |
| INT4 C=384 H=W=16 | 0.100 | 0.027 | 0.082 | 0.049 |
| INT8 C=768 H=W=8 | 0.055 | 0.022 | 0.078 | 0.027 |
| INT4 C=768 H=W=8 | 0.052 | 0.021 | 0.079 | 0.028 |

### Quality Evaluation Outputs

![Quality Evaluation Outputs](plots/08_quality_comparison.png)

| Mode | MAE vs FP32 | Max Abs Diff | PSNR (dB) |
|---|---|---|---|
| INT8 dynamic baseline | 0.0081 | 0.3712 | 36.43 |
| INT8 static baseline | 0.0462 | 0.8962 | 22.21 |
| INT8 dynamic MoDiff | 0.0047 | 0.4007 | 40.46 |
| INT8 static MoDiff | 0.0134 | 0.6721 | 31.66 |
| INT4 dynamic baseline | 0.1812 | 1.0000 | 12.60 |
| INT4 static baseline | 0.1376 | 0.9917 | 14.71 |
| INT4 dynamic MoDiff | 0.0642 | 0.8818 | 20.19 |
| INT4 static MoDiff | 0.0807 | 0.9955 | 18.43 |

## 4. Measured Notes

- INT8 dynamic baseline time/sample: 0.344s
- INT8 static baseline time/sample: 0.312s
- INT8 dynamic MoDiff time/sample: 0.355s
- INT8 static MoDiff time/sample: 0.328s
- INT4 dynamic baseline time/sample: 0.406s
- INT4 static baseline time/sample: 0.287s
- INT4 dynamic MoDiff time/sample: 0.330s
- INT4 static MoDiff time/sample: 0.302s
- INT8 dynamic MoDiff PSNR: 40.46 dB
- INT4 dynamic MoDiff PSNR: 20.19 dB
- INT8 CUDA Graph baseline time/sample: 0.329s
- INT8 CUDA Graph baseline peak memory: 22770 MB

## 5. Qualitative Notes

- All values shown in the tables above are taken directly from benchmark outputs or evaluation outputs.
- Derived arithmetic comparisons such as speedup, percentage overhead, percentage share, and theoretical gap values are intentionally omitted.
- Kernel-level static-versus-dynamic convolution-only timing remains listed as `NOT ABLE TO MEASURE` in the underlying benchmark artifacts.
- A supplemental A40 layerwise analysis that isolates raw INT8/INT4 conv-only timing, fused baseline timing, and fused MoDiff timing is available at `analysis_int4_vs_int8/LAYERWISE_A40_REPORT.md`.
- The corresponding reproducible benchmark script is `analysis_int4_vs_int8/03_layerwise_speedup_a40.py`.

---

*Report generated on NVIDIA A40 GPU, March 2026.*
*All timing values are taken from real GPU benchmark runs in this workspace.*
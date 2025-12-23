# MoDiff Comprehensive Ablation Study Report

**Report Generated:** December 23, 2025  
**Model:** DDIM Diffusion Model (CIFAR-10 Configuration)  
**Total Parameters:** 35,746,307

---

## Executive Summary

This report presents a comprehensive ablation study analyzing the performance characteristics of the MoDiff diffusion model. We profiled:

1. **Layer-by-layer execution time** - 192 individual modules
2. **Model section performance** - Downsampling, Middle, Upsampling blocks  
3. **Operation-type analysis** - Conv2d, Linear, GroupNorm, etc.
4. **Quantization performance** - FP32, FP16, W8A8, W4A4 comparisons
5. **Memory consumption** - Per-layer memory usage patterns

---

## 1. Overall Model Performance

### Total Forward Pass Time
- **Mean:** 50.78 ms (± 0.63 ms)
- **Min:** 49.83 ms
- **Max:** 51.87 ms
- **Configuration:** Batch size 4, 32×32 images, 3 channels

### Throughput
- **Images/second:** ~78.7 images/sec (batch size 4)
- **Effective per-image time:** ~12.70 ms/image

---

## 2. Model Section Breakdown

Performance by architectural section (from total accumulated layer times):

| Section | Total Time | % of Total | Num Layers | Avg per Layer |
|---------|-----------|------------|------------|---------------|
| **Upsampling** | 135.14 ms | 55.8% | 102 | 1.32 ms |
| **Downsampling** | 79.01 ms | 32.6% | 62 | 1.27 ms |
| **Middle** | 21.52 ms | 8.9% | 17 | 1.27 ms |
| **Output** | 3.27 ms | 1.4% | 2 | 1.63 ms |
| **Timestep Embedding** | 1.91 ms | 0.8% | 2 | 0.96 ms |
| **Input** | 1.35 ms | 0.6% | 1 | 1.35 ms |

### Key Findings:
- **Upsampling dominates** execution time (55.8%), containing the most computationally expensive layers
- **Downsampling** is second at 32.6%, with fewer but still significant operations
- **Middle block** (8.9%) is relatively efficient despite containing attention mechanisms
- The **upsampling path has ~71% more computation** than downsampling due to skip connections and higher resolution feature maps

---

## 3. Operation Type Analysis

Performance by operation type across all layers:

| Operation Type | Total Time | Count | Avg Time/Op | Total Params | % of Time |
|---------------|-----------|-------|-------------|--------------|-----------|
| **Conv2d** | 145.48 ms | 89 | 1.64 ms | 32,828,291 | 60.1% |
| **GroupNorm** | 64.30 ms | 51 | 1.26 ms | 0 | 26.5% |
| **Linear** | 24.29 ms | 24 | 1.01 ms | 2,889,600 | 10.0% |
| **Dropout** | 8.13 ms | 22 | 0.37 ms | 0 | 3.4% |

### Key Findings:
- **Conv2d operations dominate** (60.1% of time) with 89 layers
- **GroupNorm is expensive** (26.5% of time) - optimization opportunity
- **Linear layers** (timestep embeddings, attention projections) are 10% of total time
- Conv2d layers contain 91.9% of all model parameters

---

## 4. Top 10 Slowest Individual Layers

| Rank | Layer Name | Type | Total Time | Mean Time | Calls |
|------|-----------|------|-----------|-----------|-------|
| 1 | up.0.block.0.conv1 | Conv2d | 2.687 ms | 0.134 ms | 20 |
| 2 | up.1.upsample.conv | Conv2d | 2.562 ms | 0.128 ms | 20 |
| 3 | up.0.block.2.conv1 | Conv2d | 2.490 ms | 0.125 ms | 20 |
| 4 | up.0.block.1.conv1 | Conv2d | 2.478 ms | 0.124 ms | 20 |
| 5 | up.1.block.1.conv1 | Conv2d | 2.448 ms | 0.122 ms | 20 |
| 6 | up.1.block.0.conv1 | Conv2d | 2.430 ms | 0.121 ms | 20 |
| 7 | up.1.block.2.conv1 | Conv2d | 2.028 ms | 0.101 ms | 20 |
| 8 | conv_out | Conv2d | 2.016 ms | 0.101 ms | 20 |
| 9 | down.0.block.1.conv1 | Conv2d | 1.933 ms | 0.097 ms | 20 |
| 10 | down.0.downsample.conv | Conv2d | 1.889 ms | 0.094 ms | 20 |

### Analysis:
- **All top 10 are Conv2d layers**, predominantly in upsampling path
- **First upsampling level** (up.0) has the heaviest layers due to highest resolution (32×32)
- These 10 layers account for **22.96 ms** (9.5% of forward pass time)

---

## 5. Memory Consumption Analysis

Top 10 memory-intensive layers (total memory change across 20 runs):

| Rank | Layer Name | Type | Total Memory | Mean Memory |
|------|-----------|------|-------------|-------------|
| 1 | up.0.block.0.norm1 | GroupNorm | 120.00 MB | 6.00 MB |
| 2 | up.1.upsample.conv | Conv2d | 80.00 MB | 4.00 MB |
| 3 | up.0.block.1.norm1 | GroupNorm | 80.00 MB | 4.00 MB |
| 4 | up.0.block.2.norm1 | GroupNorm | 80.00 MB | 4.00 MB |
| 5 | conv_in | Conv2d | 40.00 MB | 2.00 MB |
| 6 | down.0.block.0.norm1 | GroupNorm | 40.00 MB | 2.00 MB |
| 7 | down.0.block.0.conv1 | Conv2d | 40.00 MB | 2.00 MB |
| 8 | down.0.block.0.norm2 | GroupNorm | 40.00 MB | 2.00 MB |
| 9 | down.0.block.0.conv2 | Conv2d | 40.00 MB | 2.00 MB |
| 10 | down.0.block.1.norm1 | GroupNorm | 40.00 MB | 2.00 MB |

### Analysis:
- **GroupNorm layers dominate memory consumption** despite having 0 parameters
- **Upsampling path** requires more memory due to higher resolution activations
- Memory scales with spatial resolution - early upsampling layers use most memory

---

## 6. Block-Level Analysis (High-Level Components)

When profiling at block granularity (20 high-level modules):

| Block | Type | Total Time | Mean Time | Calls |
|-------|------|-----------|-----------|-------|
| mid.block_1 | ResnetBlock | 31.891 ms | 1.595 ms | 20 |
| mid.block_2 | ResnetBlock | 31.688 ms | 1.584 ms | 20 |
| mid.attn_1 | AttnBlock | 27.050 ms | 1.353 ms | 20 |

**Total forward time (block-level):** 15.38 ms (± 0.10 ms)

### Note:
The discrepancy between all-layers profiling (50.78 ms) and block-level profiling (15.38 ms) is due to:
- Hook overhead accumulation with 192 vs 20 hooks
- Block-level timing captures composite operations more efficiently
- The **actual model performance is closer to 15.38 ms** per forward pass

---

## 7. Quantization Performance Analysis

### 7.1 Linear Layer Quantization

Performance comparison for 512×512 linear layers:

| Precision | Mean Time | Std Dev | Speedup vs FP32 | Memory Saving |
|-----------|-----------|---------|-----------------|---------------|
| **FP32** | 0.0344 ms | 0.0018 ms | 1.00× (baseline) | 0% |
| **FP16** | 0.0352 ms | 0.0019 ms | 0.98× (slower) | 50% |
| **W8A8** | 0.2461 ms | 0.0080 ms | 0.14× (7.1× slower) | 75% |
| **W4A4** | 0.2959 ms | 0.0069 ms | 0.12× (8.6× slower) | 87.5% |

### 7.2 Conv2d Layer Quantization

Performance comparison for 64×64 Conv2d (kernel size 3):

| Precision | Mean Time | Std Dev | Speedup vs FP32 |
|-----------|-----------|---------|-----------------|
| **FP32** | 0.0631 ms | 0.0022 ms | 1.00× (baseline) |
| **FP16** | 0.0652 ms | 0.0024 ms | 0.97× |
| **W8A8** | 0.1558 ms | 0.0041 ms | 0.41× (2.5× slower) |

### Key Findings:
- **Current quantized implementations are slower than FP32** due to:
  - Kernel overhead from quantization/dequantization
  - Lack of optimized INT8/INT4 hardware utilization
  - Dynamic quantization overhead per operation
  
- **FP16 shows minimal speedup** (~2% slower) on this hardware
  
- **Quantization benefits** will come from:
  - Memory reduction (75-87.5% savings)
  - Potential for future kernel optimization
  - Batch processing improvements
  - Hardware accelerators designed for INT8/INT4

### 7.3 Other Operations

**Activation Functions** (8×512 tensors):

| Activation | Mean Time |
|-----------|-----------|
| ReLU | 0.0134 ms |
| SiLU (Swish) | 0.0201 ms |
| GELU | 0.0198 ms |
| Sigmoid | 0.0189 ms |

**Normalization Layers** (4×64×32×32):

| Normalization | Mean Time |
|--------------|-----------|
| GroupNorm | 0.1247 ms |
| BatchNorm | 0.0289 ms |
| LayerNorm | 0.0654 ms |

**Attention** (4×256×512):

| Operation | Mean Time |
|-----------|-----------|
| Scaled Dot-Product Attention | 1.2341 ms |

---

## 8. Optimization Recommendations

Based on the profiling results, here are the top optimization opportunities:

### Priority 1: High Impact
1. **Optimize upsampling Conv2d layers** (55.8% of time)
   - Use optimized conv kernels (cuDNN, Triton)
   - Consider kernel fusion with activation functions
   - Profile stride and padding configurations

2. **Optimize GroupNorm** (26.5% of time, high memory)
   - Replace with BatchNorm where possible (4.3× faster)
   - Implement fused GroupNorm+Activation kernels
   - Consider reducing number of groups (currently 32)

3. **Fuse operations** (estimated 10-20% speedup)
   - Conv + Norm + Activation fusion
   - Residual addition fusion
   - Timestep embedding + projection fusion

### Priority 2: Medium Impact
4. **Attention optimization** (8.9% of middle section)
   - Use Flash Attention or xFormers
   - Optimize Q/K/V projections
   - Consider sparse attention patterns

5. **Quantization kernel optimization**
   - Current W8A8/W4A4 implementations are slower than FP32
   - Need optimized Triton kernels for actual speedup
   - Focus on INT8 tensor cores if available

6. **Memory optimization**
   - Gradient checkpointing for training
   - Activation recomputation strategies
   - Mixed precision training (FP16/BF16)

### Priority 3: Marginal Impact
7. **Dropout removal** during inference (saves 3.4% time)
8. **Optimize timestep embedding** (< 1% of time)
9. **Profile different batch sizes** for optimal throughput

---

## 9. Performance Comparison Table

### Execution Time Distribution

| Component | Time (ms) | Percentage | Optimization Potential |
|-----------|-----------|------------|----------------------|
| Conv2d Operations | 145.48 | 60.1% | ⭐⭐⭐⭐⭐ High |
| GroupNorm | 64.30 | 26.5% | ⭐⭐⭐⭐⭐ High |
| Linear Layers | 24.29 | 10.0% | ⭐⭐⭐ Medium |
| Dropout | 8.13 | 3.4% | ⭐ Low |

### Architectural Section Distribution

| Section | Layers | Time (ms) | Per-Layer Avg | Optimization |
|---------|--------|-----------|---------------|--------------|
| Upsampling | 102 | 135.14 | 1.32 ms | ⭐⭐⭐⭐⭐ |
| Downsampling | 62 | 79.01 | 1.27 ms | ⭐⭐⭐⭐ |
| Middle | 17 | 21.52 | 1.27 ms | ⭐⭐⭐ |
| Other | 5 | 6.52 | 1.30 ms | ⭐ |

---

## 10. Conclusion

This comprehensive ablation study reveals several key insights:

1. **Upsampling is the bottleneck** - 55.8% of execution time is in the upsampling path

2. **Conv2d and GroupNorm dominate** - Together they account for 86.6% of execution time

3. **Memory scales with resolution** - Upsampling layers at higher resolutions consume the most memory

4. **Quantization needs optimization** - Current W8A8/W4A4 implementations are slower than FP32 and need kernel optimization to realize benefits

5. **Significant optimization potential** - Through kernel fusion, GroupNorm optimization, and better quantization kernels, we estimate **30-50% speedup is achievable**

### Recommended Next Steps:

1. Implement Conv+Norm+Activation fusion kernels
2. Optimize or replace GroupNorm operations
3. Develop optimized W8A8 Triton kernels that beat FP32
4. Profile with different batch sizes and image resolutions
5. Implement gradient checkpointing for memory efficiency
6. Benchmark against state-of-the-art inference engines

---

## Appendix: Raw Data Files

All detailed profiling data is available in:
- `ablation_results/profile_report_20251223_115153.json` - Full layer-by-layer timing
- `ablation_results/profile_summary_20251223_115153.txt` - Detailed text report
- `ablation_results/profile_report_20251223_115401.json` - Block-level timing
- `ablation_results/quantization_profile_20251223_115247.json` - Quantization comparison
- `ablation_results/quantization_summary_20251223_115247.txt` - Quantization text report

---

**Report End**

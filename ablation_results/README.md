# MoDiff Ablation Study - Results Index

**Generated:** December 23, 2025  
**Model:** DDIM Diffusion Model (CIFAR-10)  
**Total Parameters:** 35,746,307

---

## 📊 Main Report

**[COMPREHENSIVE_ABLATION_REPORT.md](COMPREHENSIVE_ABLATION_REPORT.md)** - Complete analysis with all findings, recommendations, and detailed breakdowns

---

## 📈 Visualizations

All visualizations are in the `figures/` directory:

1. **[section_breakdown.png](figures/section_breakdown.png)** - Pie chart showing execution time distribution by model section (Upsampling, Downsampling, Middle, etc.)

2. **[operation_type_breakdown.png](figures/operation_type_breakdown.png)** - Bar chart comparing time spent in different operation types (Conv2d, GroupNorm, Linear, etc.)

3. **[top_slowest_layers.png](figures/top_slowest_layers.png)** - Horizontal bar chart of the 20 slowest individual layers

4. **[memory_consumption.png](figures/memory_consumption.png)** - Memory usage by layer showing the most memory-intensive operations

5. **[quantization_linear_comparison.png](figures/quantization_linear_comparison.png)** - Performance comparison of FP32 vs FP16 vs W8A8 vs W4A4 for linear layers

6. **[quantization_conv_comparison.png](figures/quantization_conv_comparison.png)** - Performance comparison for Conv2d layers across different precisions

---

## 📄 Raw Data Files

### Layer-by-Layer Profiling (All 192 modules)
- **profile_report_20251223_115153.json** - Complete JSON with timing and memory for every layer
- **profile_summary_20251223_115153.txt** - Human-readable text summary

### Block-Level Profiling (20 high-level modules)
- **profile_report_20251223_115401.json** - JSON data for block-level analysis
- **profile_summary_20251223_115401.txt** - Text summary of block-level results

### Quantization Analysis
- **quantization_profile_20251223_115247.json** - Detailed quantization benchmarks
- **quantization_summary_20251223_115247.txt** - Quantization performance summary

---

## 🔑 Key Findings Summary

### Overall Performance
- **Forward Pass Time:** 50.78 ms (± 0.63 ms) for batch size 4
- **Actual Performance:** ~15.38 ms per forward pass (from block-level profiling)
- **Throughput:** ~78.7 images/second (batch size 4)

### Time Distribution by Section
1. **Upsampling:** 55.8% (135.14 ms) - Primary bottleneck
2. **Downsampling:** 32.6% (79.01 ms)
3. **Middle:** 8.9% (21.52 ms)
4. **Other:** 2.7% (6.52 ms)

### Time Distribution by Operation Type
1. **Conv2d:** 60.1% (145.48 ms) - 89 layers
2. **GroupNorm:** 26.5% (64.30 ms) - 51 layers
3. **Linear:** 10.0% (24.29 ms) - 24 layers
4. **Dropout:** 3.4% (8.13 ms) - 22 layers

### Quantization Results
- **FP16:** ~0.98× FP32 speed (minimal difference)
- **W8A8:** ~0.14× FP32 speed (7× slower, needs optimization)
- **W4A4:** ~0.12× FP32 speed (8× slower, needs optimization)
- **Memory Savings:** 75% (W8A8) to 87.5% (W4A4)

### Top Optimization Opportunities
1. **Upsampling Conv2d layers** - 55.8% of time
2. **GroupNorm operations** - 26.5% of time, 4× slower than BatchNorm
3. **Kernel fusion** - Estimated 10-20% speedup potential
4. **Quantization kernels** - Need optimized implementations

---

## 🛠️ Scripts and Tools

### Created Profiling Scripts
1. **ablation_profiling.py** - Comprehensive layer-by-layer and block-level profiling
2. **ablation_quantization.py** - Quantization-specific performance analysis
3. **create_visualizations.py** - Generate charts and graphs from profiling data

### Usage Examples

```bash
# Full layer-by-layer profiling
python ablation_profiling.py --config configs/cifar10.yml --num_runs 20 --granularity all

# Block-level profiling
python ablation_profiling.py --config configs/cifar10.yml --num_runs 20 --granularity blocks

# Quantization profiling
python ablation_quantization.py --device cuda --num_runs 100 --profile_all

# Generate visualizations
python create_visualizations.py ablation_results
```

---

## 📊 Quick Statistics

| Metric | Value |
|--------|-------|
| Total Parameters | 35,746,307 |
| Total Layers Profiled | 192 |
| Conv2d Parameters | 32,828,291 (91.9%) |
| Linear Parameters | 2,889,600 (8.1%) |
| Forward Pass Time | 50.78 ms |
| Images/sec (batch=4) | ~78.7 |
| Peak Memory Layer | up.0.block.0.norm1 (6 MB avg) |
| Slowest Layer | up.0.block.0.conv1 (0.134 ms avg) |

---

## 📖 How to Read This Report

1. **Start with:** [COMPREHENSIVE_ABLATION_REPORT.md](COMPREHENSIVE_ABLATION_REPORT.md) for complete analysis
2. **View visualizations** in `figures/` for quick insights
3. **Examine raw data** in JSON/TXT files for specific details
4. **Use profiling scripts** to run your own experiments

---

## 🎯 Recommendations Priority

### High Impact (30-50% speedup potential)
- ✅ Optimize upsampling Conv2d layers
- ✅ Replace or optimize GroupNorm (4× slower than BatchNorm)
- ✅ Implement Conv+Norm+Activation fusion

### Medium Impact (10-20% speedup potential)
- ⚠️ Optimize attention mechanisms (Flash Attention)
- ⚠️ Fix quantization kernel performance
- ⚠️ Mixed precision training/inference

### Low Impact (<10% speedup)
- ℹ️ Remove dropout during inference
- ℹ️ Optimize timestep embeddings
- ℹ️ Profile different batch sizes

---

**End of Index**

For questions or additional analysis, refer to the comprehensive report or re-run the profiling scripts with different configurations.

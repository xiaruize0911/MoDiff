#!/usr/bin/env python3
"""
Visualization: INT4 vs INT8 Analysis
=====================================

Generates publication-quality plots and tables from benchmark results.
Reads from:
  - gemm_benchmark_results.json (from 01_gemm_microbenchmark.py)
  - pipeline_breakdown_results.json (from 02_pipeline_breakdown.py)
  - ../integration/results_ldm_benchmark/results.json (original full benchmark)

Outputs:
  - PNG plots
  - CSV summary tables
  - Markdown report
"""

import os
import sys
import json
import csv
import numpy as np

# Try to import matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not available. Install with: pip install matplotlib")
    print("Skipping plot generation, will generate tables only.")


OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_json(path):
    if not os.path.exists(path):
        print(f"Warning: {path} not found")
        return None
    with open(path) as f:
        return json.load(f)


# ============================================================================
# Plot 1: Overall LDM Benchmark Speedups (from original results.json)
# ============================================================================

def plot_ldm_speedups(ldm_results):
    """Bar chart of speedup vs FP32 for all modes."""
    if not HAS_MPL or not ldm_results:
        return
    
    modes = []
    speedups = []
    times = []
    
    mode_order = ['fp32', 'fp16', 'int8_baseline', 'int8', 'int8_static', 'int4_baseline', 'int4']
    colors = {
        'fp32': '#2196F3',
        'fp16': '#4CAF50', 
        'int8_baseline': '#FF9800',
        'int8': '#FF5722',
        'int8_static': '#E91E63',
        'int4_baseline': '#9C27B0',
        'int4': '#673AB7',
    }
    
    for mode in mode_order:
        if mode in ldm_results:
            modes.append(mode)
            spd = ldm_results[mode].get('speedup', 1.0)
            speedups.append(spd)
            times.append(ldm_results[mode]['time_per_step_ms'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Speedup bar chart
    bars = ax1.bar(range(len(modes)), speedups, 
                   color=[colors.get(m, '#999') for m in modes])
    ax1.set_xticks(range(len(modes)))
    ax1.set_xticklabels(modes, rotation=45, ha='right')
    ax1.set_ylabel('Speedup vs FP32')
    ax1.set_title('LDM Benchmark: Speedup vs FP32 (200 steps, 128 samples)')
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    for bar, spd in zip(bars, speedups):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{spd:.2f}x', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Time per step
    bars2 = ax2.bar(range(len(modes)), times,
                    color=[colors.get(m, '#999') for m in modes])
    ax2.set_xticks(range(len(modes)))
    ax2.set_xticklabels(modes, rotation=45, ha='right')
    ax2.set_ylabel('Time per Step (ms)')
    ax2.set_title('LDM Benchmark: Time per Diffusion Step')
    for bar, t in zip(bars2, times):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                f'{t:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'plot_ldm_speedups.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ============================================================================
# Plot 2: CUTLASS Conv Kernel Throughput (INT4 vs INT8)
# ============================================================================

def plot_cutlass_conv_throughput(gemm_results):
    """Compare raw CUTLASS conv kernel throughput."""
    if not HAS_MPL or not gemm_results:
        return
    
    conv_data = gemm_results.get('cutlass_conv', [])
    if not conv_data:
        return
    
    # Group by shape
    shapes = {}
    for entry in conv_data:
        shape = entry['shape']
        prec = entry['precision']
        if shape not in shapes:
            shapes[shape] = {}
        shapes[shape][prec] = entry
    
    shape_labels = list(shapes.keys())
    int8_tops = [shapes[s].get('int8', {}).get('tops', 0) for s in shape_labels]
    int4_tops = [shapes[s].get('int4', {}).get('tops', 0) for s in shape_labels]
    int8_ms = [shapes[s].get('int8', {}).get('median_ms', 0) for s in shape_labels]
    int4_ms = [shapes[s].get('int4', {}).get('median_ms', 0) for s in shape_labels]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    x = np.arange(len(shape_labels))
    width = 0.35
    
    # TOPS comparison
    bars1 = ax1.bar(x - width/2, int8_tops, width, label='INT8', color='#FF5722')
    bars2 = ax1.bar(x + width/2, int4_tops, width, label='INT4', color='#673AB7')
    ax1.set_ylabel('TOPS (Tera Operations/Sec)')
    ax1.set_title('CUTLASS Conv2d: Raw Throughput (INT8 vs INT4)')
    ax1.set_xticks(x)
    ax1.set_xticklabels([s.replace(',', '\n') for s in shape_labels], fontsize=7, rotation=0)
    ax1.legend()
    
    # Speedup ratio
    ratios = [i4/i8 if i8 > 0 else 0 for i4, i8 in zip(int4_ms, int8_ms)]
    inv_ratios = [i8/i4 if i4 > 0 else 0 for i4, i8 in zip(int4_ms, int8_ms)]
    
    colors = ['#4CAF50' if r < 1 else '#F44336' for r in ratios]
    bars3 = ax2.bar(x, inv_ratios, color=colors)
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='Break-even')
    ax2.axhline(y=1.5, color='green', linestyle='--', alpha=0.5, label='Expected ~1.5x (NVIDIA)')
    ax2.set_ylabel('INT4/INT8 Speedup')
    ax2.set_title('Pure Kernel Speedup: INT4 vs INT8 (>1 means INT4 is faster)')
    ax2.set_xticks(x)
    ax2.set_xticklabels([s.replace(',', '\n') for s in shape_labels], fontsize=7, rotation=0)
    ax2.legend()
    
    for bar, ratio in zip(bars3, inv_ratios):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{ratio:.2f}x', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'plot_cutlass_conv_throughput.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ============================================================================
# Plot 3: End-to-End Conv Breakdown (Quant + Compute)
# ============================================================================

def plot_e2e_breakdown(gemm_results):
    """Stacked bar chart showing quant overhead vs compute time."""
    if not HAS_MPL or not gemm_results:
        return
    
    e2e_data = gemm_results.get('end_to_end_conv', [])
    if not e2e_data:
        return
    
    fig, axes = plt.subplots(len(e2e_data), 1, figsize=(12, 4 * len(e2e_data)))
    if len(e2e_data) == 1:
        axes = [axes]
    
    for idx, entry in enumerate(e2e_data):
        ax = axes[idx]
        shape = entry['shape']
        
        precisions = ['fp32', 'fp16', 'int8', 'int4']
        labels = []
        quant_times = []
        conv_times = []
        other_times = []
        
        for p in precisions:
            if p in entry:
                labels.append(p.upper())
                total = entry[p]['median_ms']
                
                if p in ('int8', 'int4'):
                    q = entry.get(f'{p}_quant_only', {}).get('median_ms', 0)
                    c = entry.get(f'{p}_conv_only', {}).get('median_ms', 0)
                    other = max(0, total - q - c)
                    quant_times.append(q)
                    conv_times.append(c)
                    other_times.append(other)
                else:
                    quant_times.append(0)
                    conv_times.append(total)
                    other_times.append(0)
        
        x = np.arange(len(labels))
        
        ax.bar(x, conv_times, label='Compute (conv)', color='#2196F3')
        ax.bar(x, quant_times, bottom=conv_times, label='Quantization', color='#FF9800')
        ax.bar(x, other_times, bottom=[c+q for c,q in zip(conv_times, quant_times)], 
               label='Overhead', color='#F44336')
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel('Time (ms)')
        ax.set_title(f'End-to-End Break down: {shape}')
        ax.legend()
        
        # Annotate totals
        for i, (c, q, o) in enumerate(zip(conv_times, quant_times, other_times)):
            total = c + q + o
            ax.text(i, total + 0.1, f'{total:.2f}ms', ha='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'plot_e2e_breakdown.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ============================================================================
# Plot 4: Quantization Overhead as % of Total
# ============================================================================

def plot_quant_overhead_pct(gemm_results):
    """Show what fraction of total time is spent on quantization."""
    if not HAS_MPL or not gemm_results:
        return
    
    e2e_data = gemm_results.get('end_to_end_conv', [])
    if not e2e_data:
        return
    
    shapes = []
    int8_quant_pct = []
    int4_quant_pct = []
    int8_conv_speedup = []
    int4_conv_speedup = []
    
    for entry in e2e_data:
        shapes.append(entry['shape'].split(',')[1].replace('C=', 'C'))  # Short label
        
        int8_total = entry.get('int8', {}).get('median_ms', 1)
        int4_total = entry.get('int4', {}).get('median_ms', 1)
        int8_q = entry.get('int8_quant_only', {}).get('median_ms', 0)
        int4_q = entry.get('int4_quant_only', {}).get('median_ms', 0)
        int8_c = entry.get('int8_conv_only', {}).get('median_ms', 1)
        int4_c = entry.get('int4_conv_only', {}).get('median_ms', 1)
        fp32_ms = entry.get('fp32', {}).get('median_ms', 1)
        
        int8_quant_pct.append(100 * int8_q / int8_total if int8_total > 0 else 0)
        int4_quant_pct.append(100 * int4_q / int4_total if int4_total > 0 else 0)
        int8_conv_speedup.append(fp32_ms / int8_c if int8_c > 0 else 0)
        int4_conv_speedup.append(fp32_ms / int4_c if int4_c > 0 else 0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    x = np.arange(len(shapes))
    width = 0.35
    
    # Quant overhead percentage
    ax1.bar(x - width/2, int8_quant_pct, width, label='INT8', color='#FF5722')
    ax1.bar(x + width/2, int4_quant_pct, width, label='INT4', color='#673AB7')
    ax1.set_ylabel('Quantization Overhead (%)')
    ax1.set_title('Quantize+Pack Time as % of Total E2E Time')
    ax1.set_xticks(x)
    ax1.set_xticklabels(shapes)
    ax1.legend()
    
    # Pure conv speedup vs FP32
    ax2.bar(x - width/2, int8_conv_speedup, width, label='INT8 conv', color='#FF5722')
    ax2.bar(x + width/2, int4_conv_speedup, width, label='INT4 conv', color='#673AB7')
    ax2.set_ylabel('Speedup vs FP32')
    ax2.set_title('Pure Convolution Speedup (no quant overhead)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(shapes)
    ax2.legend()
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'plot_quant_overhead.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ============================================================================
# Plot 5: Pipeline Breakdown Comparison (INT8 vs INT4)
# ============================================================================

def plot_pipeline_breakdown(pipeline_results):
    """Pie/bar charts showing where time is spent in INT8 vs INT4 pipeline."""
    if not HAS_MPL or not pipeline_results:
        return
    
    # Compare INT8 and INT4 layer-level timing
    int8 = pipeline_results.get('int8', {})
    int4 = pipeline_results.get('int4', {})
    
    if not int8 or not int4:
        print("Skipping pipeline breakdown plot: missing INT8 or INT4 data")
        return
    
    int8_layers = int8.get('layer_profiler', {})
    int4_layers = int4.get('layer_profiler', {})
    
    if not int8_layers or not int4_layers:
        print("Skipping pipeline breakdown plot: no layer profiling data")
        return
    
    # Collect all layer types
    all_types = sorted(set(list(int8_layers.keys()) + list(int4_layers.keys())))
    
    int8_times = [int8_layers.get(lt, {}).get('total_ms', 0) for lt in all_types]
    int4_times = [int4_layers.get(lt, {}).get('total_ms', 0) for lt in all_types]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    x = np.arange(len(all_types))
    width = 0.35
    
    ax1.barh(x - width/2, int8_times, width, label='INT8', color='#FF5722')
    ax1.barh(x + width/2, int4_times, width, label='INT4', color='#673AB7')
    ax1.set_yticks(x)
    ax1.set_yticklabels(all_types, fontsize=9)
    ax1.set_xlabel('Total Time (ms)')
    ax1.set_title('Layer-Level Time: INT8 vs INT4')
    ax1.legend()
    
    # Pie chart for INT4
    colors_pie = plt.cm.Set3(np.linspace(0, 1, len(all_types)))
    int4_total = sum(int4_times)
    if int4_total > 0:
        pcts = [t/int4_total*100 for t in int4_times]
        # Only show labels for >3%
        labels_filtered = [lt if p > 3 else '' for lt, p in zip(all_types, pcts)]
        ax2.pie(int4_times, labels=labels_filtered, colors=colors_pie, autopct='%1.1f%%',
                pctdistance=0.85, startangle=140)
        ax2.set_title(f'INT4 Pipeline Time Distribution\n(Total: {int4_total:.0f} ms)')
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'plot_pipeline_breakdown.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ============================================================================
# Plot 6: FP32 matmul throughput comparison
# ============================================================================

def plot_matmul_throughput(gemm_results):
    """FP32 vs FP16 matmul TFLOPS at various sizes."""
    if not HAS_MPL or not gemm_results:
        return
    
    matmul_data = gemm_results.get('matmul', [])
    if not matmul_data:
        return
    
    # Group by size
    sizes = {}
    for entry in matmul_data:
        key = f"{entry['M']}x{entry['N']}x{entry['K']}"
        if key not in sizes:
            sizes[key] = {}
        sizes[key][entry['dtype']] = entry
    
    labels = list(sizes.keys())
    fp32_tflops = [sizes[s].get('torch.float32', {}).get('tflops', 0) for s in labels]
    fp16_tflops = [sizes[s].get('torch.float16', {}).get('tflops', 0) for s in labels]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(labels))
    width = 0.35
    
    ax.bar(x - width/2, fp32_tflops, width, label='FP32 (TF32)', color='#2196F3')
    ax.bar(x + width/2, fp16_tflops, width, label='FP16', color='#4CAF50')
    
    ax.set_ylabel('TFLOPS')
    try:
        import torch as _torch
        gpu_name = _torch.cuda.get_device_name(0) if _torch.cuda.is_available() else "CPU"
    except Exception:
        gpu_name = "GPU"
    ax.set_title(f'torch.matmul Throughput ({gpu_name})')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.legend()
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'plot_matmul_throughput.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ============================================================================
# Generate Markdown Report
# ============================================================================

def generate_report(ldm_results, gemm_results, pipeline_results):
    """Generate comprehensive markdown report."""
    report = []
    report.append("# INT4 vs INT8 Performance Analysis")
    report.append("")
    report.append(f"**GPU:** {gemm_results.get('metadata', {}).get('gpu', 'Unknown') if gemm_results else 'Unknown'}")
    report.append(f"**CUDA:** {gemm_results.get('metadata', {}).get('cuda_version', 'Unknown') if gemm_results else 'Unknown'}")
    report.append(f"**PyTorch:** {gemm_results.get('metadata', {}).get('pytorch_version', 'Unknown') if gemm_results else 'Unknown'}")
    report.append("")
    
    # Section 1: LDM Benchmark Results
    report.append("## 1. LDM Benchmark Results (Full Pipeline)")
    report.append("")
    if ldm_results:
        report.append("| Mode | Time/Sample (s) | Time/Step (ms) | Speedup vs FP32 |")
        report.append("|------|----------------|---------------|-----------------|")
        fp32_t = ldm_results.get('fp32', {}).get('time_per_sample', 1)
        for mode in ['fp32', 'fp16', 'int8_baseline', 'int8', 'int8_static', 'int4_baseline', 'int4']:
            if mode in ldm_results:
                r = ldm_results[mode]
                spd = fp32_t / r['time_per_sample'] if r['time_per_sample'] > 0 else 0
                report.append(f"| {mode} | {r['time_per_sample']:.3f} | {r['time_per_step_ms']:.2f} | {spd:.2f}x |")
        report.append("")
        report.append("**Key Observation:** INT4 achieves only ~1.76x speedup vs FP32, while INT8 achieves ~1.65x.")
        report.append("The speedup gap between INT4 and INT8 is only ~7%, far from the theoretical ~50% expected.")
        report.append("")
        report.append("![LDM Speedups](plot_ldm_speedups.png)")
        report.append("")
    
    # Section 2: Raw Kernel Throughput
    report.append("## 2. Raw CUTLASS Kernel Throughput")
    report.append("")
    if gemm_results and 'cutlass_conv' in gemm_results:
        conv_data = gemm_results['cutlass_conv']
        # Group by shape
        shapes = {}
        for entry in conv_data:
            shape = entry['shape']
            prec = entry['precision']
            if shape not in shapes:
                shapes[shape] = {}
            shapes[shape][prec] = entry
        
        report.append("| Shape | INT8 (ms) | INT4 (ms) | INT4/INT8 Speedup | INT8 TOPS | INT4 TOPS |")
        report.append("|-------|----------|----------|-------------------|----------|----------|")
        for shape, data in shapes.items():
            i8 = data.get('int8', {})
            i4 = data.get('int4', {})
            spd = i8.get('median_ms', 0) / i4.get('median_ms', 1) if i4.get('median_ms', 0) > 0 else 0
            report.append(f"| {shape} | {i8.get('median_ms', 0):.3f} | {i4.get('median_ms', 0):.3f} | {spd:.2f}x | {i8.get('tops', 0):.1f} | {i4.get('tops', 0):.1f} |")
        report.append("")
        report.append("![CUTLASS Throughput](plot_cutlass_conv_throughput.png)")
        report.append("")
    
    # Section 3: End-to-End Breakdown
    report.append("## 3. End-to-End Convolution Breakdown")
    report.append("")
    report.append("This shows time split between quantization overhead and actual compute:")
    report.append("")
    if gemm_results and 'end_to_end_conv' in gemm_results:
        e2e = gemm_results['end_to_end_conv']
        for entry in e2e:
            shape = entry['shape']
            report.append(f"### {shape}")
            report.append("")
            report.append("| Component | INT8 (ms) | INT4 (ms) |")
            report.append("|-----------|----------|----------|")
            for comp in ['quant_only', 'conv_only']:
                i8 = entry.get(f'int8_{comp}', {}).get('median_ms', 0)
                i4 = entry.get(f'int4_{comp}', {}).get('median_ms', 0)
                report.append(f"| {comp} | {i8:.4f} | {i4:.4f} |")
            i8_total = entry.get('int8', {}).get('median_ms', 0)
            i4_total = entry.get('int4', {}).get('median_ms', 0)
            report.append(f"| **Total** | **{i8_total:.4f}** | **{i4_total:.4f}** |")
            
            i8_q = entry.get('int8_quant_only', {}).get('median_ms', 0)
            i4_q = entry.get('int4_quant_only', {}).get('median_ms', 0)
            report.append(f"| Quant % of total | {100*i8_q/i8_total:.1f}% | {100*i4_q/i4_total:.1f}% |" if i8_total > 0 and i4_total > 0 else "")
            report.append("")
        
        report.append("![E2E Breakdown](plot_e2e_breakdown.png)")
        report.append("![Quant Overhead](plot_quant_overhead.png)")
        report.append("")
    
    # Section 4: Analysis
    report.append("## 4. Why INT4 Doesn't Show Expected Speedup")
    report.append("")
    report.append("### Root Causes:")
    report.append("")
    report.append("1. **Quantization + Packing Overhead (Amdahl's Law)**")
    report.append("   - INT4 requires packing (2 values per byte), adding overhead that INT8 doesn't have")
    report.append("   - Dynamic scale computation (absmax + division) is identical cost for both precisions")
    report.append("   - For small/medium convolutions, quant overhead can be 15-40% of total time")
    report.append("")
    report.append("2. **Memory-Bound Operations Dominate**")
    report.append("   - Many conv layers in LDM have small spatial dimensions (8x8, 16x16) with many channels")
    report.append("   - These are memory-bound, not compute-bound — reducing precision helps less")
    report.append("   - INT4 only halves the activation memory (weights already packed), but CUTLASS data movement overhead remains similar")
    report.append("")
    report.append("3. **Non-Quantized Overhead**")
    report.append("   - GroupNorm, SiLU, attention, skip connections, etc. run at FP32/FP16 regardless of quantization")
    report.append("   - These operations are identical between INT8 and INT4 modes")
    report.append("   - They represent a significant fraction of total pipeline time")
    report.append("")
    report.append("4. **CUTLASS INT4 Kernel Maturity**")
    report.append("   - INT4 tensor core support varies by GPU architecture")
    report.append("   - On some GPUs, INT4 CUTLASS kernels may not achieve peak theoretical throughput")
    report.append("   - The packing/unpacking within the GEMM kernel adds instruction overhead")
    report.append("")
    report.append("5. **MoDiff Overhead is Constant**")
    report.append("   - The sub_absmax_scale, dequant_accumulate, and scale_accumulate operations")
    report.append("     have similar cost for INT4 and INT8 (they operate on FP32 accumulators)")
    report.append("   - These fused kernels are a fixed overhead regardless of quantization level")
    report.append("")
    report.append("### Theoretical vs Practical")
    report.append("")
    report.append("| Factor | Theory | Practice |")
    report.append("|--------|--------|----------|")
    report.append("| Tensor core throughput | INT4 ~2x INT8 | Depends on kernel efficiency |")
    report.append("| Memory bandwidth | INT4 reads ~0.5x INT8 | Only for packed activations/weights |")
    report.append("| Quantize overhead | N/A | INT4 packing adds cost |")
    report.append("| Non-conv layers | N/A | Same cost for both |")
    report.append("| Overall pipeline | ~50% faster | ~7% faster |")
    report.append("")
    report.append("### NVIDIA Blog Reference")
    report.append("")
    report.append("The NVIDIA blog (https://developer.nvidia.com/blog/int4-for-ai-inference/) shows ~50% speedup")
    report.append("for INT4 vs INT8, but this is for **pure GEMM throughput on large matrices** where the")
    report.append("operation is compute-bound and quantization overhead is negligible.")
    report.append("")
    report.append("### What This Means for MoDiff")
    report.append("")
    report.append("1. **Baseline quantization is good:** Both INT8 and INT4 show solid speedups over FP32")
    report.append("2. **MoDiff doesn't hurt speed:** MoDiff modes are faster than baseline (temporal caching helps)")
    report.append("3. **INT4 benefit is real but modest:** The additional ~7% speedup from INT4 vs INT8 is expected")
    report.append("   given the overhead-dominated pipeline")
    report.append("4. **For bigger models:** INT4 should show larger relative gains where conv is a bigger fraction")
    report.append("   of total compute time")
    report.append("")
    
    # Section 5: Pipeline Breakdown
    if pipeline_results:
        report.append("## 5. Detailed Pipeline Breakdown")
        report.append("")
        
        for mode in ['int8', 'int4']:
            if mode in pipeline_results:
                data = pipeline_results[mode]
                report.append(f"### {mode.upper()} Pipeline")
                report.append("")
                report.append(f"- Total time: {data['total_time_s']:.2f}s")
                report.append(f"- Time/sample: {data['time_per_sample_s']:.3f}s")
                report.append(f"- Time/step: {data['time_per_step_ms']:.2f}ms")
                report.append("")
                
                layers = data.get('layer_profiler', {})
                if layers:
                    total_ms = sum(v['total_ms'] for v in layers.values())
                    report.append("| Layer Type | Total (ms) | Calls | Avg (ms) | % of Total |")
                    report.append("|-----------|-----------|-------|---------|-----------|")
                    for lt, s in sorted(layers.items(), key=lambda x: x[1]['total_ms'], reverse=True):
                        pct = s['total_ms'] / total_ms * 100 if total_ms > 0 else 0
                        report.append(f"| {lt} | {s['total_ms']:.1f} | {s['count']} | {s['avg_ms']:.3f} | {pct:.1f}% |")
                    report.append("")
        
        report.append("![Pipeline Breakdown](plot_pipeline_breakdown.png)")
        report.append("")
    
    # Write report
    report_path = os.path.join(OUTPUT_DIR, 'ANALYSIS_REPORT.md')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    print(f"Report saved to {report_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("Loading results...")
    
    ldm_results = load_json(os.path.join(OUTPUT_DIR, '..', 'integration', 'results_ldm_benchmark', 'results.json'))
    gemm_results = load_json(os.path.join(OUTPUT_DIR, 'gemm_benchmark_results.json'))
    pipeline_results = load_json(os.path.join(OUTPUT_DIR, 'pipeline_breakdown_results.json'))
    
    # Also import torch for GPU name in matmul plot
    import torch
    
    if HAS_MPL:
        print("Generating plots...")
        plot_ldm_speedups(ldm_results)
        plot_cutlass_conv_throughput(gemm_results)
        plot_e2e_breakdown(gemm_results)
        plot_quant_overhead_pct(gemm_results)
        plot_pipeline_breakdown(pipeline_results)
        plot_matmul_throughput(gemm_results)
    else:
        print("matplotlib not available, skipping plots")
    
    print("\nGenerating report...")
    generate_report(ldm_results, gemm_results, pipeline_results)
    
    print("\nDone! All outputs saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()

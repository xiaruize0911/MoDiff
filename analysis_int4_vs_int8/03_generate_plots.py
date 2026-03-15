#!/usr/bin/env python3
"""
Visualization: INT4 vs INT8 Analysis
=====================================

Generates publication-quality plots and tables from benchmark results.
Reads from:
  - gemm_benchmark_results.json (from 01_gemm_microbenchmark.py)
  - pipeline_breakdown_results.json (from 02_pipeline_breakdown.py)
  - ../integration/results/ldm/results.json (original full benchmark)

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
    # Global font sizes — applied to all plots
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
    })
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
    
    mode_order = ['fp32', 'int8_baseline', 'int8', 'int4_baseline', 'int4']
    colors = {
        'fp32':          '#B8E5FA',
        'int8_baseline': '#EEC186',
        'int8':          '#F7A6AC',
        'int4_baseline': '#EEF0A7',
        'int4':          '#F7B7D2',
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
    path = os.path.join(OUTPUT_DIR, 'plot_ldm_speedups.pdf')
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
    bars1 = ax1.bar(x - width/2, int8_tops, width, label='INT8', color='#F7A6AC')
    bars2 = ax1.bar(x + width/2, int4_tops, width, label='INT4', color='#F7B7D2')
    ax1.set_ylabel('TOPS (Tera Operations/Sec)')
    ax1.set_title('CUTLASS Conv2d: Raw Throughput (INT8 vs INT4)')
    ax1.set_xticks(x)
    ax1.set_xticklabels([s.replace(',', '\n') for s in shape_labels], fontsize=9, rotation=0)
    ax1.legend()
    
    # Speedup ratio
    ratios = [i4/i8 if i8 > 0 else 0 for i4, i8 in zip(int4_ms, int8_ms)]
    inv_ratios = [i8/i4 if i4 > 0 else 0 for i4, i8 in zip(int4_ms, int8_ms)]
    
    colors = ['#B2DBB9' if r < 1 else '#F7A6AC' for r in ratios]
    bars3 = ax2.bar(x, inv_ratios, color=colors)
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='Break-even')
    ax2.axhline(y=1.5, color='green', linestyle='--', alpha=0.5, label='Expected ~1.5x (NVIDIA)')
    ax2.set_ylabel('INT4/INT8 Speedup')
    ax2.set_title('Pure Kernel Speedup: INT4 vs INT8')
    ax2.set_xticks(x)
    ax2.set_xticklabels([s.replace(',', '\n') for s in shape_labels], fontsize=9, rotation=0)
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
        
        ax.bar(x, conv_times, label='Compute (conv)', color='#B8E5FA')
        ax.bar(x, quant_times, bottom=conv_times, label='Quantization', color='#EEC186')
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
        
        # Correct overhead: (total - conv_only) / total
        # Captures ALL non-compute cost in the actual pipeline.
        # The old formula (quant_only / total) underestimates because
        # quant_only + conv_only != total (gap = kernel scheduling + alloc overhead).
        int8_quant_pct.append(100 * (int8_total - int8_c) / int8_total if int8_total > 0 else 0)
        int4_quant_pct.append(100 * (int4_total - int4_c) / int4_total if int4_total > 0 else 0)
        int8_conv_speedup.append(fp32_ms / int8_c if int8_c > 0 else 0)
        int4_conv_speedup.append(fp32_ms / int4_c if int4_c > 0 else 0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    x = np.arange(len(shapes))
    width = 0.35
    
    # Quant overhead percentage
    ax1.bar(x - width/2, int8_quant_pct, width, label='INT8', color='#F7A6AC')
    ax1.bar(x + width/2, int4_quant_pct, width, label='INT4', color='#F7B7D2')
    ax1.set_ylabel('Overhead = (total − conv_only) / total (%)')
    ax1.set_title('True Quant Overhead: (total − conv_only) / total\n[includes quant + kernel scheduling + alloc]')
    ax1.set_xticks(x)
    ax1.set_xticklabels(shapes)
    ax1.legend()
    
    # Pure conv speedup vs FP32
    ax2.bar(x - width/2, int8_conv_speedup, width, label='INT8 conv', color='#F7A6AC')
    ax2.bar(x + width/2, int4_conv_speedup, width, label='INT4 conv', color='#F7B7D2')
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
    
    ax1.barh(x - width/2, int8_times, width, label='INT8', color='#F7A6AC')
    ax1.barh(x + width/2, int4_times, width, label='INT4', color='#F7B7D2')
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
    
    ax.bar(x - width/2, fp32_tflops, width, label='FP32 (TF32)', color='#B8E5FA')
    ax.bar(x + width/2, fp16_tflops, width, label='FP16', color='#B2DBB9')
    
    ax.set_ylabel('TFLOPS')
    try:
        import torch as _torch
        gpu_name = _torch.cuda.get_device_name(0) if _torch.cuda.is_available() else "CPU"
    except Exception:
        gpu_name = "GPU"
    ax.set_title(f'torch.matmul Throughput ({gpu_name})')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
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
        # Dynamically compute key observation numbers from actual data
        fp32_t = ldm_results.get('fp32', {}).get('time_per_sample', 1.0)
        int4_bl_t = ldm_results.get('int4_baseline', {}).get('time_per_sample', fp32_t)
        int8_bl_t = ldm_results.get('int8_baseline', {}).get('time_per_sample', fp32_t)
        int4_bl_spd = fp32_t / int4_bl_t if int4_bl_t > 0 else 1.0
        int8_bl_spd = fp32_t / int8_bl_t if int8_bl_t > 0 else 1.0
        gap_pct = abs(int4_bl_spd - int8_bl_spd) / int8_bl_spd * 100 if int8_bl_spd > 0 else 0
        report.append(f"**Key Observation:** INT4 baseline achieves ~{int4_bl_spd:.2f}x speedup vs FP32, "
                      f"while INT8 baseline achieves ~{int8_bl_spd:.2f}x.")
        report.append(f"The speedup gap between INT4 and INT8 baselines is ~{gap_pct:.1f}%, "
                      f"far from the theoretical ~50% expected due to quantization overhead, "
                      f"non-quantized layers, and memory-bound operations.")
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
    # Compute measured gap for the table (safe even if ldm_results is None)
    _fp32_t = ldm_results.get('fp32', {}).get('time_per_sample', 1.0) if ldm_results else 1.0
    _i4_bl  = ldm_results.get('int4_baseline', {}).get('time_per_sample', _fp32_t) if ldm_results else _fp32_t
    _i8_bl  = ldm_results.get('int8_baseline', {}).get('time_per_sample', _fp32_t) if ldm_results else _fp32_t
    _i4_spd = _fp32_t / _i4_bl if _i4_bl > 0 else 1.0
    _i8_spd = _fp32_t / _i8_bl if _i8_bl > 0 else 1.0
    _gap    = abs(_i4_spd - _i8_spd) / _i8_spd * 100 if _i8_spd > 0 else 0.0

    report.append("### Theoretical vs Practical")
    report.append("")
    report.append("| Factor | Theory | Practice |")
    report.append("|--------|--------|----------|")
    report.append("| Tensor core throughput | INT4 ~2x INT8 | Depends on kernel efficiency |")
    report.append("| Memory bandwidth | INT4 reads ~0.5x INT8 | Only for packed activations/weights |")
    report.append("| Quantize overhead | N/A | INT4 packing adds cost |")
    report.append("| Non-conv layers | N/A | Same cost for both |")
    report.append(f"| Overall pipeline | ~50% faster | ~{_gap:.1f}% faster |")
    report.append("")
    report.append("### NVIDIA Blog Reference")
    report.append("")
    report.append("The NVIDIA blog (https://developer.nvidia.com/blog/int4-for-ai-inference/) shows ~50% speedup")
    report.append("for INT4 vs INT8, but this is for **pure GEMM throughput on large matrices** where the")
    report.append("operation is compute-bound and quantization overhead is negligible.")
    report.append("")
    report.append("### What This Means for MoDiff")
    report.append("")
    if ldm_results:
        fp32_t = ldm_results.get('fp32', {}).get('time_per_sample', 1.0)
        int4_bl_t = ldm_results.get('int4_baseline', {}).get('time_per_sample', fp32_t)
        int8_bl_t = ldm_results.get('int8_baseline', {}).get('time_per_sample', fp32_t)
        int4_mdf_t = ldm_results.get('int4', {}).get('time_per_sample', fp32_t)
        int8_mdf_t = ldm_results.get('int8', {}).get('time_per_sample', fp32_t)
        int4_bl_spd = fp32_t / int4_bl_t if int4_bl_t > 0 else 1.0
        int8_bl_spd = fp32_t / int8_bl_t if int8_bl_t > 0 else 1.0
        int4_mdf_spd = fp32_t / int4_mdf_t if int4_mdf_t > 0 else 1.0
        int8_mdf_spd = fp32_t / int8_mdf_t if int8_mdf_t > 0 else 1.0
        gap_pct = abs(int4_bl_spd - int8_bl_spd) / int8_bl_spd * 100 if int8_bl_spd > 0 else 0
        modiff_overhead_int8_pct = (int8_bl_t - int8_mdf_t) / int8_bl_t * -100 if int8_bl_t > 0 else 0
        modiff_overhead_int4_pct = (int4_bl_t - int4_mdf_t) / int4_bl_t * -100 if int4_bl_t > 0 else 0
    else:
        gap_pct = 0.0
        modiff_overhead_int8_pct = 0.0
        modiff_overhead_int4_pct = 0.0
    report.append("1. **Baseline quantization is good:** Both INT8 and INT4 show solid speedups over FP32")
    report.append(f"2. **MoDiff adds modest runtime overhead:** Compared to baseline, MoDiff INT8 is "
                  f"~{abs(modiff_overhead_int8_pct):.1f}% slower and MoDiff INT4 is "
                  f"~{abs(modiff_overhead_int4_pct):.1f}% slower. This overhead comes from storing "
                  f"intermediate activations and computing residuals (`a_t - â_{{t+1}}`). "
                  f"Per the paper, the benefit is **quantization quality** (FID/IS scores at lower bits), "
                  f"not raw throughput.")
    report.append(f"3. **INT4 gap is real but modest:** INT4 baseline is ~{gap_pct:.1f}% faster than "
                  f"INT8 baseline, far below the theoretical ~50%. "
                  f"Overhead-dominated pipelines limit the gain.")
    report.append("4. **Paper focus:** MoDiff's core contribution is enabling 3-bit (or lower) activation "
                  "quantization without FID degradation — not raw speed. On CIFAR-10, LCQ+MoDiff "
                  "at W8/A3 achieves a similar sFID to full-precision, while vanilla quant degrades "
                  "significantly at even 6-bit activation.")
    report.append("5. **For bigger models:** INT4 should show larger relative gains where conv is a bigger "
                  "fraction of total compute time")
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
# Plot: E2E Component Breakdown Bar Chart (from experiment_results.json)
# ============================================================================

def plot_exp2_component_bars():
    """Grouped bar chart of per-component times from exp2 breakdown."""
    if not HAS_MPL:
        return

    exp_results_path = os.path.join(OUTPUT_DIR, 'experiment_results.json')
    if not os.path.exists(exp_results_path):
        print(f"Warning: {exp_results_path} not found, skipping component bars")
        return

    with open(exp_results_path) as f:
        data = json.load(f)

    exp = data.get('exp2_breakdown', {})
    modes = [m for m in ['fp32', 'int8', 'int4'] if m in exp]
    if not modes:
        print("Warning: no exp2_breakdown data found")
        return

    generic_map = {
        'Conv2d(FP32)': 'Conv2d', 'Int8Conv2d': 'Conv2d', 'Int4Conv2d': 'Conv2d',
        'Linear(FP32)': 'Linear', 'Int8Linear': 'Linear', 'Int4Linear': 'Linear',
        'Attention': 'Attention', 'GroupNorm': 'GroupNorm', 'SiLU': 'SiLU',
    }
    generic_order = ['Conv2d', 'Attention', 'Linear', 'GroupNorm', 'SiLU']
    mode_colors = {
        'fp32': '#B8E5FA', 'fp16': '#B2DBB9', 'int8': '#F7A6AC', 'int4': '#F7B7D2',
    }

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(generic_order))
    n = len(modes)
    width = 0.8 / n

    for i, mode in enumerate(modes):
        ls = exp[mode]['layer_stats']
        agg = {}
        for lt, s in ls.items():
            g = generic_map.get(lt, lt)
            agg[g] = agg.get(g, 0) + s['total_ms']
        vals = [agg.get(g, 0) for g in generic_order]
        offset = (i - n / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=mode.upper(),
                      color=mode_colors.get(mode, '#999'), edgecolor='black', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(generic_order)
    ax.set_ylabel('Total Time (ms)')
    ax.set_title('Per-Component Time: FP32 vs INT8 vs INT4')
    ax.legend(loc='upper right')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'plot_02b_component_bars.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


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
        plot_exp2_component_bars()
    else:
        print("matplotlib not available, skipping plots")
    
    print("\nGenerating report...")
    generate_report(ldm_results, gemm_results, pipeline_results)
    
    print("\nDone! All outputs saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()

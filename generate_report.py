#!/usr/bin/env python3
"""Generate a measured-data-only benchmark report and plots."""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPORT_DIR = "experiment_report"
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(os.path.join(REPORT_DIR, "plots"), exist_ok=True)

with open("integration/results/experiment_overall_v2/results.json") as f:
    overall = json.load(f)

with open("integration/results/experiment_extended/extended_results.json") as f:
    extended = json.load(f)

with open("integration/results/experiment_static_dynamic/static_dynamic_results.json") as f:
    static_dynamic = json.load(f)

with open("integration/results/experiment_static_dynamic/quality_summary.json") as f:
    quality = json.load(f)

with open("integration/results/extended/bottleneck_results.json") as f:
    bottleneck = json.load(f)

kernel_timing = extended.get("kernel_timing", {})
qt = extended.get("quantization_timing", {})
sd = static_dynamic
comps = bottleneck["model_profile"]["components"]
categories_sorted = sorted(comps.keys(), key=lambda k: -comps[k]["per_step_ms"])
hooked = bottleneck["model_profile"]["total_hooked_per_step_ms"]
total_ms = bottleneck["model_profile"]["total_per_step_ms"]

config_labels = [
    ("INT8 C=192 H=W=32", "INT8_32x192x32x32"),
    ("INT4 C=192 H=W=32", "INT4_32x192x32x32"),
    ("INT8 C=384 H=W=16", "INT8_32x384x16x16"),
    ("INT4 C=384 H=W=16", "INT4_32x384x16x16"),
    ("INT8 C=768 H=W=8", "INT8_32x768x8x8"),
    ("INT4 C=768 H=W=8", "INT4_32x768x8x8"),
]


def save_plot(fig, filename):
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "plots", filename), dpi=150, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# Plot 1: Overall model timing only
# ============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
overall_labels = ["FP32", "FP16", "INT8\nBaseline", "INT8\nMoDiff", "INT4\nBaseline", "INT4\nMoDiff"]
overall_keys = ["fp32", "fp16", "int8_baseline", "int8", "int4_baseline", "int4"]
overall_colors = ["#888888", "#5B9BD5", "#ED7D31", "#FFC000", "#70AD47", "#4472C4"]
time_per_sample = [overall[k]["time_per_sample"] for k in overall_keys]
time_per_step = [overall[k]["time_per_step_ms"] for k in overall_keys]

bars = ax1.bar(overall_labels, time_per_sample, color=overall_colors, edgecolor='black', linewidth=0.5)
ax1.set_ylabel("Time per Sample (s)")
ax1.set_title("Overall Model Latency")
ax1.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, time_per_sample):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{val:.3f}s", ha='center', va='bottom', fontsize=9)

bars = ax2.bar(overall_labels, time_per_step, color=overall_colors, edgecolor='black', linewidth=0.5)
ax2.set_ylabel("Time per Step (ms)")
ax2.set_title("Overall Per-Step Timing")
ax2.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, time_per_step):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f"{val:.2f}ms", ha='center', va='bottom', fontsize=9)

save_plot(fig, "01_overall_speedup.png")


# ============================================================================
# Plot 2: Fused vs separate kernel timings only
# ============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
kernel_labels = ["INT8\n192×32²", "INT4\n192×32²", "INT8\n384×16²", "INT4\n384×16²", "INT8\n768×8²", "INT4\n768×8²"]
x = np.arange(len(kernel_labels))
w = 0.35
fused_total = [kernel_timing[k]["fused_total_ms"] for _, k in config_labels]
separate_total = [kernel_timing[k]["separate_total_ms"] for _, k in config_labels]

ax1.bar(x - w/2, fused_total, w, label='Fused Total', color='#4472C4', edgecolor='black', linewidth=0.5)
ax1.bar(x + w/2, separate_total, w, label='Separate Total', color='#ED7D31', edgecolor='black', linewidth=0.5)
ax1.set_xticks(x)
ax1.set_xticklabels(kernel_labels)
ax1.set_ylabel("Time (ms)")
ax1.set_title("Kernel Timing: Total")
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

fused_step1 = [kernel_timing[k]["fused_step1_ms"] for _, k in config_labels]
fused_conv = [kernel_timing[k]["fused_conv_ms"] for _, k in config_labels]
ax2.bar(x - w/2, fused_step1, w, label='Fused Step1', color='#70AD47', edgecolor='black', linewidth=0.5)
ax2.bar(x + w/2, fused_conv, w, label='Fused Conv', color='#9B59B6', edgecolor='black', linewidth=0.5)
ax2.set_xticks(x)
ax2.set_xticklabels(kernel_labels)
ax2.set_ylabel("Time (ms)")
ax2.set_title("Kernel Timing: Fused Components")
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

save_plot(fig, "02_kernel_fused_vs_separate.png")


# ============================================================================
# Plot 3: Component timings only
# ============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
component_names = categories_sorted
component_vals = [comps[k]["per_step_ms"] for k in component_names]
bars = ax1.barh(component_names, component_vals, color=plt.cm.Set3(np.linspace(0, 1, len(component_names))), edgecolor='black', linewidth=0.5)
ax1.set_xlabel("Time per Step (ms)")
ax1.set_title("Per-Component UNet Breakdown")
ax1.invert_yaxis()
ax1.grid(axis='x', alpha=0.3)
for bar, val in zip(bars, component_vals):
    ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, f"{val:.3f}ms", va='center', fontsize=9)

summary_labels = ["Total Hooked", "Wall-clock Total"]
summary_vals = [hooked, total_ms]
bars = ax2.bar(summary_labels, summary_vals, color=['#4472C4', '#888888'], edgecolor='black', linewidth=0.5)
ax2.set_ylabel("Time per Step (ms)")
ax2.set_title("Profiler Summary")
ax2.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, summary_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, f"{val:.2f}ms", ha='center', va='bottom', fontsize=9)

save_plot(fig, "03_component_breakdown.png")


# ============================================================================
# Plot 4: Static vs dynamic timings only
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, prec in zip(axes, ["int8", "int4"]):
    labels = [f"{prec.upper()}\nDynamic\nBaseline", f"{prec.upper()}\nStatic\nBaseline", f"{prec.upper()}\nDynamic\nMoDiff", f"{prec.upper()}\nStatic\nMoDiff"]
    keys = [f"{prec}_dynamic_baseline", f"{prec}_static_baseline", f"{prec}_dynamic_modiff", f"{prec}_static_modiff"]
    vals = [sd[k]["time_per_sample_s"] for k in keys]
    bars = ax.bar(labels, vals, color=['#ED7D31', '#FFC000', '#4472C4', '#70AD47'], edgecolor='black', linewidth=0.5)
    ax.set_ylabel("Time per Sample (s)")
    ax.set_title(f"{prec.upper()} Static vs Dynamic")
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.004, f"{val:.3f}s", ha='center', va='bottom', fontsize=9)

save_plot(fig, "04_static_vs_dynamic.png")


# ============================================================================
# Plot 5: Quantization timing only
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 5))
dyn_ms = [qt[k]["dynamic_quant_ms"] for _, k in config_labels]
stat_ms = [qt[k]["static_quant_ms"] for _, k in config_labels]
x = np.arange(len(kernel_labels))
w = 0.35
ax.bar(x - w/2, dyn_ms, w, label='Dynamic Quant', color='#ED7D31', edgecolor='black', linewidth=0.5)
ax.bar(x + w/2, stat_ms, w, label='Static Quant', color='#70AD47', edgecolor='black', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(kernel_labels)
ax.set_ylabel("Time (ms)")
ax.set_title("Per-Layer Quantization Cost")
ax.legend()
ax.grid(axis='y', alpha=0.3)
save_plot(fig, "05_quant_overhead.png")


# ============================================================================
# Plot 6: Extended mode measured timings and memory
# ============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ext_labels = ["FP32", "FP16", "INT8\nFused", "INT8\nFused+\nMoDiff", "INT8\nCUDA\nGraph", "INT8\nSeparate", "INT4\nFused", "INT4\nFused+\nMoDiff", "INT4\nSeparate"]
ext_keys = ["fp32", "fp16", "int8_baseline", "int8", "int8_cudagraph", "int8_separate", "int4_baseline", "int4", "int4_separate"]
ext_times = [extended[k]["time_per_sample"] for k in ext_keys]
ext_mems = [extended[k]["memory_peak_mb"] for k in ext_keys]
ext_colors = ['#888888', '#5B9BD5', '#ED7D31', '#FFC000', '#9B59B6', '#E74C3C', '#70AD47', '#4472C4', '#1ABC9C']

bars = ax1.bar(ext_labels, ext_times, color=ext_colors, edgecolor='black', linewidth=0.5)
ax1.set_ylabel("Time per Sample (s)")
ax1.set_title("Extended Mode Timing")
ax1.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, ext_times):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{val:.3f}s", ha='center', va='bottom', fontsize=8)

bars = ax2.bar(ext_labels, ext_mems, color=ext_colors, edgecolor='black', linewidth=0.5)
ax2.set_ylabel("Peak Memory (MB)")
ax2.set_title("Extended Mode Peak Memory")
ax2.grid(axis='y', alpha=0.3)

save_plot(fig, "06_extended_modes.png")


# ============================================================================
# Plot 7: Cache overhead timings only
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 5))
step1 = [kernel_timing[k]["step1_cache_update_overhead_ms"] for _, k in config_labels]
conv = [kernel_timing[k]["conv_cache_update_overhead_ms"] for _, k in config_labels]
ohat = [kernel_timing[k]["ohat_update_overhead_ms"] for _, k in config_labels]
x = np.arange(len(kernel_labels))
w = 0.25
ax.bar(x - w, step1, w, label='Step1 Cache Update', color='#4472C4', edgecolor='black', linewidth=0.5)
ax.bar(x, conv, w, label='Conv Cache Update', color='#ED7D31', edgecolor='black', linewidth=0.5)
ax.bar(x + w, ohat, w, label='o_hat Update', color='#70AD47', edgecolor='black', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(kernel_labels)
ax.set_ylabel("Time (ms)")
ax.set_title("MoDiff Temporal Caching Overhead")
ax.legend()
ax.grid(axis='y', alpha=0.3)
save_plot(fig, "07_cache_overhead.png")


# ============================================================================
# Plot 8: Quality outputs only
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, prec in zip(axes, ["int8", "int4"]):
    q = quality[prec]
    keys = [f"{prec}_dynamic_baseline", f"{prec}_static_baseline", f"{prec}_dynamic_modiff", f"{prec}_static_modiff"]
    labels = ["dynamic baseline", "static baseline", "dynamic MoDiff", "static MoDiff"]
    psnr = [q[k]["psnr_vs_fp32_db"] for k in keys]
    mae = [q[k]["mae_vs_fp32"] for k in keys]
    bars = ax.bar(labels, psnr, color=['#ED7D31', '#FFC000', '#4472C4', '#70AD47'], edgecolor='black', linewidth=0.5)
    ax.set_ylabel("PSNR vs FP32 (dB)")
    ax.set_title(f"{prec.upper()} Image Quality")
    ax.grid(axis='y', alpha=0.3)
    for bar, p, m in zip(bars, psnr, mae):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.25, f"{p:.1f}dB\nMAE={m:.4f}", ha='center', va='bottom', fontsize=8)

save_plot(fig, "08_quality_comparison.png")

report = []
report.append("# MoDiff Benchmark Report: Measured Benchmark Data")
report.append("")
report.append("## Experimental Setup")
report.append("")
report.append("| Parameter | Value |")
report.append("|---|---|")
report.append("| GPU | NVIDIA A40 (48 GB) |")
report.append("| Model | Latent Diffusion Model (LDM), LSUN Churches 256×256 |")
report.append("| Diffusion Steps | 200 (DDIM) |")
report.append("| Batch Size | 32 |")
report.append("| Samples per Mode | 64 |")
report.append("| Timing Repeats (static/dynamic) | 3 |")
report.append(f"| FP32 Baseline (time/sample) | {overall['fp32']['time_per_sample']:.3f}s |")
report.append("")

report.append("## 1. Overall Model Measurements")
report.append("")
report.append("![Overall Model Measurements](plots/01_overall_speedup.png)")
report.append("")
report.append("| Mode | Total Time (s) | Samples | Time/Sample (s) | Time/Step (ms) |")
report.append("|---|---|---|---|---|")
for label, key in [
    ("FP32", "fp32"),
    ("FP16", "fp16"),
    ("INT8 Baseline (no MoDiff)", "int8_baseline"),
    ("INT8 MoDiff", "int8"),
    ("INT4 Baseline (no MoDiff)", "int4_baseline"),
    ("INT4 MoDiff", "int4"),
]:
    r = overall[key]
    report.append(f"| {label} | {r['total_time']:.3f} | {r['num_samples']} | {r['time_per_sample']:.3f} | {r['time_per_step_ms']:.2f} |")
report.append("")

report.append("## 2. Detailed Kernel and Layer Measurements")
report.append("")
report.append("### Fused and Separate Kernel Timing")
report.append("")
report.append("![Fused and Separate Kernel Timing](plots/02_kernel_fused_vs_separate.png)")
report.append("")
report.append("| Config | Fused Step1 (ms) | Fused Conv (ms) | Fused Total (ms) | Separate Step1 (ms) | Separate Conv (ms) | Separate Total (ms) |")
report.append("|---|---|---|---|---|---|---|")
for label, key in config_labels:
    r = kernel_timing[key]
    report.append(f"| {label} | {r['fused_step1_ms']:.3f} | {r['fused_conv_ms']:.3f} | {r['fused_total_ms']:.3f} | {r['separate_step1_ms']:.3f} | {r['separate_conv_ms']:.3f} | {r['separate_total_ms']:.3f} |")
report.append("")

report.append("### MoDiff Cache Update Timing")
report.append("")
report.append("![MoDiff Cache Update Timing](plots/07_cache_overhead.png)")
report.append("")
report.append("| Config | Step1 Cache Update (ms) | Conv Cache Update (ms) | o_hat Update (ms) | Step1 Extra MiB | Conv Extra MiB | Total Extra MiB |")
report.append("|---|---|---|---|---|---|---|")
for label, key in config_labels:
    r = kernel_timing[key]
    report.append(f"| {label} | {r['step1_cache_update_overhead_ms']:.3f} | {r['conv_cache_update_overhead_ms']:.3f} | {r['ohat_update_overhead_ms']:.3f} | {r['step1_cache_update_extra_mib']:.1f} | {r['conv_cache_update_extra_mib']:.1f} | {r['total_cache_update_extra_mib']:.1f} |")
report.append("")

report.append("### Per-Component Profiler Output")
report.append("")
report.append("![Per-Component Profiler Output](plots/03_component_breakdown.png)")
report.append("")
report.append("| Component | Time/Step (ms) | Invocations/Step |")
report.append("|---|---|---|")
for cat in categories_sorted:
    c = comps[cat]
    report.append(f"| {cat} | {c['per_step_ms']:.3f} | {c['count_per_step']:.1f} |")
report.append(f"| Total hooked | {hooked:.3f} | - |")
report.append(f"| Wall-clock total | {total_ms:.2f} | - |")
report.append("")
report.append("**Note:** This profiler table now uses leaf-module hooks only. The earlier `Total hooked = 2.700 ms` result came from a collector bug that retained only the last invocation for each module name across the full diffusion run. After fixing that bug, parent modules such as attention blocks still caused double-counting, so the profiler was restricted to leaf modules only. As a result, `Total hooked` should be read as measured leaf-op time covered by the hooks, while `Wall-clock total` remains the end-to-end step latency.")
report.append("")

report.append("### Extended Benchmark Measurements")
report.append("")
report.append("![Extended Benchmark Measurements](plots/06_extended_modes.png)")
report.append("")
report.append("| Mode | Time/Sample (s) | Time/Step (ms) | Peak Memory (MB) | CUDA Graphs | Captures | Replays |")
report.append("|---|---|---|---|---|---|---|")
for label, key in [
    ("FP32", "fp32"),
    ("FP16", "fp16"),
    ("INT8 Fused (Baseline)", "int8_baseline"),
    ("INT8 Fused + MoDiff", "int8"),
    ("INT8 CUDA Graph + MoDiff", "int8_cudagraph"),
    ("INT8 CUDA Graph (Baseline)", "int8_cudagraph_baseline"),
    ("INT8 Separate + MoDiff", "int8_separate"),
    ("INT8 Separate (Baseline)", "int8_separate_baseline"),
    ("INT4 Fused (Baseline)", "int4_baseline"),
    ("INT4 Fused + MoDiff", "int4"),
    ("INT4 Separate + MoDiff", "int4_separate"),
    ("INT4 Separate (Baseline)", "int4_separate_baseline"),
]:
    r = extended[key]
    is_graph = key.startswith("int8_cudagraph")
    report.append(
        f"| {label} | {r['time_per_sample']:.3f} | {r['time_per_step_ms']:.2f} | {r['memory_peak_mb']:.0f} | "
        f"{int(r.get('cuda_graph_num_graphs', 0)) if is_graph else '-'} | "
        f"{int(r.get('cuda_graph_capture_count', 0)) if is_graph else '-'} | "
        f"{int(r.get('cuda_graph_replay_count', 0)) if is_graph else '-'} |"
    )
report.append("")

report.append("## 3. Static and Dynamic Quantization Measurements")
report.append("")
report.append("### Model-Level Static and Dynamic Timing")
report.append("")
report.append("![Model-Level Static and Dynamic Timing](plots/04_static_vs_dynamic.png)")
report.append("")
report.append("| Mode | Time/Sample (s) | Time/Step (ms) | Timing Std (s) | Peak Memory (MB) | Loaded Conv Scales | Loaded Linear Scales |")
report.append("|---|---|---|---|---|---|---|")
for label, key in [
    ("INT8 Dynamic Baseline", "int8_dynamic_baseline"),
    ("INT8 Static Baseline", "int8_static_baseline"),
    ("INT8 Dynamic MoDiff", "int8_dynamic_modiff"),
    ("INT8 Static MoDiff", "int8_static_modiff"),
    ("INT4 Dynamic Baseline", "int4_dynamic_baseline"),
    ("INT4 Static Baseline", "int4_static_baseline"),
    ("INT4 Dynamic MoDiff", "int4_dynamic_modiff"),
    ("INT4 Static MoDiff", "int4_static_modiff"),
]:
    r = sd[key]
    report.append(f"| {label} | {r['time_per_sample_s']:.3f} | {r['time_per_step_ms']:.2f} | {r['timing_std_s']:.3f} | {r['memory_peak_mb']:.0f} | {int(r['loaded_conv_scales'])} | {int(r['loaded_linear_scales'])} |")
report.append("")

report.append("### Per-Layer Quantization Timing")
report.append("")
report.append("![Per-Layer Quantization Timing](plots/05_quant_overhead.png)")
report.append("")
report.append("| Config | Dynamic Quant (ms) | Static Quant (ms) | Absmax Scale (ms) | I/O Proxy (ms) |")
report.append("|---|---|---|---|---|")
for label, key in config_labels:
    r = qt[key]
    report.append(f"| {label} | {r['dynamic_quant_ms']:.3f} | {r['static_quant_ms']:.3f} | {r['absmax_scale_ms']:.3f} | {r['io_proxy_ms']:.3f} |")
report.append("")

report.append("### Quality Evaluation Outputs")
report.append("")
report.append("![Quality Evaluation Outputs](plots/08_quality_comparison.png)")
report.append("")
report.append("| Mode | MAE vs FP32 | Max Abs Diff | PSNR (dB) |")
report.append("|---|---|---|---|")
for prec in ["int8", "int4"]:
    q = quality[prec]
    for key in [f"{prec}_dynamic_baseline", f"{prec}_static_baseline", f"{prec}_dynamic_modiff", f"{prec}_static_modiff"]:
        r = q[key]
        report.append(f"| {r['label']} | {r['mae_vs_fp32']:.4f} | {r['max_abs_vs_fp32']:.4f} | {r['psnr_vs_fp32_db']:.2f} |")
report.append("")

report.append("## 4. Measured Notes")
report.append("")
report.append(f"- INT8 dynamic baseline time/sample: {sd['int8_dynamic_baseline']['time_per_sample_s']:.3f}s")
report.append(f"- INT8 static baseline time/sample: {sd['int8_static_baseline']['time_per_sample_s']:.3f}s")
report.append(f"- INT8 dynamic MoDiff time/sample: {sd['int8_dynamic_modiff']['time_per_sample_s']:.3f}s")
report.append(f"- INT8 static MoDiff time/sample: {sd['int8_static_modiff']['time_per_sample_s']:.3f}s")
report.append(f"- INT4 dynamic baseline time/sample: {sd['int4_dynamic_baseline']['time_per_sample_s']:.3f}s")
report.append(f"- INT4 static baseline time/sample: {sd['int4_static_baseline']['time_per_sample_s']:.3f}s")
report.append(f"- INT4 dynamic MoDiff time/sample: {sd['int4_dynamic_modiff']['time_per_sample_s']:.3f}s")
report.append(f"- INT4 static MoDiff time/sample: {sd['int4_static_modiff']['time_per_sample_s']:.3f}s")
report.append(f"- INT8 dynamic MoDiff PSNR: {quality['int8']['int8_dynamic_modiff']['psnr_vs_fp32_db']:.2f} dB")
report.append(f"- INT4 dynamic MoDiff PSNR: {quality['int4']['int4_dynamic_modiff']['psnr_vs_fp32_db']:.2f} dB")
report.append(f"- INT8 CUDA Graph baseline time/sample: {extended['int8_cudagraph_baseline']['time_per_sample']:.3f}s")
report.append(f"- INT8 CUDA Graph baseline peak memory: {extended['int8_cudagraph_baseline']['memory_peak_mb']:.0f} MB")
report.append("")

report.append("## 5. Qualitative Notes")
report.append("")
report.append("- All values shown in the tables above are taken directly from benchmark outputs or evaluation outputs.")
report.append("- Derived arithmetic comparisons such as speedup, percentage overhead, percentage share, and theoretical gap values are intentionally omitted.")
report.append("- Kernel-level static-versus-dynamic convolution-only timing remains listed as `NOT ABLE TO MEASURE` in the underlying benchmark artifacts.")
report.append("")
report.append("---")
report.append("")
report.append("*Report generated on NVIDIA A40 GPU, March 2026.*")
report.append("*All timing values are taken from real GPU benchmark runs in this workspace.*")

report_path = os.path.join(REPORT_DIR, "BENCHMARK_REPORT.md")
with open(report_path, "w") as f:
    f.write("\n".join(report))

print(f"Report written to: {report_path}")

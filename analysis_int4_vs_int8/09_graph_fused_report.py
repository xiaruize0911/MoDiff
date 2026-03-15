#!/usr/bin/env python3
"""Generate focused plots/report for CUDA-Graph INT8 modes and fused baseline kernels."""

import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_RESULTS = os.path.join(OUTPUT_DIR, '..', 'integration', 'results_ldm_benchmark', 'results.json')


def _load_results():
    if not os.path.exists(BENCHMARK_RESULTS):
        raise FileNotFoundError(f"Missing benchmark results: {BENCHMARK_RESULTS}")
    with open(BENCHMARK_RESULTS) as f:
        return json.load(f)


def _save_int8_graph_plot(results):
    modes = [m for m in ['int8', 'int8_graph', 'int8_baseline', 'int8_baseline_graph'] if m in results]
    if not modes:
        return None

    time_vals = [results[m]['time_per_sample'] for m in modes]
    mem_vals = [results[m].get('peak_memory_allocated_mb', 0.0) for m in modes]
    x = np.arange(len(modes))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['#F7A6AC', '#B2DBB9', '#EEC186', '#B8E5FA']

    bars = axes[0].bar(x, time_vals, color=colors[:len(modes)])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(modes, rotation=20, ha='right')
    axes[0].set_ylabel('Time / sample (s)')
    axes[0].set_title('INT8 LDM Speed: baseline vs CUDA Graph')
    for bar, val in zip(bars, time_vals):
        axes[0].text(bar.get_x() + bar.get_width() / 2, val, f'{val:.3f}s', ha='center', va='bottom', fontsize=9)

    bars = axes[1].bar(x, mem_vals, color=colors[:len(modes)])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(modes, rotation=20, ha='right')
    axes[1].set_ylabel('Peak allocated memory (MiB)')
    axes[1].set_title('INT8 LDM Memory: baseline vs CUDA Graph')
    for bar, val in zip(bars, mem_vals):
        axes[1].text(bar.get_x() + bar.get_width() / 2, val, f'{val:.0f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'plot_graph_int8_speed_memory.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


def _extract_profile_total(profile, prefix):
    total = 0.0
    for name, entry in (profile or {}).items():
        if name.startswith(prefix):
            total += entry['total_s'] * 1000.0
    return total


def _save_kernel_compare_plot(results):
    pairs = [
        ('int8_baseline', 'int8_baseline_fused', 'INT8'),
        ('int4_baseline', 'int4_baseline_fused', 'INT4'),
    ]
    labels = []
    current_q = []
    fused_q = []
    current_cdq = []
    fused_cdq = []

    for current_mode, fused_mode, label in pairs:
        if current_mode not in results or fused_mode not in results:
            continue
        labels.append(label)
        current_profile = results[current_mode].get('profile', {})
        fused_profile = results[fused_mode].get('profile', {})
        current_q.append(_extract_profile_total(current_profile, f'Baseline {label} Current Q'))
        fused_q.append(_extract_profile_total(fused_profile, f'Baseline {label} Fused Q'))
        current_cdq.append(_extract_profile_total(current_profile, f'Baseline {label} Current Compute+DQ'))
        fused_cdq.append(_extract_profile_total(fused_profile, f'Baseline {label} Fused Compute+DQ'))

    if not labels:
        return None

    x = np.arange(len(labels))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    bars1 = axes[0].bar(x - width / 2, current_q, width, label='Current', color='#EEC186')
    bars2 = axes[0].bar(x + width / 2, fused_q, width, label='Two-kernel fused', color='#B2DBB9')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel('Total Q time (ms)')
    axes[0].set_title('Baseline Q kernel time')
    axes[0].legend()
    for bars in (bars1, bars2):
        for bar in bars:
            val = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2, val, f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    bars3 = axes[1].bar(x - width / 2, current_cdq, width, label='Current', color='#F7A6AC')
    bars4 = axes[1].bar(x + width / 2, fused_cdq, width, label='Two-kernel fused', color='#B8E5FA')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel('Total compute+DQ time (ms)')
    axes[1].set_title('Baseline compute+DQ kernel time')
    axes[1].legend()
    for bars in (bars3, bars4):
        for bar in bars:
            val = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2, val, f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'plot_fused_baseline_kernel_compare.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


def _write_report(results, graph_plot, kernel_plot):
    report_lines = [
        '# CUDA-Graph INT8 and Fused-Baseline Report',
        '',
        '## INT8 speed and memory comparison',
        '',
        '| Mode | Time/sample (s) | Peak allocated (MiB) |',
        '|------|----------------:|---------------------:|',
    ]
    for mode in ['int8', 'int8_graph', 'int8_baseline', 'int8_baseline_graph']:
        if mode in results:
            entry = results[mode]
            report_lines.append(
                f"| {mode} | {entry['time_per_sample']:.3f} | {entry.get('peak_memory_allocated_mb', 0.0):.1f} |"
            )
    if graph_plot:
        report_lines.extend(['', f'![INT8 graph speed/memory]({os.path.basename(graph_plot)})'])

    report_lines.extend([
        '',
        '## Fused baseline kernel comparison',
        '',
        '| Mode | Time/sample (s) | Peak allocated (MiB) |',
        '|------|----------------:|---------------------:|',
    ])
    for mode in ['int8_baseline', 'int8_baseline_fused', 'int4_baseline', 'int4_baseline_fused']:
        if mode in results:
            entry = results[mode]
            report_lines.append(
                f"| {mode} | {entry['time_per_sample']:.3f} | {entry.get('peak_memory_allocated_mb', 0.0):.1f} |"
            )
    if kernel_plot:
        report_lines.extend(['', f'![Fused baseline kernels]({os.path.basename(kernel_plot)})'])

    report_lines.extend([
        '',
        '## File layout updates',
        '',
        '- `integration/runtime/cuda_graphs.py`: reusable CUDA Graph capture/replay helper for fixed-shape LDM sampling.',
        '- `analysis_int4_vs_int8/09_graph_fused_report.py`: focused visualization/report generator for the new graph and fused-baseline experiments.',
        '- INT8/INT4 conv baselines now support explicit `current` and `two_kernel_fused` execution modes for side-by-side profiling.',
    ])

    report_path = os.path.join(OUTPUT_DIR, 'GRAPH_FUSED_REPORT.md')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    return report_path


def main():
    results = _load_results()
    graph_plot = _save_int8_graph_plot(results)
    kernel_plot = _save_kernel_compare_plot(results)
    report_path = _write_report(results, graph_plot, kernel_plot)
    print(f'Report written to {report_path}')


if __name__ == '__main__':
    main()

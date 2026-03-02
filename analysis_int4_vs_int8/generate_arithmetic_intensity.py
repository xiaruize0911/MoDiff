#!/usr/bin/env python3
"""
Arithmetic Intensity Study: INT4 / INT8 / FP32 Quantized Convolutions & Linear Layers
========================================================================================

Arithmetic Intensity (AI) = FLOPs / Bytes_accessed

For a memory-bound kernel the roofline limit is:
    throughput ≤ peak_memory_bandwidth × AI

For a compute-bound kernel:
    throughput ≤ peak_compute

Comparing AI across precisions explains why INT4/INT8 speedups in the full pipeline
are smaller than the raw kernel compute ratio.

Reads from:
  experiment_results.json  (exp3_conv, exp4_linear data)

Outputs:
  plot_ai_conv.png
  plot_ai_linear.png
  plot_roofline_conv.png
  plot_roofline_linear.png
  table_06_arithmetic_intensity_conv.md  / .tex
  table_07_arithmetic_intensity_linear.md / .tex
"""

import os
import sys
import json
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
})

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(OUTPUT_DIR, 'experiment_results.json')

# Approximate GPU roofline numbers (A100-SXM or adjust for your GPU)
# These are used for roofline reference lines only, not for pass/fail.
PEAK_MEMORY_BW_GB_S = 900.0     # GB/s  (A100 HBM2e peak)
PEAK_FP32_TFLOPS   = 19.5       # TFLOPS (A100 FP32 tensor core)
PEAK_INT8_TOPS     = 624.0      # INT8 TOPS (A100)
PEAK_INT4_TOPS     = 1248.0     # INT4 TOPS (A100)

COLORS = {
    'fp32':  '#B8E5FA',
    'int8':  '#F7A6AC',
    'int4':  '#F7B7D2',
}


# ============================================================================
# Arithmetic Intensity Formulae
# ============================================================================

def conv_flops(N, C_in, C_out, K, H_out, W_out):
    """FLOPs for a Conv2d: 2 * multiply-adds per output element."""
    return 2 * N * C_in * C_out * K * K * H_out * W_out


def conv_bytes(N, C_in, C_out, K, H, W, H_out, W_out, precision):
    """
    Bytes accessed for Conv2d activation + weight tensors.
    precision: 'fp32' (4 B), 'int8' (1 B input+weight, 4 B output),
               'int4' (0.5 B input+weight, 4 B output).
    """
    act_elems    = N * C_in  * H    * W
    weight_elems = C_out * C_in * K * K
    out_elems    = N * C_out * H_out * W_out

    if precision == 'fp32':
        return 4 * (act_elems + weight_elems + out_elems)
    elif precision == 'int8':
        return 1 * (act_elems + weight_elems) + 4 * out_elems
    elif precision == 'int4':
        return 0.5 * (act_elems + weight_elems) + 4 * out_elems
    else:
        raise ValueError(f"Unknown precision {precision}")


def linear_flops(N, in_f, out_f):
    """FLOPs for a Linear layer."""
    return 2 * N * in_f * out_f


def linear_bytes(N, in_f, out_f, precision):
    """Bytes accessed for Linear activation + weight tensors."""
    act_elems    = N * in_f
    weight_elems = in_f * out_f
    out_elems    = N * out_f

    if precision == 'fp32':
        return 4 * (act_elems + weight_elems + out_elems)
    elif precision == 'int8':
        return 1 * (act_elems + weight_elems) + 4 * out_elems
    elif precision == 'int4':
        return 0.5 * (act_elems + weight_elems) + 4 * out_elems
    else:
        raise ValueError(f"Unknown precision {precision}")


# ============================================================================
# Build per-shape records
# ============================================================================

def build_conv_records(exp3):
    """Return list of dicts, one per conv shape."""
    bs_key = list(exp3.keys())[0]
    shapes = exp3[bs_key]
    records = []

    for sk, v in shapes.items():
        C_in  = v['C_in']
        C_out = v['C_out']
        K     = v['K']
        S     = v['S']
        P     = v.get('P', K // 2)
        H     = v['H']
        W     = v['W']
        N     = v['N']
        count = v['count']

        H_out = (H + 2 * P - K) // S + 1
        W_out = (H_out)  # square assumed

        flops  = conv_flops(N, C_in, C_out, K, H_out, W_out)
        b_fp32 = conv_bytes(N, C_in, C_out, K, H, W, H_out, W_out, 'fp32')
        b_int8 = conv_bytes(N, C_in, C_out, K, H, W, H_out, W_out, 'int8')
        b_int4 = conv_bytes(N, C_in, C_out, K, H, W, H_out, W_out, 'int4')

        # Measured throughput in GFLOPS/s
        fp32_ms  = v['fp32_ms']
        int8_ms  = v['int8_e2e_ms']
        int4_ms  = v['int4_e2e_ms']

        tput_fp32 = flops / (fp32_ms * 1e-3) / 1e9   if fp32_ms  and fp32_ms  > 0 else 0
        tput_int8 = flops / (int8_ms * 1e-3) / 1e9   if int8_ms  and int8_ms  > 0 else 0
        tput_int4 = flops / (int4_ms * 1e-3) / 1e9   if int4_ms  and int4_ms  > 0 else 0

        # Short label: Cin→Cout K×K (×count)
        label = f'{C_in}→{C_out} {K}×{K}\n(×{count})'

        records.append({
            'label':       label,
            'shape_key':   sk,
            'N': N, 'C_in': C_in, 'C_out': C_out, 'K': K,
            'H': H, 'W': W, 'H_out': H_out, 'W_out': W_out,
            'count':       count,
            'flops':       flops,
            'bytes_fp32':  b_fp32,
            'bytes_int8':  b_int8,
            'bytes_int4':  b_int4,
            'ai_fp32':     flops / b_fp32,
            'ai_int8':     flops / b_int8,
            'ai_int4':     flops / b_int4,
            'tput_fp32':   tput_fp32,
            'tput_int8':   tput_int8,
            'tput_int4':   tput_int4,
            'fp32_ms':     fp32_ms,
            'int8_ms':     int8_ms,
            'int4_ms':     int4_ms,
        })

    # Sort by descending fp32 time (most expensive first)
    records.sort(key=lambda r: r['fp32_ms'], reverse=True)
    return records


def build_linear_records(exp4):
    bs_key = list(exp4.keys())[0]
    shapes = exp4[bs_key]
    records = []

    for sk, v in shapes.items():
        in_f  = v['in_features']
        out_f = v['out_features']
        N     = v['batch_size']
        count = v['count']

        flops  = linear_flops(N, in_f, out_f)
        b_fp32 = linear_bytes(N, in_f, out_f, 'fp32')
        b_int8 = linear_bytes(N, in_f, out_f, 'int8')
        b_int4 = linear_bytes(N, in_f, out_f, 'int4')

        fp32_ms = v['fp32_ms']
        int8_ms = v['int8_baseline_ms']
        int4_ms = v['int4_baseline_ms']

        tput_fp32 = flops / (fp32_ms * 1e-3) / 1e9  if fp32_ms  and fp32_ms  > 0 else 0
        tput_int8 = flops / (int8_ms * 1e-3) / 1e9  if int8_ms  and int8_ms  > 0 else 0
        tput_int4 = flops / (int4_ms * 1e-3) / 1e9  if int4_ms  and int4_ms  > 0 else 0

        label = f'{in_f}→{out_f}\n(×{count})'

        records.append({
            'label':      label,
            'shape_key':  sk,
            'N': N, 'in_f': in_f, 'out_f': out_f, 'count': count,
            'flops':      flops,
            'bytes_fp32': b_fp32,
            'bytes_int8': b_int8,
            'bytes_int4': b_int4,
            'ai_fp32':    flops / b_fp32,
            'ai_int8':    flops / b_int8,
            'ai_int4':    flops / b_int4,
            'tput_fp32':  tput_fp32,
            'tput_int8':  tput_int8,
            'tput_int4':  tput_int4,
            'fp32_ms':    fp32_ms,
            'int8_ms':    int8_ms,
            'int4_ms':    int4_ms,
        })

    records.sort(key=lambda r: r['in_f'] * r['out_f'], reverse=True)
    return records


# ============================================================================
# Plots
# ============================================================================

def plot_ai_grouped_bars(records, title, output_filename):
    """Grouped bar chart: arithmetic intensity per shape for FP32 / INT8 / INT4."""
    labels = [r['label'] for r in records]
    ai_fp32 = [r['ai_fp32'] for r in records]
    ai_int8 = [r['ai_int8'] for r in records]
    ai_int4 = [r['ai_int4'] for r in records]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 1.2), 6))
    b1 = ax.bar(x - width, ai_fp32, width, label='FP32',  color=COLORS['fp32'], edgecolor='black', linewidth=0.4)
    b2 = ax.bar(x,         ai_int8, width, label='INT8',  color=COLORS['int8'], edgecolor='black', linewidth=0.4)
    b3 = ax.bar(x + width, ai_int4, width, label='INT4',  color=COLORS['int4'], edgecolor='black', linewidth=0.4)

    # Annotate INT8/FP32 ratio on top of INT8 bar
    for i, (a8, a32) in enumerate(zip(ai_int8, ai_fp32)):
        if a32 > 0:
            ax.text(x[i], a8 + 0.02 * max(ai_int4 + ai_int8 + ai_fp32),
                    f'{a8/a32:.1f}×', ha='center', va='bottom', fontsize=9,
                    color=COLORS['int8'], fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Arithmetic Intensity (FLOPs/Byte)')
    ax.set_title(title)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, output_filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")


def plot_roofline(records, peak_bw, peak_compute_dict, title, output_filename):
    """
    Roofline scatter: AI (x) vs measured throughput (y, GFLOPS/s).
    Overlays reference roofline lines for FP32 / INT8 / INT4.
    """
    fig, ax = plt.subplots(figsize=(11, 7))

    # Plot roofline lines
    ai_range = np.logspace(-2, 3, 500)

    rooflines = [
        ('FP32',  COLORS['fp32'], peak_compute_dict.get('fp32', PEAK_FP32_TFLOPS)   * 1e3),
        ('INT8',  COLORS['int8'], peak_compute_dict.get('int8', PEAK_INT8_TOPS)     ),
        ('INT4',  COLORS['int4'], peak_compute_dict.get('int4', PEAK_INT4_TOPS)     ),
    ]
    for prec_label, color, peak_gops in rooflines:
        mem_bound  = peak_bw * ai_range   # GB/s * FLOPs/Byte = GFLOPS/s
        compute_ca = np.full_like(ai_range, peak_gops)
        roof       = np.minimum(mem_bound, compute_ca)
        ax.plot(ai_range, roof, '--', color=color, lw=1.2, alpha=0.6, label=f'{prec_label} roofline')

    # Scatter measured points
    markers = {'fp32': 'o', 'int8': 's', 'int4': '^'}
    for r in records:
        for prec in ('fp32', 'int8', 'int4'):
            ai   = r[f'ai_{prec}']
            tput = r[f'tput_{prec}']
            if tput > 0 and ai > 0:
                ax.scatter(ai, tput, marker=markers[prec], color=COLORS[prec],
                           s=60, zorder=5)

    # Add memory bandwidth diagonal label
    x_label_ai = 5.0
    ax.text(x_label_ai, peak_bw * x_label_ai * 0.6,
            f'Mem BW limit\n({peak_bw:.0f} GB/s)',
            fontsize=9, color='gray', rotation=30, alpha=0.7)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Arithmetic Intensity (FLOPs/Byte)')
    ax.set_ylabel('Throughput (GFLOPS/s)')
    ax.set_title(title)
    ax.grid(True, which='both', alpha=0.2)
    ax.legend(loc='upper left')
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, output_filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")


def plot_ai_ratio(records, title, output_filename):
    """Bar chart of INT8/FP32 and INT4/FP32 AI ratio per shape."""
    labels = [r['label'] for r in records]
    ratio_int8 = [r['ai_int8'] / r['ai_fp32'] for r in records]
    ratio_int4 = [r['ai_int4'] / r['ai_fp32'] for r in records]

    x     = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.1), 6))
    ax.bar(x - width / 2, ratio_int8, width, label='INT8/FP32 AI ratio',
           color=COLORS['int8'], edgecolor='black', linewidth=0.4)
    ax.bar(x + width / 2, ratio_int4, width, label='INT4/FP32 AI ratio',
           color=COLORS['int4'], edgecolor='black', linewidth=0.4)
    ax.axhline(1.0, color='gray', linestyle='--', lw=1.0, alpha=0.6, label='1× (no gain)')

    for i, (r8, r4) in enumerate(zip(ratio_int8, ratio_int4)):
        ax.text(x[i] - width / 2, r8 + 0.02, f'{r8:.1f}×', ha='center', fontsize=9,
                color=COLORS['int8'])
        ax.text(x[i] + width / 2, r4 + 0.02, f'{r4:.1f}×', ha='center', fontsize=9,
                color=COLORS['int4'])

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('AI Ratio vs FP32')
    ax.set_title(title)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, output_filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")


# ============================================================================
# Tables
# ============================================================================

def _gflops_str(f):
    gf = f / 1e9
    if gf >= 1000:
        return f'{gf/1000:.1f}T'
    return f'{gf:.2f}G'


def _bytes_str(b):
    mb = b / 1e6
    if mb >= 1000:
        return f'{mb/1000:.2f}GB'
    return f'{mb:.2f}MB'


def write_md_table(records, layer_type, output_filename):
    """Write a Markdown table of arithmetic intensity data."""
    lines = []
    lines.append(f'# Arithmetic Intensity: {layer_type} Layers\n')
    lines.append('FLOPs counted as multiply-add × 2.  '
                 'Byte counts: FP32=4B/elem; INT8=1B activation+weight, 4B output; '
                 'INT4=0.5B activation+weight, 4B output.\n')
    lines.append('')

    if layer_type == 'Conv2d':
        hdr = ('| Shape | Count | FLOPs | FP32 Bytes | INT8 Bytes | INT4 Bytes '
               '| FP32 AI | INT8 AI | INT4 AI | INT8/FP32 | INT4/FP32 |')
        sep = ('|-------|-------|-------|-----------|-----------|-----------|'
               '---------|---------|---------|-----------|-----------|')
        lines.append(hdr)
        lines.append(sep)
        for r in records:
            sk = r['shape_key']
            lines.append(
                f"| {sk} | {r['count']} | {_gflops_str(r['flops'])} "
                f"| {_bytes_str(r['bytes_fp32'])} | {_bytes_str(r['bytes_int8'])} | {_bytes_str(r['bytes_int4'])} "
                f"| {r['ai_fp32']:.2f} | {r['ai_int8']:.2f} | {r['ai_int4']:.2f} "
                f"| {r['ai_int8']/r['ai_fp32']:.2f}× | {r['ai_int4']/r['ai_fp32']:.2f}× |"
            )
    else:
        hdr = ('| Shape | Count | FLOPs | FP32 Bytes | INT8 Bytes | INT4 Bytes '
               '| FP32 AI | INT8 AI | INT4 AI | INT8/FP32 | INT4/FP32 |')
        sep = ('|-------|-------|-------|-----------|-----------|-----------|'
               '---------|---------|---------|-----------|-----------|')
        lines.append(hdr)
        lines.append(sep)
        for r in records:
            sk = r['shape_key']
            lines.append(
                f"| {sk} | {r['count']} | {_gflops_str(r['flops'])} "
                f"| {_bytes_str(r['bytes_fp32'])} | {_bytes_str(r['bytes_int8'])} | {_bytes_str(r['bytes_int4'])} "
                f"| {r['ai_fp32']:.2f} | {r['ai_int8']:.2f} | {r['ai_int4']:.2f} "
                f"| {r['ai_int8']/r['ai_fp32']:.2f}× | {r['ai_int4']/r['ai_fp32']:.2f}× |"
            )

    path = os.path.join(OUTPUT_DIR, output_filename)
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  Saved {path}")


def write_tex_table(records, layer_type, output_filename):
    """Write a LaTeX table of arithmetic intensity data."""
    lines = []
    lines.append(r'\begin{table}[h]')
    lines.append(r'  \centering')
    lines.append(r'  \caption{Arithmetic Intensity: ' + layer_type + r' Layers. '
                 r'Bytes counted as: FP32=4B/elem; INT8=1B act+weight, 4B output; INT4=0.5B act+weight, 4B output.}')
    lines.append(r'  \label{tab:ai_' + layer_type.lower().replace(' ', '_') + r'}')
    lines.append(r'  \begin{tabular}{lrrrrrrrrr}')
    lines.append(r'    \toprule')
    lines.append(r'    Shape & FLOPs & \multicolumn{3}{c}{Bytes} & \multicolumn{3}{c}{AI (FLOPs/B)} & INT8/FP32 & INT4/FP32 \\')
    lines.append(r'    \cmidrule(lr){3-5}\cmidrule(lr){6-8}')
    lines.append(r'    & & FP32 & INT8 & INT4 & FP32 & INT8 & INT4 & & \\')
    lines.append(r'    \midrule')
    for r in records:
        sk = r['shape_key'].replace('_', r'\_').replace(',', ', ')
        lines.append(
            f"    {sk} & {_gflops_str(r['flops'])} "
            f"& {_bytes_str(r['bytes_fp32'])} & {_bytes_str(r['bytes_int8'])} & {_bytes_str(r['bytes_int4'])} "
            f"& {r['ai_fp32']:.2f} & {r['ai_int8']:.2f} & {r['ai_int4']:.2f} "
            f"& {r['ai_int8']/r['ai_fp32']:.2f}$\\times$ & {r['ai_int4']/r['ai_fp32']:.2f}$\\times$ \\\\"
        )
    lines.append(r'    \bottomrule')
    lines.append(r'  \end{tabular}')
    lines.append(r'\end{table}')

    path = os.path.join(OUTPUT_DIR, output_filename)
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  Saved {path}")


# ============================================================================
# Main
# ============================================================================

def main():
    if not os.path.exists(RESULTS_PATH):
        print(f"ERROR: {RESULTS_PATH} not found")
        sys.exit(1)

    with open(RESULTS_PATH) as f:
        data = json.load(f)

    exp3 = data.get('exp3_conv', {})
    exp4 = data.get('exp4_linear', {})

    if not exp3:
        print("WARNING: exp3_conv not found in results, skipping conv analysis")
    if not exp4:
        print("WARNING: exp4_linear not found in results, skipping linear analysis")

    # -------------------------------------------------------------------------
    # Conv arithmetic intensity
    # -------------------------------------------------------------------------
    if exp3:
        print("\n=== Conv2d Arithmetic Intensity ===")
        conv_recs = build_conv_records(exp3)

        print("  Generating AI bar chart...")
        plot_ai_grouped_bars(
            conv_recs,
            title='Arithmetic Intensity: Conv2d Layers (FP32 / INT8 / INT4)',
            output_filename='plot_ai_conv.png'
        )

        print("  Generating AI ratio chart...")
        plot_ai_ratio(
            conv_recs,
            title='AI Ratio vs FP32: Conv2d\n(Shows data-movement reduction from lower precision)',
            output_filename='plot_ai_ratio_conv.png'
        )

        print("  Generating roofline chart...")
        plot_roofline(
            conv_recs,
            peak_bw=PEAK_MEMORY_BW_GB_S,
            peak_compute_dict={
                'fp32': PEAK_FP32_TFLOPS * 1e3,   # GFLOPS/s
                'int8': PEAK_INT8_TOPS,
                'int4': PEAK_INT4_TOPS,
            },
            title='Roofline: Conv2d Layers\n(dashes = roofline limits, dots = measured)',
            output_filename='plot_roofline_conv.png'
        )

        print("  Writing Conv2d markdown table...")
        write_md_table(conv_recs, 'Conv2d', 'table_06_arithmetic_intensity_conv.md')

        print("  Writing Conv2d LaTeX table...")
        write_tex_table(conv_recs, 'Conv2d', 'table_06_arithmetic_intensity_conv.tex')

    # -------------------------------------------------------------------------
    # Linear arithmetic intensity
    # -------------------------------------------------------------------------
    if exp4:
        print("\n=== Linear Arithmetic Intensity ===")
        lin_recs = build_linear_records(exp4)

        print("  Generating AI bar chart...")
        plot_ai_grouped_bars(
            lin_recs,
            title='Arithmetic Intensity: Linear Layers (FP32 / INT8 / INT4)',
            output_filename='plot_ai_linear.png'
        )

        print("  Generating AI ratio chart...")
        plot_ai_ratio(
            lin_recs,
            title='AI Ratio vs FP32: Linear\n(Shows data-movement reduction from lower precision)',
            output_filename='plot_ai_ratio_linear.png'
        )

        print("  Generating roofline chart...")
        plot_roofline(
            lin_recs,
            peak_bw=PEAK_MEMORY_BW_GB_S,
            peak_compute_dict={
                'fp32': PEAK_FP32_TFLOPS * 1e3,
                'int8': PEAK_INT8_TOPS,
                'int4': PEAK_INT4_TOPS,
            },
            title='Roofline: Linear Layers\n(dashes = roofline limits, dots = measured)',
            output_filename='plot_roofline_linear.png'
        )

        print("  Writing Linear markdown table...")
        write_md_table(lin_recs, 'Linear', 'table_07_arithmetic_intensity_linear.md')

        print("  Writing Linear LaTeX table...")
        write_tex_table(lin_recs, 'Linear', 'table_07_arithmetic_intensity_linear.tex')

    print("\nDone! All arithmetic intensity outputs saved to:", OUTPUT_DIR)


if __name__ == '__main__':
    main()

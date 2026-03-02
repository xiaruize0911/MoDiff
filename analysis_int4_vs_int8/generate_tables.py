#!/usr/bin/env python3
"""
Generate Markdown and LaTeX result tables from experiment_results.json.
Each table is saved as a separate .md and .tex file (mirroring the plot files).
"""

import os
import json
import numpy as np

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(OUTPUT_DIR, 'experiment_results.json')

# ── helpers ──────────────────────────────────────────────────────────────────

def save(name: str, md: str, tex: str):
    """Write .md and .tex files and print a confirmation."""
    md_path  = os.path.join(OUTPUT_DIR, f'{name}.md')
    tex_path = os.path.join(OUTPUT_DIR, f'{name}.tex')
    with open(md_path,  'w') as f: f.write(md)
    with open(tex_path, 'w') as f: f.write(tex)
    print(f'  Saved {md_path}')
    print(f'  Saved {tex_path}')


def tex_escape(s: str) -> str:
    return s.replace('_', r'\_').replace('%', r'\%').replace('&', r'\&').replace('×', r'$\times$')


class MDTable:
    """Simple Markdown table builder."""
    def __init__(self, headers: list[str]):
        self.headers = headers
        self.rows: list[list[str]] = []

    def add(self, *cells):
        self.rows.append([str(c) for c in cells])

    def render(self) -> str:
        sep = '|' + '|'.join(['---'] * len(self.headers)) + '|'
        header = '| ' + ' | '.join(self.headers) + ' |'
        lines = [header, sep]
        for row in self.rows:
            lines.append('| ' + ' | '.join(row) + ' |')
        return '\n'.join(lines) + '\n'


class TexTable:
    """Simple LaTeX booktabs table builder."""
    def __init__(self, headers: list[str], caption: str = '', label: str = ''):
        self.headers = headers
        self.caption = caption
        self.label = label
        self.rows: list[tuple[list[str], bool]] = []   # (cells, midrule_after)

    def add(self, *cells, midrule: bool = False):
        self.rows.append(([tex_escape(str(c)) for c in cells], midrule))

    def render(self) -> str:
        ncols = len(self.headers)
        col_spec = 'l' + 'r' * (ncols - 1)
        head_str = ' & '.join(tex_escape(h) for h in self.headers) + r' \\'
        lines = [
            r'\begin{table}[htbp]',
            r'\centering',
            r'\small',
            rf'\caption{{{tex_escape(self.caption)}}}',
            rf'\label{{tab:{self.label}}}',
            rf'\begin{{tabular}}{{{col_spec}}}',
            r'\toprule',
            head_str,
            r'\midrule',
        ]
        for cells, mr in self.rows:
            lines.append(' & '.join(cells) + r' \\')
            if mr:
                lines.append(r'\midrule')
        lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
        return '\n'.join(lines) + '\n'


# ── Table 1 : Full Pipeline Speedup ──────────────────────────────────────────

def table_01(data):
    exp   = data['exp1_pipeline']
    modes = [m for m in ['fp32', 'int8_baseline', 'int8', 'int4_baseline', 'int4'] if m in exp]
    fp32_t = exp['fp32']['time_per_sample']

    headers = ['Mode', 'Time/Sample (ms)', 'Time/Step (ms)', 'Speedup vs FP32']

    md  = MDTable(headers)
    tex = TexTable(headers,
                   caption='Full pipeline inference speed (200 DDIM steps, 128 samples, batch size 32, LSUN Churches 256×256).',
                   label='pipeline_speedup')

    for m in modes:
        r    = exp[m]
        t_ms = r['time_per_sample'] * 1000
        spd  = fp32_t / r['time_per_sample']
        md .add(m, f'{t_ms:.1f}', f"{r['time_per_step_ms']:.2f}", f'{spd:.3f}×')
        tex.add(m, f'{t_ms:.1f}', f"{r['time_per_step_ms']:.2f}", f'{spd:.3f}' + r'$\times$')

    header_md = '## Table 1 – Full Pipeline Speedup\n\n'
    save('table_01_pipeline_speedup', header_md + md.render(), tex.render())


# ── Table 2 : Per-Component Breakdown ────────────────────────────────────────

def table_02(data):
    exp = data['exp2_breakdown']
    generic_map = {
        'Conv2d(FP32)': 'Conv2d', 'Int8Conv2d': 'Conv2d', 'Int4Conv2d': 'Conv2d',
        'Linear(FP32)': 'Linear', 'Int8Linear': 'Linear', 'Int4Linear': 'Linear',
        'Attention': 'Attention', 'GroupNorm': 'GroupNorm', 'SiLU': 'SiLU',
    }
    generic_order = ['Conv2d', 'Attention', 'Linear', 'GroupNorm', 'SiLU']
    exp_modes = [m for m in ['fp32', 'fp16', 'int8', 'int4'] if m in exp]

    def agg(mode):
        ls = exp[mode]['layer_stats']
        out = {}
        for lt, s in ls.items():
            g = generic_map.get(lt, lt)
            out[g] = out.get(g, 0) + s['total_ms']
        return out

    # Dynamic headers based on available modes
    ms_headers  = [f'{m.upper()} (ms)' for m in exp_modes]
    rat_headers = [f'{m.upper()}/FP32' for m in exp_modes if m != 'fp32']
    headers = ['Component'] + ms_headers + rat_headers

    md  = MDTable(headers)
    tex = TexTable(headers,
                   caption=(r'Per-component cumulative time (50 steps $\times$ 2 batches $\times$ 32 samples, bs=32). '
                            r'FP32 = no autocast; INT8/INT4 = FP16 autocast (matching Exp.\ 1 conditions).'),
                   label='component_breakdown')

    aggs = {m: agg(m) for m in exp_modes}
    fp32_agg = aggs['fp32']

    for g in generic_order:
        ms_vals  = [f"{aggs[m].get(g, 0):.1f}" for m in exp_modes]
        rat_vals = []
        for m in exp_modes:
            if m == 'fp32':
                continue
            f = fp32_agg.get(g, 0)
            v = aggs[m].get(g, 0)
            rat_vals.append(f'{v/f:.2f}×' if f else 'N/A')
        md .add(g, *ms_vals, *rat_vals)
        tex.add(g,
                *ms_vals,
                *[r.replace('×', r'$\times$') for r in rat_vals])

    # Wall-time summary row
    wall_fp32 = exp['fp32']['total_time_s']
    wall_ms   = [f"{exp[m]['total_time_s']:.2f}s" for m in exp_modes]
    wall_rats = [f"{wall_fp32/exp[m]['total_time_s']:.2f}×" for m in exp_modes if m != 'fp32']
    md .add('**Wall time**', *wall_ms, *wall_rats)
    tex.add('Total (wall)',  *wall_ms, *[r.replace('×', r'$\times$') for r in wall_rats])

    header_md = ('## Table 2 – Per-Component Pipeline Breakdown\n\n'
                 '_FP32 = no autocast; INT8/INT4 = FP16 autocast._\n\n')
    save('table_02_component_breakdown', header_md + md.render(), tex.render())


# ── Table 3 : Per Conv-Layer-Shape ───────────────────────────────────────────

def table_03(data):
    exp = data['exp3_conv']
    bs_key = list(exp.keys())[0]
    shapes = exp[bs_key]
    sorted_shapes = sorted(shapes.items(), key=lambda x: x[1]['fp32_ms'], reverse=True)

    headers = ['Layer (Cin→Cout, K, S)', 'Count', 'FP32 (ms)', 'INT8 E2E (ms)', 'INT8 ×', 'INT4 E2E (ms)', 'INT4 ×']

    md  = MDTable(headers)
    tex = TexTable(headers,
                   caption='Per-shape conv-layer benchmark (batch size 8). E2E includes quantization overhead.',
                   label='conv_layer')

    for sk, v in sorted_shapes:
        parts  = sk.split(',')
        cin    = parts[0].split('=')[1]
        cout   = parts[1].split('=')[1]
        k      = parts[2].split('=')[1]
        s_val  = parts[3].split('=')[1]
        count  = int(parts[4].split('=')[1])
        label  = f'{cin}→{cout}, K={k}, S={s_val}'

        fp32   = v['fp32_ms']
        i8     = v['int8_e2e_ms']
        i4     = v['int4_e2e_ms'] or 0
        s8     = v['int8_speedup_vs_fp32']
        s4     = v['int4_speedup_vs_fp32'] or 0

        md .add(label, count, f'{fp32:.4f}', f'{i8:.4f}', f'{s8:.2f}×',
                f'{i4:.4f}' if i4 else 'N/A', f'{s4:.2f}×' if s4 else 'N/A')
        tex.add(label, str(count), f'{fp32:.4f}', f'{i8:.4f}',
                f'{s8:.2f}' + r'$\times$',
                f'{i4:.4f}' if i4 else '---',
                f'{s4:.2f}' + r'$\times$' if s4 else '---')

    # Weighted summary row
    total_fp32 = sum(v['fp32_ms'] * int(sk.split('Count=')[1].split(',')[0].split(')')[0])
                     for sk, v in sorted_shapes)
    total_int8 = sum(v['int8_e2e_ms'] * int(sk.split('Count=')[1].split(',')[0].split(')')[0])
                     for sk, v in sorted_shapes)
    total_int4 = sum((v['int4_e2e_ms'] or v['fp32_ms']) *
                     int(sk.split('Count=')[1].split(',')[0].split(')')[0])
                     for sk, v in sorted_shapes)

    # Recompute using parsed count
    total_fp32 = sum(v['fp32_ms'] * v['count'] for _, v in sorted_shapes)
    total_int8 = sum(v['int8_e2e_ms'] * v['count'] for _, v in sorted_shapes)
    total_int4 = sum((v['int4_e2e_ms'] or v['fp32_ms']) * v['count'] for _, v in sorted_shapes)

    w8 = total_fp32 / total_int8
    w4 = total_fp32 / total_int4

    # MD footer
    footer_md = (f'\n**Weighted-average speedup (by layer count):** '
                 f'INT8 {w8:.2f}×, INT4 {w4:.2f}×\n')

    header_md = '## Table 3 – Per Conv-Layer-Shape Analysis\n\n'
    save('table_03_conv_layer', header_md + md.render() + footer_md, tex.render())


# ── Table 4 : Per Linear-Layer-Shape ─────────────────────────────────────────

def table_04(data):
    exp    = data['exp4_linear']
    bs_key = list(exp.keys())[0]
    shapes = exp[bs_key]

    headers = ['Shape (in→out)', 'Count', 'FP32 (ms)', 'INT8-base (ms)', 'INT8-MoDiff (ms)', 'INT4-base (ms)', 'INT4-MoDiff (ms)']

    md  = MDTable(headers)
    tex = TexTable(headers,
                   caption='Per-shape linear-layer benchmark (batch size 8). All shapes are time-embedding projections.',
                   label='linear_layer')

    for sk, v in shapes.items():
        parts = sk.split(',')
        inf   = parts[0].split('=')[1]
        outf  = parts[1].split('=')[1]
        count = parts[2].split('=')[1]
        label = f'{inf}→{outf}'
        md .add(label, count,
                f"{v['fp32_ms']:.4f}",
                f"{v['int8_baseline_ms']:.4f}", f"{v['int8_modiff_ms']:.4f}",
                f"{v['int4_baseline_ms']:.4f}", f"{v['int4_modiff_ms']:.4f}")
        tex.add(label, count,
                f"{v['fp32_ms']:.4f}",
                f"{v['int8_baseline_ms']:.4f}", f"{v['int8_modiff_ms']:.4f}",
                f"{v['int4_baseline_ms']:.4f}", f"{v['int4_modiff_ms']:.4f}")

    # Summary row
    avg = lambda key: np.mean([v[key] for v in shapes.values()])
    f32   = avg('fp32_ms')
    i8b   = avg('int8_baseline_ms')
    i8m   = avg('int8_modiff_ms')
    i4b   = avg('int4_baseline_ms')
    i4m   = avg('int4_modiff_ms')

    footer_md = (
        f'\n**Averages:** FP32 {f32:.4f} ms, '
        f'INT8-base {i8b:.4f} ms ({f32/i8b:.2f}× vs FP32), '
        f'INT8-MoDiff {i8m:.4f} ms, '
        f'INT4-base {i4b:.4f} ms ({f32/i4b:.2f}× vs FP32), '
        f'INT4-MoDiff {i4m:.4f} ms\n'
    )

    header_md = '## Table 4 – Per Linear-Layer-Shape Analysis\n\n'
    save('table_04_linear_layer', header_md + md.render() + footer_md, tex.render())


# ── Table 5 : Batch Size Ablation ────────────────────────────────────────────

def table_05(data):
    exp    = data['exp5_ablation']
    bsizes = sorted([int(k) for k in exp['fp32'].keys() if exp['fp32'][k] is not None])

    # 5a – time per sample
    headers_t = ['Batch Size', 'FP32 (ms)', 'INT8 (ms)', 'INT8/FP32', 'INT4 (ms)', 'INT4/FP32']
    md_t  = MDTable(headers_t)
    tex_t = TexTable(headers_t,
                     caption='Batch-size ablation: time per sample (50 DDIM steps).',
                     label='batch_latency')

    for bs in bsizes:
        f   = exp['fp32'][str(bs)]['time_per_sample'] * 1000
        i8d = exp['int8'].get(str(bs))
        i4d = exp['int4'].get(str(bs))
        i8  = i8d['time_per_sample'] * 1000 if i8d else None
        i4  = i4d['time_per_sample'] * 1000 if i4d else None
        s8  = f / i8 if i8 else None
        s4  = f / i4 if i4 else None
        md_t.add(bs, f'{f:.1f}',
                 f'{i8:.1f}' if i8 else 'OOM', f'{s8:.3f}×' if s8 else 'OOM',
                 f'{i4:.1f}' if i4 else 'OOM', f'{s4:.3f}×' if s4 else 'OOM')
        tex_t.add(str(bs), f'{f:.1f}',
                  f'{i8:.1f}' if i8 else r'\textit{OOM}',
                  f'{s8:.3f}' + r'$\times$' if s8 else r'\textit{OOM}',
                  f'{i4:.1f}' if i4 else r'\textit{OOM}',
                  f'{s4:.3f}' + r'$\times$' if s4 else r'\textit{OOM}')

    # 5b – throughput
    headers_th = ['Batch Size', 'FP32 (samp/s)', 'INT8 (samp/s)', 'INT4 (samp/s)']
    md_th  = MDTable(headers_th)
    tex_th = TexTable(headers_th,
                      caption='Batch-size ablation: throughput (samples/second, 50 DDIM steps).',
                      label='batch_throughput')

    for bs in bsizes:
        f   = exp['fp32'][str(bs)]['throughput_samples_per_sec']
        i8d = exp['int8'].get(str(bs))
        i4d = exp['int4'].get(str(bs))
        i8  = i8d['throughput_samples_per_sec'] if i8d else None
        i4  = i4d['throughput_samples_per_sec'] if i4d else None
        md_th.add(bs, f'{f:.2f}',
                  f'{i8:.2f}' if i8 else 'OOM',
                  f'{i4:.2f}' if i4 else 'OOM')
        tex_th.add(str(bs), f'{f:.2f}',
                   f'{i8:.2f}' if i8 else r'\textit{OOM}',
                   f'{i4:.2f}' if i4 else r'\textit{OOM}')

    header_md = '## Table 5a – Batch Size Ablation: Time per Sample\n\n'
    md_combined = (
        header_md + md_t.render() +
        '\n## Table 5b – Batch Size Ablation: Throughput\n\n' + md_th.render()
    )
    tex_combined = tex_t.render() + '\n' + tex_th.render()

    save('table_05_batch_ablation', md_combined, tex_combined)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print('Loading experiment results...')
    with open(RESULTS_PATH) as f:
        data = json.load(f)

    print('Generating tables...')
    table_01(data)
    table_02(data)
    table_03(data)
    table_04(data)
    table_05(data)
    print(f'\nDone! All tables saved in: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()

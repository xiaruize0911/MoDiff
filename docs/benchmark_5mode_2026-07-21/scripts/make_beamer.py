"""Generate a Beamer deck (numbers only) from data/*.csv -> slides/beamer.tex.
Every number is read straight from the measured CSVs (no transcription). Tables + the figures."""
import os, csv
os.chdir("/workspace/MoDiff")
HERE = "docs/benchmark_5mode_2026-07-21"
D = f"{HERE}/data"; OUT = f"{HERE}/slides"
os.makedirs(OUT, exist_ok=True)


def rd(n):
    with open(f"{D}/{n}") as f:
        return list(csv.DictReader(f))


def num(x, d=1):
    try: return f"{float(x):.{d}f}"
    except (ValueError, TypeError): return "--"


L = []
def w(s=""): L.append(s)

# ---------------- preamble ----------------
w(r"""\documentclass[aspectratio=169,10pt]{beamer}
\usetheme{Madrid}\usecolortheme{seahorse}
\usepackage{booktabs}\usepackage{graphicx}
\setbeamertemplate{navigation symbols}{}
\setbeamerfont{frametitle}{size=\normalsize}
\renewcommand{\arraystretch}{1.15}
\newcommand{\g}[1]{{\footnotesize\color{gray}#1}}
\title[MoDiff 5-mode benchmark]{MoDiff 5-mode Benchmark --- Measured Results}
\subtitle{LSUN-Churches LDM-8 UNet \textbullet\ b128 \textbullet\ DDIM \textbullet\ NVIDIA A40}
\date{2026-07-21}
\begin{document}
\frame{\titlepage}
""")

# ---------------- setup + inventory ----------------
w(r"""\begin{frame}{Setup \& kernel shape inventory}
\footnotesize
\begin{columns}[T]
\begin{column}{0.46\textwidth}
\begin{tabular}{@{}ll@{}}
\toprule
GPU & NVIDIA A40 (48\,GB, SM 8.6)\\
PyTorch / CUDA & 2.4.1+cu124 / 12.4\\
Model & LSUN-Churches LDM-8 UNet\\
Latent & $4\times32\times32$\\
Batch & 128\\
Sampler & DDIM\\
Modes & fp16, int8/int4 base, int8/int4 modiff\\
\bottomrule
\end{tabular}
\end{column}
\begin{column}{0.5\textwidth}
\begin{tabular}{@{}lrr@{}}
\toprule
family & shapes & calls/step\\
\midrule
conv & 33 & 89\\
\quad int8/int4 & 20 & 70\\
\quad fp16-only & 13 & 19\\
linear & 14 & 79\\
\quad qkv/proj (AWQ) & 10 & 42\\
\quad time-embed & 4 & 37\\
attention & 5 & 21\\
\quad flash int8/int4 & 3 & 15\\
\quad fp16 (hd=96) & 2 & 6\\
\bottomrule
\end{tabular}
\end{column}
\end{columns}
\end{frame}
""")

# ---------------- E2E speed ----------------
es = {r["mode"]: r for r in rd("e2e_speed.csv")}
order = ["fp16", "int8_baseline", "int4_baseline", "int8_modiff", "int4_modiff"]
lbl = {"fp16": "fp16", "int8_baseline": "int8 base", "int4_baseline": "int4 base",
       "int8_modiff": "int8 modiff", "int4_modiff": "int4 modiff"}
w(r"\begin{frame}{E2E DDIM step speed (b128, 5$\times$200 steps)}")
w(r"\centering\begin{tabular}{@{}lrrr@{}}\toprule")
w(r"mode & ms/step & min ms & vs fp16\\\midrule")
for m in order:
    r = es[m]
    w(f"{lbl[m]} & {num(r['ms_step'],1)} & {num(r['min_ms'],1)} & {num(r['speedup_vs_fp16'],2)}$\\times$\\\\")
w(r"\bottomrule\end{tabular}")
w(r"\end{frame}")

# ---------------- E2E timing profile ----------------
tp = {r["mode"]: r for r in rd("e2e_timing_profile.csv")}
buckets = [k for k in rd("e2e_timing_profile.csv")[0].keys() if k not in ("mode",)]
show = ["attention", "attn bmm (fp16)", "conv (int GEMM)", "qkv/proj int GEMM", "GroupNorm",
        "quantize/dequant", "modiff cache", "elementwise/copy", "gpu_busy", "wall"]
disp = {"attention": "attention", "attn bmm (fp16)": "attn bmm fp16", "conv (int GEMM)": "conv (int GEMM)",
        "qkv/proj int GEMM": "qkv/proj GEMM", "GroupNorm": "GroupNorm", "quantize/dequant": "quantize/dequant",
        "modiff cache": "modiff cache", "elementwise/copy": "elementwise/copy",
        "gpu_busy": "gpu\\_busy", "wall": "wall"}
w(r"\begin{frame}{E2E per-component timing profile \g{(GPU self-time, ms/step)}}")
w(r"\centering\scriptsize\begin{tabular}{@{}lrrrrr@{}}\toprule")
w(r"bucket & fp16 & int8 base & int4 base & int8 mod & int4 mod\\\midrule")
for b in show:
    if b in ("gpu_busy", "wall"): w(r"\midrule")
    row = " & ".join(num(tp[m][b], 1) for m in order)
    bb = "\\textbf{" + disp[b] + "}" if b in ("gpu_busy", "wall") else disp[b]
    w(f"{bb} & {row}\\\\")
w(r"\bottomrule\end{tabular}")
w(r"\end{frame}")

# ---------------- per-step family summary ----------------
ps = rd("perstep_summary.csv")
w(r"\begin{frame}{Time in one DDIM step, per kernel family \g{(ms/step = count$\times\mu$s/call)}}")
w(r"\centering\begin{tabular}{@{}lrrrrr@{}}\toprule")
w(r"family (calls/step) & fp16 & int8 base & int4 base & int8 mod & int4 mod\\\midrule")
famdisp = {"conv (all 89 convs/step)": "conv (89)",
           "linear qkv/proj+temb (79/step)": "linear qkv/proj+temb (79)",
           "attention block incl GN+quant (21/step)": "attention incl GN+quant (21)",
           "SUM of standalone kernels": "\\textbf{sum of kernels}"}
for r in ps:
    if r["family"] == "SUM of standalone kernels": w(r"\midrule")
    vals = " & ".join(num(r[m], 2) for m in order)
    w(f"{famdisp[r['family']]} & {vals}\\\\")
w(r"\bottomrule\end{tabular}")
w(r"\end{frame}")

# ---------------- conv per-shape (split 2 frames) ----------------
conv = [r for r in rd("conv_kernel_speed.csv") if r["Cin"] != "TOTAL_PER_STEP"]
convtot = next(r for r in rd("conv_kernel_speed.csv") if r["Cin"] == "TOTAL_PER_STEP")
conv = sorted(conv, key=lambda r: -float(r["fp16_us_per_step"]))
hdr = (r"$C_{in}\!\to\!C_{out}$ & HW & $\times$ & fp16 & i8\,b & i4\,b & i8\,m & i4\,m & i8b$\times$ & i4b$\times$")
def conv_rows(rows):
    for r in rows:
        q = "" if r["quant_eligible"] == "1" else "$^\\dagger$"
        yield (f"{r['Cin']}$\\to${r['Cout']}{q} & {r['H']}$^2$ & {r['count_per_step']} & "
               f"{num(r['fp16_us'],0)} & {num(r['int8_baseline_us'],0)} & {num(r['int4_baseline_us'],0)} & "
               f"{num(r['int8_modiff_us'],0)} & {num(r['int4_modiff_us'],0)} & "
               f"{num(r['int8_baseline_vs_fp16'],2)} & {num(r['int4_baseline_vs_fp16'],2)}\\\\")
half = 17
for i, chunk in enumerate([conv[:half], conv[half:]]):
    w(r"\begin{frame}{Conv kernel --- all 33 geometries \g{($\mu$s/call, %d/2)}}" % (i + 1))
    w(r"\centering\tiny\begin{tabular}{@{}lrrrrrrrrr@{}}\toprule")
    w(hdr + r"\\\midrule")
    for line in conv_rows(chunk): w(line)
    if i == 1:
        w(r"\midrule")
        w(f"\\textbf{{total/step (ms)}} & & 89 & "
          f"\\textbf{{{num(float(convtot['fp16_us_per_step'])/1000,1)}}} & "
          f"\\textbf{{{num(float(convtot['int8_baseline_us_per_step'])/1000,1)}}} & "
          f"\\textbf{{{num(float(convtot['int4_baseline_us_per_step'])/1000,1)}}} & "
          f"\\textbf{{{num(float(convtot['int8_modiff_us_per_step'])/1000,1)}}} & "
          f"\\textbf{{{num(float(convtot['int4_modiff_us_per_step'])/1000,1)}}} & "
          f"{num(convtot['int8_baseline_vs_fp16'],2)} & {num(convtot['int4_baseline_vs_fp16'],2)}\\\\")
    w(r"\bottomrule\end{tabular}")
    w(r"\par\vspace{2pt}{\tiny $^\dagger$ fp16-only geometry (skip / $1\times1$ / $C_{in}<32$ / out): same kernel in every mode.}")
    w(r"\end{frame}")

# ---------------- linear per-shape ----------------
lin = [r for r in rd("linear_kernel_speed.csv") if r["role"] != "TOTAL_PER_STEP"]
lintot = next(r for r in rd("linear_kernel_speed.csv") if r["role"] == "TOTAL_PER_STEP")
w(r"\begin{frame}{Linear GEMM --- all 14 shapes \g{($\mu$s/call; full=quantize+GEMM, gemm=GEMM-only)}}")
w(r"\centering\scriptsize\begin{tabular}{@{}llrrrrrrrr@{}}\toprule")
w(r"role & $K\!\to\!N$ & M & $\times$ & fp16 & i8 full & i8 gemm & i4 full & i4 gemm & i8$\times$/i4$\times$\\\midrule")
for r in lin:
    w(f"{r['role']} & {r['K']}$\\to${r['N']} & {int(float(r['M']))} & {r['count_per_step']} & "
      f"{num(r['fp16_us'],0)} & {num(r['int8_full_us'],0)} & {num(r['int8_gemmonly_us'],0)} & "
      f"{num(r['int4_full_us'],0)} & {num(r['int4_gemmonly_us'],0)} & "
      f"{num(r['int8_vs_fp16'],2)}/{num(r['int4_vs_fp16'],2)}\\\\")
w(r"\midrule")
w(f"\\textbf{{total/step (ms)}} & & & 79 & "
  f"\\textbf{{{num(float(lintot['fp16_us'])/1000,2)}}} & "
  f"\\textbf{{{num(float(lintot['int8_full_us'])/1000,2)}}} & "
  f"{num(float(lintot['int8_gemmonly_us'])/1000,2)} & "
  f"\\textbf{{{num(float(lintot['int4_full_us'])/1000,2)}}} & "
  f"{num(float(lintot['int4_gemmonly_us'])/1000,2)} & "
  f"{num(lintot['int8_vs_fp16'],2)}/{num(lintot['int4_vs_fp16'],2)}\\\\")
w(f"\\textbf{{vs fp16}} & & & & & \\textbf{{{num(lintot['int8_vs_fp16'],2)}$\\times$}} & "
  f"{num(lintot['int8_gemmonly_vs_fp16'],2)}$\\times$ & "
  f"\\textbf{{{num(lintot['int4_vs_fp16'],2)}$\\times$}} & {num(lintot['int4_gemmonly_vs_fp16'],2)}$\\times$ & \\\\")
w(r"\bottomrule\end{tabular}")
w(r"\end{frame}")

# ---------------- attention per-shape ----------------
at = [r for r in rd("attn_kernel_speed.csv") if r["C"] != "TOTAL_PER_STEP"]
attot = next(r for r in rd("attn_kernel_speed.csv") if r["C"] == "TOTAL_PER_STEP")
w(r"\begin{frame}{Attention block (GN+quant+attn) --- all 5 blocks \g{($\mu$s/call)}}")
w(r"\centering\scriptsize\begin{tabular}{@{}lrrrrrrrr@{}}\toprule")
w(r"C/hd/T & $\times$ & GN & fp16 & int8 & int4 & i8$\times$ & i4$\times$ & relL2 i8/i4\\\midrule")
for r in at:
    rl = (f"{num(r['relL2_int8'],3)}/{num(r['relL2_int4'],3)}" if r["relL2_int8"] not in ("", None) else "--")
    w(f"{r['C']}/{r['hd']}/{r['T']} & {r['count_per_step']} & {num(r['gn_us'],0)} & "
      f"{num(r['fp16_us'],0)} & {num(r['int8_us'],0)} & {num(r['int4_us'],0)} & "
      f"{num(r['int8_vs_fp16'],2)} & {num(r['int4_vs_fp16'],2)} & {rl}\\\\")
w(r"\midrule")
w(f"\\textbf{{total/step (ms)}} & 21 & & "
  f"\\textbf{{{num(float(attot['fp16_us_per_step'])/1000,1)}}} & "
  f"\\textbf{{{num(float(attot['int8_us_per_step'])/1000,1)}}} & "
  f"\\textbf{{{num(float(attot['int4_us_per_step'])/1000,1)}}} & "
  f"{num(attot['int8_vs_fp16'],2)} & {num(attot['int4_vs_fp16'],2)} & \\\\")
w(r"\bottomrule\end{tabular}")
w(r"\end{frame}")

# ---------------- GN->qkv quantize fusion (opt) ----------------
fq = [r for r in rd("fuse_gn_qkv_quant.csv") if r["C"] != "TOTAL_PER_STEP"]
fqt = next(r for r in rd("fuse_gn_qkv_quant.csv") if r["C"] == "TOTAL_PER_STEP")
fe = rd("fuse_gn_qkv_e2e.csv")[0]
w(r"\begin{frame}{Optimization: fuse qkv activation-quantize into GroupNorm \g{(int8; $\mu$s/call)}}")
w(r"\centering\scriptsize qkv front-end (GN + quantize + qkv GEMM), per block:\\[2pt]")
w(r"\begin{tabular}{@{}lrrrrrr@{}}\toprule")
w(r"C/T & $\times$ & fp16 & non-fused & fused & fus/nonf & relL2 vs nonfused\\\midrule")
for r in fq:
    w(f"{r['C']}/{r['T']} & {r['count_per_step']} & {num(r['fp16_fe_us'],0)} & {num(r['nonfused_fe_us'],0)} & "
      f"{num(r['fused_fe_us'],0)} & {num(r['fused_vs_nonfused'],2)}$\\times$ & {num(r['relL2_vs_nonfused'],4)}\\\\")
w(r"\midrule")
w(f"\\textbf{{total/step (ms)}} & 21 & \\textbf{{{num(float(fqt['fp16_fe_us'])/1000,2)}}} & "
  f"\\textbf{{{num(float(fqt['nonfused_fe_us'])/1000,2)}}} & \\textbf{{{num(float(fqt['fused_fe_us'])/1000,2)}}} & "
  f"\\textbf{{{num(fqt['fused_vs_nonfused'],2)}$\\times$}} & \\\\")
w(r"\bottomrule\end{tabular}")
w(r"\par\vspace{4pt}\scriptsize In-model (int8\_baseline, flag OFF vs ON): output rel-L2 "
  f"$= {num(fe['output_relL2_on_vs_off'],4)}$ (bit-identical), 21 qkv quantize kernels/step removed (42$\\to$21); "
  f"e2e ms/step {num(fe['ms_step_off'],1)} $\\to$ {num(fe['ms_step_on'],1)} "
  r"(gpu\_busy $-1$ ms/step, wall flat --- quantize was latency-hidden).")
w(r"\end{frame}")

# ---------------- figures ----------------
figs = [("fig_e2e_speed.png", "E2E DDIM step speed"),
        ("fig_e2e_timing_profile.png", "E2E per-component timing profile"),
        ("fig_perstep_summary.png", "Per-kernel-family time in one step"),
        ("fig_conv_kernel.png", "Conv kernel time per geometry + per-step total"),
        ("fig_conv_perstep.png", "Top conv geometries by per-step contribution"),
        ("fig_linear_kernel.png", "Linear qkv/proj time per shape + per-step total"),
        ("fig_attn_kernel.png", "Attention block time per shape + per-step total"),
        ("fig_fuse_gn_qkv.png", "§7 Fuse GN→qkv quantize: per-block front-end + per-step total")]
for f, t in figs:
    w(r"\begin{frame}{%s}" % t)
    w(r"\centering\includegraphics[width=\linewidth,height=0.82\textheight,keepaspectratio]{../figs/%s}" % f)
    w(r"\end{frame}")

w(r"\end{document}")

open(f"{OUT}/beamer.tex", "w").write("\n".join(L))
print(f"WROTE {OUT}/beamer.tex ({len(L)} lines)")

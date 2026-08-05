"""Assemble the full measurement report: every table, every figure, a glossary, no analysis.

Distinct from make_data_report.py, which covers the five benchmark suites only. This one adds the
nsys in-context measurements and the three diagrams, and it explains the vocabulary -- what a CV
is measured over, what gpu_busy_frac means, why an int4 channel dimension is halved -- because a
reader cannot use a table whose column headings are undefined.

It contains no interpretation: no rankings, no recommendations, no "what this means". Terminology
and measurement configuration are described; conclusions are not drawn.

The document is generated from the JSON and the trace summaries, never transcribed.
"""
import argparse
import collections
import glob
import io
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
D = "final_report_2026-07-28"
# The label must name the mode, not just its bit width. These are the *_baseline modes, i.e. MoDiff
# temporal caching DISABLED -- the plain W8A8/W4A4 PTQ configuration. Labelling them "INT8"/"INT4"
# read as though the report's headline speedups were MoDiff results; they are the opposite, and per
# the paper's Table 2 (LSUN-Church, W8) A8 is exactly the regime where MoDiff adds nothing (4.24 vs
# 3.85 FID). Its value is at A4 (355.85 -> 3.97). The MoDiff modes are internal "int8"/"int4".
MODES = [("fp16", "FP16"), ("int8_baseline", "INT8 (MoDiff off)"),
         ("int4_baseline", "INT4 (MoDiff off)")]


def rel(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def run(script, args):
    r = subprocess.run([sys.executable, os.path.join(HERE, script)] + args,
                       capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit("%s failed:\n%s\n%s" % (script, r.stdout[-1500:], r.stderr[-1500:]))
    return r.stdout


def banners(text, level="###"):
    out, lines, i = [], text.splitlines(), 0
    while i < len(lines):
        ln = lines[i]
        if set(ln.strip()) == {"="} and len(ln.strip()) > 10:
            if i + 1 < len(lines) and lines[i + 1].strip():
                out.append("%s %s" % (level, lines[i + 1].strip()))
                i += 3 if (i + 2 < len(lines) and set(lines[i + 2].strip()) == {"="}) else 2
                continue
            i += 1
            continue
        out.append(ln)
        i += 1
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def e2e_runs_table(files):
    ds = [json.load(open(rel(f))) for f in files]
    out = ["| mode | " + " | ".join("run %d" % (i + 1) for i in range(len(ds)))
           + " | mean ms/batch | cross-run CV | within-run CV | vs FP16 |",
           "|---|" + "---:|" * (len(ds) + 4)]
    means = {}
    for m, lbl in MODES:
        v = [d["modes"][m]["stats"]["mean"] / 1e3 for d in ds]
        cvi = sum(d["modes"][m]["stats"]["cv_pct"] for d in ds) / len(ds)
        mu = sum(v) / len(v)
        means[m] = mu
        sd = (sum((x - mu) ** 2 for x in v) / (len(v) - 1)) ** 0.5 if len(v) > 1 else 0.0
        out.append("| %s | %s | %.1f | %.2f%% | %.2f%% | %s |"
                   % (lbl, " | ".join("%.1f" % x for x in v), mu, sd / mu * 100, cvi,
                      "1.000×" if m == "fp16" else "%.3f×" % (means["fp16"] / mu)))
    return "\n".join(out), ds[0]


def nsys_attention_table(nd):
    """In-context GPU time per attention shape, averaged over the runs, from the trace summaries."""
    rows = collections.defaultdict(dict)
    for f in sorted(glob.glob(os.path.join(nd, "nsys_attention_*_b128_run*.txt"))):
        m = re.match(r".*nsys_attention_(.+)_b128_run(\d)\.txt", f)
        mode, run = m.group(1), int(m.group(2))
        for line in open(f):
            g = re.match(r"\|\s+attention\s+(C\d+)\s+(T\d+)\s+\|\s+[\d.]+\s+\|.*?\|\s+\d+\s+\|"
                         r"\s+([\d.]+)\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)", line)
            if g:
                rows[(mode, g.group(1) + "/" + g.group(2))][run] = (
                    float(g.group(3)), float(g.group(4)), float(g.group(5)))
    out = ["| mode | shape | GPU µs/call (run 1 / 2 / 3) | mean | CV | CPU issue µs/call |"
           " GPU / issue |", "|---|---|---|---:|---:|---:|---:|"]
    for (mode, shape), rr in sorted(rows.items(),
                                    key=lambda kv: (kv[0][0], -int(kv[0][1].split("/T")[1]))):
        if len(rr) < 3:
            continue
        g = [rr[r][0] for r in sorted(rr)]
        mu = sum(g) / len(g)
        sd = (sum((x - mu) ** 2 for x in g) / (len(g) - 1)) ** 0.5
        iss = sum(rr[r][1] for r in rr) / len(rr)
        rat = sum(rr[r][2] for r in rr) / len(rr)
        out.append("| %s | %s | %s | %.1f | %.2f%% | %.1f | %.2f |"
                   % (mode, shape, " / ".join("%.1f" % x for x in g), mu, sd / mu * 100,
                      iss, rat))
    return "\n".join(out)


def nsys_idle_table(nd):
    out = ["| mode | traces | GPU busy | GPU idle | median gap | gaps > 50 µs |",
           "|---|---:|---:|---:|---:|---|"]
    for m, lbl in MODES:
        busy, idle, med, big = [], [], [], []
        for f in sorted(glob.glob(os.path.join(nd, "nsys_all_%s_b128_run*.txt" % m))):
            t = open(f).read()
            b = re.search(r"GPU busy\s+[\d.]+ ms\s+\(([\d.]+)%", t)
            i = re.search(r"GPU idle\s+[\d.]+ ms\s+\(([\d.]+)%", t)
            g = re.search(r"median ([\d.]+) µs", t)
            n = re.search(r"gaps > 50 µs: (\d+), totalling [\d.]+ ms \(([\d.]+)%", t)
            if b:
                busy.append(float(b.group(1)))
            if i:
                idle.append(float(i.group(1)))
            if g:
                med.append(float(g.group(1)))
            if n:
                big.append((int(n.group(1)), float(n.group(2))))
        if not busy:
            continue
        out.append("| %s | %d | %.1f%% | %.1f%% | %.2f µs | %d, %.1f%% of the span |"
                   % (lbl, len(busy), sum(busy) / len(busy), sum(idle) / len(idle),
                      sum(med) / len(med),
                      round(sum(b for b, _ in big) / len(big)),
                      sum(p for _, p in big) / len(big)))
    return "\n".join(out)


def hierarchy_table(nd, e2e_files):
    """Level 1 of the e2e breakdown: share of wall time by layer type."""
    sys.path.insert(0, HERE)
    import csv
    ds = [json.load(open(rel(f))) for f in e2e_files]
    out = ["| layer type | " + " | ".join(l for _, l in MODES) + " |",
           "|---|" + "---:|" * len(MODES)]
    per = {}
    for m, _ in MODES:
        acc = collections.Counter()
        n = 0
        for c in sorted(glob.glob(os.path.join(nd, "nsys_all_%s_b128_run*.kernrange.csv" % m))):
            n += 1
            for r in csv.reader(open(c)):
                l1 = r[0].split("|")[1] if r[0].startswith("L|") else "outside layers"
                acc[l1] += float(r[2])
        per[m] = (acc, max(n, 1))
    kinds = sorted({k for m, _ in MODES for k in per[m][0]},
                   key=lambda k: -per["fp16"][0].get(k, 0))
    walls = {m: sum(d["modes"][m]["stats"]["mean"] for d in ds) / len(ds) / 1e3
             for m, _ in MODES}
    for k in kinds:
        cells = []
        for m, _ in MODES:
            acc, n = per[m]
            tot = sum(acc.values()) or 1
            ms = acc.get(k, 0) / tot * walls[m]
            cells.append("%.0f ms  (%.0f%%)" % (ms, ms / walls[m] * 100))
        out.append("| %s | %s |" % (k, " | ".join(cells)))
    return "\n".join(out)


GLOSSARY = """
## Glossary

Terms used in the column headings and figures.

**mean of the round medians.** Every timing does `warmup`, then `rounds` x `iters` timed calls.
Inside a round the median of the `iters` samples is taken, which drops a single scheduler hiccup;
the reported value is the mean of those round medians. Medians are also in the JSON and differ by
under 0.4%.

**CV (coefficient of variation).** stdev / mean, as a percent, computed over the round medians —
so it measures whether re-running would give the same number, not how much individual calls
jitter. The latter is `within_round_cv_pct`. CV is dimensionless, so a 2000 µs kernel and a 20 µs
one can be compared.

**95% CI.** A Student-t interval on the mean of the round medians. At 8 rounds the normal
approximation would be about 20% too narrow. Speedups carry a delta-method interval of their own.

**stability.** A label on CV: `tight` ≤ 1%, `ok` ≤ 3%, `NOISY` > 3%.

**cross-run vs within-run.** Within-run CV is computed inside one process. Cross-run CV compares
independent invocations. They are not the same number and the report gives both where available.

**gpu_busy_frac.** Summed kernel self-time divided by the measured wall time of the same layer.
Below 1 the GPU was idle inside the measured interval.

**CPU issue cost.** The wall time the CPU spends issuing a layer's kernels — Python plus the aten
dispatch chain. It is roughly constant per call regardless of tensor size.

**GPU / issue.** From the nsys traces: GPU time attributed to a layer divided by the wall time of
its NVTX range. The range is CPU-side and closes when issuing finishes, so above 1 the CPU is
running ahead of the device and below 1 it is not.

**profiler self-time, and scaling.** The profiler reports each kernel's own GPU time. Its total
differs from the measured wall clock, so stage tables are scaled by `profiler_scale` = wall /
profiler total. Every timing in this report is measured once WITHOUT a profiler and attributed in
a separate profiled run.

**entry point vs kernel.** An entry point is the C++ function Python calls, e.g.
`gemm_w4a4_awq_bias_res`. A kernel is what runs on the GPU, e.g. `gemm_w4a4_kernel_awq`. One entry
may launch several kernels, and the same kernel may serve several entries.

**`*_qout` entries.** The fused steady-state attention entries, which write their output already
quantized. In the first few steps of a run other entries fire (`_vt`, `_vt_static`, fp16 SDPA);
over 200 steps the fused ones are what execute, so cross-mode tables select them.

**packed int4.** Two 4-bit values share one int8 byte, so a packed channel dimension is half the
logical one: `(128, 32, 32, 96) int8` carries 192 logical channels.

**T, C, hd.** T is tokens per attention block (H×W of the feature map), C is channels, hd is head
dimension. The quantized kernels pad hd from 24 to 32 at T=1024.

**layer kinds.** `attention` is an AttentionBlock; `resblock_plain` a ResBlock without resize;
`resblock_updown` a ResBlock carrying an Upsample or Downsample. These are UNet module types, not
op types — conv and linear live inside all three.

**op classes.** Kernels are grouped by name into attention core, QKV/output projection,
convolution, GroupNorm+quantize, K/V gather, attention output quantize, and an explicit
elementwise/other bucket. The mapping is in `ck_stages.py`; CUTLASS names a convolution
`..._fprop_...` and a matmul `..._gemm_...`, so matching is on specific tokens rather than
substrings.

**`quantized?` = `no — dtype only`.** Marks a shape whose entry point is torch's own op in all
three modes, i.e. it was never quantized. Its cross-mode ratio reflects the input dtype the
surrounding pipeline supplied, not quantization.

**NVTX range.** A named marker pushed around each layer so nsys can attribute GPU work to it.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernels", required=True)
    ap.add_argument("--layers", required=True)
    ap.add_argument("--e2e-suite", required=True)
    ap.add_argument("--e2e-run", action="append", required=True)
    ap.add_argument("--nsys-dir", default="docs/final_report_2026-07-28/data/nsys")
    ap.add_argument("--out", default="docs/MEASUREMENT_REPORT_2026-08-01.md")
    ap.add_argument("--commit", default="")
    a = ap.parse_args()

    ks = json.load(open(rel(a.kernels)))
    nd = rel(a.nsys_dir)
    tables = banners(run("ck_final_numbers.py",
                         ["--kernels", a.kernels, "--layers", a.layers,
                          "--e2e", a.e2e_suite, "--top", "200"]))
    tables = "\n".join(l for l in tables.splitlines() if not l.startswith("gpu="))
    profile = banners(run("ck_attention_profile.py", ["--layers", a.layers, "--no-plot"]),
                      level="####")
    st = run("ck_report_numbers.py", ["--e2e", a.e2e_suite, "--layers", a.layers])
    # Strip the banner lines BEFORE converting banners to headings: this block begins with a
    # "=" rule, and banners() would take the line after it -- the table's own header row -- as a
    # section title, emitting "### | stage | FP16 | ...".
    raw = st[st.index("2. WHOLE-MODEL"):st.index("3. LAYER LEVEL")].rsplit("=" * 40, 1)[0]
    stages = "\n".join(l for l in raw.splitlines()
                       if not l.startswith("2. WHOLE-MODEL") and set(l.strip()) != {"="})
    stages = ("### Whole-model time by stage\n\nProfiler self-time scaled to the measured "
              "median wall time, ms per batch.\n" + stages.strip())
    e2e_tab, _ = e2e_runs_table(a.e2e_run)

    doc = f"""# Measurement report — FP16 / INT8 / INT4

**Hardware** {ks['gpu']} · **Model** LSUN-churches LDM, 21 AttentionBlocks + 35 ResBlocks
**Batch** 128 throughout · **End to end** 200 DDIM steps
{('**Commit** `' + a.commit + '`') if a.commit else ''}

Data and figures, with the vocabulary explained. No analysis: nothing here ranks, recommends or
concludes. Generated by `integration/benchmarks/report/make_measurement_report.py`.

**Measurement protocol.** Every timing is taken twice over: once WITHOUT a profiler, which is the
number reported, and once WITH one, which supplies the per-kernel attribution. Each is warmed up
and repeated — kernel and layer suites do {ks['warmup']} warmup calls then {ks['rounds']} rounds x
{ks['iters_per_round']} timed calls; the end-to-end suite does 3 warmup samples then several
profiler-free repeats, and that whole procedure is itself run three times.

{GLOSSARY}

## Source data

| file | contents |
|---|---|
| [`{D}/data/{os.path.basename(a.kernels)}`]({D}/data/{os.path.basename(a.kernels)}) | attention / conv / linear kernel suites: every call signature, distribution, per-kernel profile |
| [`{D}/data/{os.path.basename(a.layers)}`]({D}/data/{os.path.basename(a.layers)}) | per-layer suite: every (kind, shape) x mode, distribution, per-kernel profile |
| [`{D}/data/{os.path.basename(a.e2e_suite)}`]({D}/data/{os.path.basename(a.e2e_suite)}) | end-to-end suite, 9 profiler-free repeats + 1 profiled run per mode |
| `{D}/data/e2e_plots_b128_run{{1,2,3}}.json` | the three independent end-to-end invocations |
| `{D}/data/nsys/nsys_attention_*_b128_run{{1,2,3}}.txt` | 9 nsys traces, attention layers annotated |
| `{D}/data/nsys/nsys_all_*_b128_run{{1,2,3}}.txt` | 9 nsys traces, every layer annotated |

`.nsys-rep` and `.png` are gitignored; regenerate with the scripts below.

| script | role |
|---|---|
| [`kernel_suites_bench.py`](../integration/benchmarks/report/kernel_suites_bench.py) | kernel suites |
| [`layer_pipeline_bench.py`](../integration/benchmarks/report/layer_pipeline_bench.py) | layer suite |
| [`e2e_three_mode_bench.py`](../integration/benchmarks/report/e2e_three_mode_bench.py) | end-to-end suite |
| [`nsys_layer_trace.py`](../integration/benchmarks/report/nsys_layer_trace.py) | nsys traces + idle analysis |
| [`setup_nsys.sh`](../integration/benchmarks/report/setup_nsys.sh) | provisions the Nsight Systems CLI |
| [`ck_bench_stats.py`](../integration/benchmarks/report/ck_bench_stats.py) | the statistics shared by all suites |
| [`ck_final_numbers.py`](../integration/benchmarks/report/ck_final_numbers.py) · [`ck_attention_profile.py`](../integration/benchmarks/report/ck_attention_profile.py) · [`ck_report_numbers.py`](../integration/benchmarks/report/ck_report_numbers.py) | tables |
| [`make_pipeline_diagram.py`](../integration/benchmarks/report/make_pipeline_diagram.py) · [`make_e2e_hierarchy.py`](../integration/benchmarks/report/make_e2e_hierarchy.py) · [`make_checkpoint_report_plots.py`](../integration/benchmarks/report/make_checkpoint_report_plots.py) | figures |
| [`ck_audit.py`](../integration/benchmarks/report/ck_audit.py) · [`ck_verify_report.py`](../integration/benchmarks/report/ck_verify_report.py) | checks |

## Figures

### Kernels inside the attention layer

![attention kernel profile]({D}/plots/fig_attn_kernel_profile.png)

Stack = summed kernel self-time; the black tick is the measured pipeline latency. Where they
differ the layer is CPU-dispatch bound (see `gpu_busy_frac` in the glossary).

### What each attention kernel does, and the tensors between them

![attention pipeline diagram]({D}/plots/fig_attn_pipeline_diagram.png)

C192/32², T=1024. Kernels and times are from the layer profile; shapes are the logical dataflow.

### End to end: layer type → op class → kernel

![e2e hierarchy]({D}/plots/fig_e2e_hierarchy.png)

Attribution by NVTX range from the nsys traces, scaled to the mean profiler-free wall time.

### Suite overviews

![e2e]({D}/plots/fig_final_e2e.png)

![layers]({D}/plots/fig_final_layers.png)

![speedup matrix]({D}/plots/fig_final_speedup_matrix.png)

## End to end, three independent invocations

{e2e_tab}

{stages}

## In context: attention layers during a real run

GPU time attributed to each attention shape by NVTX range, over three traces per mode.

{nsys_attention_table(nd)}

### GPU idle on the timeline

{nsys_idle_table(nd)}

### Share of end-to-end time by layer type

{hierarchy_table(nd, a.e2e_run)}

## Suites: full tables

{tables}

## Attention layer: per-kernel profile

{profile}

## Measurement conditions

- `models/ldm/lsun_churches256/model.ckpt` is an 856-byte stub whose `state_dict` has 0 entries,
  loaded with `strict=False`; all weights are randomly initialised. No figure here is an
  image-quality measurement. The INT4 static calibration carries one shared scale (34.6463) across
  all 21 layers.
- Nsight Compute counters are unavailable: `ncu` returns `ERR_NVGPUCTRPERM`, the driver has
  `RmProfilingAdminOnly=1` and the container has no `CAP_SYS_ADMIN`. Nsight Systems traces
  normally, since CUPTI's Activity API is not gated by that permission; its CPU sampling is
  disabled because `perf_event_paranoid=4`.
- 12 of the 33 FP16-mode torch convs receive fp32 inputs, against 2 of 13 in the quantized modes.
- Within one process, `cv_pct` does not predict cross-session reproducibility for launch-bound
  rows: re-running the suites moved 29 of 237 comparable kernel rows by more than 10%, worst
  +31.4%, and that row reported `cv_pct` 0.94%.
- Container rebuilt 2026-08-01 (`omegaconf`, `einops`, `pytorch_lightning`, `tqdm`, `matplotlib`
  reinstalled; torch 2.4.1+cu124 unchanged).
"""
    out = rel(a.out)
    with open(out, "w") as f:
        f.write(doc)
    print("wrote %s (%d lines, %d table rows)"
          % (out, doc.count("\n") + 1, sum(1 for l in doc.splitlines() if l.startswith("|"))))


if __name__ == "__main__":
    main()

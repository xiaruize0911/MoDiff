"""Generate the data-only final report: tables, figures, and links to the source JSON.

The report is ASSEMBLED, not written. Every table comes from ck_final_numbers.py and
ck_attention_profile.py running against the measured JSON, so the document cannot drift from the
data and cannot pick up a transcription slip -- which is the entire reason the previous revision's
tables were generated too.

Contains no interpretation: no rankings, no recommendations, no "what this means". The only prose
is measurement configuration and the factual notes needed to keep a column from being read as
something it is not (which keys were never quantized, why the capture-window call counts are not a
steady-state mix). Those are properties of the data, not conclusions drawn from it.

Usage:
  python3 make_data_report.py --kernels ... --layers ... --e2e ... --out docs/FINAL_REPORT.md
"""
import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))


def rel(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def run(script, args):
    r = subprocess.run([sys.executable, os.path.join(HERE, script)] + args,
                       capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit("%s failed:\n%s\n%s" % (script, r.stdout[-2000:], r.stderr[-2000:]))
    return r.stdout


def banners_to_headings(text, level="###"):
    """The generators separate sections with ==== banner lines. Left as-is those become setext H1
    in markdown (a line of '=' under text promotes it), which would silently restructure the
    document, so they are converted to explicit ATX headings."""
    out, lines = [], text.splitlines()
    i = 0
    while i < len(lines):
        ln = lines[i]
        if set(ln.strip()) == {"="} and len(ln.strip()) > 10:
            # banner; the title is the next line, optionally followed by a closing banner
            if i + 1 < len(lines) and lines[i + 1].strip():
                title = lines[i + 1].strip()
                j = i + 2
                if j < len(lines) and set(lines[j].strip()) == {"="}:
                    j += 1
                out.append("%s %s" % (level, title))
                i = j
                continue
            i += 1
            continue
        out.append(ln)
        i += 1
    # collapse 3+ blank lines
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernels", required=True)
    ap.add_argument("--layers", required=True)
    ap.add_argument("--e2e", required=True)
    ap.add_argument("--out", default="docs/FINAL_REPORT_2026-08-01.md")
    ap.add_argument("--commit", default="")
    a = ap.parse_args()

    ks = json.load(open(rel(a.kernels)))
    lay = json.load(open(rel(a.layers)))
    e2e = json.load(open(rel(a.e2e)))

    n_kern = sum(len(v) for rec in ks["modes"].values() for v in rec.values())
    n_lay = sum(len(v) for v in lay["modes"].values())
    n_e2e = sum(m["stats"]["n"] for m in e2e["modes"].values())

    tables = banners_to_headings(run("ck_final_numbers.py",
                                     ["--kernels", a.kernels, "--layers", a.layers,
                                      "--e2e", a.e2e, "--top", "200"]))
    tables = "\n".join(l for l in tables.splitlines() if not l.startswith("gpu="))
    profile = banners_to_headings(run("ck_attention_profile.py",
                                      ["--layers", a.layers, "--no-plot"]), level="####")

    # The whole-model stage decomposition lives in ck_report_numbers.py, not ck_final_numbers.py.
    # Splice its section in rather than duplicating the kernel->stage attribution in a second
    # place: that mapping is adversarial (CUTLASS names a convolution "..._fprop_..." and a matmul
    # "..._gemm_..."), and two copies would eventually disagree.
    stages_raw = run("ck_report_numbers.py", ["--e2e", a.e2e, "--layers", a.layers])
    start = stages_raw.index("2. WHOLE-MODEL TIME BY STAGE")
    end = stages_raw.index("3. LAYER LEVEL")
    stages = banners_to_headings(stages_raw[start:end].rsplit("=" * 40, 1)[0])
    stages = "### Whole-model time by stage (profiler self-time, ms per batch)\n\n" + \
             "\n".join(l for l in stages.splitlines()
                       if not l.startswith("2. WHOLE-MODEL"))

    D = "final_report_2026-07-28"
    kb, lb, eb = (os.path.basename(a.kernels), os.path.basename(a.layers),
                  os.path.basename(a.e2e))

    doc = f"""# Final report (data) — FP16 / INT8 / INT4, five measurement levels

**Hardware** {ks['gpu']} · **Model** LSUN-churches LDM, 21 AttentionBlocks + 35 ResBlocks
**Batch** {ks['batch']} everywhere · **End to end** {e2e['steps']} DDIM steps, {e2e['repeats']} repeats after 3 warmup samples
**Kernel and layer suites** warmup {ks['warmup']}, then {ks['rounds']} rounds x {ks['iters_per_round']} timed calls
{('**Commit** `' + a.commit + '`') if a.commit else ''}

Tables and figures only. This document is generated by
`integration/benchmarks/report/make_data_report.py` directly from the JSON below; the analysis that
accompanied the previous revision is in git history.

## Source data

| file | contents |
|---|---|
| [`{D}/data/{kb}`]({D}/data/{kb}) | suites 1-3: every kernel call signature, full timing distribution, per-kernel profile |
| [`{D}/data/{lb}`]({D}/data/{lb}) | suite 4: every (layer kind, shape) x mode, distribution + per-kernel profile + role rollup |
| [`{D}/data/{eb}`]({D}/data/{eb}) | suite 5: per-repeat wall times, `route_check`, full per-kernel profile per mode |

Each JSON row carries `stats` with `n`, `mean`, `median`, `stdev`, `sem`, `cv_pct`, `min`, `max`,
`spread_pct`, `ci95_lo/hi`, `iqr`, `within_round_cv_pct`, and the raw per-round `samples`.

| script | role |
|---|---|
| [`kernel_suites_bench.py`](../integration/benchmarks/report/kernel_suites_bench.py) | suites 1-3 driver |
| [`layer_pipeline_bench.py`](../integration/benchmarks/report/layer_pipeline_bench.py) | suite 4 driver |
| [`e2e_three_mode_bench.py`](../integration/benchmarks/report/e2e_three_mode_bench.py) | suite 5 driver |
| [`ck_bench_stats.py`](../integration/benchmarks/report/ck_bench_stats.py) | timing + statistics shared by all five |
| [`ck_final_numbers.py`](../integration/benchmarks/report/ck_final_numbers.py) | emits the tables below |
| [`ck_attention_profile.py`](../integration/benchmarks/report/ck_attention_profile.py) | emits the attention-layer profile + figure |
| [`ck_verify_report.py`](../integration/benchmarks/report/ck_verify_report.py) | checks every quoted figure against the JSON |
| [`make_data_report.py`](../integration/benchmarks/report/make_data_report.py) | assembles this document |

**Counts.** {n_kern} kernel signatures, {n_lay} layer measurements, {n_e2e} end-to-end repeats.

## How to read the numbers

Central values are the **mean of the round medians**; intervals are **Student-t 95%** on that mean.
Within a round the median of the timed calls is taken, so `cv_pct` is round-to-round
reproducibility, not within-round jitter (`within_round_cv_pct` in the JSON covers that).
`stability` is a label on `cv_pct`: `tight` <= 1%, `ok` <= 3%, `NOISY` > 3%.

Two annotations appear in the tables and are properties of the data, not judgements:

- **`quantized?` = `no — dtype only`** marks a key whose entry point is torch's own op in *all
  three* modes, so it was never quantized. Its cross-mode ratio reflects the input dtype the
  surrounding pipeline supplied (fp32 in fp16 mode, fp16 in the quantized modes): the same entry at
  the same conv shape reads 1381.8 us against 579.5 us. Those keys are excluded from the quoted
  ranges.
- **`calls_per_sample` in the JSON is a capture-window count, not a steady-state mix.** The
  attention route differs in the first steps: at T=1024 the 25 calls of a 5-step window split
  10 / 5 / 10 across `flash_attn_*_vt`, `_vt_static` and the fused `*_qout` entry. Over 200 steps
  the fused entry is what runs (the e2e profile shows that kernel firing 1000 times = 5 blocks x
  200 steps). Cross-mode tables therefore select the fused `*_qout` entry.

## Figures

![e2e](final_report_2026-07-28/plots/fig_ck0801_e2e.png)

![layers](final_report_2026-07-28/plots/fig_ck0801_layers.png)

![speedup matrix](final_report_2026-07-28/plots/fig_ck0801_speedup_matrix.png)

![attention stages](final_report_2026-07-28/plots/fig_ck0801_attn_stages.png)

![attention kernel profile](final_report_2026-07-28/plots/fig_attn_kernel_profile.png)

## Suites 1-5: tables

{tables}

{stages}

## Attention layer: per-kernel profile

{profile}

## Measurement conditions

- `models/ldm/lsun_churches256/model.ckpt` is an 856-byte stub whose `state_dict` has 0 entries,
  loaded with `strict=False`; all weights are randomly initialised. No figure here is an
  image-quality measurement. The INT4 static calibration carries one shared scale (34.6463) across
  all 21 layers.
- Nsight Compute counters are unavailable: `ncu` 2024.1.1 returns `ERR_NVGPUCTRPERM`. All
  attribution is CUDA-event timing plus profiler self-time.
- 12 of the 33 FP16-mode torch convs receive fp32 inputs, against 2 of 13 in the quantized modes.
- The e2e repeats are not independent: they trend monotonically within a run (FP16 20550 -> 20638
  ms, INT8 14577 -> 14747 ms across 9 repeats), so the t-interval assumes more independence than
  the samples have. Per-repeat values are listed with the e2e table and in the JSON.
- Container rebuilt 2026-08-01 (`omegaconf`, `einops`, `pytorch_lightning`, `tqdm`, `matplotlib`
  reinstalled; torch 2.4.1+cu124 unchanged).
- Stage tables are profiler self-time scaled to the measured **median** wall time; the scaled sums
  reproduce it to +-0.00%.
"""
    out = rel(a.out)
    with open(out, "w") as f:
        f.write(doc)
    print("wrote %s (%d lines, %d table rows)"
          % (out, doc.count("\n") + 1, sum(1 for l in doc.splitlines() if l.startswith("|"))))


if __name__ == "__main__":
    main()

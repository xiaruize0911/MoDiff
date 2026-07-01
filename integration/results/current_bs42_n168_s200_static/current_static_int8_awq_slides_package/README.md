# Static INT8/AWQ Slides Package

This package contains the Beamer source and plot assets for:

`current_static_int8_awq_slides.tex`

Workload: batch size 42 separated kernels, static quantization.

The deck intentionally excludes full-pipeline benchmark timing. It puts
implementation/kernel-call details before the expanded separated-kernel results.

## Contents

- `current_static_int8_awq_slides.tex`: main Beamer source.
- `current_static_int8_awq_slides.pdf`: compiled slide deck with embedded PGFPlots figures.
- `pgf_plots/`: native PGFPlots figures with numbered category labels.
- `reports/`: markdown reports used as source context.
- `data/`: expanded 5-round benchmark CSV/JSON.
- `scripts/`: benchmark script used for the expanded separated-kernel run.
- `Makefile`: builds the PDF when `pdflatex` is installed.

## Build

Run:

```bash
make
```

Expected output:

`current_static_int8_awq_slides.pdf`

If `pdflatex` is not installed, install a LaTeX distribution such as TeX Live first. The deck uses the `pgfplots` package.

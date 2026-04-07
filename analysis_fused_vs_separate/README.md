# Fused vs Separate MoDiff Benchmark

This folder contains a dedicated fused-vs-separate benchmark suite for MoDiff.

It covers two levels and now supports two activation-quantization modes:

- `dynamic`
  - activation scales are recomputed online from the current tensor / residual
  - useful for measuring the pure dynamic MoDiff path

- `static`
  - one fixed activation scale is reused per quantized layer
  - whole-model runs load matching saved scales when available, otherwise auto-calibrate fresh scales from representative DDIM sampling calls

The benchmark suite itself still compares **fused vs separate kernels** inside either quantization mode.

It covers two levels:

- `01_layerwise_fused_vs_separate_all_shapes.py`
  - enumerates **all unique Conv2d shapes** exercised by the LSUN-Churches LDM UNet
  - benchmarks the **modulated MoDiff hot path** for fused vs separate implementations
  - reports INT8 and INT4 timings for Step1, Conv, and Total
  - uses warmup iterations plus timed iterations × repeats

- `02_modelwise_fusion_benchmark.py`
  - benchmarks the **whole LSUN-Churches LDM model**
  - compares fused vs separate MoDiff implementations for INT8 and INT4
  - uses warmup runs plus timed iterations × repeats
  - resets MoDiff state before every timed call for fairness

## Output folders

By default, the scripts write into whatever `--output-dir` you pass. For clean mode-separated artifacts, use distinct directories such as:

- `analysis_fused_vs_separate/dynamic_quant/layerwise_results/`
- `analysis_fused_vs_separate/dynamic_quant/modelwise_results/`
- `analysis_fused_vs_separate/static_quant/layerwise_results/`
- `analysis_fused_vs_separate/static_quant/modelwise_results/`

Each script emits JSON plus a Markdown report. The layerwise runner also writes a CSV table.

## Typical usage

### Layerwise, all shapes

```bash
python analysis_fused_vs_separate/01_layerwise_fused_vs_separate_all_shapes.py \
  --quant-mode dynamic \
  --output-dir analysis_fused_vs_separate/dynamic_quant/layerwise_results
```

### Layerwise, static mode

```bash
python analysis_fused_vs_separate/01_layerwise_fused_vs_separate_all_shapes.py \
  --quant-mode static \
  --output-dir analysis_fused_vs_separate/static_quant/layerwise_results
```

### Layerwise smoke test

```bash
python analysis_fused_vs_separate/01_layerwise_fused_vs_separate_all_shapes.py \
  --quant-mode dynamic \
  --max-shapes 2 \
  --warmup 5 \
  --iters 20 \
  --timed-repeats 3
```

### Whole model

```bash
python analysis_fused_vs_separate/02_modelwise_fusion_benchmark.py \
  --ckpt /path/to/model.ckpt \
  --quant-mode dynamic \
  --output-dir analysis_fused_vs_separate/dynamic_quant/modelwise_results
```

### Whole model, static mode

```bash
python analysis_fused_vs_separate/02_modelwise_fusion_benchmark.py \
  --ckpt /path/to/model.ckpt \
  --quant-mode static \
  --output-dir analysis_fused_vs_separate/static_quant/modelwise_results
```

### Whole-model smoke test

```bash
python analysis_fused_vs_separate/02_modelwise_fusion_benchmark.py \
  --ckpt /path/to/model.ckpt \
  --quant-mode dynamic \
  --precision int8 \
  --steps 10 \
  --batch-size 4 \
  --timing-iterations 1 \
  --timing-repeats 2 \
  --warmup-runs 1
```

## Notes

- The whole-model benchmark expects an LSUN-Churches LDM checkpoint. The current workspace does **not** include one, so pass `--ckpt` explicitly when running it.
- Optional calibration files are looked up at:
  - `integration/calibration/int8_calibration.pt`
  - `integration/calibration/int4_calibration.pt`
- In `dynamic` mode those files are ignored by design.
- In `static` mode the full-model benchmark loads matching saved scales when possible; if a supplied scale file is absent or has no matching conv keys, the script auto-calibrates fresh scales and saves them beside the modelwise results.
- The layerwise benchmark does **not** require the checkpoint because it benchmarks synthetic tensors shaped from the UNet architecture inventory.

# Memory Transfer Analysis Report -- Real Measurement

**Model**: LSUN-Churches LDM (U-Net diffusion model)  
**Steps**: 50  |  **Batch**: 4  
**Method**: Forward-pass hooks (measured) + analytical Q/DQ round-trips  

---

## Summary

| Mode | Total HBM (GB) | vs FP32 | Weight (GB) | Cache total (GB) | Q/DQ (GB) |
|------|--------------|--------|------------|-----------------|----------|
| FP32 | 66.14 | --- | 54.79 | 0.00 | 0.00 |
| INT8 Standard | 54.18 | +18.1% | 16.51 | 0.00 | 26.32 |
| INT4 Standard | 47.13 | +28.7% | 10.60 | 0.00 | 25.17 |
| INT8 MoDiff | 68.82 | -4.0% | 16.51 | 12.88 | 28.07 |
| INT4 MoDiff | 61.76 | +6.6% | 10.60 | 12.88 | 26.92 |

## Key Findings

- **INT8 Standard** saves +18.1% HBM vs FP32 (weight bytes: 54.8 GB -> 16.5 GB, 3.3x compression); Q/DQ overhead: 26.32 GB
- **INT4 Standard** saves +28.7% HBM vs FP32 (weight bytes: 54.8 GB -> 10.6 GB, 5.2x compression); Q/DQ overhead: 25.17 GB
- **INT8 MoDiff** total: 68.82 GB (-4.0% vs FP32); cache overhead: 12.88 GB; Q/DQ overhead: 28.07 GB
- **INT4 MoDiff** total: 61.76 GB (+6.6% vs FP32); cache overhead: 12.88 GB; Q/DQ overhead: 26.92 GB

## Why MoDiff Is Still Fast

MoDiff's speedup comes from 4x/8x GEMM throughput (tensor cores process
INT8/INT4 residuals, not FP32 activations), not from reduced DRAM bandwidth.
The residuals (a_t - a_hat_{t+1}) have ~10x smaller range, enabling INT4
quantization with FP32-level output quality (Theorem 1 of the MoDiff paper).

## Measurement Method

### Directly Measured (forward-pass hooks, `tensor.nbytes`)
- `input[0]` — FP32 activation (4 B/elem)
- quantised weight buffer: `weight_int8` (1 B/elem), `weight_packed` (0.5 B/elem),
  `weight_fp16` for linear layers (2 B/elem)
- `output` — FP32 (4 B/elem)
- MoDiff `a_hat_cache` read + write; `o_hat_cache` read (from next-step perspective)
  (o_hat write IS the output write, not double-counted)

### Analytically Added (Q/DQ kernel-boundary round-trips)
Every CUDA kernel call produces a new tensor allocation; for conv activations
(smallest: 4×192×32×32 = 786 KB >> L2), these round-trip through HBM:

**INT8 Conv standard:**
  `x(FP32) → scale_quantize_int8 → x_int8 → CUTLASS → out_raw → ×w_scale → out`
  Extra = write+read `x_int8` (INT8, ×¼) + write+read `out_raw` (FP32)
  = `2×(act/4) + 2×out`

**INT4 Conv standard:**
  Extra = write+read `x_packed` (INT4, ×⅛) + write+read `out_raw`
  = `2×(act/8) + 2×out`

**MoDiff Conv modulated (extra on top of standard):**
  `sub_absmax_scale` writes `_residual_buf` (FP32, =act);
  `scale_quantize_int8` reads it; `dequant_accumulate_int8` reads it again
  + `a_hat_cache` is read a **second time** by `dequant_accumulate_int8`
  Extra mod = `3×act + a_hat_nbytes`

**FP16-Linear layers:** activations are ≤12 KB → L2-resident; counted as 0.

## Output Files

| File | Description |
|------|-------------|
| `memory_transfer_analysis.json` | Raw numbers for all modes |
| `plot_memory_total_io.png` | Total IO bar chart |
| `plot_memory_breakdown.png` | Stacked component breakdown |
| `plot_memory_savings.png` | Savings vs FP32 |
| `plot_memory_cumulative.png` | IO accumulation over timesteps |
| `plot_memory_per_step.png` | Per-step bar chart |
| `table_memory_summary.md/tex` | Full summary table |
| `table_memory_per_step.md/tex` | Per-step IO table |

# Memory Transfer Analysis Report -- Real Measurement

**Model**: LSUN-Churches LDM (U-Net diffusion model)  
**Steps**: 50  |  **Batch**: 4  
**Method**: Forward-pass hooks (measured)  

---

## Summary

| Mode | Total HBM (GB) | vs FP32 | Weight (GB) | Cache total (GB) |
|------|--------------|--------|------------|-----------------|
| FP32 | 66.16 | --- | 54.81 | 0.00 |
| INT8 Standard | 27.90 | +57.8% | 16.55 | 0.00 |
| INT4 Standard | 22.00 | +66.8% | 10.64 | 0.00 |
| INT8 MoDiff | 49.84 | +24.7% | 16.55 | 21.94 |
| INT4 MoDiff | 43.94 | +33.6% | 10.64 | 21.94 |

## Key Findings

- **INT8 Standard** reads 54.8 GB -> 16.5 GB weight bytes (3.3x compression); 
- **INT4 Standard** reads 54.8 GB -> 10.6 GB weight bytes (5.2x compression); 
- **INT8 MoDiff** total: 49.84 GB (+24.7% vs FP32); cache overhead: 21.94 GB;
- **INT4 MoDiff** total: 43.94 GB (+33.6% vs FP32); cache overhead: 21.94 GB;

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

### Conv-only modulated kernel split (measured labels)
- **Kernel-1** (`step1_quantize_*_fprop`): reads input + `a_hat_cache`; writes
  quantized activation + updated `a_hat_cache` + `_residual_buf`
- **Kernel-2** (`conv2d_*_fprop_o_hat`): reads quantized activation + quantized
  weights + `weight_scale_channel` + previous `o_hat_cache`; writes updated `o_hat_cache`
- Output file: `table_memory_conv_kernel_split.md` and `plot_memory_conv_kernel_split.png`

## Output Files

| File | Description |
|------|-------------|
| `memory_transfer_analysis.json` | Raw numbers for all modes |
| `plot_memory_total_io.png` | Total IO bar chart |
| `plot_memory_breakdown.png` | Stacked component breakdown |
| `plot_memory_savings.png` | Savings vs FP32 |
| `plot_memory_cumulative.png` | IO accumulation over timesteps |
| `plot_memory_per_step.png` | Per-step bar chart |
| `plot_memory_conv_kernel_split.png` | Conv kernel-1 vs kernel-2 IO |
| `table_memory_summary.md/tex` | Full summary table |
| `table_memory_per_step.md/tex` | Per-step IO table |
| `table_memory_conv_kernel_split.md` | Conv kernel-1/kernel-2 IO table |

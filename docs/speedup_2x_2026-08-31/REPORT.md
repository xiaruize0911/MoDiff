# Pipeline ≥2× vs fp16 (2026-08-31)

Churches LDM-8, NVIDIA A40, batch 128, DDIM 50, reset MoDiff state **outside** the CUDA-event timer. Same process.

## Result

| arm | ms/step | vs fp16 |
|---|--:|--:|
| fp16 | 105.75 | 1.00× |
| W8A8 PTQ | 64.78 | 1.63× |
| **W4A4 PTQ** | **50.96** | **2.08×** |
| W4A4 MoDiff K=1 | 70.14 | 1.51× |
| W4A4 replay K=4 | 51.87 | 2.04× |

Target for 2.00×: **52.87 ms/step**. W4A4 PTQ (`int4_baseline`) is under it.

Data: `data/e2e_four_arm.json`.

## What changed

`OptimizedInt4Conv2d._has_weight_zp` used `torch.any(zp != 0)` on the CUDA buffer at every conv. That is a device reduction plus a host sync, and it is illegal during CUDA-graph capture.

It is now a Python flag, refreshed when `weight_zp` is assigned (init zeros → False; AdaRound / tests that replace the buffer still arm correctly). The forward path never reads the GPU zp just to decide the no-op.

Before this cache, W4A4 PTQ on the same protocol was **56–57 ms** (1.80–1.83× vs fp16 ~102.5). After: **50.96 ms**.

CUDA-graph capture of INT4 PTQ now succeeds (`UNetCudaGraphManager` phase `standard`). Extra saving is ~0.5–1 ms; not required for 2×.

## What did not work (left opt-in / unused)

- `MODIFF_QUANT_SKIP_OUT=1` wrapping skip 1×1s and `out.2`: uncalibrated was 50 ms but relL2 2.0 (noise); live-calib was **slower** than shipped PTQ (54.3 ms) and relL2 2.0. Default stays off.
- `torch.compile` on INT4 PTQ: Inductor `Or.expand` crash.
- Replay K=8/16: ≥2× but W4A4 images smear (historical relL2 0.55 at K=8). Not the 2× arm.

## Quality

The 2× arm is shipped **W4A4 PTQ**, not replay. This tree has `int4_calibration_realckpt.pt` (qdiff file absent). n=4 latent relL2 vs fp16 ≈ 0.64. Historical qdiff W4A4 PTQ was ~0.47–0.50. Images are the usual W4A4-PTQ softening, not an empty network. W8A8 PTQ remains 1.63× if you need that quality.

Grid: `plots/verify_2x_grid.png`.

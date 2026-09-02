# Skip-K on Stable Diffusion 1.5

Freeze `a_hat`/`o_hat` writes on K−1 of every K steps. GN+quantize+conv still run. Attention is SpatialTransformer (QKV/proj = Linear + fp16). `MODIFF_QUANT_ATTN=0`, `MODIFF_LINEAR=0`. Uncalibrated activations (SmoothQuant NaN on C=320 `in_conv`).

NVIDIA A40 · DDIM 50 · prompt `a photograph of a church on a hill` · n=4 quality · batch 8 speed.

| K | relL2 vs fp16 | e2e ms/step | vs K=1 | layer 320→320 ms |
|---|--:|--:|--:|--:|
| 1 | 0.063 | 170.21 | 1.000× | 0.565 |
| 2 | 0.066 | 170.22 | 1.000× | 0.547 |
| 5 | 0.069 | 169.96 | 1.001× | 0.540 |
| 10 | 0.072 | 169.90 | 1.002× | 0.530 |
| 20 | 0.075 | 169.89 | 1.002× | 0.528 |
| 50 | 0.090 | 169.91 | 1.002× | 0.529 |

K=20 is visually the same as K=1. K=50 still the same buildings, relL2 0.090. E2E ceiling **0.2%**. Grid: `plots/skip_k_sd15_grid.png`.

Churches (same skip, production attn W8A8) saturates at ~3% e2e by K=4 and quality falls through PTQ at K=50. SD1.5 quality is more robust because skip-K does not touch SpatialTransformer Linears; that is also why the e2e win is smaller.

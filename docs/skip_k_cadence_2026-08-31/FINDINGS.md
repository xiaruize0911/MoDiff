# Skip-K write freeze (K=1,2,5,10)

Freeze `a_hat`/`o_hat` stores on K−1 of every K modulated steps. GN+quantize+conv still run. No replay. No FID.

NVIDIA A40 · W8A8 MoDiff · LSUN-churches · DDIM 50 · `MODIFF_CACHE_SKIP_K` CUDA path.

## Quality (n=6, seed 20260805)

| K | relL2 vs fp16 | write skip |
|---|--:|--:|
| 1 | 0.120 | 0% |
| 2 | 0.133 | 51% |
| 5 | 0.176 | 82% |
| 10 | 0.254 | 92% |

K=2 is visually the same buildings as K=1. K=10 changes structure (column 3 drops the two red spires). Grid: `plots/skip_k_grid.png`.

## Speed

End-to-end, batch 128, median of 2 timed trials:

| K | ms/step | vs K=1 |
|---|--:|--:|
| 1 | 93.62 | 1.000× |
| 2 | 92.40 | 1.013× (−1.30%) |
| 5 | 91.88 | 1.019× (−1.85%) |
| 10 | 91.91 | 1.019× (−1.83%) |

K=5 and K=10 are the same within trial noise. Write-only, so the e2e ceiling is ~2%.

One-layer 192→192 32×32, 200 steps: 1.068 → 1.023 / 0.997 / 0.983 ms (1.04× / 1.07× / 1.09×).

# Skip-K: higher K, Churches + SD1.5

Freeze `a_hat` / `o_hat` stores on K−1 of every K modulated steps. GN+quantize+conv still run. No replay. No FID.

## Churches LDM-8 (production W8A8)

A40 · batch 128 · DDIM 50 · `MODIFF_QUANT_ATTN=1` · static delta. Each timed trial is a full sample from a reset MoDiff state.

| mode | ms/step | vs fp16 | vs K=1 | relL2 vs fp16 | write skip |
|---|--:|--:|--:|--:|--:|
| fp16 | 102.47 | 1.00× | — | 0.000 | — |
| W8A8 PTQ | 64.80 | **1.58×** | — | 0.324 | — |
| K=1 | 85.60 | 1.20× | 1.000× | 0.108 | 0% |
| K=2 | 84.21 | 1.22× | 1.017× | 0.128 | 51% |
| K=4 | 83.38 | 1.23× | 1.027× | 0.156 | 76% |
| K=5 | 83.25 | 1.23× | 1.028× | 0.165 | 82% |
| K=10 | 83.02 | 1.23× | 1.031× | 0.255 | 92% |
| K=20 | 82.34 | 1.24× | 1.040× | 0.322 | 96% |
| K=50 | 82.98 | 1.23× | 1.032× | 0.368 | 100%* |
| K=100 | 83.06 | 1.23× | 1.031× | 0.368 | 100%* |

\*t=T still commits; `_write_ahat_now` is only on modulated steps. K=50 and K=100 on a 50-step DDIM are the same freeze-after-T path.

Speed already plateaus at **K=4** (~83 ms). K=20–100 stay in that band (K=20’s 82.34 is a separate retime, CV 0.25%). Raising K does not close the 18 ms gap to PTQ.

Quality: K=20 matches PTQ (0.322 vs 0.324). K=50/100 go **past** PTQ (0.368) and look muddy — freeze-after-T. Grid: `plots/quality_grid.png` (K≤10) and `plots/quality_grid_highk.png` (K=20/50/100).

One-layer 192→192: 1.055 → 0.962 / 0.957 / 0.955 ms at K=20/50/100 (1.10× vs K=1). Layer keeps picking up a few tens of µs; e2e does not.

## Stable Diffusion 1.5

A40 · batch 8 · latent 4×64×64 · DDIM 50 · prompt `a photograph of a church on a hill`. MoDiff on convs only (`MODIFF_QUANT_ATTN=0`, `MODIFF_LINEAR=0`). Uncalibrated / dynamic activations (no SmoothQuant file). Write counter stays 0 on that path; skip still changes the samples.

| K | ms/step | vs K=1 | relL2 vs fp16 | layer 320→320 ms |
|---|--:|--:|--:|--:|
| 1 | 170.21 | 1.000× | 0.063 | 0.565 |
| 2 | 170.22 | 1.000× | 0.066 | 0.547 |
| 5 | 169.96 | 1.001× | 0.069 | 0.540 |
| 10 | 169.90 | 1.002× | 0.072 | 0.530 |
| 20 | 169.89 | 1.002× | 0.075 | 0.528 |
| 50 | 169.91 | 1.002× | 0.090 | 0.529 |

E2E ceiling **0.2%**. Layer saturates at ~1.07× by K=10. Quality holds through K=20; K=50 is still the same buildings (relL2 0.090). Grid: `docs/skip_k_sd15_2026-08-31/plots/skip_k_sd15_grid.png`.

SD1.5 is more skip-tolerant than Churches because SpatialTransformer (184 Linears + fp16 attention) is a larger fraction of the step — skip-K does not touch those Linears. That is also why the e2e win is 10× smaller than Churches’ 3%.

## Takeaway

Raising K past 4–10 does not buy e2e speed on either model. On Churches it only spends quality: K=20 ≈ PTQ, K=50 = freeze-after-T and worse than PTQ. On SD1.5 quality survives to K=50, but the step is already attention-bound so skip is 0.2%.

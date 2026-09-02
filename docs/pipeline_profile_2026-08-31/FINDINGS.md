# Full-pipeline profile: does the 1.58× layer win show up e2e?

Protocol: LSUN-Churches LDM-8, A40, batch 128, DDIM 50, 1 warmup + 2 timed (CV ≤ 0.23%). Production path: `MODIFF_LINEAR=0`, static delta, `int8_calibration_realckpt.pt`, attention W8A8 fused epilogue. Skip is the same MoDiff path with `MODIFF_CACHE_SKIP_K=5` (skip `a_hat`/`o_hat` stores on 4/5 steps; GN+conv still run). Kernel buckets from torch profiler, scaled to unprofiled wall time.

## Wall clock

| mode | ms/step | vs fp16 | vs PTQ |
|---|--:|--:|--:|
| fp16 | 102.23 | 1.00× | — |
| W8A8 PTQ | 64.77 | **1.58×** | 1.00× |
| W8A8 MoDiff | 72.20 | 1.42× | 0.90× |
| W8A8 skip (K=5) | 69.13 | 1.48× | 0.94× |

The pipeline **does** speed up vs fp16. Skip recovers 3.07 ms vs MoDiff commit (−4.3%), still 4.36 ms behind PTQ. The single-layer 1.58× (MoDiff vs PTQ on 192→192 32×32) does **not** appear e2e.

## Kernel buckets (ms/step)

| bucket | fp16 | PTQ | MoDiff | skip K=5 |
|---|--:|--:|--:|--:|
| GroupNorm+SiLU family | 20.90 | 10.83 | 17.41 | 15.22 |
| GEMM / conv | 46.66 | 37.13 | 37.81 | 37.07 |
| attention | 11.35 | 9.01 | 8.86 | 8.79 |
| elementwise / copy | 19.59 | 5.77 | 6.11 | 6.07 |
| other | 3.73 | 2.03 | 2.00 | 1.98 |

GN+GEMM summed over the UNet: fp16 67.6 → PTQ 48.0 → MoDiff 55.2 → skip 52.3. Skip’s 3 ms comes almost entirely from GN (−2.2 ms) plus a small GEMM nudge; leftover (attention + copy) is unchanged (~17 ms).

Data: `data/e2e.json`, `data/buckets.json`.

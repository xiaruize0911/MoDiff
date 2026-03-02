## Table 2 – Per-Component Pipeline Breakdown

_FP32 = no autocast; INT8/INT4 = FP16 autocast._

| Component | FP32 (ms) | INT8 (ms) | INT4 (ms) | INT8/FP32 | INT4/FP32 |
|---|---|---|---|---|---|
| Conv2d | 3274.4 | 2948.6 | 2517.0 | 0.90× | 0.77× |
| Attention | 3512.1 | 511.4 | 532.5 | 0.15× | 0.15× |
| Linear | 68.6 | 183.8 | 223.1 | 2.68× | 3.25× |
| GroupNorm | 266.5 | 268.1 | 283.1 | 1.01× | 1.06× |
| SiLU | 29.3 | 29.2 | 29.8 | 1.00× | 1.02× |
| **Wall time** | 9.86s | 6.24s | 5.99s | 1.58× | 1.65× |

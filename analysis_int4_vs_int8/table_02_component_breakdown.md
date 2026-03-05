## Table 2 – Per-Component Pipeline Breakdown

_FP32 = no autocast; INT8/INT4 = FP16 autocast._

| Component | FP32 (ms) | INT8 (ms) | INT4 (ms) | INT8/FP32 | INT4/FP32 |
|---|---|---|---|---|---|
| Conv2d | 1437.7 | 1296.2 | 1122.1 | 0.90× | 0.78× |
| Attention | 952.9 | 259.1 | 266.3 | 0.27× | 0.28× |
| Linear | 81.1 | 465.3 | 542.8 | 5.74× | 6.70× |
| GroupNorm | 127.8 | 156.9 | 162.3 | 1.23× | 1.27× |
| SiLU | 22.8 | 35.6 | 58.7 | 1.56× | 2.57× |
| **Wall time** | 3.86s | 3.73s | 3.78s | 1.04× | 1.02× |

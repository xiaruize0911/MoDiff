## Table 4 – Per Linear-Layer-Shape Analysis

| Shape (in→out) | Count | FP32 (ms) | INT8-base (ms) | INT8-MoDiff (ms) | INT4-base (ms) | INT4-MoDiff (ms) |
|---|---|---|---|---|---|---|
| 192→768 | 1 | 0.0316 | 0.0535 | 0.1700 | 0.0532 | 0.1687 |
| 768→768 | 15 | 0.0347 | 0.0525 | 0.1718 | 0.0525 | 0.1721 |
| 768→384 | 6 | 0.0353 | 0.0524 | 0.1711 | 0.0529 | 0.1719 |
| 768→1536 | 15 | 0.0345 | 0.0528 | 0.1713 | 0.0524 | 0.1704 |

**Averages:** FP32 0.0340 ms, INT8-base 0.0528 ms (0.64× vs FP32), INT8-MoDiff 0.1710 ms, INT4-base 0.0527 ms (0.64× vs FP32), INT4-MoDiff 0.1708 ms

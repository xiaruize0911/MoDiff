## Table 4 – Per Linear-Layer-Shape Analysis

| Shape (in→out) | Count | FP32 (ms) | INT8-base (ms) | INT8-MoDiff (ms) | INT4-base (ms) | INT4-MoDiff (ms) |
|---|---|---|---|---|---|---|
| 192→768 | 1 | 0.0340 | 0.0516 | 0.1704 | 0.0534 | 0.1660 |
| 768→768 | 15 | 0.0367 | 0.0533 | 0.1744 | 0.0536 | 0.1750 |
| 768→384 | 6 | 0.0369 | 0.0529 | 0.1763 | 0.0530 | 0.1749 |
| 768→1536 | 15 | 0.0368 | 0.0531 | 0.1736 | 0.0524 | 0.1702 |

**Averages:** FP32 0.0361 ms, INT8-base 0.0527 ms (0.68× vs FP32), INT8-MoDiff 0.1736 ms, INT4-base 0.0531 ms (0.68× vs FP32), INT4-MoDiff 0.1716 ms

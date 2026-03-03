# Per-Timestep DRAM Transfer -- Measured

| Mode | Total (GB) | Per-Step (MB) | vs FP32 | Weight Save | Cache/step (MB) |
|------|-----------|--------------|--------|------------|----------------|
| FP32 | 66.14 | 1322.9 | 1.00x | 0% | 0.0 |
| INT8 Standard | 54.18 | 1083.7 | 0.82x | 70% | 0.0 |
| INT4 Standard | 47.13 | 942.6 | 0.71x | 81% | 0.0 |
| INT8 MoDiff | 68.82 | 1376.3 | 1.04x | 70% | 257.7 |
| INT4 MoDiff | 61.76 | 1235.2 | 0.93x | 81% | 257.7 |
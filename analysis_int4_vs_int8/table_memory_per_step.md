# Per-Timestep DRAM Transfer -- Measured

| Mode | Total (GB) | Per-Step (MB) | vs FP32 | Weight Save | Cache/step (MB) |
|------|-----------|--------------|--------|------------|----------------|
| FP32 | 66.16 | 1323.2 | 1.00x | 0% | 0.0 |
| INT8 Standard | 27.90 | 558.0 | 0.42x | 70% | 0.0 |
| INT4 Standard | 22.00 | 439.9 | 0.33x | 81% | 0.0 |
| INT8 MoDiff | 49.84 | 996.8 | 0.75x | 70% | 438.8 |
| INT4 MoDiff | 43.94 | 878.7 | 0.66x | 81% | 438.8 |
# Memory Transfer Summary -- Real Measurement

**Steps:** 50  |  **Batch:** 4

> Measured via forward-pass hooks recording `tensor.nbytes` for each layer call.

| Mode | Weight (GB) | Act FP32 (GB) | Output (GB) | Cache Rd (GB) | Cache Wr (GB) | **Total (GB)** | **vs FP32** |
|------|------------|--------------|------------|-------------|-------------|--------------|------------|
| FP32 | 54.81 | 6.57 | 4.79 | 0.00 | 0.00 | **66.16** | --- |
| INT8 Standard | 16.55 | 6.57 | 4.79 | 0.00 | 0.00 | **27.90** | +57.8% |
| INT4 Standard | 10.64 | 6.57 | 4.79 | 0.00 | 0.00 | **22.00** | +66.8% |
| INT8 MoDiff | 16.55 | 6.57 | 4.79 | 12.72 | 9.22 | **49.84** | +24.7% |
| INT4 MoDiff | 10.64 | 6.57 | 4.79 | 12.72 | 9.22 | **43.94** | +33.6% |

> Positive = less DRAM than FP32.

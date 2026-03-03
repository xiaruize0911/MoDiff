# Memory Transfer Summary -- Real Measurement

**Steps:** 50  |  **Batch:** 4

> Measured via forward-pass hooks recording `tensor.nbytes` for each layer call.
> Q/DQ overhead computed analytically from kernel-boundary tensor sizes.

| Mode | Weight (GB) | Act FP32 (GB) | Output (GB) | Cache Rd (GB) | Cache Wr (GB) | Q/DQ (GB) | **Total (GB)** | **vs FP32** |
|------|------------|--------------|------------|-------------|-------------|----------|--------------|------------|
| FP32 | 54.79 | 6.57 | 4.79 | 0.00 | 0.00 | 0.00 | **66.14** | --- |
| INT8 Standard | 16.51 | 6.57 | 4.79 | 0.00 | 0.00 | 26.32 | **54.18** | +18.1% |
| INT4 Standard | 10.60 | 6.57 | 4.79 | 0.00 | 0.00 | 25.17 | **47.13** | +28.7% |
| INT8 MoDiff | 16.51 | 6.57 | 4.79 | 8.19 | 4.70 | 28.07 | **68.82** | -4.0% |
| INT4 MoDiff | 10.60 | 6.57 | 4.79 | 8.19 | 4.70 | 26.92 | **61.76** | +6.6% |

> Positive = less DRAM than FP32.

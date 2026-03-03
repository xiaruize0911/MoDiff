# Key Memory Transfer Ratios

| Mode | Total IO | vs FP32 | Weight Compression | GEMM Act Compression | Cache Cost |
|------|---------|---------|-------------------|---------------------|------------|
| FP32 | 582.5 GB | 1x | 1x | 1x | - |
| INT8 Standard | 418.1 GB | 1.39x | 4.0x | 4.0x | - |
| INT4 Standard | 390.7 GB | 1.49x | 8.0x | 8.0x | - |
| INT8 MoDiff | 989.7 GB | 0.59x | 4.0x | 4.0x | 571.6 GB |
| INT4 MoDiff | 962.3 GB | 0.61x | 8.0x | 8.0x | 571.6 GB |

**Weight Compression**: ratio of FP32 weight bytes to quantised weight bytes.  
**GEMM Act Compression**: ratio of FP32 activation bytes to bytes loaded by tensor cores.  
**Cache Cost**: extra FP32 bytes for MoDiff per-layer a_hat + o_hat caches (total).  

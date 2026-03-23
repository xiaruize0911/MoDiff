# Layer-Level Quantization Timing Report

**Date**: 2026-03-22 16:43:55
**GPU**: NVIDIA A40
**Batch Size**: 32

This report compares the current dynamic activation quantization path against a static-scale quantization path using the same CUTLASS quantization kernels.

Interpretation notes:
- **Dynamic quantization**: includes the per-tensor absmax/scale discovery inside the hot path.
- **Static quantization**: reuses a fixed precomputed activation scale and only performs quantize/(pack) work.
- **IO proxy**: tensor copy used as a lower-bound proxy for memory movement during quantization.
- **Compute estimate**: `static_quant_ms - io_proxy_ms`, clipped at zero. This is an upper-bound style estimate of arithmetic/packing overhead.

## INT8 Dynamic vs Static Quantization

| Shape | Dynamic (ms) | Static (ms) | Dynamic overhead | Absmax+scale (ms) | IO proxy (ms) | Compute est. (ms) | Dominant |
| --- | --- | --- | --- | --- | --- | --- | --- |
| INT8_32x192x32x32 | 0.196 | 0.056 | +0.140ms (+251.9%) | 0.149 | 0.094 | 0.000 | io |
| INT8_32x384x16x16 | 0.104 | 0.030 | +0.074ms (+243.8%) | 0.082 | 0.049 | 0.000 | io |
| INT8_32x768x8x8 | 0.055 | 0.022 | +0.032ms (+144.6%) | 0.078 | 0.027 | 0.000 | io |

## INT4 Dynamic vs Static Quantization

| Shape | Dynamic (ms) | Static (ms) | Dynamic overhead | Absmax+scale (ms) | IO proxy (ms) | Compute est. (ms) | Dominant |
| --- | --- | --- | --- | --- | --- | --- | --- |
| INT4_32x192x32x32 | 0.190 | 0.050 | +0.140ms (+278.6%) | 0.150 | 0.094 | 0.000 | io |
| INT4_32x384x16x16 | 0.100 | 0.027 | +0.073ms (+266.6%) | 0.082 | 0.049 | 0.000 | io |
| INT4_32x768x8x8 | 0.052 | 0.021 | +0.031ms (+145.0%) | 0.079 | 0.028 | 0.000 | io |

## Key takeaways

- The gap between dynamic and static quantization isolates the cost of discovering a fresh activation scale in the hot path.
- The IO proxy vs static quantization comparison indicates whether quantization is primarily memory-movement limited or arithmetic/packing limited.
- If the IO proxy is close to the static quantization time, quantization is effectively IO-bound; if the compute estimate is larger, arithmetic/packing is the main contributor.
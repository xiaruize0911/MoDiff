# Layer-Level Quantization Timing Report

**Date**: 2026-03-19 14:11:01
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
| INT8_32x192x32x32 | 0.197 | 0.056 | +0.141ms (+249.6%) | 0.151 | 0.095 | 0.000 | io |
| INT8_32x384x16x16 | 0.105 | 0.030 | +0.074ms (+245.0%) | 0.093 | 0.050 | 0.000 | io |
| INT8_32x768x8x8 | 0.055 | 0.024 | +0.031ms (+131.1%) | 0.089 | 0.036 | 0.000 | io |

## INT4 Dynamic vs Static Quantization

| Shape | Dynamic (ms) | Static (ms) | Dynamic overhead | Absmax+scale (ms) | IO proxy (ms) | Compute est. (ms) | Dominant |
| --- | --- | --- | --- | --- | --- | --- | --- |
| INT4_32x192x32x32 | 0.190 | 0.051 | +0.140ms (+275.5%) | 0.151 | 0.095 | 0.000 | io |
| INT4_32x384x16x16 | 0.101 | 0.031 | +0.070ms (+227.8%) | 0.094 | 0.050 | 0.000 | io |
| INT4_32x768x8x8 | 0.080 | 0.034 | +0.046ms (+134.9%) | 0.122 | 0.031 | 0.003 | io |

## Key takeaways

- The gap between dynamic and static quantization isolates the cost of discovering a fresh activation scale in the hot path.
- The IO proxy vs static quantization comparison indicates whether quantization is primarily memory-movement limited or arithmetic/packing limited.
- If the IO proxy is close to the static quantization time, quantization is effectively IO-bound; if the compute estimate is larger, arithmetic/packing is the main contributor.
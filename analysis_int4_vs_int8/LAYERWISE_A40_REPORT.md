# A40 Layerwise INT8 vs INT4 Speedup Report

**Generated:** 2026-03-29 05:47:58
**GPU:** NVIDIA A40
**PyTorch:** 2.6.0+cu124
**CUDA (PyTorch):** 12.4
**Iterations:** 100
**Warmup:** 20

## What MoDiff is

MoDiff is **error-compensated modulated quantization across diffusion timesteps**.
Instead of quantizing the full activation $a_t$ at every step, it caches the next-step approximation and quantizes the residual:

$$
\hat{a}_t = Q(a_t - \hat{a}_{t+1}) + \hat{a}_{t+1}, \qquad
\hat{o}_t = A(Q(a_t - \hat{a}_{t+1})) + \hat{o}_{t+1}
$$

That makes quantization **more accurate**, especially at lower activation bitwidths. It does **not** automatically guarantee a 2x latency speedup, because the residual path, cache traffic, quantization, and dequant/accumulate work still cost time.

## Summary table

| Shape | Raw conv-only | Baseline fused | MoDiff fused (static) | MoDiff fused (dynamic) |
| --- | --- | --- | --- | --- |
| N=32, C=192, H=W=32 | 1.95x | 1.45x | 1.30x | 1.22x |
| N=32, C=384, H=W=16 | 1.79x | 1.43x | 1.29x | 1.21x |
| N=32, C=768, H=W=8 | 1.86x | 1.64x | 1.49x | 1.39x |

## Detailed measurements

### N=32, C=192, H=W=32

| Stage | INT8 (ms) | INT4 (ms) | INT4 / INT8 speedup |
| --- | --- | --- | --- |
| Raw conv-only | 0.228 | 0.117 | 1.95x |
| Baseline fused static | 0.376 | 0.260 | 1.45x |
| MoDiff fused static total | 0.513 | 0.394 | 1.30x |
| MoDiff fused dynamic total | 0.653 | 0.534 | 1.22x |

| MoDiff breakdown | INT8 (ms) | INT4 (ms) | INT4 / INT8 speedup |
| --- | --- | --- | --- |
| Static step1 | 0.149 | 0.143 | 1.04x |
| Static conv | 0.363 | 0.250 | 1.45x |
| Dynamic step1 | 0.289 | 0.284 | 1.02x |
| Dynamic conv | 0.363 | 0.250 | 1.45x |

### N=32, C=384, H=W=16

| Stage | INT8 (ms) | INT4 (ms) | INT4 / INT8 speedup |
| --- | --- | --- | --- |
| Raw conv-only | 0.133 | 0.074 | 1.79x |
| Baseline fused static | 0.210 | 0.147 | 1.43x |
| MoDiff fused static total | 0.277 | 0.215 | 1.29x |
| MoDiff fused dynamic total | 0.351 | 0.290 | 1.21x |

| MoDiff breakdown | INT8 (ms) | INT4 (ms) | INT4 / INT8 speedup |
| --- | --- | --- | --- |
| Static step1 | 0.077 | 0.074 | 1.04x |
| Static conv | 0.200 | 0.141 | 1.42x |
| Dynamic step1 | 0.151 | 0.148 | 1.02x |
| Dynamic conv | 0.201 | 0.142 | 1.41x |

### N=32, C=768, H=W=8

| Stage | INT8 (ms) | INT4 (ms) | INT4 / INT8 speedup |
| --- | --- | --- | --- |
| Raw conv-only | 0.149 | 0.080 | 1.86x |
| Baseline fused static | 0.189 | 0.115 | 1.64x |
| MoDiff fused static total | 0.223 | 0.149 | 1.49x |
| MoDiff fused dynamic total | 0.259 | 0.186 | 1.39x |

| MoDiff breakdown | INT8 (ms) | INT4 (ms) | INT4 / INT8 speedup |
| --- | --- | --- | --- |
| Static step1 | 0.040 | 0.039 | 1.04x |
| Static conv | 0.182 | 0.110 | 1.65x |
| Dynamic step1 | 0.076 | 0.074 | 1.03x |
| Dynamic conv | 0.183 | 0.112 | 1.63x |

## Interpretation

- If INT4 were delivering a clean 2x benefit, the **raw conv-only** line would already be close to 2x.
- In practice, the speedup typically shrinks from raw conv to fused baseline to fused MoDiff because the extra work is increasingly **memory-traffic heavy** and less sensitive to the nominal tensor-core throughput ratio.
- The MoDiff `step1` path is especially important: it includes residual handling, quantization, and cache maintenance. That work is much less likely to scale as 2x when moving from INT8 to INT4.
- So if the report shows only a modest end-to-end INT4 advantage, that is consistent with a pipeline where **raw compute is not the only bottleneck**.

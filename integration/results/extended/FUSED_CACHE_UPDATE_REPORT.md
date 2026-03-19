# Fused Cache Update Overhead Report

**Date**: 2026-03-19 14:11:00
**GPU**: NVIDIA A40
**Batch Size**: 32
**Timesteps**: 200

This report isolates the cost of MoDiff cache updates while preserving the same fused launch structure as the production kernels.

Compared kernels:
- **Step1 fused**: `sub_absmax_scale + quantize (+ optional a_hat update)`
- **Conv fused**: `conv + dequant (+ optional o_hat update)`

## INT8 Cache Update Overhead

| Shape | Step1 w/ cache (ms) | Step1 no cache (ms) | a_hat update cost | Conv w/ cache (ms) | Conv no cache (ms) | o_hat update cost | Extra IO from cache update (MiB) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| INT8_32x192x32x32 | 0.288 | 0.193 | +0.094ms (+48.9%) | 0.363 | 0.319 | +0.044ms (+13.8%) | 72.0 |
| INT8_32x384x16x16 | 0.148 | 0.101 | +0.047ms (+46.7%) | 0.199 | 0.177 | +0.022ms (+12.4%) | 36.0 |
| INT8_32x768x8x8 | 0.074 | 0.051 | +0.022ms (+43.3%) | 0.180 | 0.169 | +0.012ms (+7.0%) | 18.0 |

## INT4 Cache Update Overhead

| Shape | Step1 w/ cache (ms) | Step1 no cache (ms) | a_hat update cost | Conv w/ cache (ms) | Conv no cache (ms) | o_hat update cost | Extra IO from cache update (MiB) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| INT4_32x192x32x32 | 0.280 | 0.186 | +0.094ms (+50.2%) | 0.251 | 0.208 | +0.043ms (+20.5%) | 72.0 |
| INT4_32x384x16x16 | 0.144 | 0.097 | +0.047ms (+48.4%) | 0.142 | 0.120 | +0.023ms (+18.8%) | 36.0 |
| INT4_32x768x8x8 | 0.071 | 0.051 | +0.019ms (+38.0%) | 0.112 | 0.100 | +0.012ms (+11.6%) | 18.0 |

## Memory-IO model

- **Step1 cache update (`a_hat`)**: additional float32 read + float32 write per input activation, i.e. about **8 bytes / element** beyond the no-cache fused baseline.
- **Conv cache update (`o_hat`)**: additional float32 read of the existing `o_hat_cache` per output activation, i.e. about **4 bytes / element** beyond the no-cache fused baseline.
- These are lower-bound tensor traffic estimates for the cache-update delta itself; they intentionally ignore small scalar buffers and assume the same quantized compute path in both variants.

## Takeaways

- The Step1 cache update isolates the cost of writing the temporal activation cache (`a_hat`).
- The Conv cache update isolates the cost of reading and accumulating into the temporal output cache (`o_hat`).
- Comparing these no-cache fused baselines against the production fused kernels shows how much of MoDiff hot-path time is spent on cache maintenance rather than quantized compute.
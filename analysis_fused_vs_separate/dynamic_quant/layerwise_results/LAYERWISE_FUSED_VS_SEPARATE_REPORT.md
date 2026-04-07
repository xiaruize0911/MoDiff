# Layerwise fused-vs-separate MoDiff benchmark (dynamic quantization)

**Date**: 2026-04-07 03:10:34
**GPU**: NVIDIA A40
**Config**: `/workspace/MoDiff/models/ldm/lsun_churches256/config.yaml`
**Batch Size**: 32
**Quant Mode**: dynamic

This benchmark isolates one **modulated MoDiff update** per unique Conv2d shape observed in the LSUN-Churches LDM UNet.

Timing notes:
- Each value is the synchronized per-call average over 3 timed repeats × 20 iterations, after 5 warm-up iterations.
- `a_hat` and `o_hat` buffers are reset to a fixed zero state before every timed call, outside the timed region.
- The layerwise benchmark isolates the MoDiff hot path where fusion matters most: residual update + quantization + conv-side dequant/accumulate.
- All unique Conv2d shapes are enumerated, but only shapes that match the repository's quantized-conversion rules are benchmarked; excluded shapes are still reported separately.
- First-step warmup behavior is intentionally left to the whole-model benchmark.
- Activation quantization mode: **dynamic** (per-call dynamic scale recomputed from the current activation tensor).

## Weighted aggregate over one UNet forward

| Precision | Fused Step1 (ms) | Separate Step1 (ms) | Step1 speedup | Fused Conv (ms) | Separate Conv (ms) | Conv speedup | Fused Total (ms) | Separate Total (ms) | Fusion speedup | Benchmarked calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| INT8 | 8.665 | 30.834 | 3.56x | 12.012 | 14.215 | 1.18x | 20.355 | 43.833 | 2.15x | 70 |
| INT4 | 8.501 | 27.737 | 3.26x | 8.007 | 10.375 | 1.30x | 16.347 | 38.289 | 2.34x | 70 |

## INT8 per-shape results

| Shape | Total Count | Supported Count | Unsupported Count | Fused Step1 (ms) | Separate Step1 (ms) | Step1 speedup | Fused Conv (ms) | Separate Conv (ms) | Conv speedup | Fused Total (ms) | Separate Total (ms) | Fusion speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32x32 \| 192->192 \| k3x3 s1x1 p1x1 | 7 | 7 | 0 | 0.2864 | 0.9074 | 3.17x | 0.3625 | 0.4561 | 1.26x | 0.6482 | 1.3627 | 2.10x |
| 16x16 \| 384->384 \| k3x3 s1x1 p1x1 | 8 | 8 | 0 | 0.1476 | 0.4746 | 3.22x | 0.2019 | 0.2506 | 1.24x | 0.3461 | 0.7236 | 2.09x |
| 32x32 \| 384->384 \| k3x3 s1x1 p1x1 | 2 | 2 | 0 | 0.5686 | 1.7682 | 3.11x | 0.6972 | 0.8804 | 1.26x | 1.2640 | 2.6476 | 2.09x |
| 32x32 \| 384->192 \| k3x3 s1x1 p1x1 | 2 | 2 | 0 | 0.5691 | 1.7678 | 3.11x | 0.4274 | 0.5206 | 1.22x | 0.9942 | 2.2871 | 2.30x |
| 32x32 \| 576->192 \| k3x3 s1x1 p1x1 | 1 | 1 | 0 | 0.8504 | 2.6291 | 3.09x | 0.5841 | 0.6776 | 1.16x | 1.4324 | 3.3060 | 2.31x |
| 2x2 \| 768->768 \| k3x3 s1x1 p1x1 | 13 | 13 | 0 | 0.0217 | 0.1676 | 7.74x | 0.0777 | 0.0801 | 1.03x | 0.0860 | 0.2234 | 2.60x |
| 16x16 \| 768->384 \| k3x3 s1x1 p1x1 | 2 | 2 | 0 | 0.2863 | 0.9074 | 3.17x | 0.2982 | 0.3463 | 1.16x | 0.5813 | 1.2525 | 2.15x |
| 8x8 \| 384->384 \| k3x3 s1x1 p1x1 | 9 | 9 | 0 | 0.0343 | 0.1720 | 5.02x | 0.0572 | 0.0651 | 1.14x | 0.0906 | 0.2176 | 2.40x |
| 4x4 \| 768->768 \| k3x3 s1x1 p1x1 | 8 | 8 | 0 | 0.0215 | 0.1744 | 8.11x | 0.0801 | 0.0871 | 1.09x | 0.0958 | 0.2331 | 2.43x |
| 16x16 \| 576->384 \| k3x3 s1x1 p1x1 | 1 | 1 | 0 | 0.2151 | 0.6895 | 3.21x | 0.2671 | 0.3164 | 1.18x | 0.4811 | 1.0043 | 2.09x |
| 8x8 \| 768->768 \| k3x3 s1x1 p1x1 | 2 | 2 | 0 | 0.0722 | 0.2459 | 3.41x | 0.1836 | 0.2081 | 1.13x | 0.2558 | 0.4538 | 1.77x |
| 16x16 \| 192->192 \| k3x3 s1x1 p1x1 | 2 | 2 | 0 | 0.0719 | 0.2449 | 3.40x | 0.1001 | 0.1255 | 1.25x | 0.1716 | 0.3698 | 2.16x |
| 8x8 \| 768->384 \| k3x3 s1x1 p1x1 | 2 | 2 | 0 | 0.0721 | 0.2454 | 3.40x | 0.0876 | 0.0935 | 1.07x | 0.1608 | 0.3380 | 2.10x |
| 2x2 \| 1536->768 \| k3x3 s1x1 p1x1 | 3 | 3 | 0 | 0.0178 | 0.1404 | 7.88x | 0.1419 | 0.1453 | 1.02x | 0.1528 | 0.1948 | 1.27x |
| 8x8 \| 1152->384 \| k3x3 s1x1 p1x1 | 1 | 1 | 0 | 0.1112 | 0.3609 | 3.25x | 0.1198 | 0.1265 | 1.06x | 0.2290 | 0.4855 | 2.12x |
| 4x4 \| 1536->768 \| k3x3 s1x1 p1x1 | 2 | 2 | 0 | 0.0308 | 0.1426 | 4.63x | 0.1443 | 0.1503 | 1.04x | 0.1743 | 0.2380 | 1.37x |
| 16x16 \| 192->384 \| k3x3 s1x1 p1x1 | 1 | 1 | 0 | 0.0719 | 0.2459 | 3.42x | 0.1702 | 0.2189 | 1.29x | 0.2420 | 0.4648 | 1.92x |
| 4x4 \| 384->384 \| k3x3 s1x1 p1x1 | 2 | 2 | 0 | 0.0173 | 0.1407 | 8.14x | 0.0454 | 0.0481 | 1.06x | 0.0546 | 0.1878 | 3.44x |
| 4x4 \| 1152->768 \| k3x3 s1x1 p1x1 | 1 | 1 | 0 | 0.0210 | 0.1447 | 6.88x | 0.1120 | 0.1184 | 1.06x | 0.1300 | 0.1936 | 1.49x |
| 4x4 \| 384->768 \| k3x3 s1x1 p1x1 | 1 | 1 | 0 | 0.0178 | 0.1447 | 8.12x | 0.0464 | 0.0543 | 1.17x | 0.0613 | 0.1893 | 3.09x |
| 2x2 \| 1536->768 \| k1x1 s1x1 p0x0 | 3 | 0 | 3 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | skip_connection, pointwise_1x1 |
| 32x32 \| 384->192 \| k1x1 s1x1 p0x0 | 2 | 0 | 2 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | skip_connection, pointwise_1x1 |
| 16x16 \| 768->384 \| k1x1 s1x1 p0x0 | 2 | 0 | 2 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | skip_connection, pointwise_1x1 |
| 8x8 \| 768->384 \| k1x1 s1x1 p0x0 | 2 | 0 | 2 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | skip_connection, pointwise_1x1 |
| 4x4 \| 1536->768 \| k1x1 s1x1 p0x0 | 2 | 0 | 2 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | skip_connection, pointwise_1x1 |
| 32x32 \| 576->192 \| k1x1 s1x1 p0x0 | 1 | 0 | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | skip_connection, pointwise_1x1 |
| 32x32 \| 192->4 \| k3x3 s1x1 p1x1 | 1 | 0 | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | final_out |
| 32x32 \| 4->192 \| k3x3 s1x1 p1x1 | 1 | 0 | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | in_channels<32 |
| 16x16 \| 576->384 \| k1x1 s1x1 p0x0 | 1 | 0 | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | skip_connection, pointwise_1x1 |
| 16x16 \| 192->384 \| k1x1 s1x1 p0x0 | 1 | 0 | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | skip_connection, pointwise_1x1 |
| 8x8 \| 1152->384 \| k1x1 s1x1 p0x0 | 1 | 0 | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | skip_connection, pointwise_1x1 |
| 4x4 \| 1152->768 \| k1x1 s1x1 p0x0 | 1 | 0 | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | skip_connection, pointwise_1x1 |
| 4x4 \| 384->768 \| k1x1 s1x1 p0x0 | 1 | 0 | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | skip_connection, pointwise_1x1 |

## INT4 per-shape results

| Shape | Total Count | Supported Count | Unsupported Count | Fused Step1 (ms) | Separate Step1 (ms) | Step1 speedup | Fused Conv (ms) | Separate Conv (ms) | Conv speedup | Fused Total (ms) | Separate Total (ms) | Fusion speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32x32 \| 192->192 \| k3x3 s1x1 p1x1 | 7 | 7 | 0 | 0.2805 | 0.8340 | 2.97x | 0.2511 | 0.3456 | 1.38x | 0.5304 | 1.1810 | 2.23x |
| 16x16 \| 384->384 \| k3x3 s1x1 p1x1 | 8 | 8 | 0 | 0.1444 | 0.4321 | 2.99x | 0.1437 | 0.1919 | 1.34x | 0.2851 | 0.6258 | 2.20x |
| 32x32 \| 384->384 \| k3x3 s1x1 p1x1 | 2 | 2 | 0 | 0.5568 | 1.6292 | 2.93x | 0.5182 | 0.7019 | 1.35x | 1.0727 | 2.3337 | 2.18x |
| 32x32 \| 384->192 \| k3x3 s1x1 p1x1 | 2 | 2 | 0 | 0.5564 | 1.6291 | 2.93x | 0.2991 | 0.3917 | 1.31x | 0.8527 | 2.0223 | 2.37x |
| 32x32 \| 576->192 \| k3x3 s1x1 p1x1 | 1 | 1 | 0 | 0.8328 | 2.4257 | 2.91x | 0.3789 | 0.4737 | 1.25x | 1.2105 | 2.8992 | 2.40x |
| 2x2 \| 768->768 \| k3x3 s1x1 p1x1 | 13 | 13 | 0 | 0.0222 | 0.1368 | 6.17x | 0.0441 | 0.0539 | 1.22x | 0.0634 | 0.1965 | 3.10x |
| 16x16 \| 768->384 \| k3x3 s1x1 p1x1 | 2 | 2 | 0 | 0.2804 | 0.8335 | 2.97x | 0.1889 | 0.2378 | 1.26x | 0.4669 | 1.0724 | 2.30x |
| 8x8 \| 384->384 \| k3x3 s1x1 p1x1 | 9 | 9 | 0 | 0.0320 | 0.1440 | 4.49x | 0.0409 | 0.0529 | 1.29x | 0.0726 | 0.2043 | 2.82x |
| 4x4 \| 768->768 \| k3x3 s1x1 p1x1 | 8 | 8 | 0 | 0.0229 | 0.1475 | 6.45x | 0.0459 | 0.0551 | 1.20x | 0.0647 | 0.2158 | 3.33x |
| 16x16 \| 576->384 \| k3x3 s1x1 p1x1 | 1 | 1 | 0 | 0.2109 | 0.6334 | 3.00x | 0.1761 | 0.2243 | 1.27x | 0.3838 | 0.8590 | 2.24x |
| 8x8 \| 768->768 \| k3x3 s1x1 p1x1 | 2 | 2 | 0 | 0.0700 | 0.2195 | 3.14x | 0.1136 | 0.1393 | 1.23x | 0.1837 | 0.3587 | 1.95x |
| 16x16 \| 192->192 \| k3x3 s1x1 p1x1 | 2 | 2 | 0 | 0.0702 | 0.2193 | 3.12x | 0.0691 | 0.0944 | 1.37x | 0.1393 | 0.3149 | 2.26x |
| 8x8 \| 768->384 \| k3x3 s1x1 p1x1 | 2 | 2 | 0 | 0.0702 | 0.2196 | 3.13x | 0.0564 | 0.0627 | 1.11x | 0.1269 | 0.2875 | 2.26x |
| 2x2 \| 1536->768 \| k3x3 s1x1 p1x1 | 3 | 3 | 0 | 0.0182 | 0.1192 | 6.55x | 0.0766 | 0.0792 | 1.03x | 0.0880 | 0.1780 | 2.02x |
| 8x8 \| 1152->384 \| k3x3 s1x1 p1x1 | 1 | 1 | 0 | 0.1088 | 0.3278 | 3.01x | 0.0718 | 0.0777 | 1.08x | 0.1789 | 0.4098 | 2.29x |
| 4x4 \| 1536->768 \| k3x3 s1x1 p1x1 | 2 | 2 | 0 | 0.0306 | 0.1200 | 3.92x | 0.0788 | 0.0861 | 1.09x | 0.1089 | 0.1717 | 1.58x |
| 16x16 \| 192->384 \| k3x3 s1x1 p1x1 | 1 | 1 | 0 | 0.0701 | 0.2198 | 3.13x | 0.1244 | 0.1738 | 1.40x | 0.1954 | 0.3948 | 2.02x |
| 4x4 \| 384->384 \| k3x3 s1x1 p1x1 | 2 | 2 | 0 | 0.0182 | 0.1200 | 6.61x | 0.0332 | 0.0385 | 1.16x | 0.0461 | 0.1647 | 3.57x |
| 4x4 \| 1152->768 \| k3x3 s1x1 p1x1 | 1 | 1 | 0 | 0.0208 | 0.1214 | 5.82x | 0.0625 | 0.0692 | 1.11x | 0.0821 | 0.1768 | 2.15x |
| 4x4 \| 384->768 \| k3x3 s1x1 p1x1 | 1 | 1 | 0 | 0.0187 | 0.1203 | 6.43x | 0.0335 | 0.0416 | 1.24x | 0.0498 | 0.1708 | 3.43x |
| 2x2 \| 1536->768 \| k1x1 s1x1 p0x0 | 3 | 0 | 3 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | skip_connection, pointwise_1x1 |
| 32x32 \| 384->192 \| k1x1 s1x1 p0x0 | 2 | 0 | 2 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | skip_connection, pointwise_1x1 |
| 16x16 \| 768->384 \| k1x1 s1x1 p0x0 | 2 | 0 | 2 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | skip_connection, pointwise_1x1 |
| 8x8 \| 768->384 \| k1x1 s1x1 p0x0 | 2 | 0 | 2 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | skip_connection, pointwise_1x1 |
| 4x4 \| 1536->768 \| k1x1 s1x1 p0x0 | 2 | 0 | 2 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | skip_connection, pointwise_1x1 |
| 32x32 \| 576->192 \| k1x1 s1x1 p0x0 | 1 | 0 | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | skip_connection, pointwise_1x1 |
| 32x32 \| 192->4 \| k3x3 s1x1 p1x1 | 1 | 0 | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | final_out |
| 32x32 \| 4->192 \| k3x3 s1x1 p1x1 | 1 | 0 | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | in_channels<32 |
| 16x16 \| 576->384 \| k1x1 s1x1 p0x0 | 1 | 0 | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | skip_connection, pointwise_1x1 |
| 16x16 \| 192->384 \| k1x1 s1x1 p0x0 | 1 | 0 | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | skip_connection, pointwise_1x1 |
| 8x8 \| 1152->384 \| k1x1 s1x1 p0x0 | 1 | 0 | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | skip_connection, pointwise_1x1 |
| 4x4 \| 1152->768 \| k1x1 s1x1 p0x0 | 1 | 0 | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | skip_connection, pointwise_1x1 |
| 4x4 \| 384->768 \| k1x1 s1x1 p0x0 | 1 | 0 | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | skip_connection, pointwise_1x1 |

## Shapes with runtime issues or exclusions

| Shape | INT8 status | INT4 status |
| --- | --- | --- |
| 2x2 \| 1536->768 \| k1x1 s1x1 p0x0 | skip_connection, pointwise_1x1 | skip_connection, pointwise_1x1 |
| 32x32 \| 384->192 \| k1x1 s1x1 p0x0 | skip_connection, pointwise_1x1 | skip_connection, pointwise_1x1 |
| 16x16 \| 768->384 \| k1x1 s1x1 p0x0 | skip_connection, pointwise_1x1 | skip_connection, pointwise_1x1 |
| 8x8 \| 768->384 \| k1x1 s1x1 p0x0 | skip_connection, pointwise_1x1 | skip_connection, pointwise_1x1 |
| 4x4 \| 1536->768 \| k1x1 s1x1 p0x0 | skip_connection, pointwise_1x1 | skip_connection, pointwise_1x1 |
| 32x32 \| 576->192 \| k1x1 s1x1 p0x0 | skip_connection, pointwise_1x1 | skip_connection, pointwise_1x1 |
| 32x32 \| 192->4 \| k3x3 s1x1 p1x1 | final_out | final_out |
| 32x32 \| 4->192 \| k3x3 s1x1 p1x1 | in_channels<32 | in_channels<32 |
| 16x16 \| 576->384 \| k1x1 s1x1 p0x0 | skip_connection, pointwise_1x1 | skip_connection, pointwise_1x1 |
| 16x16 \| 192->384 \| k1x1 s1x1 p0x0 | skip_connection, pointwise_1x1 | skip_connection, pointwise_1x1 |
| 8x8 \| 1152->384 \| k1x1 s1x1 p0x0 | skip_connection, pointwise_1x1 | skip_connection, pointwise_1x1 |
| 4x4 \| 1152->768 \| k1x1 s1x1 p0x0 | skip_connection, pointwise_1x1 | skip_connection, pointwise_1x1 |
| 4x4 \| 384->768 \| k1x1 s1x1 p0x0 | skip_connection, pointwise_1x1 | skip_connection, pointwise_1x1 |

# Layerwise fused-vs-separate MoDiff benchmark (static quantization)

**Date**: 2026-04-07 03:14:42
**GPU**: NVIDIA A40
**Config**: `/workspace/MoDiff/models/ldm/lsun_churches256/config.yaml`
**Batch Size**: 32
**Quant Mode**: static

This benchmark isolates one **modulated MoDiff update** per unique Conv2d shape observed in the LSUN-Churches LDM UNet.

Timing notes:
- Each value is the synchronized per-call average over 3 timed repeats × 20 iterations, after 5 warm-up iterations.
- `a_hat` and `o_hat` buffers are reset to a fixed zero state before every timed call, outside the timed region.
- The layerwise benchmark isolates the MoDiff hot path where fusion matters most: residual update + quantization + conv-side dequant/accumulate.
- All unique Conv2d shapes are enumerated, but only shapes that match the repository's quantized-conversion rules are benchmarked; excluded shapes are still reported separately.
- First-step warmup behavior is intentionally left to the whole-model benchmark.
- Activation quantization mode: **static** (per-shape static scale calibrated once from the synthetic activation tensor).

## Weighted aggregate over one UNet forward

| Precision | Fused Step1 (ms) | Separate Step1 (ms) | Step1 speedup | Fused Conv (ms) | Separate Conv (ms) | Conv speedup | Fused Total (ms) | Separate Total (ms) | Fusion speedup | Benchmarked calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| INT8 | 4.894 | 23.732 | 4.85x | 12.029 | 14.239 | 1.18x | 16.352 | 36.632 | 2.24x | 70 |
| INT4 | 4.724 | 21.456 | 4.54x | 8.016 | 10.372 | 1.29x | 12.451 | 31.061 | 2.49x | 70 |

## INT8 per-shape results

| Shape | Total Count | Supported Count | Unsupported Count | Fused Step1 (ms) | Separate Step1 (ms) | Step1 speedup | Fused Conv (ms) | Separate Conv (ms) | Conv speedup | Fused Total (ms) | Separate Total (ms) | Fusion speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32x32 \| 192->192 \| k3x3 s1x1 p1x1 | 7 | 7 | 0 | 0.1497 | 0.7521 | 5.03x | 0.3633 | 0.4564 | 1.26x | 0.5119 | 1.2063 | 2.36x |
| 16x16 \| 384->384 \| k3x3 s1x1 p1x1 | 8 | 8 | 0 | 0.0776 | 0.3859 | 4.97x | 0.2024 | 0.2512 | 1.24x | 0.2767 | 0.6353 | 2.30x |
| 32x32 \| 384->384 \| k3x3 s1x1 p1x1 | 2 | 2 | 0 | 0.2945 | 1.4822 | 5.03x | 0.6967 | 0.8839 | 1.27x | 0.9938 | 2.3624 | 2.38x |
| 32x32 \| 384->192 \| k3x3 s1x1 p1x1 | 2 | 2 | 0 | 0.2946 | 1.4819 | 5.03x | 0.4270 | 0.5206 | 1.22x | 0.7186 | 2.0005 | 2.78x |
| 32x32 \| 576->192 \| k3x3 s1x1 p1x1 | 1 | 1 | 0 | 0.4397 | 2.2118 | 5.03x | 0.5846 | 0.6779 | 1.16x | 1.0223 | 2.8874 | 2.82x |
| 16x16 \| 768->384 \| k3x3 s1x1 p1x1 | 2 | 2 | 0 | 0.1495 | 0.7520 | 5.03x | 0.2994 | 0.3479 | 1.16x | 0.4467 | 1.0973 | 2.46x |
| 2x2 \| 768->768 \| k3x3 s1x1 p1x1 | 13 | 13 | 0 | 0.0223 | 0.0922 | 4.13x | 0.0777 | 0.0801 | 1.03x | 0.0815 | 0.1385 | 1.70x |
| 8x8 \| 384->384 \| k3x3 s1x1 p1x1 | 9 | 9 | 0 | 0.0231 | 0.0921 | 3.98x | 0.0575 | 0.0653 | 1.14x | 0.0764 | 0.1313 | 1.72x |
| 4x4 \| 768->768 \| k3x3 s1x1 p1x1 | 8 | 8 | 0 | 0.0216 | 0.0921 | 4.27x | 0.0800 | 0.0871 | 1.09x | 0.0862 | 0.1397 | 1.62x |
| 16x16 \| 576->384 \| k3x3 s1x1 p1x1 | 1 | 1 | 0 | 0.1134 | 0.5681 | 5.01x | 0.2677 | 0.3174 | 1.19x | 0.3786 | 0.8832 | 2.33x |
| 8x8 \| 768->768 \| k3x3 s1x1 p1x1 | 2 | 2 | 0 | 0.0404 | 0.1947 | 4.82x | 0.1840 | 0.2092 | 1.14x | 0.2231 | 0.4023 | 1.80x |
| 16x16 \| 192->192 \| k3x3 s1x1 p1x1 | 2 | 2 | 0 | 0.0403 | 0.1943 | 4.82x | 0.1000 | 0.1255 | 1.25x | 0.1375 | 0.3184 | 2.32x |
| 8x8 \| 768->384 \| k3x3 s1x1 p1x1 | 2 | 2 | 0 | 0.0404 | 0.1947 | 4.82x | 0.0878 | 0.0938 | 1.07x | 0.1266 | 0.2948 | 2.33x |
| 2x2 \| 1536->768 \| k3x3 s1x1 p1x1 | 3 | 3 | 0 | 0.0219 | 0.0911 | 4.17x | 0.1421 | 0.1455 | 1.02x | 0.1460 | 0.1709 | 1.17x |
| 4x4 \| 1536->768 \| k3x3 s1x1 p1x1 | 2 | 2 | 0 | 0.0203 | 0.0819 | 4.03x | 0.1445 | 0.1506 | 1.04x | 0.1594 | 0.2115 | 1.33x |
| 8x8 \| 1152->384 \| k3x3 s1x1 p1x1 | 1 | 1 | 0 | 0.0594 | 0.2931 | 4.94x | 0.1198 | 0.1267 | 1.06x | 0.1765 | 0.4173 | 2.36x |
| 16x16 \| 192->384 \| k3x3 s1x1 p1x1 | 1 | 1 | 0 | 0.0404 | 0.1948 | 4.82x | 0.1710 | 0.2190 | 1.28x | 0.2086 | 0.4127 | 1.98x |
| 4x4 \| 384->384 \| k3x3 s1x1 p1x1 | 2 | 2 | 0 | 0.0189 | 0.0773 | 4.09x | 0.0453 | 0.0482 | 1.06x | 0.0536 | 0.1184 | 2.21x |
| 4x4 \| 1152->768 \| k3x3 s1x1 p1x1 | 1 | 1 | 0 | 0.0190 | 0.0790 | 4.17x | 0.1121 | 0.1184 | 1.06x | 0.1206 | 0.1638 | 1.36x |
| 4x4 \| 384->768 \| k3x3 s1x1 p1x1 | 1 | 1 | 0 | 0.0191 | 0.0786 | 4.11x | 0.0465 | 0.0539 | 1.16x | 0.0537 | 0.1171 | 2.18x |
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
| 32x32 \| 192->192 \| k3x3 s1x1 p1x1 | 7 | 7 | 0 | 0.1440 | 0.6829 | 4.74x | 0.2512 | 0.3451 | 1.37x | 0.3933 | 1.0246 | 2.61x |
| 16x16 \| 384->384 \| k3x3 s1x1 p1x1 | 8 | 8 | 0 | 0.0746 | 0.3490 | 4.68x | 0.1436 | 0.1914 | 1.33x | 0.2146 | 0.5376 | 2.51x |
| 32x32 \| 384->384 \| k3x3 s1x1 p1x1 | 2 | 2 | 0 | 0.2829 | 1.3478 | 4.76x | 0.5175 | 0.7016 | 1.36x | 0.7992 | 2.0470 | 2.56x |
| 32x32 \| 384->192 \| k3x3 s1x1 p1x1 | 2 | 2 | 0 | 0.2830 | 1.3474 | 4.76x | 0.2990 | 0.3934 | 1.32x | 0.5799 | 1.7396 | 3.00x |
| 32x32 \| 576->192 \| k3x3 s1x1 p1x1 | 1 | 1 | 0 | 0.4227 | 2.0112 | 4.76x | 0.3795 | 0.4724 | 1.24x | 0.7995 | 2.4828 | 3.11x |
| 16x16 \| 768->384 \| k3x3 s1x1 p1x1 | 2 | 2 | 0 | 0.1440 | 0.6826 | 4.74x | 0.1898 | 0.2380 | 1.25x | 0.3305 | 0.9153 | 2.77x |
| 2x2 \| 768->768 \| k3x3 s1x1 p1x1 | 13 | 13 | 0 | 0.0215 | 0.0811 | 3.77x | 0.0442 | 0.0541 | 1.22x | 0.0632 | 0.1192 | 1.88x |
| 8x8 \| 384->384 \| k3x3 s1x1 p1x1 | 9 | 9 | 0 | 0.0220 | 0.0827 | 3.77x | 0.0407 | 0.0527 | 1.29x | 0.0587 | 0.1144 | 1.95x |
| 4x4 \| 768->768 \| k3x3 s1x1 p1x1 | 8 | 8 | 0 | 0.0220 | 0.0806 | 3.67x | 0.0460 | 0.0542 | 1.18x | 0.0621 | 0.1254 | 2.02x |
| 16x16 \| 576->384 \| k3x3 s1x1 p1x1 | 1 | 1 | 0 | 0.1089 | 0.5152 | 4.73x | 0.1769 | 0.2258 | 1.28x | 0.2826 | 0.7386 | 2.61x |
| 8x8 \| 768->768 \| k3x3 s1x1 p1x1 | 2 | 2 | 0 | 0.0432 | 0.1744 | 4.04x | 0.1146 | 0.1388 | 1.21x | 0.1501 | 0.3101 | 2.07x |
| 16x16 \| 192->192 \| k3x3 s1x1 p1x1 | 2 | 2 | 0 | 0.0389 | 0.1743 | 4.48x | 0.0689 | 0.0953 | 1.38x | 0.1059 | 0.2670 | 2.52x |
| 8x8 \| 768->384 \| k3x3 s1x1 p1x1 | 2 | 2 | 0 | 0.0388 | 0.1749 | 4.50x | 0.0565 | 0.0640 | 1.13x | 0.0940 | 0.2357 | 2.51x |
| 2x2 \| 1536->768 \| k3x3 s1x1 p1x1 | 3 | 3 | 0 | 0.0220 | 0.0820 | 3.73x | 0.0768 | 0.0799 | 1.04x | 0.0817 | 0.1308 | 1.60x |
| 4x4 \| 1536->768 \| k3x3 s1x1 p1x1 | 2 | 2 | 0 | 0.0188 | 0.0730 | 3.88x | 0.0788 | 0.0852 | 1.08x | 0.0930 | 0.1388 | 1.49x |
| 8x8 \| 1152->384 \| k3x3 s1x1 p1x1 | 1 | 1 | 0 | 0.0574 | 0.2640 | 4.60x | 0.0725 | 0.0793 | 1.09x | 0.1271 | 0.3404 | 2.68x |
| 16x16 \| 192->384 \| k3x3 s1x1 p1x1 | 1 | 1 | 0 | 0.0388 | 0.1747 | 4.50x | 0.1250 | 0.1737 | 1.39x | 0.1616 | 0.3461 | 2.14x |
| 4x4 \| 384->384 \| k3x3 s1x1 p1x1 | 2 | 2 | 0 | 0.0182 | 0.0697 | 3.82x | 0.0344 | 0.0396 | 1.15x | 0.0470 | 0.0964 | 2.05x |
| 4x4 \| 1152->768 \| k3x3 s1x1 p1x1 | 1 | 1 | 0 | 0.0182 | 0.0715 | 3.92x | 0.0625 | 0.0682 | 1.09x | 0.0722 | 0.1068 | 1.48x |
| 4x4 \| 384->768 \| k3x3 s1x1 p1x1 | 1 | 1 | 0 | 0.0187 | 0.0699 | 3.75x | 0.0331 | 0.0420 | 1.27x | 0.0474 | 0.0990 | 2.09x |
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

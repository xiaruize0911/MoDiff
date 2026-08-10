# Raw measurement output

Verbatim stdout from the runs FINDINGS.md summarises. `logs/` is gitignored repo-wide
(`.gitignore`: `*.log`), so the raw tables are kept here instead of being lost to it.

## `ab_paired_k1`

```

K=1, batch 128, 200 steps, 4 paired repeats, NVIDIA A40

| arm | ms/step (median) | CV% | updown fused/step |
|---|---:|---:|---:|
| ON  (fix)    | 105.32 | 0.364 | 8.00 |
| OFF (before) | 106.56 | 0.250 | 0.00 |

paired deltas (OFF - ON), ms/step: +1.42, +1.32, +1.17, +1.15
median +1.24 ms/step recovered by the fix
stdev of the paired delta: 0.128 ms
```

## `ab_paired_k4`

```
Converting UNet linear layers to INT8 (37 linear layers)...
Loading static calibration from integration/calibration/int8_calibration.pt
✓ Loaded 70 INT8 conv layer scales (static quantization enabled)
✓ Loaded 37 INT8 linear layer scales
✓ Converted 21 AttentionBlocks to QUANTIZED standard attention (W8A8 QKᵀ/AV, STATIC)
✓ Quantized 42 Linear layers to W8A8 (modiff=True)
✓ Calibrated 42 W8A8 linear activation scales (static)
✓ Wired SiLU fusion for 62 quantized conv layers
✓ Wired Upsample->quantize fusion for 16 Upsample layers

K=4, batch 128, 200 steps, 4 paired repeats, NVIDIA A40

| arm | ms/step (median) | CV% | updown fused/step |
|---|---:|---:|---:|
| ON  (fix)    | 99.60 | 0.357 | 8.00 |
| OFF (before) | 99.99 | 0.165 | 6.00 |

paired deltas (OFF - ON), ms/step: +0.70, +0.50, +0.28, +0.30
median +0.40 ms/step recovered by the fix
stdev of the paired delta: 0.197 ms
```

## `bench_kernel_shapes`

```
batch 128, NVIDIA A40, int8 store, identity SmoothQuant

| shape | dir | unfused ms | fused dyn ms | fused static ms | dyn speedup |
|---|---|---:|---:|---:|---:|
| 192x32x32 | -1 | 0.852 | 0.799 | 0.509 | 1.07x |
| 384x16x16 | -1 | 0.478 | 0.336 | 0.195 | 1.42x |
| 384x8x8 | -1 | 0.216 | 0.091 | 0.048 | 2.39x |
| 768x4x4 | -1 | 0.078 | 0.059 | 0.024 | 1.34x |
| 768x2x2 | +1 | 0.065 | 0.044 | 0.019 | 1.48x |
| 768x4x4 | +1 | 0.287 | 0.179 | 0.136 | 1.61x |
| 384x8x8 | +1 | 0.625 | 0.460 | 0.390 | 1.36x |
| 384x16x16 | +1 | 2.057 | 2.244 | 1.963 | 0.92x |
| **all 8** | | **4.660** | **4.211** | **3.284** | **1.11x** |

per step at K=1: +0.448 ms recovered (4.660 -> 4.211)
the reduction launch costs 0.927 ms; a K>1 reuse step pays 3.284 ms
```

## `bench_unpack_int4_widen`

```
A. equivalence: pack+widen vs int8 quantize at code_ceiling=7

| shape | scale regime | max|code| | differing elements |
|---|---|---:|---:|
| 192x32x32 | exact 7/absmax | 7 | 0 |
| 192x32x32 | 4x too fine (stale/clip) | 7 | 0 |
| 384x32x32 | exact 7/absmax | 7 | 0 |
| 384x32x32 | 4x too fine (stale/clip) | 7 | 0 |
| 384x16x16 | exact 7/absmax | 7 | 0 |
| 384x16x16 | 4x too fine (stale/clip) | 7 | 0 |
| 768x16x16 | exact 7/absmax | 7 | 0 |
| 768x16x16 | 4x too fine (stale/clip) | 7 | 0 |
| 384x8x8 | exact 7/absmax | 7 | 0 |
| 384x8x8 | 4x too fine (stale/clip) | 7 | 0 |

bit-identical on every case

B. cost of the widening pass, batch 128

| shape | layers | per call µs | per step ms |
|---|---:|---:|---:|
| 192x32x32 | 14 | 121.4 | 1.700 |
| 384x32x32 | 4 | 239.7 | 0.959 |
| 384x16x16 | 16 | 62.2 | 0.996 |
| 768x16x16 | 4 | 121.3 | 0.485 |
| 384x8x8 | 12 | 16.0 | 0.192 |
| 768x8x8 | 6 | 32.5 | 0.195 |
| 768x4x4 | 10 | 9.1 | 0.091 |
| 1152x8x8 | 2 | 47.3 | 0.095 |
| 1536x4x4 | 2 | 16.1 | 0.032 |
| **all 70** | | | **4.744** |

added per step: 4.74 ms on top of the ~105 ms/step W8A4 currently runs at (4.5%)

The two routes produce the same codes, so this is a pure re-encoding of the same rule: 4-bitness moves from a clamp parameter into the storage format. The ms above is what that costs.
```

## `test_kernel_correctness`

```
A/B/C/D: static bit-identity, published scale, codes, a_hat

| shape | dir | quant | A static | B scale rel | C max code err | D a_hat max |
|---|---|---|---|---:|---:|---:|
| 192x32x32 | -1 | int4 | ok | 3.11e-08 | 0 | 0.000e+00 |
| 192x32x32 | -1 | int8 | ok | 1.84e-08 | 0 | 9.537e-07 |
| 384x16x16 | -1 | int4 | ok | 2.94e-08 | 0 | 0.000e+00 |
| 384x16x16 | -1 | int8 | ok | 5.01e-08 | 1 | 5.518e-02 |
| 384x8x8 | -1 | int4 | ok | 1.14e-07 | 0 | 0.000e+00 |
| 384x8x8 | -1 | int8 | ok | 7.54e-08 | 0 | 4.768e-07 |
| 768x4x4 | -1 | int4 | ok | 3.40e-08 | 0 | 0.000e+00 |
| 768x4x4 | -1 | int8 | ok | 3.68e-09 | 0 | 1.192e-07 |
| 768x2x2 | +1 | int4 | ok | 1.28e-07 | 0 | 0.000e+00 |
| 768x2x2 | +1 | int8 | ok | 5.64e-08 | 0 | 7.629e-06 |
| 768x4x4 | +1 | int4 | ok | 2.94e-08 | 0 | 0.000e+00 |
| 768x4x4 | +1 | int8 | ok | 3.09e-08 | 1 | 1.338e-01 |
| 384x8x8 | +1 | int4 | ok | 1.07e-07 | 0 | 0.000e+00 |
| 384x8x8 | +1 | int8 | ok | 8.18e-08 | 0 | 3.815e-06 |
| 384x16x16 | +1 | int4 | ok | 2.62e-08 | 1 | 2.249e+00 |
| 384x16x16 | +1 | int8 | ok | 9.29e-09 | 1 | 1.240e-01 |

E: a4 on an int8 store (this IS W8A4)
  int4 store, a4=False: max|code| = 7, want 7 -> ok
  int4 store, a4=True: max|code| = 7, want 7 -> ok
  int8 store, a4=False: max|code| = 127, want 127 -> ok
  int8 store, a4=True: max|code| = 7, want 7 -> ok

F: report_next
  int4: codes on GIVEN scale ok, published 1.1082 vs 1.1082 ok, inv ok
  int8: codes on GIVEN scale ok, published 20.1060 vs 20.1060 ok, inv ok

PASS
```

## `test_wiring_8of8`

```
| K | UNet forwards | fused calls | fused/step | want |
| 1 | 7 | 56 | 8.00 | 8 |
| 4 | 7 | 56 | 8.00 | 8 |
PASS
```


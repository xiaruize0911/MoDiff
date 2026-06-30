# Nsight CUDA I/O Usage Analysis

Profiles: `integration/results/nsys_memory_repro_20260624_fp16_baseline/profiles/fp16_s50_b168.sqlite` ...
DDIM steps: `50`

## Total CUDA I/O

| Mode | Total MiB | Total Count | Total ms | D2D MiB | D2D Count | Extra MiB vs FP16 | Extra D2D MiB vs FP16 |
|---|---:|---:|---:|---:|---:|---:|---:|
| fp16 | 2,807.7 | 2104 | 285.3 | 111.7 | 210 | 0.0 | 0.0 |
| int8 | 11,312.5 | 7640 | 315.2 | 8,615.8 | 1264 | 8,504.8 | 8,504.1 |
| int8_baseline | 4,611.8 | 3438 | 440.4 | 1,915.2 | 910 | 1,804.1 | 1,803.5 |
| int4 | 4,660.1 | 7360 | 439.1 | 1,963.4 | 984 | 1,852.4 | 1,851.7 |
| int4_baseline | 4,611.8 | 3438 | 392.0 | 1,915.2 | 910 | 1,804.1 | 1,803.5 |

## Memcpy Kind Breakdown

| Mode | Kind | MiB | Count | ms |
|---|---|---:|---:|---:|
| fp16 | Host-to-Device | 2,570.0 | 1461 | 248.8 |
| fp16 | Device-to-Host | 126.0 | 433 | 35.8 |
| fp16 | Device-to-Device | 111.7 | 210 | 0.7 |
| int8 | Device-to-Device | 8,615.8 | 1264 | 33.4 |
| int8 | Host-to-Device | 2,570.6 | 5943 | 247.5 |
| int8 | Device-to-Host | 126.0 | 433 | 34.3 |
| int8_baseline | Host-to-Device | 2,570.6 | 2095 | 401.5 |
| int8_baseline | Device-to-Device | 1,915.2 | 910 | 8.3 |
| int8_baseline | Device-to-Host | 126.0 | 433 | 30.6 |
| int4 | Host-to-Device | 2,570.6 | 5943 | 398.0 |
| int4 | Device-to-Device | 1,963.4 | 984 | 8.4 |
| int4 | Device-to-Host | 126.0 | 433 | 32.7 |
| int4_baseline | Host-to-Device | 2,570.6 | 2095 | 350.1 |
| int4_baseline | Device-to-Device | 1,915.2 | 910 | 8.3 |
| int4_baseline | Device-to-Host | 126.0 | 433 | 33.6 |

## Runtime Count And Time

This table names the CUDA runtime API associated with the memcpy events. In this capture all recorded CUDA memcpy traffic is issued through `cudaMemcpyAsync_v3020`.

| Mode | Runtime name | MiB | Count | ms |
|---|---|---:|---:|---:|
| fp16 | `cudaMemcpyAsync_v3020` | 2,807.7 | 2104 | 285.3 |
| int8 | `cudaMemcpyAsync_v3020` | 11,312.5 | 7640 | 315.2 |
| int8_baseline | `cudaMemcpyAsync_v3020` | 4,611.8 | 3438 | 440.4 |
| int4 | `cudaMemcpyAsync_v3020` | 4,660.1 | 7360 | 439.1 |
| int4_baseline | `cudaMemcpyAsync_v3020` | 4,611.8 | 3438 | 392.0 |

## Largest D2D Size Buckets

These buckets show repeated GPU-to-GPU copies with identical byte sizes. Repetition across 50 DDIM steps exposes per-step copy patterns.

### fp16

| Count | Recorded API name | MiB Each | Total MiB | Approx count/step |
|---:|---|---:|---:|---:|
| 54 | `cudaMemcpyAsync_v3020 (54)` | 0.0 | 0.2 | 1.08 |
| 48 | `cudaMemcpyAsync_v3020 (48)` | 0.0 | 0.1 | 0.96 |
| 20 | `cudaMemcpyAsync_v3020 (20)` | 0.0 | 0.0 | 0.40 |
| 12 | `cudaMemcpyAsync_v3020 (12)` | 2.6 | 31.5 | 0.24 |
| 10 | `cudaMemcpyAsync_v3020 (10)` | 1.1 | 11.2 | 0.20 |
| 10 | `cudaMemcpyAsync_v3020 (10)` | 4.5 | 45.0 | 0.20 |
| 10 | `cudaMemcpyAsync_v3020 (10)` | 0.0 | 0.1 | 0.20 |
| 8 | `cudaMemcpyAsync_v3020 (8)` | 1.0 | 8.0 | 0.16 |
| 6 | `cudaMemcpyAsync_v3020 (6)` | 0.3 | 1.7 | 0.12 |
| 6 | `cudaMemcpyAsync_v3020 (6)` | 0.0 | 0.0 | 0.12 |
| 4 | `cudaMemcpyAsync_v3020 (4)` | 0.0 | 0.0 | 0.08 |
| 4 | `cudaMemcpyAsync_v3020 (4)` | 0.0 | 0.0 | 0.08 |

### int8

| Count | Recorded API name | MiB Each | Total MiB | Approx count/step |
|---:|---|---:|---:|---:|
| 282 | `cudaMemcpyAsync_v3020 (282)` | 0.0 | 0.8 | 5.64 |
| 256 | `cudaMemcpyAsync_v3020 (256)` | 0.0 | 0.4 | 5.12 |
| 108 | `cudaMemcpyAsync_v3020 (108)` | 0.0 | 0.1 | 2.16 |
| 88 | `cudaMemcpyAsync_v3020 (88)` | 1.0 | 86.6 | 1.76 |
| 46 | `cudaMemcpyAsync_v3020 (46)` | 20.2 | 931.5 | 0.92 |
| 46 | `cudaMemcpyAsync_v3020 (46)` | 7.9 | 362.2 | 0.92 |
| 42 | `cudaMemcpyAsync_v3020 (42)` | 5.1 | 212.6 | 0.84 |
| 40 | `cudaMemcpyAsync_v3020 (40)` | 31.5 | 1,260.0 | 0.80 |
| 40 | `cudaMemcpyAsync_v3020 (40)` | 3.9 | 157.5 | 0.80 |
| 38 | `cudaMemcpyAsync_v3020 (38)` | 63.0 | 2,394.0 | 0.76 |
| 32 | `cudaMemcpyAsync_v3020 (32)` | 0.5 | 15.8 | 0.64 |
| 30 | `cudaMemcpyAsync_v3020 (30)` | 0.0 | 0.2 | 0.60 |

### int8_baseline

| Count | Recorded API name | MiB Each | Total MiB | Approx count/step |
|---:|---|---:|---:|---:|
| 282 | `cudaMemcpyAsync_v3020 (282)` | 0.0 | 0.8 | 5.64 |
| 256 | `cudaMemcpyAsync_v3020 (256)` | 0.0 | 0.4 | 5.12 |
| 108 | `cudaMemcpyAsync_v3020 (108)` | 0.0 | 0.1 | 2.16 |
| 46 | `cudaMemcpyAsync_v3020 (46)` | 20.2 | 931.5 | 0.92 |
| 42 | `cudaMemcpyAsync_v3020 (42)` | 5.1 | 212.6 | 0.84 |
| 30 | `cudaMemcpyAsync_v3020 (30)` | 0.0 | 0.2 | 0.60 |
| 18 | `cudaMemcpyAsync_v3020 (18)` | 1.3 | 22.8 | 0.36 |
| 12 | `cudaMemcpyAsync_v3020 (12)` | 0.0 | 0.1 | 0.24 |
| 12 | `cudaMemcpyAsync_v3020 (12)` | 0.0 | 0.0 | 0.24 |
| 12 | `cudaMemcpyAsync_v3020 (12)` | 2.6 | 31.5 | 0.24 |
| 10 | `cudaMemcpyAsync_v3020 (10)` | 1.1 | 11.2 | 0.20 |
| 10 | `cudaMemcpyAsync_v3020 (10)` | 4.5 | 45.0 | 0.20 |

### int4

| Count | Recorded API name | MiB Each | Total MiB | Approx count/step |
|---:|---|---:|---:|---:|
| 282 | `cudaMemcpyAsync_v3020 (282)` | 0.0 | 0.8 | 5.64 |
| 256 | `cudaMemcpyAsync_v3020 (256)` | 0.0 | 0.4 | 5.12 |
| 108 | `cudaMemcpyAsync_v3020 (108)` | 0.0 | 0.1 | 2.16 |
| 46 | `cudaMemcpyAsync_v3020 (46)` | 20.2 | 931.5 | 0.92 |
| 42 | `cudaMemcpyAsync_v3020 (42)` | 5.1 | 212.6 | 0.84 |
| 32 | `cudaMemcpyAsync_v3020 (32)` | 0.5 | 15.8 | 0.64 |
| 30 | `cudaMemcpyAsync_v3020 (30)` | 0.0 | 0.2 | 0.60 |
| 30 | `cudaMemcpyAsync_v3020 (30)` | 1.0 | 29.5 | 0.60 |
| 18 | `cudaMemcpyAsync_v3020 (18)` | 1.3 | 22.8 | 0.36 |
| 12 | `cudaMemcpyAsync_v3020 (12)` | 0.0 | 0.1 | 0.24 |
| 12 | `cudaMemcpyAsync_v3020 (12)` | 0.0 | 0.0 | 0.24 |
| 12 | `cudaMemcpyAsync_v3020 (12)` | 2.6 | 31.5 | 0.24 |

### int4_baseline

| Count | Recorded API name | MiB Each | Total MiB | Approx count/step |
|---:|---|---:|---:|---:|
| 282 | `cudaMemcpyAsync_v3020 (282)` | 0.0 | 0.8 | 5.64 |
| 256 | `cudaMemcpyAsync_v3020 (256)` | 0.0 | 0.4 | 5.12 |
| 108 | `cudaMemcpyAsync_v3020 (108)` | 0.0 | 0.1 | 2.16 |
| 46 | `cudaMemcpyAsync_v3020 (46)` | 20.2 | 931.5 | 0.92 |
| 42 | `cudaMemcpyAsync_v3020 (42)` | 5.1 | 212.6 | 0.84 |
| 30 | `cudaMemcpyAsync_v3020 (30)` | 0.0 | 0.2 | 0.60 |
| 18 | `cudaMemcpyAsync_v3020 (18)` | 1.3 | 22.8 | 0.36 |
| 12 | `cudaMemcpyAsync_v3020 (12)` | 0.0 | 0.1 | 0.24 |
| 12 | `cudaMemcpyAsync_v3020 (12)` | 0.0 | 0.0 | 0.24 |
| 12 | `cudaMemcpyAsync_v3020 (12)` | 2.6 | 31.5 | 0.24 |
| 10 | `cudaMemcpyAsync_v3020 (10)` | 1.1 | 11.2 | 0.20 |
| 10 | `cudaMemcpyAsync_v3020 (10)` | 4.5 | 45.0 | 0.20 |

## Plots

### Total CUDA memcpy I/O

![Total CUDA memcpy I/O](plots/total_cuda_io.png)

### CUDA memcpy I/O by transfer kind

![CUDA memcpy I/O by transfer kind](plots/cuda_io_by_kind.png)

### D2D memcpy event count

![D2D memcpy event count](plots/d2d_count.png)

### Baseline D2D copies by repeated tensor size

![Baseline D2D copies by repeated tensor size](plots/d2d_top_sizes_baselines.png)

### D2D traffic by repeated tensor-size bucket

![D2D traffic by repeated tensor-size bucket](plots/d2d_size_heatmap.png)

## Interpretation

- `int4` and `int4_baseline` have very different resident tracked memory, but similar total CUDA memcpy I/O.
- Baseline extra I/O versus FP16 is almost entirely D2D, meaning on-GPU tensor movement rather than host transfer.
- INT8 MoDiff has the largest D2D because its MoDiff static path keeps cache updates plus INT8 quantized-island movement.
- INT4 MoDiff and INT4 baseline are close in memcpy bytes because both run the same low-bit island pattern; the MoDiff cache cost appears mainly as resident cache memory, not a large additional memcpy volume.

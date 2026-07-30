# INT8 attention：全 layer 确定性量化路线

`MODIFF_INT8_QKV_EPILOGUE` 默认开启，不再包含 `auto` 选择。INT8 的
`MODIFF_FLASH_GATE`、`MODIFF_FLASH_PACKED` 和
`MODIFF_INT8_PACKED_PERSISTENT` 同样只接受确定性的 on/off；传入 `auto`
会直接报错。21 个 attention layer 在静态 scale 冻结后都运行量化 QKV、量化
attention 和 W8A8 projection。设置 `MODIFF_INT8_QKV_EPILOGUE=0` 仅用于显式
回退和 A/B。`MODIFF_FLASH_PACKED` 默认关闭，只有显式设为 1 才进入诊断路线，
不再做逐 layer autotune。

T16/T4 首次加载时仍用 8 次观测收集静态 Q/K/V scale；scale 冻结后的稳态
推理全部走量化 kernel。这个观测窗口不包含在下面的稳态 benchmark 中。

## 数据流

大 shape（T1024/T256/T64）：

```text
GroupNorm→INT8
  → W8A8 QKV GEMM + bias + per-column INT8 requant
  → fused K gather / V transpose（Q 留在 packed QKV）
  → direct-Q INT8 FlashAttention + projection requant
  → W8A8 projection + bias + residual
```

小 shape（T16/T4）：

```text
GroupNorm→INT8
  → W8A8 QKV GEMM + bias + per-column INT8 requant
  → packed-INT8 small attention + projection requant
  → W8A8 projection + bias + residual
```

## A40 batch 128

20 warmups，5 rounds × 60 iterations，报告各轮中位数。

| Tokens | Instances | FP16 layer | INT8 layer | Speedup |
|---:|---:|---:|---:|---:|
| 1024 | 5 | 3101.73 µs | 2837.12 µs | 1.093× |
| 256 | 5 | 1078.75 µs | 918.94 µs | 1.174× |
| 64 | 5 | 411.39 µs | 234.59 µs | 1.754× |
| 16 | 5 | 216.26 µs | 179.87 µs | 1.202× |
| 4 | 1 | 199.23 µs | 103.23 µs | 1.930× |
| **21-block weighted** | **21** | **24.240 ms** | **20.956 ms** | **1.157×** |

当前仍未达到 1.5× 目标。T1024 的 5 个 block 占 INT8 加权延迟约
`14.19 ms / 20.96 ms`，但单层只有 1.093×；其中 INT8 FlashAttention
本身约 1.558 ms，是下一阶段最重要的优化对象。

```text
T1024  FP16 ███████████████████████████████  3102
       INT8 ████████████████████████████     2837

T256   FP16 ███████████                      1079
       INT8 █████████                         919

T64    FP16 ████                              411
       INT8 ██                                235

T16    FP16 ██                                216
       INT8 ██                                180

T4     FP16 ██                                199
       INT8 █                                 103   (µs)
```

T1024 INT8 的主要 kernel 时间为：FlashAttention `1557.98 µs`、QKV
W8A8 INT8 epilogue `380.56 µs`、projection W8A8 `363.35 µs`、K/V producer
`313.35 µs`、GroupNorm+quantize `281.42 µs`。因此继续优化总体延迟时，应优先处理
T1024 FlashAttention，而不是对已经很快的 T64/T4 再加分支。

## Correctness

- Quantized small-attention vs FP32 reference:
  - T16: bit exact.
  - T4: relative L2 `0.000165`, max INT8-code difference `1`.
- Existing nine-shape attention correctness: all pass; INT8 relative error
  `0.0075–0.0163`, below `0.05`.
- Three fixed-seed 50-step DDIM comparisons: seeds 1234, 5678, 9012 all have
  latent relative L2 `0.0`, below `0.02`.

Machine-readable summary:
[`data/int8_all_quantized_summary.json`](data/int8_all_quantized_summary.json).

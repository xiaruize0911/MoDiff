# INT8 Attention two-phase optimization

## Outcome

Phase 1 的 register-P Flash specialization 保持 bit-exact，资源为
`REG=64, STACK=0, LOCAL=0`，但 T1024 kernel 从 `1645.04 µs` 回退到
`1768.23 µs`，因此默认关闭。

Phase 2 将 W8A8 QKV epilogue 改为直接产生 Q、padded K 和转置 V。T1024
使用 `head × QKV × hd_pad` 权重布局；进一步 fine-tune 后，T256 使用保持
原始 `hd48` 列数的 compact-segment epilogue。两个 shape 都删除
`from_i8_kv_tiled_kernel`；T64 固定保留旧路线。没有运行时 autotune。

## Validation

- Q/K/V layout、padding 和最终 Flash qout 对旧路线逐 bit 一致。
- batch 1/4/128 与非默认 CUDA stream 验证通过。
- 连续 20 次输出一致，无 pipeline race。
- 九个 attention correctness shape 全部通过，INT8 relative L2 为
  `0.0076–0.0160`，低于 `0.05`。
- seeds 1234、5678、9012 的 batch-4、50-step DDIM latent relative L2
  均为 `0.0`，低于 `0.02`。
- 所有 INT8 配置传入 `auto` 都会明确报错。

## Benchmark

A40 batch 128；静态 scale 已冻结；20 warmups；5 rounds × 60 iterations；
正式完整 layer benchmark 独立运行两次，以下为两次结果的中位数。

| Tokens | Instances | FP16 | Previous INT8 | Fine-tuned INT8 | Speedup |
|---:|---:|---:|---:|---:|---:|
| 1024 | 5 | 3118.59 µs | 2785.70 µs | 2772.99 µs | 1.125× |
| 256 | 5 | 1078.00 µs | 916.91 µs | 861.58 µs | 1.251× |
| 64 | 5 | 411.76 µs | 232.61 µs | 231.36 µs | 1.780× |
| 16 | 5 | 215.02 µs | 179.06 µs | 179.60 µs | 1.197× |
| 4 | 1 | 196.91 µs | 105.28 µs | 105.30 µs | 1.870× |
| **21-block weighted** | **21** | **24.314 ms** | **20.677 ms** | **20.333 ms** | **1.196×** |

```text
FP16          24.314 ms |████████████████████████|
INT8 before   20.677 ms |████████████████████    |
INT8 after    20.333 ms |████████████████████    |
1.5x target   16.209 ms |████████████████        |
```

T256 fine-tune 相对上一版 INT8 weighted 提升 `1.017×`，通过 1% 门槛。
从最初 `20.907 ms` 基线累计提升 `1.028×`；相对 FP16 达到 `1.196×`。
距离 1.5× 目标仍有约 `4.12 ms`；最大瓶颈
仍是 T1024 INT8 FlashAttention，约 `1.55 ms/layer`。

Machine-readable results:
[`data/int8_two_phase_optimization.json`](data/int8_two_phase_optimization.json).

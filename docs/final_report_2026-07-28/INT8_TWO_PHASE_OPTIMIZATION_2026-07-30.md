# INT8 Attention two-phase optimization

> **更正（2026-08-03）：本文中所有 latent 层面的证据作废。**
> 文中每一个「latent relative L2」和整层「bit-exact」结论都是恒真的。`UNetModel.out[-1]` 是
> `zero_module`（`ldm/modules/diffusionmodules/openaimodel.py:745`），`AttentionBlock.proj_out`
> 也是（`:345`）；本树的 checkpoint 是 856 字节的空壳，`state_dict` 有 0 个条目、以
> `strict=False` 加载，所以这两处权重一直是零。于是每个 attention block 都是对输入的
> bit-exact 恒等映射，而 UNet 对任何输入都预测**恒为零**，采样 latent 只由初始噪声和 DDIM
> schedule 决定。任何改动（无论对错）都必然得到 latent relative L2 = `0`。实证：把全部 21 个
> attention block 的输出强制替换为常数，`forward` 触发 420 次，latent 仍然逐位相同。
> `integration/tests/golden/e2e_*_vacuous.pt` 五个 golden 也是证据 —— fp16、int8、int4 三者逐位相同。
>
> **仍然成立的**：kernel 级正确性结果 —— 九个 attention shape 对照「用同一批量化 code 算出的
> fp32 参考」（`scripts/qattn_correctness.py`），以及直接在 kernel 输出上测的 code 差异/ 相对
> L2。这些不需要 checkpoint，也不经过任何 zero_module。
> **同样不受影响**：本文所有计时。kernel 开销与数据无关，shape 和 launch 序列都是真的。
>
> 脚本已修（比较前会激活这些零初始化层并断言可观测性），重跑即可得到有意义的判定。完整说明见
> [`docs/gn_qkv_fusion_2026-08-03/FINDINGS.md`](../gn_qkv_fusion_2026-08-03/FINDINGS.md) 第 5 节。


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

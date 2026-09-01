# Scheme A only — B/C 已从代码删除，A 重测

2026-09-01 · LDM-8 LSUN-Churches · NVIDIA A40 · torch 2.4.1+cu124  
W8A8 CUTLASS · batch 128 计时 · n=4 latent · DDIM η=0 · seed `20260805`  
`MODIFF_LINEAR=0` · static delta · flash attn · warmup=1 · `ATTN_REPLAY_K=1`

计时：CUDA event，MoDiff 状态在 timer 外 reset。质量是 latent relL2，不是 FID。

样张：[`plots/a_grid.png`](plots/a_grid.png)  
数据：[`data/remeasure_a.json`](data/remeasure_a.json)

---

## 代码

B（`REPLAY_K>1`：跳过步不算 conv，`out = o_hat_冻结 + skip`）和 C（再 `DROP_OHAT=1`：`out = skip`）已从生产路径删掉：

- `OptimizedInt8Conv2d` / `OptimizedInt4Conv2d`：`_replay_residual` / `_peek_replay` / `_replay_out` 及全部 early-out
- `FusedResBlock`：`MODIFF_REPLAY_BLOCK` peek / skip-in / skip-out
- CUDA graph：`residual_replay` 相位

现在每步都是完整 MoDiff（方案 A）：GN+Q+INT8 conv，写入 `a_hat`/`o_hat`，`out = o_hat_新 + skip(x)`。

`MODIFF_REPLAY_K`、`MODIFF_REPLAY_DROP_OHAT`、`MODIFF_REPLAY_BLOCK` 设了也无效。`CACHE_SKIP_K`（仍算、不写 cache）还在，默认 1。

---

## 重测（删 B/C 之后，同一协议）

墙钟相对 **fp16 S=50**（5369 ms/sample）。2.00× 线 = 2684 ms。

| 臂 | 步数 | ms/step | trials | ms/sample | vs fp16 S=50 | relL2 vs fp16 S=50 |
|---|--:|--:|--:|--:|--:|--:|
| fp16 | 50 | 107.37 | 107.21 / 107.54 | 5369 | 1.00× | 0 |
| fp16 | 25 | 107.88 | 107.77 / 107.99 | 2697 | 1.99× | 0.198 |
| **A 完整 MoDiff** | **50** | **75.25** | 75.22 / 75.28 | **3763** | **1.43×** | **0.110** |
| **A 完整 MoDiff** | **25** | **77.48** | 77.41 / 77.54 | **1937** | **2.77×** | **0.217** |

同 schedule 对照：

| | relL2 |
|---|--:|
| A S=50 vs fp16 S=50 | **0.110** |
| A S=25 vs fp16 S=25 | **0.116** |
| fp16 S=25 vs fp16 S=50 | 0.198 |
| A S=25 vs fp16 S=50 | 0.217 |

A S=25 相对 fp16 S=50 的 0.217 主要是 DDIM 25 vs 50（fp16 自己付 0.198）。MoDiff 量化误差在两种步数上都是 ~0.11。

相对上次（B/C 还在、A 已经是 `REPLAY_K=1`）同协议：A S=50 当时 74.89 ms / 1.43× / relL2 0.091；现在 75.25 ms / 1.43× / 0.110。墙钟同一档；n=4 relL2 有样本噪声。

---

## CACHE_SKIP_K 扫描（DDIM 50）

B/C 删掉之后，A 上还能动的 K 只有 `MODIFF_CACHE_SKIP_K`：每步仍跑 GN+Q+conv，只在 `step_count % K != 0` 时不写 `a_hat`/`o_hat`。K=1 = 不跳。

fp16 本轮 = **107.13 ms/step**。样张 [`plots/k_sweep.png`](plots/k_sweep.png)，数据 [`data/k_sweep.json`](data/k_sweep.json)。

| K | 不写 cache 比例 | ms/step | vs fp16 | vs K=1 | relL2 vs fp16 | relL2 vs K=1 |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 0 | 74.57 | **1.44×** | 1.00× | **0.120** | 0 |
| 2 | 50% | 72.82 | 1.47× | 1.02× | 0.145 | 0.053 |
| 3 | 67% | 72.82 | 1.47× | 1.02× | 0.168 | 0.096 |
| 5 | 80% | 72.55 | 1.48× | 1.03× | 0.185 | 0.121 |
| 7 | 86% | 72.43 | 1.48× | 1.03× | 0.217 | 0.172 |
| 10 | 90% | 72.58 | 1.48× | 1.03× | 0.253 | 0.218 |

K=2 相对 K=1 省 **1.75 ms**，之后基本饱和（K=7 再抠 0.4 ms）。50 步上到不了 2×。质量从 K=1 起单调变差；K≥7 肉眼开始糊。默认保持 **K=1**。

---

## 结论

- **生产路径只有 A，skip K=1。** 每步都算、都写 cache。
- **50 步：1.43–1.44×**，图接近 fp16。
- **加大 CACHE_SKIP_K 几乎不更快**（封顶 ~1.48×），质量掉。
- **≥2×：A S=25，2.77×**。不要用跳 cache 去凑 50 步 2×。

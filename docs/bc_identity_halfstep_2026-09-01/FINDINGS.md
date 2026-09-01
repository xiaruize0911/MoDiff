# B 与 C 不是同一件事；半步 DDIM 才是更干净的 2×

2026-09-01 · LDM-8 Churches · A40 · W8A8 · seed `20260805` · n=4 latent · batch 128 计时

数据：[`data/bc_identity.json`](data/bc_identity.json)  
样张：[`plots/bc_halfstep_grid.png`](plots/bc_halfstep_grid.png)

---

## 层上的等式（同一输入 x）

跳过步上 ResBlock 的 `out_conv`：

```
out_B = o_hat_冻结 + skip(x)
out_C = skip(x)
out_B − out_C = o_hat_冻结          # 精确，不是近似
```

所以 `||B−C|| / ||B|| = ||o_hat|| / ||o_hat + skip||`。若这个量 ≪ 1，丢掉 `o_hat` 是小扰动，B ≈ C。实测不是。

在 B、K=2 的一次 n=4 生成上挂钩（875 次 skip / 840 次 commit）：

| | n | median | p10 | p90 |
|---|--:|--:|--:|--:|
| skip  `\|\|B−C\| / \|B\|\|` | 875 | **0.485** | 0.156 | 0.705 |
| skip  `\|\|o_hat\| / \|skip\|\|` | 875 | **0.573** | 0.153 | 1.039 |
| skip  cos(`o_hat`, `skip`) | 875 | **0.044** | −0.222 | 0.182 |
| commit 同一比值（刚算完的残差） | 840 | 0.488 | 0.151 | 0.706 |

commit 与 skip 的分布几乎一样：冻结并没有把 `o_hat` 变小，只是不再更新。`o_hat` 和 `skip` 接近正交，量级相当。C 不是「B 减一个小偏置」，是把 ResBlock 输出换成一个几乎无关、差不多大的向量。

最重的几层 skip 上 median `||B−C||/||B||`：`output_blocks.5.0` 0.757，`input_blocks.10.0` 0.753，`input_blocks.1.0` 0.707。没有一层的中位数接近 0。

---

## 端到端：B 贴近 A，C 贴近色块

同一 seed、同一 loaded 模型，latent relL2：

| | vs fp16 S=50 | vs A S=50 | B vs C |
|---|--:|--:|--:|
| **A** 完整 MoDiff S=50 | 0.091 | 0 | — |
| **B K=2** | 0.198 | 0.167 | **0.932** |
| **C K=2** | 0.905 | 0.954 | 0.932 |
| **B K=3** | 0.258 | 0.252 | **0.953** |
| **C K=3** | 1.109 | 1.108 | 0.953 |

B 离 A 只有 0.17；C 离 A、离 B 都是 ~0.93。C 更接近其它 C（C K=2 vs C K=3 = 0.26），不接近任何 B。图上 B 仍是教堂，C 是色块——不是同一种失败的两个名字。

---

## 半步 DDIM，以及「直接跳层」

| arm | ms/sample | vs fp16 S=50 墙钟 | relL2 vs fp16 S=50 | 图 |
|---|--:|--:|--:|---|
| fp16 S=50 | 5351 | 1.00× | 0 | 参考 |
| A S=50 | 3744 | 1.43× | 0.091 | 接近 fp16 |
| B K=2 S=50 | 2765 | 1.94× | 0.198 | 教堂清晰 |
| C K=2 S=50 | 2700 | 1.98× | 0.905 | 色块 |
| B K=3 S=50 | 2462 | 2.17× | 0.258 | 结构在 |
| **fp16 S=25** | 2693 | **1.99×** | 0.203 | 仍清晰 |
| **A S=25** | **1933** | **2.77×** | 0.225 | 仍清晰 |
| B freeze after t=T | 1833 | 2.92× | 0.745 | 噪点 |
| C skip residual after t=T | 1695 | 3.16× | 1.421 | 色块 |
| C skip + S=25 | 919 | 5.82× | 1.257 | 色块 |

公平的质量对照：A S=25 vs **同 schedule 的** fp16 S=25 是 **0.104**，和 A S=50 vs fp16 S=50 的 0.091 同一档。A@25 相对 fp16@50 的 0.225，绝大部分是 DDIM 25 vs 50（fp16 自己就要付 0.203），不是 MoDiff 坏了。

B K=2 的 latent 离 A S=25 只有 0.113——「50 步里隔步冻残差」和「25 步全算」落在附近。C 对两者都是 ~0.93。

直接跳层（t=T 之后 `out = skip(x)`，K=∞ + DROP_OHAT）无论 50 步还是 25 步都是色块。把 B 冻到 t=T 之后不再更新，是 0.745 的噪点，不是教堂。

---

## Skip-step 变化量

ResBlock 相邻两次访问（通常隔 1 个 DDIM step）。median，n=4：

| | Δo_hat | Δskip | Δout | Δo_hat/Δskip | Δout / \|out\| |
|---|--:|--:|--:|--:|--:|
| **A** 每步都算 | 13.06 | 15.36 | 20.59 | **0.72** | 0.067 |
| **B K=2 skip** | **0** | 4.29 | 4.29 | 0 | 0.015 |
| **B K=2 commit** | 23.22 | 26.85 | 36.41 | 0.72 | 0.133 |
| **C K=2 skip** | 0 | 284 | 365 | 0 | ~1 |
| **B K=3 skip** | 0 | 4.04 | 4.04 | 0 | — |

B 的 skip 步：`Δout / Δskip = 0.9998`（就是 `Δout = Δskip`）。C 公式相同，但 `x` 已经漂了，Δskip 比 B 大约 **66×**。A 每步卷积增量 `Δo_hat` 是 `Δskip` 的 0.72 倍——跳过的不是小量。`Δo_hat / \|o_hat\|` 在 A 上 median 0.079（累计残差每步改 ~8%）。

数据：[`data/skip_step_delta.json`](data/skip_step_delta.json)

「为什么不直接跳过这个 DDIM step？」——可以，而且比 B 更好。1.5% 只是冻卷积之后 **一个 ResBlock** 的 `Δout`。同一套 B K=2 skip 步上，UNet 的 ε 仍变 **0.94%**（A 每步 1.8%），latent x 仍变 **1.4%**。就算 ε 完全复用（`Δε=0`），DDIM 积分仍因 α 变化把 x 挪 **1.5%**。

直接跳过整步 = A S=25：墙钟 1916 ms，和「50 步里隔步复用 ε」的 1940 ms 一样快，质量更好（vs A：0.197 vs 0.236）。B K=2 仍跑 50 次 UNet（attention + skip 都在），只要 2764 ms。

数据：[`data/skip_the_step.json`](data/skip_the_step.json)

---

## 结论

**B 和 C 都没有意义，不要当作模式用。** 默认保持 `MODIFF_REPLAY_K=1`、`MODIFF_REPLAY_DROP_OHAT=0`。

- **C**：丢掉 `o_hat` 就是丢掉卷积支路。skip 步公式和 B 一样是 `Δout=Δskip`，但 level 差一个与 skip 同量级的正交项，端到端 relL2 0.93，图是色块。
- **B**：50 步时刻表上冻 conv，看起来像「少做一半」。真正少做一半工作是 **A S=25**（2.77× vs B K=3 的 2.17×；relL2 0.225 vs 0.258）。B 的 skip 步 ResBlock 只动 1.5%，是因为 conv 冻了，不是因为这一步 DDIM 可以省掉。

Knobs 留在树上当阴性对照，不再推荐、不再扫 K。

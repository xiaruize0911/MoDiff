# W8A8 跳步方案：B/C 撤回，A S=25 才是 ≥2×

2026-09-01 · LDM-8 LSUN-Churches · NVIDIA A40 · torch 2.4.1+cu124  
W8A8 CUTLASS · batch 128 计时 · n=4 latent 质量 · DDIM η=0 · seed `20260805`  
`MODIFF_LINEAR=0` · static delta · flash attn · warmup=1 · `ATTN_REPLAY_K=1`

计时：CUDA event，MoDiff 状态在 timer 外 reset。质量是 latent relL2，不是 FID。

样张由 `scripts/measure_bc.py` 写出 `plots/bc_halfstep_grid.png`（本树未入库）。

---

## 结论

1. **B、C 都不是模式。** 默认 `REPLAY_K=1` `DROP_OHAT=0`。
2. **≥2× 且图仍是 MoDiff 教堂：A S=25**（完整 MoDiff，25 步），墙钟 **2.77×** vs fp16 S=50。
3. A 的 **每步** ~74 ms（1.43×）已经接近上限；50 步上到不了 2×。

---

## 1. A / B / C 是什么

ResBlock：`out = 卷积支路 + skip(x)`。MoDiff 把卷积支路做成累计状态 `o_hat`。

```
A  每步都算    out = o_hat_新 + skip(x)
B  跳过步      out = o_hat_冻结 + skip(x)     # 不算 conv，还加残差
C  跳过步      out = skip(x)                 # 不算 conv，丢掉 o_hat
```

同一输入上：`out_B − out_C = o_hat`（精确）。跳过步上两者的 **变化** 都是 `Δout = Δskip`，差的是输出坐在 `o_hat` 上还是坐在 0 上。

---

## 2. 总表（同一 loaded 模型）

墙钟相对 **fp16 S=50**（5351 ms/sample）。2.00× 线 = 2676 ms。

| 臂 | 步数 | ms/step | ms/sample | vs fp16 S=50 | relL2 vs fp16 S=50 | 图 |
|---|--:|--:|--:|--:|--:|---|
| fp16 | 50 | 107.02 | 5351 | 1.00× | 0 | 参考 |
| **A 完整 MoDiff** | 50 | 74.89 | 3744 | 1.43× | **0.091** | 接近 fp16 |
| B K=2 冻残差 | 50 | 55.29 | 2765 | 1.94× | 0.198 | 教堂清晰 |
| C K=2 丢 o_hat | 50 | 54.00 | 2700 | 1.98× | **0.905** | 色块 |
| B K=3 冻残差 | 50 | 49.24 | 2462 | 2.17× | 0.258 | 结构在 |
| C K=3 丢 o_hat | 50 | 47.33 | 2366 | 2.26× | **1.109** | 色块 |
| fp16 | 25 | 107.71 | 2693 | 1.99× | 0.203 | 仍清晰 |
| **A 完整 MoDiff** | **25** | 77.30 | **1933** | **2.77×** | 0.225 | 仍清晰 |
| B 冻到 t=T 之后 | 50 | 36.66 | 1833 | 2.92× | 0.745 | 噪点 |
| C 跳残差 after t=T | 50 | 33.89 | 1695 | 3.16× | 1.421 | 色块 |
| C + S=25 | 25 | 36.76 | 919 | 5.82× | 1.257 | 色块 |

A S=25 vs **同 schedule** 的 fp16 S=25：relL2 **0.104**，和 A S=50 vs fp16 S=50 的 0.091 同一档。相对 fp16 S=50 的 0.225 主要是 DDIM 25 vs 50（fp16 自己付 0.203）。

数据：[`data/bc_identity.json`](data/bc_identity.json)

---

## 3. B 不是 C

层上（B K=2，875 次 skip / 840 次 commit）：

| | median | p10 | p90 |
|---|--:|--:|--:|
| skip `\|\|B−C\| / \|B\|\|` | **0.485** | 0.156 | 0.705 |
| skip `\|\|o_hat\| / \|skip\|\|` | 0.573 | 0.153 | 1.039 |
| skip cos(`o_hat`, `skip`) | **0.044** | −0.222 | 0.182 |
| commit 同一比值 | 0.488 | 0.151 | 0.706 |

`o_hat` 与 `skip` 同量级、近正交。丢掉它不是小扰动。

端到端 latent relL2：

| | vs fp16 S=50 | vs A S=50 | B vs C |
|---|--:|--:|--:|
| A S=50 | 0.091 | 0 | — |
| B K=2 | 0.198 | 0.167 | **0.932** |
| C K=2 | 0.905 | 0.954 | 0.932 |
| B K=3 | 0.258 | 0.252 | **0.953** |
| C K=3 | 1.109 | 1.108 | 0.953 |

B 贴近 A；C 贴近色块。

---

## 4. Skip 步上实际变了多少

ResBlock `out_conv` 相邻两次访问，median：

| | Δo_hat | Δskip | Δout | Δo_hat/Δskip | Δout/‖out‖ |
|---|--:|--:|--:|--:|--:|
| A 每步都算 | 13.06 | 15.36 | 20.59 | **0.72** | 6.7% |
| **B K=2 skip** | **0** | 4.29 | **4.29** | 0 | 1.5% |
| B K=2 commit | 23.22 | 26.85 | 36.41 | 0.72 | 13.3% |
| C K=2 skip | 0 | **284** | **365** | 0 | ~100% |

B skip：`Δout/Δskip = 0.9998`（就是 `Δout = Δskip`）。A 每步被跳掉的卷积增量是 `Δskip` 的 0.72 倍。C 公式相同，但 `x` 已漂，Δskip 约为 B 的 **66×**。

数据：[`data/skip_step_delta.json`](data/skip_step_delta.json)

1.5% 只是冻 conv 之后 **一个 ResBlock**。同一 B skip 步上，UNet 的 ε 仍变 **0.94%**（A 每步 1.8%），latent x 仍变 **1.4%**。ε 完全复用时 `Δε=0`，DDIM 积分仍因 α 把 x 挪 **1.5%**。

| 做法 | ms/sample | relL2 vs A S=50 |
|---|--:|--:|
| B K=2（50 次 UNet，跳 conv） | 2764 | 0.222 |
| 隔步复用 ε（25 次 UNet + 50 次积分） | 1940 | 0.236 |
| **A S=25（25 次完整 MoDiff）** | **1916** | **0.197** |

「直接跳过这个 step」= A S=25，不是 B。

数据：[`data/skip_the_step.json`](data/skip_the_step.json)

---

## 5. A 每步还能不能再快

A ≈ 74 ms。PTQ（无 MoDiff）≈ 65 ms。2× @ S=50 需要 ≈ 53 ms。

Kernel 桶（另一进程，A=72.20 / PTQ=64.77 / fp16=102.23，[`pipeline_profile`](../pipeline_profile_2026-08-31/FINDINGS.md)）：

| bucket | fp16 | PTQ | A | A−PTQ |
|---|--:|--:|--:|--:|
| GEMM / conv | 46.7 | 37.1 | 37.8 | +0.7 |
| GN+SiLU | 20.9 | 10.8 | **17.4** | **+6.6** |
| attention | 11.4 | 9.0 | 8.9 | −0.1 |
| elem/copy | 19.6 | 5.8 | 6.1 | +0.3 |
| other | 3.7 | 2.0 | 2.0 | 0 |

A 比 PTQ 多的 ~8 ms 几乎全在 GN/`a_hat`。这条线已关（藏、叠、压、隔步不写、B/C）。

还开着的都小：`FUSE_QKV_I8` +0.79 ms；CUDA graph ~0.5 ms（capture 需先预分配 attention 的 `torch.tensor`）；Stream-K 上限 ≲1–2 ms。加起来 A 仍是 ~71 ms、**1.45×**。50 步上到不了 2×。

---

## 默认

```
MODIFF_REPLAY_K=1
MODIFF_REPLAY_DROP_OHAT=0
MODIFF_CACHE_SKIP_K=1
MODIFF_ATTN_REPLAY_K=1
```

≥2×：完整 MoDiff，DDIM **25** 步。FID 若仍锁 50 步，A 就是 1.43×，不要用 B 去凑 2×。

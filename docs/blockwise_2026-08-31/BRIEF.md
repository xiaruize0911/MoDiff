# Blockwise 量化简报

NVIDIA A40 · LSUN-Churches LDM-8 · 50 步 DDIM · n=24 · 3 个种子。全文 **G 一律指输入通道数**：
权重块 = G 个通道 × 全部 9 个 tap，激活块 = 一个 (n,h,w) 上的 G 个通道。两者共享同一组 C 边界。

结论一句话：

> **G=32 通道**是合适的块大小，但**只值得用在权重上**。激活做 blockwise 在所有块大小上都没有收益
> （曲线对 G 完全平），在现行 refresh 节奏下反而更差。权重 blockwise 在 W8A8 上值 ~1.9×。
> 但现有 epilogue 无法承载任何 blockwise，唯一能跑的实现在 G=32 时让 conv 慢 **5.0×**——
> 整条三-kernel 路径换算成对 fp16 是 **0.38×**，即比不量化还慢。这是一个 mainloop 工程，不是一个开关。

英文完整版见 [FINDINGS.md](FINDINGS.md)。

---

## 0. 先说噪声

四次独立运行恰好共用了同一个 baseline arm（同配置、同种子、不同进程）：

| arm | 复现值 | 极差 |
|---|---|--:|
| W8A8 shipped, r=4 | .0273 / .0266 / .0258 / .0240 | **.0033** |
| W4A4 shipped, r=4 | .2788 / .2794 / .2801 / .2791 | .0013 |

所以流水线跨进程不是 bit-确定的，**W8A8 上小于 ~0.003 的差异读不出来**。下文每条结论都过了这条线，
或明确标注没过。

## 1. 为什么现在不是 blockwise

EVT epilogue 是 `o_hat[elem] += acc * alpha * weight_scale[k]`：`alpha` 是标量广播，
`weight_scale` 是沿**输出**通道的行广播。

- 输出通道是 GEMM 的 N 维，能从 reduction 里提出来 → 所以权重一直是 per-output-channel。
- 块 scale 沿 reduction 轴 `K = Cin*R*S` 变化。epilogue 只看到累加完的 accumulator，
  **数学上**无法事后修正 → 不是实现缺口。
- per-token（每个输入像素一个 scale）也救不了 3×3：一个输出行取自 9 个不同 scale 的输入像素。
  只有 1×1 conv 能在 epilogue 里吃 per-M scale。

`grep -rE 'group_scale|block_scale|blockwise' csrc/` = **0 命中**。

例外：**`a_hat` 不受此限**——它只被 elementwise kernel 碰，从不是 GEMM 操作数。

## 2. 用现有 kernel 做出的精确 blockwise

D2 epilogue 是对 `o_hat` 的 read-modify-write，所以**按通道块逐块调 conv**、每块带自己的 `alpha`
和自己的 per-(块, 输出通道) `weight_scales`，累加出来就是精确的 blockwise 反量化。不是近似：

```
split-K blockwise vs 参考实现: relerr 3.7e-04   （残差是 fp16 o_hat 累加）
```

## 3. 代价：conv 上的相对 speedup

A40, B=128，**全部 20 个 UNet ResBlock conv 形状**、按每步调用次数加权（62 calls/step），
与 [`conv_kernel_sweep_2026-08-28`](../conv_kernel_sweep_2026-08-28/FINDINGS.md) §5 同一组形状同一套权重。

### 融合后 2-kernel 对 2-kernel：MoDiff vs int8 baseline

上面对 fp16 的比较回答的是"量化值不值"，而不是"时序缓存要付多少代价"——后者需要拿 MoDiff 的融合对
去比 **baseline 的**融合对，两边各两个 kernel，都用生产形态。
[`scripts/fused_pair.py`](scripts/fused_pair.py)。

| arm | GN 阶段 | conv |
|---|---|---|
| fp16 | `group_norm_silu_nhwc` | `F.conv2d` fp16 |
| baseline int8 | `group_norm_silu_quantize_nhwc_fast` | `conv2d_int8_evt_bias_residual_fp16` (D1) |
| MoDiff int8 | `group_norm_silu_delta_quantize_nhwc` | `conv2d_int8_evt_o_hat` (D2) |

**先说一个必须交代的不对称。** fast-reduce 变体
（[`gn_fast_reduce_2026-08-16`](../gn_fast_reduce_2026-08-16)，128–512 线程、pair-major pass 1）
只移植到了 **baseline** 的入口。树里没有 `group_norm_silu_delta_quantize_nhwc_fast`，
所以生产的 baseline 跑 fast reduction，生产的 MoDiff 跑 plain。我第一次做这张表时是 plain 对 plain，
得出 MoDiff 比 baseline **快** 1.04×——那不是生产配置。两个 baseline 变体都列在下面。

频次加权，20 个 UNet 形状、62 calls/step：

| kernel | ms/step | |
|---|--:|---|
| baseline GN 阶段 `_fast`（生产） | 6.62 | |
| baseline GN 阶段 plain | 13.51 | fast-reduce 在这里值 **2.04×** |
| MoDiff GN 阶段（delta + `a_hat`） | 11.32 | 对 baseline `_fast` 是 **0.585×** |
| baseline conv (D1) | 21.86 | |
| MoDiff conv (D2) | 22.06 | 对 D1 是 **0.991×** |

| 2-kernel 路径 | ms/step | vs fp16 | vs baseline |
|---|--:|--:|--:|
| fp16 | 45.30 | 1.000× | 0.629× |
| **baseline int8（生产）** | 28.49 | **1.590×** | 1.000× |
| baseline int8, GN_FAST=0 | 35.38 | 1.281× | 0.805× |
| **MoDiff int8（生产）** | 33.38 | **1.357×** | **0.853×** |
| MoDiff blockwise G=64 | 66.60 | 0.680× | 0.428× |
| MoDiff blockwise G=32 | 119.71 | 0.378× | 0.238× |

所以在生产配置下 **MoDiff 的两-kernel 路径是 baseline 的 0.853×**，
即慢 17%。conv 基本打平（0.991×，
和 [`conv_kernel_sweep`](../conv_kernel_sweep_2026-08-28/FINDINGS.md) 对融合 EVT 测到的 0.966× 一致）。
那 4.9 ms 的差距**全部**在 GN 阶段。

**而这个差距是字节数，不是调优。** GN 阶段的有效带宽，对 A40 的 696 GB/s：

| kernel | 字节/输入元素 | ms | GB/s | 占峰值 |
|---|--:|--:|--:|--:|
| fp16 GN+SiLU | 4（读2 + 写2） | 14.64 | 179 | 26% |
| baseline plain | 3（读2 + 写1） | 13.51 | 145 | 21% |
| baseline `_fast` | 3（读2 + 写1） | 6.62 | 296 | 43% |
| **MoDiff delta** | **7**（读2 x + 读2 `a_hat` + 写1 + 写2 `a_hat`） | 11.32 | **405** | **58%** |

MoDiff 的 delta kernel 是四个里带宽效率**最高**的——58% 峰值，而 baseline `_fast` 只有 43%。
它在墙钟上输，纯粹因为时序缓存迫使它每元素搬 **7 字节而不是 3**：必须读 `a_hat` 再写回去。
这是方法本身的代价，不是漏掉的优化。

这也就框住了上限。把 M1 拉到满带宽 696 GB/s，它需要 6.58 ms，
MoDiff 路径变成 28.64 ms =
**baseline 的 0.995×**——刚好打平。所以：

> **MoDiff 的逐步 kernel 在任何调优水平下都赢不了 baseline。** 完美的 GN 阶段只能打平。
> MoDiff 的速度理由必须建立在**整块跳过计算**（replay）上，这也正是
> [`cache_schemes_report_2026-08-28`](../cache_schemes_report_2026-08-28/BRIEF.md) 测到的
> replay 2.65× / skip ~1.00×。它逐步的理由是质量理由：和 baseline 同一速度档，但激活位宽更低。

![融合对](plots/fig9_fused_pair.png)

#### 撤回："把 fast-reduce 移植到 delta kernel"是错的

本节早先的版本建议把 fast-reduce 移植到 MoDiff 的 delta kernel，并估了最多 4.7 ms/step。
这个建议撤回——两半都错。[`scripts/gn_decomposition.py`](scripts/gn_decomposition.py)。

**fast-reduce 是什么。** 就是 GroupNorm 组统计量那趟 reduction 的一个 block-size 策略，别无其他
（[`csrc/gn_block_size.h`](../../csrc/gn_block_size.h)、
[`baseline/norm/group_norm_silu.cu:552`](../../csrc/baseline/norm/group_norm_silu.cu:552)）：

| | block_size |
|---|---|
| generic | 32，翻倍直到覆盖 `group_size`，上限 1024 |
| fast | 128，当 `block_size*12 < group_size` 时翻倍，上限 512 —— 约每线程六对 |

两边 grid 都是 `N * num_groups`；数学完全相同，只有 reduction **顺序**变了。收益纯粹是 occupancy——
一个 group 填不满 1024 线程时那是灾难性的——所以形状越小收益越大（8×8 和 4×4 上 4.5–4.9×，2×2 上 1.1×）。

**为什么 MoDiff 不需要它。** 生产的 delta kernel 根本不走 group-major 分解：`gn_launch_group_stats`
默认走 **channel-major**（`BLK = C/K`），它的 block size 与 batch 无关，所以没有 fast-reduce 要修的那个
occupancy 问题。带同一套 fast 启发式的 group-major delta kernel **已经存在**——
`group_norm_silu_delta_quantize_nhwc_fused`，用 `MODIFF_GN_GROUPMAJOR=1` 可达——并且作为
"已测为回退"的死代码留在树里。在 20 个真实形状、B=128 上正面对比：

| GN 阶段变体 | ms/step | |
|---|--:|---|
| channel-major（生产） | 11.14 | |
| group-major + fast-reduce | 21.91 | **0.508×** —— 移植版慢约 2× |

这既确认了已提交的"回退"判断，也和带宽表早就暗示的一致：MoDiff 的 GN kernel 在 58% 峰值，
baseline `_fast` 只有 43%，它从来不是没调好的那个。**对 baseline 那 4.9 ms 的差距是每元素 7 vs 3 字节，
任何 reduction 策略的改动都碰不到它。** `gn_block_size.h` 从另一个方向得出同样结论，并写在头注里：
该策略对 MoDiff 默认关闭，而且"前提已被驳倒，应保持关闭"。

**唯一还剩的东西很小。** group-major 在很小的空间尺寸上确实赢（2×2 和 4×4 上 1.05–2.6×），
而 `HW <= 16 → group-major` 在这组形状上把两者分得干干净净。逐形状分派值
0.23 ms/step（1.021×）——真实、可复现，但大概不值一个分派分支，
因为它帮到的恰好是对总数几乎没有贡献的那些形状。


blockwise 顺带一提（各自套在自己的 conv 上）：MoDiff G=64 0.680× fp16、
G=32 0.378×；baseline G=64 0.723×、
G=32 0.388×。两边都输给 fp16，和之前一致。注意 baseline 的 blockwise 行
只是**计时代理**：D1 是写出而不是累加，nb 次调用不会求和、结果是错的——正确的 baseline blockwise
需要一个会累加的 D1 或者额外一趟 reduction，只会更贵。

### 到底在量什么：三个 kernel，不是一个

这棵树里一个 ResBlock conv 是**三个阶段**，而只有 int8 arm 三个都要付：

| | 阶段 | kernel |
|---|---|---|
| K1 | GroupNorm + SiLU | `group_norm_silu_nhwc` |
| K2 | 对 `a_hat` 求 delta 并量化、写 `a_hat` | `step1_static_quantize_fprop_silu` —— **仅 int8** |
| K3 | conv | `conv2d_int8_evt_o_hat`，fp16 arm 是 `F.conv2d` |

生产路径把 K1+K2 融成 `group_norm_silu_delta_quantize_nhwc`，这也是
[`conv_layer_microbench`](../cache_schemes_report_2026-08-28/scripts/conv_layer_microbench.py)
把 conv path 记成两个 kernel 的原因。两种拆法都量了。
[`scripts/path_kernels.py`](scripts/path_kernels.py)。

频次加权，20 个 UNet 形状、62 calls/step：

| kernel | ms/step |
|---|--:|
| K1 GN+SiLU（两个 arm 都有） | 14.05 |
| K2 quantize（仅 int8） | 8.19 |
| K1+K2 融合（生产） | **11.23** —— 省 11.02 |
| K3 conv, int8 | 21.59 |
| K3 conv, fp16 | 30.83 |

| 路径 | ms/step | **vs fp16 路径** |
|---|--:|--:|
| fp16 (K1 + K3) | 44.89 | 1.000× |
| int8，三个 kernel (K1+K2+K3) | 43.84 | **1.024×** |
| int8，K1+K2 融合（生产） | 32.82 | **1.368×** |
| int8 blockwise G=64 | 66.47 | 0.675× |
| int8 blockwise G=32 | 119.63 | 0.375× |

三点，第一点是对本文档自己之前说法的修正：

1. **只看 conv 会高估路径 speedup。** conv 单独是 1.43×，
   但路径上诚实的数字是 **1.37×** —— int8 还得跑 K2，fp16 不用。
2. **不做 GN+quantize 融合的话，int8 几乎赢不了 fp16：只有 1.02×。**
   K2 要 8.19 ms，而 conv 那边只赢 9.24 ms，
   量化那一步几乎把收益全吃掉。融合不是"在已有收益上再优化"——**它本身就是收益的来源**。
3. **融合后的 K1+K2 比 fp16 的 K1 还便宜**（11.23 vs 14.05 ms）。
   这一阶段是带宽瓶颈，融合 kernel 写出的是 1 字节的码，而 fp16 的 GN 要写 2 字节的 `normed`，
   所以量化连 norm 那步一起加速了。1.37× 里有很大一部分来自这里。

对 blockwise 来说，路径视角比纯 conv 视角**稍微没那么惨**（这里 0.68×/0.38×，纯 conv 是 0.56×/0.29×），
只是因为 K1+K2 不随块数翻倍。但在每个 G 上它都还是输给 fp16。**而且这些 blockwise 总数是下界的下界**：
真实实现需要 K2 输出逐块 absmax 而不是一个标量，那笔额外 reduction 没有建模 —— K2 是按原样计时的。

![路径 kernel](plots/fig8_path_kernels.png)

**只看 conv kernel（K3）对 fp16 的 speedup**（量化的意义就在这里）。fp16 基准 = channels-last fp16 的 `F.conv2d`、
开 `cudnn.benchmark`，即这棵树在
[`kernel_speedup.py`](../bench_report_2026-08-13_postzp/scripts/kernel_speedup.py) 里用的
`torch_conv2d_fp16` 约定。

| | ms/step | **vs fp16** | vs int8 per-tensor |
|---|--:|--:|--:|
| fp16 | 31.11 | 1.000× | 0.692× |
| **int8 per-tensor（现行）** | 21.51 | **1.446×** | 1.000× |
| int8 blockwise G=64 | 55.45 | **0.561×** | 0.388× |
| int8 blockwise G=32 | 108.77 | **0.286×** | 0.198× |
| int8 blockwise G=16 | 215.57 | **0.144×** | 0.100× |

> per-tensor int8 对 fp16 是 **1.45×**。改成 blockwise 不只是把这 1.45× 还回去，而是变成
> **比 fp16 慢 1.8×–6.9×**。测过的每一个块大小上，blockwise int8 conv 都比干脆不量化更慢。

对 fp16 的盈亏平衡点在 G=64 以上：能覆盖全部形状的最粗那一档就已经输给 fp16 了。逐形状看，
只有少数空间大的 conv 在 G=128 还站在 1.0× 以上（如 `384->384 32x32` 的 1.30×）。

1.446× 这个基准数有两点要注意：

* 它**低于已提交的 conv suite 1.78×**（[`KERNEL_SPEEDUP.md`](../bench_report_2026-08-13_postzp/KERNEL_SPEEDUP.md)），
  而那份文档自己解释了为什么它偏乐观：它 fp16 arm 的 12 条 conv 记录激活是 fp32 进来的，
  autocast 的 fp32→fp16 转换被算进了计时区间。这里两边都喂 fp16 channels-last，是纯算术比较。
* 这里的 int8 arm 是 **MoDiff 的 `o_hat` RMW** conv（要读写 `o_hat`）。已提交的 sweep 把它定在
  baseline int8 EVT 的 0.966×，所以非 MoDiff 的 int8 conv 在这个 harness 里大约是对 fp16 1.50×。

**对 int8 per-tensor：**

| | ms/step | vs fused | 慢多少 |
|---|--:|--:|--:|
| fused per-tensor（现行） | 21.51 | 1.000× | — |
| G=64 | 55.45 | **0.388×** | 2.58× |
| G=32 | 108.77 | **0.198×** | 5.06× |
| G=16 | 215.57 | **0.100×** | 10.02× |

`Cin` 必须能被 G 整除，所以只有 G∈{16,32,64} 覆盖全部形状（192 和 576 不被 128/256 整除）；
更粗的 G 逐形状测了但不进加权总和。**这已经是下界**（切块拷贝没计入计时）。

### 随 (B, N, H, W) 怎么变

一个加权总数看不出趋势，所以按已提交 sweep 的同一个默认点 `B=128, N=384, H=16, W=16` 逐轴单独扫。
[`scripts/axis_sweep.py`](scripts/axis_sweep.py)。`N` 同时是 Cin 和 Cout。

![逐轴扫描](plots/fig7_axis_sweep.png)

**B — batch**（N=384, H=16, W=16）

| B | fp16 ms | int8 ms | bw G=64 ms | bw G=32 ms | **int8/fp16** | bw64/fp16 | bw32/fp16 |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 8 | 0.070 | 0.040 | 0.104 | 0.207 | 1.77× | 0.68× | 0.34× |
| 16 | 0.115 | 0.079 | 0.199 | 0.395 | 1.46× | 0.58× | 0.29× |
| 32 | 0.193 | 0.130 | 0.382 | 0.747 | 1.48× | 0.50× | 0.26× |
| 64 | 0.353 | 0.238 | 0.656 | 1.288 | 1.49× | 0.54× | 0.27× |
| 128 | 0.708 | 0.457 | 1.158 | 2.278 | 1.55× | 0.61× | 0.31× |
| 256 | 1.409 | 0.879 | 2.195 | 4.280 | 1.60× | 0.64× | 0.33× |

**N — 通道数 (Cin=Cout)**（B=128, H=16, W=16）

| N | fp16 ms | int8 ms | bw G=64 ms | bw G=32 ms | **int8/fp16** | bw64/fp16 | bw32/fp16 |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 128 | 0.104 | 0.078 | 0.161 | 0.320 | 1.33× | 0.65× | 0.33× |
| 192 | 0.206 | 0.209 | 0.408 | 0.805 | **0.98×** | 0.50× | 0.26× |
| 256 | 0.340 | 0.232 | 0.549 | 1.088 | 1.46× | 0.62× | 0.31× |
| 384 | 0.708 | 0.459 | 1.155 | 2.288 | 1.54× | 0.61× | 0.31× |
| 512 | 1.277 | 0.763 | 2.008 | 3.914 | 1.67× | 0.64× | 0.33× |
| 768 | 2.920 | 1.643 | 4.376 | 8.577 | 1.78× | 0.67× | 0.34× |
| 1152 | 6.533 | 3.490 | 9.690 | 19.136 | 1.87× | 0.67× | 0.34× |
| 1536 | 11.585 | 6.158 | 17.009 | 33.335 | 1.88× | 0.68× | 0.35× |

**H — 高 (W=16)**（B=128, N=384, W=16）

| H | fp16 ms | int8 ms | bw G=64 ms | bw G=32 ms | **int8/fp16** | bw64/fp16 | bw32/fp16 |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 2 | 0.114 | 0.079 | 0.201 | 0.397 | 1.45× | 0.57× | 0.29× |
| 4 | 0.194 | 0.139 | 0.395 | 0.731 | 1.40× | 0.49× | 0.27× |
| 8 | 0.356 | 0.243 | 0.660 | 1.291 | 1.46× | 0.54× | 0.28× |
| 16 | 0.718 | 0.469 | 1.167 | 2.274 | 1.53× | 0.62× | 0.32× |
| 32 | 1.457 | 0.891 | 2.217 | 4.332 | 1.64× | 0.66× | 0.34× |

**W — 宽 (H=16)**（B=128, N=384, H=16）

| W | fp16 ms | int8 ms | bw G=64 ms | bw G=32 ms | **int8/fp16** | bw64/fp16 | bw32/fp16 |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 2 | 0.114 | 0.079 | 0.204 | 0.397 | 1.44× | 0.56× | 0.29× |
| 4 | 0.195 | 0.137 | 0.392 | 0.737 | 1.42× | 0.50× | 0.26× |
| 8 | 0.358 | 0.242 | 0.662 | 1.290 | 1.48× | 0.54× | 0.28× |
| 16 | 0.715 | 0.469 | 1.172 | 2.283 | 1.52× | 0.61× | 0.31× |
| 32 | 1.505 | 0.892 | 2.225 | 4.309 | 1.69× | 0.68× | 0.35× |

加权总数看不到的四点：

1. **blockwise 在任何轴上都没赢过 fp16。** G=64 全程落在 0.49–0.68×，G=32 在 0.26–0.35×，
   跨所有 B/N/H/W 都是。这个结论不依赖形状 —— 参数空间里没有一个角落让 blockwise 划得来。
2. **int8 的收益随通道数增长**：N=128 时 1.33×，单调升到 N=1536 的 1.88×。int8 conv 更吃张量核，
   GEMM 越"胖"它越占便宜。
3. **N=192 是个坑：int8 只有 0.98×，输给 fp16。** 这正是
   [`conv_kernel_sweep_2026-08-28`](../conv_kernel_sweep_2026-08-28/FINDINGS.md) §3 发现的
   半空 N-tile（`N=192` 和 `N=256` 一样贵、97 TFLOPS、全扫描最差点）—— 而且它足以把整个量化收益抹平。
   192 在这里是真实通道数：20 个 UNet conv 形状里有 5 个的 Cin 或 Cout 是 192，20 形状表也吻合
   （`192->192 32x32` 1.06×、`192->192 16x16` 1.00×）。
4. **B、H、W 几乎不影响。** 三者都是平到微升（1.40–1.77×），而且 H 和 W 表现完全一样 ——
   本来就该如此，两者都以 `B*H*W` 进入 GEMM-M。B=8 那点偏高（1.77×）是因为 fp16 在那里是
   launch-bound，不是 int8 变好了。

逐形状看，G=32 时慢 3.8×–6.7×，而且它跟的是 **G，不是块数**：`1536->768 2x2` 切 48 块、
`192->192 32x32` 切 6 块，却分别落在 0.167× 和 0.262×。所以"代价 = 块数 × fused，因为 epilogue
每块重跑一遍"这个朴素模型是**错的**。真实关系是 `nb × (一个 Cin=G 的独立 conv)`，而那个独立调用
同时吃两笔罚款。在 `768->768 8x8`、G=32、nb=24 上测：

| | µs | |
|---|--:|---|
| fused, K=6912 | 380.3 | |
| fused 的 1/nb（理想每次调用） | 15.8 | 免费切分该花多少 |
| Cin=G 独立 conv, K=288 | 117.3 | 理想值的 **7.4×** |
| 只有 epilogue（`o_hat` 大小的 fp16 RMW，无 GEMM） | 68.1 | 每次调用的 **58%** |

所以 epilogue 重跑是较大的一项，但只是勉强过半：另外 42% 是 **K 太薄的 GEMM** ——每次调用的
reduction 只有 `K = G*R*S = 288` 而不是 6912，摊不开 mainloop。两项的比例随形状变（空间大的
epilogue 主导，空间小、通道深的薄 GEMM 主导），这正是为什么总和上的倍数反而比较均匀。

参照：已提交的 conv-set 基准是 W8A8 full 32.47 ms/step、W4A4 full 21.47。那套 harness
（独立 L=8 chain）还包含 quantize 步，跟上面 21.51 ms 的纯 conv kernel 数字不可直接比；
blockwise 的倍数只作用在 conv 那部分，所以端到端一步的变慢会小于 5×。

![代价](plots/fig4_cost.png)

## 4. 决定性的一张表：哪个张量在付钱

| G（通道） | 256 | 128 | 64 | 32 | 16 | shipped |
|---|--:|--:|--:|--:|--:|--:|
| W8A8 **只激活**, r=4 | .0431 | .0452 | .0485 | .0510 | .0532 | **.0240** |
| W8A8 **只激活**, r=1 | .0236 | .0230 | .0228 | .0227 | .0232 | **.0226** |
| W8A8 **只权重**, r=4 | .0268 | .0192 | .0228 | **.0139** | .0133 | .0258 |
| W4A4 **只激活**, r=4 | .2391 | .2390 | .2399 | .2403 | .2415 | **.2791** |
| W4A4 **只权重**, r=4 | .2760 | .2725 | .2608 | **.2419** | .2416 | .2801 |

横着读激活那两行：**是平的**。W8A8/r=1 全部落在 .0227–.0236，baseline .0226，跨度 .0009 而噪声
底 .0014——等于没有。W4A4 激活行也是平的（.2391–.2415），但整体比 baseline 低 .040，而这份收益
最粗的那一档就已经全给了，`token act`（每像素一个 scale、完全不分块）同样给到 .2383。

> **激活 blockwise 的收益不来自“块”。** W4A4 上那点收益是“离开 per-tensor”换来的，per-token 就够。

权重那两行确实有斜率且过了噪声底：W8A8 .0258 → .0139（1.9×），W4A4 .2801 → .2419（1.16×）。
G=16 再往下只多给 .0006 / .0003，都在噪声内。

![归因](plots/fig5_attribution.png)

## 5. 机制：紧的块 scale 吃不下 delta 增长

delta scale 由 `|delta|max` 算出后**保持 4 步**（`DELTA_REFRESH=4`）。被 clip 的 delta 码比例：

| | shipped | G=256 | G=128 | G=64 | G=32 | G=16 |
|---|--:|--:|--:|--:|--:|--:|
| W8A8, r=4 | 0.000% | 0.21% | 0.35% | 0.59% | 1.00% | 1.72% |
| W8A8, r=1 | 0% | 0% | 0% | 0% | 0% | 0% |

per-tensor scale 由全局最差的块决定，对其他块都偏松，窗口内 delta 长大还有余量；per-block scale
天生就紧，一长就 clip，块越细越紧越 clip。r=1 时 scale 取自当步 delta 自己的 absmax，clip 恒为 0
——所以整个效应的**符号随节奏翻转**。W4A4 网格本身太粗，粒度压过 clip，所以那边照样赢。

## 6. 建议

**块大小 G=32 通道**：唯一有斜率的那条曲线（权重）的拐点；G=16 只多给噪声级的 .0006/.0003 却再贵一倍；
32 整除所有 UNet 通道数（192/384/576/768/1152/1536），不需要 padding；32 个 int8 通道 = 32 B =
两次 16 B 向量访存，块边界不会切断 `uint4`；scale 元数据占 int8 权重字节的 0.7%（int4 1.4%）。

**范围：只做权重。** 这是收窄而不是坏消息——只需要 per-K-block 的**权重** scale 的 mainloop，比全
blockwise 小得多：权重 scale 是静态的、load 时就知道、可以排成合并访存；而激活块 scale 必须每步由
GN/delta-quantize kernel 产出再喂进 mainloop。

**不要上 split-K 版本。** 它不只是相对 per-tensor int8 贵 5.0×，整条三-kernel 路径落到 **0.38× fp16**——
这一层不量化反而更快。拿这个去换一个本来就比 W4A4 小 10 倍的 W8A8 误差项上的 1.9×，不值。可选项只有两个：

1. **先放着，改看 refresh 节奏。** 同进程同种子配对比较，三次运行里 r=1 都比 r=4 好
   （+.0033 / +.0033 / +.0014），符号一致但两次只是刚过 .0033 的跨进程噪声底，所以算
   **有希望但未确认**，动手前值得单独跑一次配对实验。它的吸引力在于这是个环境变量而不是 kernel，
   而且这棵树已经为了让逐步 refresh 变便宜做了 free absmax reporting。W4A4 上同一比较是
   +.0006 / −.0001 / +.0007，节奏没用。
2. **写一个融合的 blockwise-权重 mainloop。** 一次 epilogue，权重 scale 在 reduction 内按 K 块折进去。
   开销应是每个 K-tile 多一次 scale 读取，而不是 split-K 那 5.0×（既重跑 epilogue，又把 K 从 6912 砍到 288）。工程量与现有手搭的
   `ImplicitGemmConvolutionEVT` 相当（CUTLASS 4.6.1 没有 EVT-on-conv，这里也没有库可抄）。

**blockwise 救不了 W4A4。** 最好的 blockwise W4A4 是 .1495，而 W8A8 是 .0259——还差 5.8×。
代价上，拿已提交的 conv-set 数字（W8A8 full 32.47、W4A4 full 21.47）套上实测的 split-K 倍数，
blockwise W4A4 在 G=32 约是纯 W8A8 的 3.3×、G=16 约 6.6×。这个倍数是在 **int8** conv 上测的，
int4 的 split-K 罚款没测，所以 W4A4 的代价是外推。要低比特激活，粒度不是那个杠杆。

## 7. 未解

- **W4A4 上 absmax 权重端到端打赢 mse 权重**（G=16: .1495 vs .1827），尽管 mse 的权重 Frobenius
  误差更低（.1185 vs .1319）。“重建误差更低 → 端到端更好”在这里反了，而现行 int4 clip search 正是
  建立在这个前提上。需要单独查。
- W4A4 权重/激活的超加性（1.16× × 1.17× → 1.53×）测到了但没解释。
- 全部只有 relL2，没跑 FID。W8A8 的效应贴着噪声底，这个样本量的 FID 也分不开；W4A4 的
  .279 → .150 够大，值得补一次 FID，但没做。
- MoDiff 的 GN 阶段已在 A40 带宽的 58%，它对 baseline 那 4.9 ms/step 的赤字是时序缓存强制的
  7 vs 3 字节。没有已知杠杆能补上——fast-reduce 已查证并驳倒（§3）。能补上的是**根本不搬 `a_hat`**，
  也就是 replay 在做的事。
- `a_hat` blockwise 是免费的（§1）却没测。鉴于激活粒度是平的，它大概率无关，但它是唯一零成本的 blockwise。

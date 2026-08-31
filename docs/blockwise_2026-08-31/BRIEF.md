# Blockwise 量化简报

NVIDIA A40 · LSUN-Churches LDM-8 · 50 步 DDIM · n=24 · 3 个种子。全文 **G 一律指输入通道数**：
权重块 = G 个通道 × 全部 9 个 tap，激活块 = 一个 (n,h,w) 上的 G 个通道。两者共享同一组 C 边界。

结论一句话：

> **G=32 通道**是合适的块大小，但**只值得用在权重上**。激活做 blockwise 在所有块大小上都没有收益
> （曲线对 G 完全平），在现行 refresh 节奏下反而更差。权重 blockwise 在 W8A8 上值 ~1.9×。
> 但现有 epilogue 无法承载任何 blockwise，唯一能跑的实现在 G=32 时让 conv 路径慢 **5.0×**——
> 这是一个 mainloop 工程，不是一个开关。

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

| | ms/step | vs fused | 慢多少 |
|---|--:|--:|--:|
| fused per-tensor（现行） | 21.51 | 1.000× | — |
| G=64 | 55.22 | **0.390×** | 2.57× |
| G=32 | 108.51 | **0.198×** | 5.04× |
| G=16 | 215.33 | **0.100×** | 10.01× |

`Cin` 必须能被 G 整除，所以只有 G∈{16,32,64} 覆盖全部形状（192 和 576 不被 128/256 整除）；
更粗的 G 逐形状测了但不进加权总和。**这已经是下界**（切块拷贝没计入计时）。

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

**不要上 split-K 版本。** 拿 conv 路径 5.0× 去换一个本来就比 W4A4 小 10 倍的 W8A8 误差项上的
1.9×，不值。可选项只有两个：

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
- `a_hat` blockwise 是免费的（§1）却没测。鉴于激活粒度是平的，它大概率无关，但它是唯一零成本的 blockwise。

# 融合 GN→QKV 这条路已经被生产路线超越了

**GPU** NVIDIA A40 · **Batch** 128 · **模式** int8_baseline · **Commit** `85ad97d`
**脚本** [`gn_qkv_route_ab.py`](scripts/gn_qkv_route_ab.py)、[`check_output_degeneracy.py`](scripts/check_output_degeneracy.py)、[`trace_route_dispatch.py`](scripts/trace_route_dispatch.py)
**数据** [`data/gn_qkv_route_ab.json`](data/gn_qkv_route_ab.json)

结论：**不要打开 `MODIFF_FUSE_GN_QKV_INT8`，也不要给 INT4 做对应变体。** 实测四条路线，生产
路线在每个 shape 上都最快。这条建议（上一轮对测量报告的分析里排第 ①）是错的，下面是推翻它的数据。

---

## 1. 两个 flag 在稳态下根本走不到

`forward` 的分派顺序（[quantized_std_attention.py:799-833](../../integration/fused_ops/quantized_std_attention.py:799)）：

```
_int4_layout_epilogue_forward   -> return        (INT4，全部 shape)
_int4_qkv_epilogue_forward      -> return
_int8_qkv_epilogue_forward      -> return        (INT8，_fq_frozen2 之后全部 shape)
if self._route1: ...                             <- MODIFF_ROUTE1，到不了
if self._fuse_gn_qkv_i8: ...                     <- MODIFF_FUSE_GN_QKV_INT8，到不了
```

INT8 稳态一律在第三个分支返回，INT4 在第一个。两个 opt-in 分支只在标定窗口内（`_fq_frozen2`
还是 False 时）可达。所以「把 flag 打开」在生产配置下是空操作 —— 必须同时关掉
`MODIFF_INT8_QKV_EPILOGUE` 才能观察到它们。

源码注释里的 1.37×/1.15× 来自 `7b5b431`（**2026-07-23**），而 W8A8 QKV int8-epilogue 路线是
一周后（07-30）才落地的。那两个数字对照的是当时的基线，不是今天的生产路线。

## 2. 四路线同进程实测

同一个 module 实例，按 route 轮转顺序计时（每轮换头，8 轮 × 60 iters，12 轮预热），
所以时钟爬升和热漂移平摊到每条路线。CV 全部 ≤ 0.71%。

| shape | 实例 | P 生产 | R1 `MODIFF_ROUTE1` | A `MODIFF_FUSE_GN_QKV_INT8` | N 07-30 前基线 |
|---|--:|--:|--:|--:|--:|
| C192/32² T=1024 | 5 | **2722.8** | 2892.0 (0.942×) | 3020.6 (0.901×) | 2970.5 (0.917×) |
| C384/16² T=256 | 5 | **884.9** | 1085.5 (0.815×) | 1172.8 (0.754×) | 1001.7 (0.883×) |
| C384/8² T=64 | 5 | **263.4** | 不合格 | 不合格 | 299.2 (0.880×) |
| C768/4² T=16 | 5 | **221.7** | 不合格 | 不合格 | 232.1 (0.955×) |
| C768/2² T=4 | 1 | **118.4** | 不合格 | 不合格 | 187.4 (0.632×) |

「不合格」= 融合路线要求 `T % 128 == 0 且 C % 8 == 0`，只有 T=1024/256 满足。

生产路线全胜。而且 **A 相对今天的 N 基线是 0.983×（T=1024）和 0.854×（T=256）** —— 那个 1.37×
连对着旧基线都复现不出来了，因为 `_qkv_from_gn` 自己在这期间也融了（GN+quantize 已经并成一个
kernel），旧基线早就不是当初那个旧基线。

## 3. 为什么会输：两个融合互斥

T=1024 的 kernel 组成（µs/layer call）：

| | P 生产 | R1 |
|---|--:|--:|
| GN 统计 / GN+quantize | `group_norm_silu_quantize` **257.0** | `gn_accum` 108.0 + `gn_finalize` 2.6 + Fill 3.4 |
| QKV | `gemm_w8a8_kernel_awq_out_i8` **610.0** | `ImplicitGemmConvolutionFusionPerSampleEVT` **580.1** |
| K/V gather + transpose | — （融进 QKV epilogue） | `from_i8_kv_tiled_kernel` **307.5** |
| **norm→QKV 小计** | **867.0** | **1001.6** |
| flash | 1364.2 | 1410.7 |
| out proj (+bias+res) | 344.4 | 345.0 |

融合本身是有效的：R1 的 norm+QKV 段是 694.1 µs，比 P 的 867.0 快 **1.25×** —— 注释里那个方向
是真的。输的原因在下一行：生产路线的 `gemm_w8a8_awq_qkv_i8_layouts` 把 **Q/K/Vt 三个 layout 的
生成融进了 GEMM epilogue**，整段 K/V gather+transpose 直接不存在；R1 融了 GN 就得把这 307.5 µs
赔回来，赔的比省的多。

两者互斥的原因是结构性的：GN 融合要求 per-sample 的 scale/bias 在 mainloop 里施加，这只有
CUTLASS 的 `ImplicitGemmConvolutionFusion` 做得到（一个 **conv**）；而它的 epilogue 是
`LinearCombination`，没法同时吐出 attention 要的三种 layout。

T=256 更糟：fp16 的 fused conv（383.5）本身就慢过 P 的 int8 GEMM（311.2），因为这个 shape 已经
够 compute-heavy，int8 tensor core 的 2× 吞吐吃得到，再加 129.6 的 K/V gather，直接 0.815×。

## 4. 剩下的真实余量：约 1.2%，需要一个新 kernel

要同时拿到两个融合，需要一个新东西：**int8 GEMM，prologue 施加 per-sample GN scale/bias，
epilogue 直接吐 Q/K/Vt 三种 layout**。乐观估计（假设它能达到 fp16 fused conv 的 580.1 µs 并且
layout 输出像 P 一样近乎免费）：

- T=1024：867.0 → 约 694 µs，省 ~173 µs/call
- ×5 个 block ×200 步 = **~173 ms**，占 INT8 端到端 14164 ms 的 **1.22%**
- T=256：融合版反而慢 45 µs/call，没有余量

为 1.2% 写一个带 GN prologue 和三路 layout epilogue 的 CUTLASS int8 kernel，不划算。**建议放弃
①。** 上一轮把它排第一是因为把注释里的 1.37× 当成了现值。

## 5. 顺带发现：本树里 attention 层的数值校验是空的

四条路线跑的是**完全不同**的 kernel（`trace_route_dispatch.py` 逐 entry point 确认），输出却
25M 个元素全部 bit 相同。原因不是巧合：

- `AttentionBlock.proj_out` 是 `zero_module`（[openaimodel.py:345](../../ldm/modules/diffusionmodules/openaimodel.py:345)）
- stub checkpoint 的 `state_dict` 是空的、`strict=False`，所以 `proj_out` 权重**保持全零**
  （实测 `proj.qweight` 49152 个元素全 0，bias 全 0）
- 于是 AttentionBlock 的输出 **bit-exact 等于它的输入**（实测 `torch.equal(out, x) == True`）

也就是说本树里每个 attention block 都是恒等映射，attention 内部算什么都不影响输出。推论：

- 任何在**层级或端到端**层面校验 attention 改动的检查都是恒真的 —— 包括
  `int8_hd24_layer_ab.py` 的 `output_bit_exact`，以及 `INT8_QKV_EPILOGUE_RECHECK_2026-07-30.md`
  里那句「Candidate versus previous INT8 latent relative L2 is 0 for every seed」。它们对任何
  改动都会通过，不构成证据。
- 仍然有效的是 kernel 级的合成张量对照（`qattn_correctness.py` 那类，对 fp32 参考跑 9 个
  shape，INT8 相对误差 0.0076–0.0161）。attention 的正确性靠的是这一层，不是端到端那句。
- **计时不受影响**：这些 kernel 的开销与数据无关，shape 和 launch 序列都是真的。本文以及测量
  报告里的性能数字照样成立。

测量报告已经声明过「所有权重随机初始化，本报告没有任何图像质量measurement」
（[MEASUREMENT_REPORT_2026-08-01.md:783](../MEASUREMENT_REPORT_2026-08-01.md:783)）；这里是它更
锋利的版本 —— 不只是「质量没测」，而是 attention 的层级/端到端数值校验**在结构上不可能失败**。

### 5b. 追查下去：问题比 attention 大得多

修的时候发现 attention 只是表层。把 21 个 attention proj 全部激活以后，latent **仍然**逐位不变。
逐层排查（[`debug_activation.py`](scripts/debug_activation.py)、
[`trace_route_dispatch.py`](scripts/trace_route_dispatch.py)）定位到真正的原因：

**`UNetModel.out[-1]` 也是 `zero_module`**（[openaimodel.py:745](../../ldm/modules/diffusionmodules/openaimodel.py:745)）。
实测 `unet.out[2].weight` 6912 个元素全 0，**UNet 对任何输入的 ε 预测恒为零**（`absmax 0.0`）。
所以采样 latent 只由初始噪声和 DDIM schedule 决定 —— **本树里任何基于 latent 的检查，对 UNet 中
任何位置的任何改动都是恒真的**，不只是 attention，conv / GroupNorm / 量化全都一样。

铁证：把全部 21 个 attention block 的输出替换为常数，`forward` 触发 **420 次**，latent 仍然
**逐位相同**；`integration/tests/golden/` 里 07-27 存下的五个 golden（fp16 / int8 / int8_baseline /
int4 / int4_baseline）**互相逐位相同**，absmax 全是 141.61083984375。fp16 和 int4 逐位相同不是
保真度结论，是被比较的量根本不依赖被测对象。

还有第二个独立的坑：**stub 的权重每个进程都不一样**。checkpoint 是空的，所有权重来自默认初始化，
而 torch 的全局 RNG 是每进程非确定性播种的（实测两次 `torch.initial_seed()` 分别是
7152681835639687281 和 12024912460555673961，qkv 权重完全不同）。零输出掩盖了这一点；一旦激活，
逐位相同的代码重跑 golden 会报 rel_err ≈ 0.4。所以修复必须**同时**做两件事：构建前播种、构建后激活。

### 5c. 修了什么，以及哪一类修不了

新增 [`integration/utils/attention_identity_guard.py`](../../integration/utils/attention_identity_guard.py)：

| 函数 | 作用 |
|---|---|
| `zeroed_modules(model)` | 找出所有会吞掉输入的零权重模块（含 `QuantLinearWxAx` 与 `OptimizedInt8/4Conv2d` 的各自存储格式） |
| `assert_unet_output_observable(unet)` | **行为式**断言：UNet 预测恒零就抛异常。这条骗不过去 |
| `assert_attention_observable(model)` | 结构式断言，便宜 |
| `seed_model_construction()` | 构建前播种，让随机网络可复现 |
| `activate_zeroed_modules(model)` | 就地恢复 `zero_module` 抹掉的默认初始化 |
| `prepare_for_comparison(model)` | 激活 + 断言，两行接入 |

实现上有三个坑必须踩过（都已处理，见文件内注释）：**(1)** 必须在 `torch.inference_mode()` 外调用，
否则 autocast 的 fp16 权重缓存会让写入在整个 region 内被忽略；**(2)** 每个模块的随机种子取自
**规范化名字**而非单一随机流 —— fp16 有 57 个待激活模块、量化模式有 92 个，共用一个流会让同一层在
不同模式拿到不同权重；**(3)** 量化模块必须**先抽 fp 权重再量化它**，不能直接抽整数 code，否则
fp16 和 int4 拿到的是毫无关系的两组权重。

验证（[`verify_guard.py`](scripts/verify_guard.py)，三个模式全过）：断言在原始模型上都抛异常 →
激活 57/92 个模块 → 断言通过 → 破坏 attention 输出后 latent 动了（relL2 0.27–0.31）。

**同模式 A/B 等价性检查：修好了。** `e2e_output_check.py` 实测：代码不变三个进程都是
`rel_err=0.0000`（修前是「任何改动都 0」）；换掉 attention 路线（`MODIFF_FLASH_GATE=off`）得到
`0.0011` —— 能分辨了，且正确判定在 2% 容差内。加 `--no-activate-zeroed` 则同样的改动是
`0.0000`，完全瞎。

**跨模式（fp16 vs 量化）精度比较：激活救不了，已改为如实标注。** 因为
`integration/calibration/*.pt` 是对着**未激活**的网络标定的，激活后每个激活值 scale 都是错的，
结果被 scale 失配主导。实测单次 UNet forward 的 rel-vs-fp16：INT8 **0.84**、INT4 **0.55** ——
INT8 名义上比 INT4 还差，且两者都远大于量化误差该有的量级。要让这类比较有意义，必须对激活后的
网络重新标定。`test_std_attn_e2e.py` 现在明确声明自己只是**接线检查**，并把读者引向 kernel 级测试。

改动清单：

- `integration/utils/attention_identity_guard.py`（新）
- `integration/tests/e2e_output_check.py` — 接入 guard + 构建播种；golden 按激活状态分键；
  另外补了缺失的 `src/taming-transformers` 路径（**原来根本跑不起来**）
- `integration/tests/test_std_attn_e2e.py` — 改为接线检查并说明为何数字不可解读
- `integration/tests/golden/` — 五个 07-27 的 golden 重命名为 `*_vacuous.pt` 并加
  [`README.md`](../../integration/tests/golden/README.md) 说明；它们本身就是证据
- `docs/final_report_2026-07-28/scripts/` — `int8_hd24_layer_ab.py`（另修了一个让它无法运行的
  import 路径 bug）、`int8_hd24_exact_quality.py`、`int8_qkv_epilogue_quality.py`、
  `int4_optimization_quality.py` 全部接入 guard
- 五个 `*_2026-07-30.md` 文档顶部加了更正声明，撤回 latent 层面的证据、保留 kernel 层面的
- `MEASUREMENT_REPORT_2026-08-01.md` 的 caveat 加强

## 6. 环境

容器在 08-01 之后又被重置过一次：`omegaconf`、`einops`、`pytorch_lightning`、`tqdm`、
`matplotlib` 再次缺失。`pip install -r requirements.txt` 即可恢复 —— 里面已经钉了
`torchmetrics==0.6.0`，这个钉子是必须的：`pytorch-lightning==1.4.2` 会 import
`torchmetrics.utilities.data.get_num_classes`，新版 torchmetrics 已经把它删了。逐个手装依赖
（而不是走 requirements.txt）会漏掉这个约束并撞上 ImportError。

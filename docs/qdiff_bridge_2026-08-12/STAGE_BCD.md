# Stage B, C and D: the QKV shape, what warm-up costs, and dropping MoDiff from the Linears

Three of the advisor's asks from 2026-08-11, answered with measurements. A40, LSUN-churches.

---

# Stage B — "QKV 是 Convolution 不是 Linear，测一下这个 shape 行不行"

**He is right about the premise.** `openaimodel.py:337,345` build the attention block's `qkv` and
`proj_out` as **`nn.Conv1d` with kernel_size=1**, not `Linear`.
[token_major_attention.py:147-155](integration/fused_ops/token_major_attention.py:147) reshapes them
to `nn.Linear` — `[3C,C,1] → [3C,C]`, a pure axis drop, bit-identical — and `convert_linears_to_wxax`
then makes them `QuantLinearWxAx`.

**And the reshape is the right call**, by a wide margin. Weighted over all 21 blocks at batch 128:

| form | ms/step |
|---|---:|
| `nn.Linear` fp16 (what the tree does) | **6.997** |
| `nn.Conv1d` k=1 fp16 (the model's original form) | 30.334 |

cuDNN handles these 1×1 Conv1d shapes badly — 4.3× worse. Keeping the conversion is worth more than
anything else measured here.

## The answer to "行不行": the GEMM yes, the path no

| shape (M×K×N) | blk | pad | linear fp16 | int8 total | = quant | + gemm | tot/fp16 | **gemm/fp16** |
|---|--:|---|---:|---:|---:|---:|---:|---:|
| 131072×192×576 (qkv) | 5 | 192×640 **+11%** | 0.3840 | 0.5548 | 0.1431 | 0.4089 | 0.69× | 0.94× |
| 131072×192×192 (proj) | 5 | 192×256 **+33%** | 0.1924 | 0.2964 | 0.1405 | 0.1540 | **0.65×** | **1.25×** |
| 32768×384×1152 (qkv) | 5 | +0% | 0.4144 | 0.3220 | 0.0715 | 0.2477 | 1.29× | 1.67× |
| 32768×384×384 (proj) | 5 | +0% | 0.1307 | 0.1759 | 0.0714 | 0.1026 | 0.74× | 1.27× |
| 8192×384×1152 (qkv) | 5 | +0% | 0.0868 | 0.0998 | 0.0198 | 0.0785 | 0.87× | 1.11× |
| 8192×384×384 (proj) | 5 | +0% | 0.0593 | 0.0581 | 0.0199 | 0.0371 | 1.02× | 1.60× |
| 2048×768×2304 (qkv) | 5 | +0% | 0.0769 | 0.0718 | 0.0084 | 0.0616 | 1.07× | 1.25× |
| 2048×768×768 (proj) | 5 | +0% | 0.0421 | 0.0379 | 0.0083 | 0.0264 | 1.11× | 1.60× |
| 512×768×2304 (qkv, mid) | 1 | +0% | 0.0307 | 0.0247 | 0.0082 | 0.0214 | 1.24× | 1.44× |
| 512×768×768 (proj, mid) | 1 | +0% | 0.0330 | 0.0247 | 0.0086 | 0.0213 | 1.34× | 1.55× |

**Weighted over 21 blocks: int8 total 8.133 ms against fp16's 6.997 — 0.86×, a net LOSS.
But the int8 GEMM alone is 5.626 ms — 1.24×, a WIN.**

The difference is the standalone `quantize_act_int8` pass: **2.432 ms of the 8.133**, 30% of the int8
cost. On the C=192 projection it is 47% of the total and single-handedly flips a **1.25× GEMM win into
a 0.65× loss**.

## What this means

1. **The int8 tensor-core GEMM is fine.** It beats cuBLAS fp16 on 10 of 10 shapes when the quantize is
   excluded, by 0.94–1.67×. The shape is not the problem the advisor suspected it might be.
2. **The quantize pass is the problem**, and it is already solved elsewhere in this tree. The conv
   path never pays it — `step1_static_quantize_fprop` fuses quantize into the producer — and the
   landed int8-qkv fusion (`gemm_w8a8_awq_o_hat_out_i8`) removes it for the qkv on the 10 hd=48
   blocks by having the *previous* GEMM emit int8 directly. **Extending that to the 5 hd=24 blocks
   and to `proj` is where the remaining ~2.4 ms is**, and it is a fusion problem, not a kernel-tuning
   problem.
3. **Padding costs real work on exactly the worst blocks.** `wxax_linear.py:64-68` pads K to a
   multiple of 64 and N to a multiple of 128. Seven of ten shapes pad by 0%, but the five hd=24
   blocks — the most expensive tier — pad **576→640 (+11%)** on qkv and **192→256 (+33%)** on proj.
4. This independently supports dropping MoDiff from the projections: the int8 projection path is a
   net loss *before* MoDiff's delta traffic is counted at all.

Data: `data/qkv_shape_bench.json`. Caveat: the `conv1d` column is cuDNN-algorithm dependent and swung
between runs (the 2048×768 rows read 0.41/0.13 in one run and 1.60/1.57 in another); the Linear-vs-
Conv1d conclusion is robust to that, individual cells are not.

---

# Stage C — "Warmup 花了多少时间"

Three separate mechanisms. They answer differently, and only one is inside the reported ms/step.

## (1) Attention self-calibration — 8 forwards — **EXCLUDED**

`MODIFF_ATTN_CALIB_STEPS` (default 8). Every harness discards a full run before measuring
(`dynamic_delta_ab.py:107`, `differential_timing.py:286`), and nothing ever resets the freeze flags,
so this is **once per process, not once per sample**, and it is not in any reported number.

## (2) Conv MoDiff warm-up rounds — **INCLUDED, +1.27 ms/step at S=50**

`MODIFF_WARMUP_STEPS` (default 5) = 4 *extra* quantize+conv passes over all 70 convs, on t=T only.

| | forward 1 (t=T) |
|---|---:|
| W=1 | 43.1–47.9 ms |
| W=5 | 94.6–111.8 ms |
| **delta** | **+63.39 ms** (stdev 4.63, 4/4 pairs) |

The warm-up rounds are **57% of the t=T forward**. Amortized: **+1.27 ms/step at S=50**, or
+0.32 ms/step at S=200. Steps t<T are unaffected (median 26.7 vs 27.1 — a wash), which is the
expected shape of the result and a check that the knob did what it claims.

> **Two earlier attempts at this measurement failed, and the reason is worth recording.** Rebuilding
> per setting gave 27.6 / 19.3 / 27.9 ms/step for W=1/3/5 — *non-monotonic*, the middle setting 8 ms
> faster than both ends. A paired A/B on one model still gave deltas of +0.74 / −5.60 / +2.68 / +2.28,
> stdev 3.8 on a median of 1.5. The cause was **not** contention — the GPU was idle between runs. It
> is **clock ramp**: this A40 idles at 210 MHz and 50-step batch-8 runs are short enough to bounce
> between clock states. The fix was to measure where the effect is large (forward 1, where the rounds
> are actually paid) and divide, rather than to measure a 1% effect against 15% noise.

## (3) The post-freeze kernel switch — **essentially free**

This was the advisor's specific suspicion ("你可能要换kernel重新launch"): at the freeze boundary
`_ensure_route1`, `_qkv_inv_out_scale` and `_ensure_fused` fold weights and build scale vectors, and
from forward 9 onward a structurally different set of CUDA kernels runs.

| | ms | vs steady |
|---|---:|---:|
| forward 1 | 131.30 | **+103.73** |
| forwards 2–8 (calibrating route) | 32.30 mean | +4.73 / forward |
| forward 9 (**first frozen**) | 28.45 | **+0.88** |
| steady (forwards 10+) | 27.57 | — |

**The freeze boundary costs +0.88 ms, once.** Across three runs it measured −0.10, −0.18 and +0.88 —
i.e. indistinguishable from zero. The weight folding and scale-vector construction are real but
negligible.

Total one-off warm-up per process: **136.8 ms ≈ 5 forwards**, of which **forward 1 alone is 76%** —
and that is cuDNN autotuning and first-touch allocation, not the route change. All of it is excluded
from every reported ms/step.

> The decomposition matters. Reporting the lump (`calibrating_mean − steady = +17.15 ms/forward`)
> would attribute three quarters of a one-time allocation cost to the route change and make the
> advisor's hypothesis look confirmed when it is refuted.

## Answers, short

| ask | answer |
|---|---|
| How much does warm-up cost? | 136.8 ms once per process, **excluded** from ms/step; plus **1.27 ms/step at S=50** for the conv rounds, which **is** included |
| Is there a kernel-relaunch cost at the freeze? | **No** — +0.88 ms, once, within noise of zero |
| Is the biggest warm-up cost the calibration? | **No** — 76% is forward 1's cuDNN autotune |

Data: `data/warmup_cost.json`.

## Reproducing

```bash
python docs/qdiff_bridge_2026-08-12/scripts/qkv_shape_bench.py --batch 128 --iters 50
python docs/qdiff_bridge_2026-08-12/scripts/warmup_cost.py --steps 50 --batch 8 --pairs 4
```

---

# Stage D — flip `MODIFF_LINEAR` off by default

The advisor: *"如果说它会让它变慢，我们可以把 Linear 去掉，我以前试过，其实也还好。"* Done, in
`benchmark_ldm.py`. **This is a real trade, not a free win**, so both halves are here.

## The quality cost, re-measured on the corrected model

The condition for flipping was that the quality comparison be re-made after recalibration — W4A4 was
the setting where `MODIFF_LINEAR` was said to be visually load-bearing. DDIM 50, batch 8, 3 paired
seeds, latent relL2 vs a per-seed fp16 reference:

| configuration | ON (=1) | OFF (=0) | OFF/ON | seeds ON wins |
|---|---:|---:|---:|---|
| W8A8, qdiff scales | 0.0503 | 0.0612 | **1.22×** | **3/3** |
| W8A8, shipped scales | 0.0530 | 0.0606 | 1.14× | 1/3 |
| W4A4 | 0.3602 | 0.4176 | 1.16× | **3/3** |

**A hypothesis going in was wrong, and it is worth recording.** I expected MoDiff-on-Linear to be
partly *compensating* for the bad absmax activation scale, so that fixing the scale would shrink its
benefit. It did not — the benefit **grew** (1.14× → 1.22×, and from 1/3 seeds to 3/3). This also
reproduces the 2026-08-06 measurement closely (0.0607 → 0.0508 then, 0.0503 → 0.0612 now).

## What turning it off buys

* **1.059× → 1.371× vs fp16** — ~29% more throughput (differential timing, batch 128, 200 steps × 5
  repeats, `docs/current_state_2026-08-12`).
* **The fused int8-output epilogue on all 21 attention blocks.** `benchmark_ldm.py:751` asserted that
  MoDiff sets `_out_i8 False` and disables it; this run **measured** it, and the flip moves
  qout-eligible from **0/21 to 21/21**.
* Stage B above: the int8 projection GEMM path is **0.86× cuBLAS fp16 at these shapes anyway** — a net
  loss before MoDiff's delta traffic is counted at all.

## Why the trade is judged worth taking

At W8A8, **0.0612 is still inside the band a well-behaved 8-bit activation quantizer should occupy**
(`docs/modiff_correctness_2026-08-03` calls 0.0393 exactly that). So this is 18% of an already-small
error in exchange for 29% throughput and an unlocked fusion.

The "recognisable churches vs fog" concern was always about **W4A4**, and there *both* arms are bad
(0.36 and 0.42) — that configuration is not a recommendation either way.

`MODIFF_LINEAR=1` restores the old behaviour. The default should be revisited if the linear MoDiff
path ever gains a GEMM `o_hat`-accumulate epilogue, which would remove most of the speed cost.

## No committed measurement changes meaning

All four measurement harnesses set `MODIFF_LINEAR` **explicitly** per arm — `differential_timing.py`
(in its per-arm env dicts), `profile_layers_and_model.py:218`, `profile_blocks.py:218`, and
`dynamic_delta_ab.py` (which only mentions it in a comment). So no previously committed number
silently changes. Verified end to end: with the env var unset, 42 wxax linears build at
`modiff=False` and qout-eligible reads 21/21.

Data: `data/linear_modiff_ab.json`. `ms_per_step` in that file is indicative only — arms are built
sequentially in one process, and this A40 idles at 210 MHz (see Stage C).

```bash
python docs/qdiff_bridge_2026-08-12/scripts/linear_modiff_ab.py --steps 50 --seeds 3
```

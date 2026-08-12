# Current-state profile and benchmark, 2026-08-12

Absolute state of the tree as it stands. **No pre/post comparison in this document** — for the
verification that the `csrc/` split changed nothing, see `docs/postsplit_benchmark_2026-08-12/`.

All numbers measured 2026-08-12 on an idle **NVIDIA A40**, LSUN-churches LDM, **batch 128**, DDIM.
Data lives in `docs/postsplit_benchmark_2026-08-12/data/`; the figures here regenerate from it offline.

**Three instruments, three different scopes. They do not sum to one another** — that is a property of
the instruments, not an inconsistency:

| instrument | scope | what it is authoritative for |
|---|---|---|
| differential | whole model, profiler-free wall clock, 200 steps × 5 repeats | **e2e ms/step** |
| layer harness | CUDA events on live dispatch targets, covers 0.64–0.88 of the step | **shares** per layer/block |
| Perfetto trace | 8 steps, bucketed offline | **which CUDA kernel**, within one capture |

---

## 1. End to end, by configuration

![e2e](plots/00_e2e_by_arm.png)

| arm | ms/step | vs fp16 | CV% | spread% |
|---|---:|---:|---:|---:|
| fp16 (autocast) | 104.30 | 1.000× | 0.07 | 0.15 |
| **W8A8 PTQ** (no MoDiff anywhere) | **71.80** | **1.453×** | 0.38 | 0.98 |
| MoDiff conv only, K=4 | 76.09 | 1.371× | 0.25 | 0.57 |
| MoDiff conv only, K=1 | 81.66 | 1.277× | 0.16 | 0.43 |
| MoDiff conv+proj, K=1 *(the paper's datapath)* | 104.01 | 1.003× | 0.10 | 0.23 |
| MoDiff conv+proj, K=4 | 98.45 | 1.059× | 0.12 | 0.33 |
| … + projection refresh schedule *(opt-in)* | 95.68 | 1.090× | 0.13 | 0.33 |
| … + route (b) qkv int8→flash *(opt-in)* | **95.09** | **1.097×** | 0.25 | 0.60 |

**The headline fact about this configuration space:** plain W8A8 PTQ is the fastest thing here at
**1.453×**, and *every* MoDiff arm is slower than it. The paper's datapath (conv+proj, K=1) is
**1.003×** — i.e. it costs everything quantization gains and lands back at fp16. The two opt-ins claw
back 8.9 ms of that, reaching 1.097×, still 24.3 ms behind PTQ.

MoDiff's cost is concentrated in the **projections**: conv-only K=4 is 1.371×, and adding MoDiff to the
42 attention projections takes it to 1.059×. That is 22.4 ms/step for the projection delta path.

*Caveat on absolute values:* two runs in the same container agree to 0.21 ms; runs a day apart differ
by ~1.3 ms. Treat the third digit as session-relative, and take any difference smaller than that from a
paired A/B instead.

---

## 2. Per attention block — all 21

![blocks](plots/01_per_attention_block.png)

`W8A8 conv+proj`, batch 128. The 21 blocks fall into exactly **four shape tiers**, and cost tracks the
tier with almost no within-tier variation (≤ 0.04 ms):

| tier | blocks | idx | ms/step each | tier total | share |
|---|--:|---|---:|---:|---:|
| C192 T1024 hd24 | 5 | 0, 1, 18, 19, 20 | 5.29–5.32 | **26.55** | **62%** |
| C384 T256 hd48 | 5 | 2, 3, 15, 16, 17 | 2.23–2.26 | 11.22 | 26% |
| C384 T64 hd48 | 5 | 4, 5, 12, 13, 14 | 0.64–0.65 | 3.23 | 8% |
| C768 T16 hd96 | 6 | 6–11 | 0.18–0.33 | 1.85 | 4% |
| **total** | **21** | | | **42.85** | |

**Five blocks are 62% of all attention time.** They are the hd=24 tier — the same five that route (b)
cannot reach (the int8 gather needs `hd % 16 == 0`, and 24 bytes/token fails it), and the same five
where the 8-byte loader was measured 2.11× slower than the mma kernel. Any further attention work is
about these five blocks or it is about 38% of the budget.

The 6 blocks at hd=96 never run the custom flash at all (`_resolve_flash` requires `hd ≤ 48` and
`T % 64 == 0`), so they fall back to PyTorch SDPA — and cost 4% of attention, which is why that has
never been worth fixing.

### Per projection

42 linears, **8.75 ms/step total**, and they are strongly asymmetric within each block: the qkv
projection costs 0.0002–0.009 ms while the output projection costs 0.056–0.97. The largest six output
projections (blocks 0, 1, 18, 19, 20 at ~0.96 ms each) are again the hd=24 tier.

---

## 3. Per conv layer — all 70 called

![layers](plots/02_per_conv_layer.png)

| config | conv total | layers called | min | median | max |
|---|---:|--:|---:|---:|---:|
| W8A8 conv+proj | 39.96 | 70 | 0.003 | 0.263 | 3.316 |

Cost concentrates at the **high-resolution ends** — the input blocks and output blocks — while the
low-resolution middle is nearly free. The single most expensive conv is 3.316 ms (`conv130`, an output
block), 8.3% of all conv time on its own.

**70 of 140 quantized conv modules are never called during sampling.** This is unexplained and has been
open since 2026-08-11; `fusion_audit.py` reports `n_conv_modules` and `n_conv_layers_called` separately
because of it. Any per-layer conv work should target the 70 that actually run.

### By kind

| config | wall | coverage | conv | attn (score path) | proj (42 linears) | updown |
|---|---:|---:|---:|---:|---:|---:|
| W8A8 PTQ | 71.92 | 0.635 | 22.22 | 19.63 | 0.00 | 3.85 |
| W8A8 conv-only | 79.13 | 0.845 | 40.42 | 19.74 | 0.00 | 6.71 |
| **W8A8 conv+proj** | **101.73** | **0.880** | **39.96** | **34.09** | **8.75** | **6.67** |
| W8A4 conv+proj | 102.39 | 0.870 | 39.68 | 34.00 | 8.73 | 6.66 |
| W4A4 conv+proj | 95.44 | 0.866 | 28.50 | 22.46 | **27.01** | 4.65 |

Two structural facts fall straight out of this table:

* **W8A4 and W8A8 are the same datapath.** conv 39.68 vs 39.96, attn 34.00 vs 34.09. The activation
  width is a clamp, not a different kernel.
* **W4A4's projections cost 27.01 ms**, 3.1× W8A8's 8.75 — the int4 projections' `o_hat` traffic. This
  is why at W4A4 turning `MODIFF_LINEAR` on or off was the difference between recognisable churches and
  fog, while at W8A8/W8A4 it was visually indistinguishable.

**Read shares within a row, not totals.** 12–37% of the step sits outside the timed dispatchers
(ResBlock arithmetic, `x_upd`, elementwise glue), so `wall` here is not the e2e number in §1.

---

## 4. Per kernel — the current default plus both opt-ins

![kernels](plots/03_per_kernel.png)

`modiff_full_k4_projk4_qkvi8`, batch 128, **91.62 ms/step of GPU time** in the trace.

| bucket | ms/step | calls | distinct kernels |
|---|---:|---:|--:|
| conv | 27.43 | 84 | 10 |
| delta_quantize | 15.84 | 130 | 4 |
| linear_gemm | 15.08 | 85 | 8 |
| **elementwise** | **11.59** | **339** | 7 |
| attention | 9.81 | 21 | 3 |
| norm_quantize | 8.91 | 178 | 5 |
| attn_quantize | 2.94 | 10 | 2 |
| other | 0.01 | 5 | 4 |

Top kernels:

| kernel | ms/step | calls |
|---|---:|---:|
| `cutlass ImplicitGemmConvolutionEVT` (int8 conv) | 13.77 | 35 |
| `cutlass ImplicitGemmConvolutionEVT` (2nd instantiation) | 11.59 | 35 |
| `gn_apply_delta_quantize_flat_vec2_kernel` | 10.72 | 83 |
| `gemm_w8a8_kernel_awq` | 10.62 | 32 |
| `flash_attn_int8_mma_kernel_t` | 7.01 | 5 |
| `gn_stats_partials_chanmajor_kernel` | 4.70 | 83 |
| `at::native::elementwise_kernel` | 4.05 | 120 |
| `gemm_w8a8_kernel_awq_out_i8` | 3.91 | 10 |
| `group_norm_silu_delta_quantize_resize_nhwc_kernel` | 3.56 | 10 |
| `at::native::vectorized_elementwise_kernel` | 2.62 | 69 |
| `static_quantize_and_update_ahat_kernel_int8_half_cache_vec2` | 2.61 | 21 |
| `flash_attn_int8_packed_mma_kernel` | 2.54 | 10 |

**What is fused and what is not.** Roughly 78 of the 91.6 ms sits in kernels that each do several
operations per launch: conv + EVT epilogue, GN + delta-subtract + quantize + `a_hat` write, GEMM +
`o_hat` + bias + residual, QKᵀ + softmax + AV with scores never leaving SRAM. **The two genuinely
unfused items are `gn_stats_partials_chanmajor` (4.70 ms) and the elementwise glue (11.59 ms over 339
calls).**

The elementwise bucket is the **largest unfused item in the model** — bigger than the whole
`attn_quantize` bucket, bigger than `norm_quantize` — and no one has ever targeted it. `flash_attn_int8_mma_kernel_t`
at 7.01 ms over just 5 calls is the hd=24 attention tier again, from the other instrument.

---

## What this data does NOT cover

Stated so nobody reads a gap as a zero:

1. **Per-kernel traces exist for 3 arms captured today** (`int8_ptq`, `…_projk4`, `…_qkvi8`). The other
   7 arms in `trace_buckets_postsplit.json` come from **earlier captures**. They remain representative —
   the SASS gate proves the device code is byte-identical — but they were not re-captured today.
2. **The layer/block profile covers 6 configs, and the two opt-in arms are not among them.** The
   per-block table above is `W8A8 conv+proj`, i.e. *before* the projection refresh and route (b). Those
   two change the projection and qkv paths, so per-block attention numbers would shift somewhat.
3. **No W8A4/W4A4 e2e arm.** `differential_timing.py`'s `CONFIGS` has neither; §1 is W8A8 (and fp16)
   only, while §3's per-kind table has W8A4/W4A4 from the layer harness. The two sections are not
   directly comparable.
4. **Batch 128 only, DDIM, LSUN-churches.** No batch sweep, no other model.
5. **No quality numbers here.** This is speed only. relL2/FID live in `docs/act_bits_2026-08-05`,
   `docs/delta_clip_2026-08-06`, `docs/fid_2026-08-05`.

## Reproducing

```bash
bash docs/postsplit_benchmark_2026-08-12/scripts/run_all.sh   # ~33 min, produces all the data
python docs/current_state_2026-08-12/scripts/make_plots.py    # these figures, offline
```

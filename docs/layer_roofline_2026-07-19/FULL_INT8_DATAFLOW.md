# Full-int8 dataflow: unpadded-output GEMM (kill the N-pad slice copies)

Date: 2026-07-19 · Model: LSUN churches LDM UNet · GPU: A40 · Batch: 128 · DDIM.

## Motivation

The task was "full-int8 dataflow" — extend the int8-output fusion into int8
conv→conv chaining and int8 residual adds to shrink the ~23% elementwise+quantize
buckets.

The investigation contradicted that premise, so the plan changed:

- The int8 conv **already deep-fuses dequant** (`conv2d_int8_fprop_deepfuse_*`):
  there is **no separate dequant kernel** in the profile, so there are no
  "dequant round-trips" to reclaim by chaining convs.
- A steady-state copy trace (patching `.contiguous`/`.to`, with warmup) showed the
  real reclaimable traffic is **N-padding slice-copies**, not dequant:

  | copy site | MB / fwd @ b8 | what |
  |---|---|---|
  | `token_major_attention.py:269` | 47.2 | qkv GEMM output `out[:, :N].contiguous()` |
  | `token_major_attention.py:234` | 15.7 | proj GEMM output `out[:, :N].contiguous()` |
  | `fused_resblock.py:171` | 17.7 | residual fp32→fp16 cast (genuine) |
  | mod_scale/shift reshape | 0.6 | tiny |

  The two slices are the padded **C192** attention linears (qkv N 576→640, proj
  192→256): the AWQ-tiling GEMM pads N up to `CTA_N=128`, writes `[M, N_pad]`, then
  the caller slices `[:, :N].contiguous()` — a full copy. At b128 these scale 16×
  to **~1 GB/step**.

## The fix

Make the int8/int4 GEMMs write the **unpadded** `[M, n_out]` result directly,
skipping the padded columns, so the slice+copy disappears.

- `csrc/kernels/gemm_wxax.cu`: added `n_out` to `gemm_w8a8_kernel_awq` /
  `gemm_w4a4_kernel_awq` epilogues — store bounds-checked (`col0 < n_out`;
  `n_out` even so a `__half2` pair is all-in or all-out). New wrappers
  `gemm_w8a8_awq_nout` / `gemm_w4a4_awq_nout` allocate `[M, n_out]` and pass it;
  the original wrappers pass `n_out = N` (dense, byte-identical to before).
- Wired into `_qkv_from_gn` / `_apply_proj` (`token_major_attention.py`) and
  `QuantLinearWxAx._gemm` (`wxax_linear.py`): when `_awqt_N != out_features`, call
  the `_nout` variant and drop the slice.

## Correctness

`gemm_*_awq_nout` is **bit-exact** vs the old `gemm + slice` (maxdiff 0.0) on all
churches qkv/proj shapes, int8 and int4. Only that pair was swapped, so the model
output is unchanged by construction.

## Results (b128, 30 warm-up + 5×200 steps, mean ms/step)

Data: `data/full_int8_dataflow_b128.csv`, `data/bench5_confirm_b128.csv`.

| version | before | after | Δ | vs fp16 |
|---|---|---|---|---|
| int8_baseline | 178.1 | **173.3** | −2.7% | **1.10×** |
| int8_modiff | 201.1 | **196.6** | −2.2% | 0.97× |
| int4_baseline | 175.7 | **171.8** | −2.2% | **1.11×** |
| int4_modiff | 202.5 | **197.1** | −2.7% | 0.97× |

fp16 reference = 190.2 ms (warm). The baselines moved from 1.07×/1.08× to
1.10×/1.11×. The MoDiff temporal-cache versions improved too but stay ~0.97×
(the a_hat/o_hat caching adds elementwise overhead not recovered at this config).

The steady-state copy trace after the fix confirms the 47 MB + 16 MB slice copies
are **gone**; only the (genuine) residual cast and tiny reshapes remain.

## Profile after the fix (int8_baseline, GPU-busy 172.1 ms/step)

Data: `data/detailed_int8_baseline_b128.csv`.

| bucket | ms | % |
|---|---|---|
| attention softmax | 41.9 | 24.4 |
| attention QKᵀ/AV bmm (fp16) | 40.2 | 23.3 |
| elementwise / copy | 28.4 | 16.5 |
| conv (int GEMM) | 22.5 | 13.0 |
| GroupNorm (fused GN→int8) | 21.8 | 12.7 |
| upsample / concat | 4.7 | 2.7 |
| qkv/proj int GEMM | 4.7 | 2.7 |
| other fp16 GEMM | 3.6 | 2.1 |
| quantize / dequant | 3.0 | 1.7 |

- **quantizable compute (conv + qkv/proj int GEMM) = 27.1 ms (16%)** — the only
  part int8/int4 speeds up. Amdahl ceiling ≈ 1.19×; we are at 1.10×.
- **fp16 / memory-bound (attn + GN + elementwise) = 137 ms (80%)** — unchanged by
  quantization. Attention alone is **48%** and stays fp16 (every quantized-
  attention path measured is slower than fp16 SDPA).
- `elementwise/copy` fell from ~32 ms (pre-fix) to 28.4 ms — the slice copies.

## Conclusion

The unpadded-output GEMM removed ~1 GB/step of N-pad slice copies and gave a real,
bit-exact **2–3% e2e** gain across all four quantized versions, pushing the
baselines from ~1.07× to ~1.10–1.11× vs fp16 — as far toward the 1.19× Amdahl
ceiling as the copy layer allows. The only remaining lever with real headroom is
attention (48%), which stays fp16 because quantizing it is slower every way tried.

### Caveats
- The clean fp16 5×200 rerun was killed by SIGTERM (an earlier foreground wait hit
  its timeout); the warm 190.2 ms quick-measure is used as the reference (matches
  the prior 188 ms). The 349 ms fp16 line inside `bench5_confirm_b128.csv` is the
  cold-clock artifact (fp16 ran first) — not the reference.
- int4 "before" numbers are the same-session no-fix means (175.7 / 202.5) for
  clean attribution, not the older prior means.

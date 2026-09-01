# Scheme A (full MoDiff): remaining speed headroom

2026-09-01 · W8A8 · A40 · batch 128 · DDIM 50 · `LINEAR=0` · `REPLAY_K=1`

A is the production path: every step GN → delta-quantize → INT8 conv, write `a_hat`/`o_hat`. Current: **~74 ms/step, 1.43× vs fp16**. S=25 of this path is already the ≥2× wall-clock answer (2.77× vs fp16 S=50). This note is only “can the 74 ms come down?”

Buckets from [pipeline_profile_2026-08-31](../pipeline_profile_2026-08-31/FINDINGS.md) (A = 72.20, PTQ = 64.77, fp16 = 102.23):

| bucket | fp16 | PTQ | **A** | A − PTQ |
|---|--:|--:|--:|--:|
| GEMM / conv | 46.7 | 37.1 | **37.8** | +0.7 |
| GroupNorm+SiLU | 20.9 | 10.8 | **17.4** | **+6.6** |
| attention | 11.4 | 9.0 | **8.9** | −0.1 |
| elementwise / copy | 19.6 | 5.8 | **6.1** | +0.3 |
| other | 3.7 | 2.0 | **2.0** | 0 |

The 8 ms A-vs-PTQ gap is almost entirely the MoDiff GN+`a_hat` path. Conv GEMM is already at PTQ. Fusion already took the copy bucket (3.4×).

To hit 2× **at S=50** you need ~53 ms, i.e. **−21 ms from A**. That 21 ms is not inside A: the only 8 ms of MoDiff-specific tax is `a_hat`, and every lossless/lossy transform of `a_hat` is closed (C11–C20, D, B, C).

---

## Closed on A (do not reopen)

| lever | why closed |
|---|---|
| B / C skip-compute | dominated by A S=25; C is color blocks |
| D skip `a_hat` store | S ≈ 2 ms, e2e stays 1.48× |
| hide / overlap `a_hat` (C11, C12) | SM occupancy; conv is a worse place for the store |
| int8/fp8/downsample `a_hat` (C15, C20) | accumulator-pair invariant |
| MoDiff GN `fast_reduce` (C10) | chanmajor already wins at b128; forcing group-major is slower |
| attention residual replay | 2× at PTQ quality |

---

## Still on the table (all small)

| lever | honest size | note |
|---|---|---|
| **`MODIFF_FUSE_QKV_I8=1`** | **+0.79 ms/step** | already wired, quality-neutral at W8A8 (±1.1% latent, n=32). Opt-in today. |
| **CUDA graph** | **~0.5 ms** (PTQ analog: 64.87 → 64.41) | MoDiff capture fails: attention epilogue lazy-`torch.tensor` (`quantized_std_attention.py` `_qkv_inv_scale_t` / `_int8_qkv_inv_out`). Fix is pre-allocate on first eager step. Not a 2× lever. |
| Stream-K / fill idle SMs | **≲ 1–2 ms bound** | 768 2×2 wastes 71% SM-slots but is 73 µs × 12 = 0.9 ms/step; 192 32×32 already 97% full. Unmeasured e2e; ceiling is the leftover of small tiles. |
| C3 flash T=1024 hd=24 | maybe 1–2 of the **9 ms** attn bucket | four approaches already lost; next is MMA fragment layout. |
| C18 vec4 apply | 0.045 ms | measured, deliberately not landed |

Optimistic sum of the cheap ones (FUSE_QKV + graph + Stream-K upper bound) ≈ **2–3 ms** → A at ~71–72 ms, **1.45–1.47×**. Still not 2× at 50 steps. On A S=25 that is ~50–75 ms/sample on a 1916 ms sample (~3%).

---

## What to do

1. **Do not hunt 2× inside A’s 50-step kernel.** The ≥2× path is A S=25.
2. If A’s *ms/step* still matters (FID protocol stays DDIM 50 at 1.43×): turn on `FUSE_QKV_I8`, then make the attention scale tensors capture-legal so CUDA graph can apply. That is the whole cheap list.
3. Stream-K and C3 are real engineering, not research holes, and they do not change the 2× story.

# Along-C B=32 int8 `a_hat`: faster than fp16 `a_hat`

LSUN-churches LDM-KL-8, W8A8, A40, batch 128, 50 DDIM steps.
`MODIFF_AHAT_BLOCK=32 MODIFF_AHAT_BITS=16 MODIFF_IMODE=0 MODIFF_DELTA_MODE=static
MODIFF_LINEAR=0 MODIFF_AHAT_REFRESH=0`.

## Result

Three independent runs of `scripts/block32_real.py`:

| run | fp16 `a_hat` | B=32 int8 `a_hat` | speedup | relL2 |
|---|---:|---:|---:|---:|
| 1 | 81.744 | **79.947** | **1.022x** | 0.052 |
| 2 | 81.737 | **80.300** | **1.018x** | 0.073 |
| 3 | 81.302 | **80.150** | **1.014x** | 0.102 |

int8 is faster in every run, by 1.15-1.80 ms/step. Within-arm spread is 0.35 ms (int8) and
0.44 ms (fp16), so the gap is not noise. Images are indistinguishable
(`plots/block32_real.png`). relL2 is not reproducible to better than about +-0.03 run to run,
so treat 0.05-0.10 as one number; it is still below the 0.129 the Python prototype produced.

`a_hat` storage is 1 byte per element plus one fp32 scale per 32 channels = 1.125 B/elem,
against 2 B/elem for fp16: a 1.78x smaller cache **and** less time.

Starting point for this pass was 84.95 ms/step (0.96x, i.e. 4.3% *slower* than fp16).

## Where the 5 ms came from

Kernel-time deltas against the fp16 arm, `scripts/block32_why_slower.py`, 20 DDIM steps:

| bucket | before | after | note |
|---|---:|---:|---|
| `gn_apply` vec2 -> vec4 | +3.85 | **-1.00** | see below |
| eager `other` (t=T pack) | +3.33 | +0.25 | `ahat_pack_block_nhwc` |
| `ahat_commit_block` | +1.52 | +0.46 | vectorized; resize GN cannot fuse it |
| `gn_resize` | -1.04 | -1.38 | int8 store is cheaper than fp16 |
| **total** | **+3.8** | **-1.23** | |

Three separate causes, in the order they were worth fixing:

**1. Profile window length.** A 5-step profile put the total delta at +0.35 ms/step while
the 50-step wall clock said +3.5. The t=T step has no `a_hat` read and the one-time pack
lands inside it, so a short window both dilutes the per-step cost and hides the one-time
cost. `block32_why_slower.py` now defaults to 20 steps, and `block32_cpu_gap.py` exists to
confirm a wall-clock gap is GPU-side before chasing it (it was: host time was -0.42 ms).

**2. The t=T pack.** The entire +3.3 ms/step of eager `other` was `_pack_ahat_along_c`:
eight eager passes over every calibrated layer's full `a_hat`. It is one-time work, so the
+3.3 is that ~68 ms spread over the profile's 20 steps -- in the 50-step benchmark it is
1.4 ms/step. `ahat_pack_block_nhwc` does it in one coalesced pass, 5 ms, and `other` drops
to +0.25.

I first blamed this on the Python along-C fake-quant in `_maybe_quantize_ahat` and gated it
behind `MODIFF_AHAT_BLOCK_FAKE=1` -- the profile did not move at all, because in this config
there are no layers for it to run on. Every conv layer that holds an `a_hat` cache (70 of
them) is calibrated and takes the int8 blockwise path; the fake-quant branch is dead here.
The gate is harmless and still the right default, but it bought nothing. Worth remembering
that a plausible-looking eager op is not evidence until the profile moves.

**3. Occupancy, not instruction count.** This was the whole `gn_apply` story and the least
obvious part. A standalone replica of the kernel's loop (`scripts/ahat_variants.cu`) showed
the int8 B=32 epilogue at **0.87x** of the fp16 one -- faster, as the 25%-lower byte count
predicts -- while the real kernel was 1.06x. The difference was register pressure: `b32`
was a runtime condition, so the dead generic `a_hat` path stayed in the
`AhatI8=true` instantiation and cost 46 registers against fp16's 34, dropping the
blocks/SM limit from 7 to 5. Making it the compile-time template parameter `AhatB32`
brought it to 34 registers and the kernel to 0.977x.

Things that were tried and did *not* matter: replacing `roundf`+clamp+`cvt` with PRMT/FADD
magic-number int8 conversion (I2F/F2I are eighth-rate on GA10x, but removing four of them
per pair changed nothing measurable), and `__reduce_max_sync` instead of four
`__shfl_xor_sync`. Both are kept because they are not worse and they shrink the code, but
neither is why this got faster.

**4. vec4.** With occupancy fixed, the remaining lever was the group reduce, which the vec2
layout runs once per 2 channels. `gn_apply_delta_quantize_flat_vec4_b32_kernel` gives each
thread 4 channels, so 8 lanes cover a B=32 group: half the reduces, half the scale loads
and reciprocals, and one 4-byte `a_hat` access instead of two 2-byte ones. 0.977x -> 0.819x
on the isolated kernel.

This is specific to int8, not generic vectorization. The probe includes an fp16-vec4
control, and it is **slower** than fp16-vec2 (1.049x): fp16 `a_hat` is already 4 bytes per
lane at vec2 and has no group reduce to amortize, so widening only costs it registers.
vec8 (4 lanes per group) is also worse than vec4, 0.796x vs 0.779x.

Requires `CPG % 4 == 0` so all four channels share one mean/inv_std. Churches C=192 and
C=576 have CPG 6 and 18 and stay on the vec2 B=32 kernel.

## Correctness

`scripts/verify_b32_paths.py` checks both dispatches against a PyTorch reference that
mirrors the kernel op for op (including its fp16 round-trip before SiLU and `roundf`'s
half-away-from-zero). C=384/768 exercise vec4, C=192/576 exercise vec2:

- `yq`: exact.
- `a_hat` codes: max difference 1 LSB, on 0% of elements above that. `__fdividef(127, g)`
  is an approximate reciprocal, so values sitting on a rounding boundary can move one code.
- scales: within 1.5e-8.

`scripts/smoke_ahat_modes.py` runs the other `a_hat` storage modes that share these device
helpers and recompiled with them: along-C B=16 (generic blockwise, not the B=32 fast path),
per-tensor held int8, and I-MoDiff int16. All produce finite latents at their established
relL2 levels.

While writing that smoke, one pre-existing bug surfaced in this uncommitted change set: the
earlier restructure of the three vec2 kernels into `ahat_quant_update2_w<false>` plus an
external `ahat_store2` is wrong for I-MoDiff, whose `ahat_inv` is 0 by construction
(`scale[0]==0` signals the integer datapath), so the external store wrote zeros into an
int16 buffer. Fixed by keeping I-MoDiff, per-tensor int8 and fp16 inside
`ahat_quant_update2_w<WriteAhat>` and using the split form only for blockwise. Note this did
**not** move I-MoDiff's E2E number (relL2 3.2310 -> 3.2307), so whatever makes I-MoDiff's
latent diverge end to end is elsewhere; its kernel invariants pass
(`integration/tests/test_imode.py`, 4/4). That remains open and outside this work.

## What is still on the table

`ahat_commit_block` is +0.46 ms/step. `group_norm_silu_delta_quantize_resize_nhwc` is
group-major (one block per (sample, GN group)), and a B=32 along-C group does not nest
inside a GN group for any of the CPG values here, so the resnap cannot be fused into it
without splitting that kernel into a stats pass plus a pair-major apply pass. The int8
resize store already saves 1.38 ms, so the net is -0.92 either way.

## Do not

- Do not hold `a_hat` scales across writes. Every write must resnap from the new per-group
  amax. Held scales pass isolated kernel tests and then produce rainbow noise end to end
  (relL2 2.22): after the first write, quiet groups keep tiny t=T scales, activations grow,
  `a_hat` saturates, and the cache stops moving.
- Do not run `python setup.py build_ext --inplace` for a change confined to the `a_hat`
  path; it recompiles every CUTLASS conv (~32 min). `scripts/rebuild_ahat.sh` rebuilds the
  three `.cu` files plus `pybind.cpp` and relinks in ~5 min.

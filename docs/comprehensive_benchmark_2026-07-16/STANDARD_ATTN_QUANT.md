# MoDiff fully-quantized STANDARD attention (W8A8 / W4A4) — 2026-07-16

Flash attention is **removed on every layer/mode** (it is fp16-only and opaque). MoDiff is a
fully-quantized method, so attention runs as **standard materialized attention** (QKᵀ → softmax → AV,
scores in HBM) whose score path is quantized to int8 (W8A8) and int4 (W4A4), consistent with the
quantized convs and linears. **Effectiveness is measured vs the fp16 *standard* (math) attention** — the
flash-SDPA baseline from `REPORT.md` is a different (faster, unquantizable) design and out of scope here.
A40, batch 32, churches; kernels in `csrc/kernels/attn_quant_gemm.cu` (batched over B·H).

## Effective path (fused quantize · fp16 scores · fused softmax)

The score path is fully fused to be e2e-effective — no PyTorch quantize, no fp32 score dump:
1. **`quantize_attn_qkv`** — one CUDA launch does per-token int8/int4 Q/K quantize (`sq,sk[BH,T]`),
   per-channel-over-T V absmax + transpose + quantize/pack to channel-major `Vt` (`sv[BH,hd_pad]`), and
   hd-padding — replacing ~5 ms of PyTorch absmax/round/pad/transpose/pack elementwise ops.
2. **fp16 pre-scaled scores** — the int8/int4 QKᵀ epilogue applies `sq·sk·(1/√d)` and stores **fp16**
   `S[BH,T,T]` (logits are in fp16 range), instead of fp32 raw int32 scores. This **halves** the T×T
   score memory vs the earlier fp32 dump and lets softmax read fp16 directly.
3. **fused softmax/requant** — reads the pre-scaled fp16 logits (no dequant), online max/exp/sum,
   writes int8 P∈[0,127] (or packed int4 P∈[0,7]) + per-row `sp`.

## Kernels (built + validated)

int8: `attn_qk_int8` (batched int8 QKᵀ → **fp16 pre-scaled** scores) · `attn_softmax_requant` (fp16 online
softmax → int8 P∈[0,127] + per-row sp) · `attn_av_int8` (int8 P·Vtᵀ → fp16, dequant sp·sv).
int4: `attn_qk_int4` / `attn_softmax_requant4` (packed int4 P∈[0,7]) / `attn_av_int4` via `m16n8k64.s4`.
Fused quantize: `quantize_attn_qkv`. Small-T blocks (T<256, e.g. 4×4/8×8) fall back to fp16 standard attention.

**Correctness** ([`integration/tests/test_attn_eff.py`](../../integration/tests/test_attn_eff.py), fused path
vs fp32 reference): full **int8** attention rel **0.020–0.030**; full **int4** attention rel 0.34–0.43 (int4
Q·K·V + 8-level int4 P — inherently lossy). `test_kernel_correctness` ALL PASS.

## Speed — effective vs fp16 standard attention

Per real attention block (BH=256, [`data/attn_kernel_speed.csv`](data/attn_kernel_speed.csv), full fused path):

| block (BH,T,hd) | fp16 std µs | int8 µs | int4 µs | int8 | int4 |
|---|--:|--:|--:|--:|--:|
| **C192 T1024 hd24** (dominant) | 13467 | 5278 | 4845 | **2.55×** | **2.78×** |
| C384 T256 hd48 | 911 | 882 | 828 | 1.03× | 1.10× |
| C768 T64 hd96 | 87 | 336 | 330 | 0.26×→fp16 | 0.26×→fp16 |

**int8/int4 quantized standard attention is 2.55×/2.78× faster than fp16 standard attention on the dominant
T=1024 block** (int8 QKᵀ/AV tensor cores at 2× + int4 at 4×, fp16 scores halving score IO, fused quantize +
softmax); modest on the mid block; the tiny T=64 block uses the fp16 fallback.

## e2e quality (vs fp16 standard attention, batch 8)

| path (isolated: fp16 conv + quantized attn) | 5 steps | 20 steps |
|---|--:|--:|
| **int8** standard attention | 0.008 | **0.015** (quality-safe) |
| int4 standard attention | 0.14 | 0.30 (compounds) |

- **int8 attention is quality-safe e2e** (rel 0.015 over 20 DDIM steps).
- **int4 attention error compounds over the trajectory** (0.14→0.30) — by design the target for **MoDiff
  temporal-delta compensation** (the method caches/compensates quant error across steps). Reported, not gated.

## C — effective large-M linear (AWQ N-pad)

The dominant C192 qkv linear (M=32768,K=192,N=576) lost to cuBLAS on our `gemm_w8a8` (0.41×, short-K,
memory-bound, no split-K). AWQ's w8a8 has a large-M tiling but needs N%128; padding N→640 (offline weight/
wscale pad, slice output) gives near-parity: **C192 qkv 0.41×→0.97×, proj 0.62×→0.97×** vs fp16 cuBLAS.
`QuantLinearWxAx` now routes int8 (in%64, out≥128) through AWQ (N-padded when needed). (int4 keeps `gemm_w4a4`.)

## E — full 6-mode pipeline (standard quantized attention, batch 32)

Speed ([`data/pipeline_speed.csv`](data/pipeline_speed.csv), GPU-busy; wall stdev ≤0.3) + peak memory:

| mode | wall | GPU-busy | vs fp16 | peak MiB | vs fp16 mem |
|---|--:|--:|--:|--:|--:|
| fp32 | 102.65 | 101.79 | 0.55× | 4920 | +13% |
| fp16 (standard attn) | 56.00 | 55.04 | 1.00× | 4369 | — |
| int8 base | 59.48 | 59.04 | **0.94×** | 4387 | ≈parity |
| int8 modiff | 67.62 | 66.73 | 0.83× | 4794 | +10% |
| int4 base | **54.74** | 54.16 | **1.02×** | **3996** | **−8.5%** |
| int4 modiff | 60.18 | 59.25 | 0.93× | 4405 | +0.8% |

**Effective result: int4 base now beats fp16 standard attention e2e (1.02×, 54.74 < 56.00 ms/step) at −8.5%
peak memory, and int8 base is at near-parity (0.94×)** — up from the pre-fusion 0.77×/0.78×. The three
overheads that made the earlier materialized path e2e-negative are gone:
1. **PyTorch Q/K/V quantize → fused `quantize_attn_qkv`**: profile
   ([`data/kernel_profile.csv`](data/kernel_profile.csv)) elementwise **12.9→4.6 ms** (int8 base) vs fp16 7.6.
2. **fp32 scores → fp16 pre-scaled scores**: analytical attention IO
   ([`data/pipeline_io_analytic.csv`](data/pipeline_io_analytic.csv)) **int8 13957→8496 MiB, int4 12491→7029**,
   both now **below fp16 11430** — quantizing the score path finally *reduces* IO. Measured peak follows
   (int4 base 3996 < fp16 4369).
3. **fused softmax** reads fp16 logits (no dequant pass).

The residual gap for int8 base (0.94×) is the O(T²) softmax, which is ~dtype-invariant and shared by all
modes (profile: attention softmax ~16.5 ms in every quantized mode); int4 clears it because its QKᵀ/AV at 4×
tensor-core throughput + half the operand IO more than pays for the softmax.

## Analytical total IO ([`data/pipeline_io_analytic.csv`](data/pipeline_io_analytic.csv))

| precision | conv | linear | attn | total MiB/step | vs fp16 |
|---|--:|--:|--:|--:|--:|
| fp16 | 1330 | 648 | 11430 | 13408 | 1.00× |
| int8 | 847 | 648 | 8496 | 9991 | 0.75× |
| int4 | 605 | 648 | 7029 | 8283 | 0.62× |

## Relationship to the flash-SDPA finding (REPORT.md)

`REPORT.md` showed fp16 **flash** SDPA is ~9× faster than fp16 **math** attention. This milestone deliberately
uses **standard (math) attention** because the fully-quantized MoDiff method needs a quantizable score path.
Net: the quantized standard attention beats the fp16 *standard* baseline (kernels 2.5–2.8×; e2e int4 1.02×,
int8 0.94×), but standard attention (fp16 or int) is still slower than fp16 flash — the method trades flash
speed for full-pipeline quantization + MoDiff error compensation.

## Status
Complete: A (flash removed) · B0–B3 (int8+int4 QKᵀ/softmax/AV kernels) · C (AWQ-N-pad linear) · D (wiring) ·
E (6-mode speed/IO/profile) · **effective path (fused quantize + fp16 scores + fused softmax)**. Verdict:
correct + int8 quality-safe fully-quantized standard attention; kernels 2.5–2.8× vs fp16 standard in isolation
and **e2e-positive** (int4 base 1.02× at −8.5% peak memory, int8 base 0.94× near-parity).

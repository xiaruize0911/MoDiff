# MoDiff fully-quantized STANDARD attention (W8A8 / W4A4) — 2026-07-16

Flash attention is **removed on every layer/mode** (it is fp16-only and opaque). MoDiff is a
fully-quantized method, so attention runs as **standard materialized attention** (QKᵀ → softmax → AV,
scores in HBM) whose score path is quantized to int8 (W8A8) and int4 (W4A4), consistent with the
quantized convs and linears. **Effectiveness is measured vs the fp16 *standard* (math) attention** — the
flash-SDPA baseline from `REPORT.md` is a different (faster, unquantizable) design and out of scope here.
A40, batch 32, churches; kernels in `csrc/kernels/attn_quant_gemm.cu` (batched over B·H).

## Kernels (built + validated)

int8: `attn_qk_int8` (batched int8 QKᵀ → fp32 raw scores) · `attn_softmax_requant` (dequant sq·sk·1/√d,
fp32 online softmax → int8 P∈[0,127] + per-row sp) · `attn_av_int8` (int8 P·Vᵀ → fp16, dequant sp·sv).
int4: `attn_qk_int4` / `attn_softmax_requant4` (packed int4 P∈[0,7]) / `attn_av_int4` via `m16n8k64.s4`.
Small-T blocks (T<256, e.g. 4×4/8×8) fall back to fp16 standard attention.

**Correctness** (`test_qk/sm/av/av4`): QKᵀ int matmul **exact** (rel 0); softmax rel 0.017–0.027 vs
`F.softmax`; full **int8** attention rel **0.015–0.030** vs fp16; full **int4** attention rel 0.34–0.43
(int4 Q·K·V + 8-level int4 P — inherently lossy). `test_kernel_correctness` ALL PASS.

## Speed — effective vs fp16 standard attention

Per real attention block (BH=256, [`data/attn_kernel_speed.csv`](data/attn_kernel_speed.csv)):

| block (BH,T,hd) | fp16 std µs | int8 µs | int4 µs | int8 | int4 |
|---|--:|--:|--:|--:|--:|
| **C192 T1024 hd24** (dominant) | 13457 | 6027 | 5541 | **2.23×** | **2.43×** |
| C384 T256 hd48 | 908 | 778 | 735 | 1.17× | 1.24× |
| C768 T64 hd96 | 70 | 160 | 155 | 0.44×→fp16 | 0.44×→fp16 |

**int8/int4 quantized standard attention is 2.2–2.4× faster than fp16 standard attention on the dominant
T=1024 block** (int8 QKᵀ/AV tensor cores at 2× + int4 at 4×, with the fp32 softmax shared); modest on the
mid block; the tiny T=64 block uses the fp16 fallback. This is the plan's effectiveness gate — met for the
blocks that matter.

## e2e quality (vs fp16 standard attention, batch 8)

| path (isolated: fp16 conv + quantized attn) | 5 steps | 20 steps |
|---|--:|--:|
| **int8** standard attention | 0.008 | **0.015** (quality-safe) |
| int4 standard attention | 0.14 | 0.30 (compounds) |

- **int8 attention is quality-safe e2e** (rel 0.015 over 20 DDIM steps).
- **int4 attention error compounds over the trajectory** (0.14→0.30) — by design the target for **MoDiff
  temporal-delta compensation** (the method caches/compensates quant error across steps). Reported, not gated.
- Full `int8_baseline` (int8 conv + int8 attn) e2e 0.41 is **conv-dominated** (int8 conv compounding over 20
  steps), not the attention — a separate, pre-existing effect the MoDiff modes address.

## Relationship to the flash-SDPA finding (REPORT.md)

`REPORT.md` showed fp16 **flash** SDPA is ~9× faster than fp16 **math** attention (and our int8 flash was
uncompetitive). This milestone deliberately uses **standard (math) attention** instead, because the
fully-quantized MoDiff method needs a quantizable score path. Net: the quantized standard attention beats the
fp16 *standard* baseline 2.2–2.4×, but standard attention (fp16 or int) is slower than fp16 flash — the method
trades flash speed for full-pipeline quantization + MoDiff error compensation.

## C — effective large-M linear (AWQ N-pad)

The dominant C192 qkv linear (M=32768,K=192,N=576) lost to cuBLAS on our `gemm_w8a8` (0.41×, short-K,
memory-bound, no split-K). AWQ's w8a8 has a large-M tiling but needs N%128; padding N→640 (offline weight/
wscale pad, slice output) gives near-parity: **C192 qkv 0.41×→0.97×, proj 0.62×→0.97×** vs fp16 cuBLAS.
`QuantLinearWxAx` now routes int8 (in%64, out≥128) through AWQ (N-padded when needed). Memory-bound short-K
still can't beat fp16 (0.97× ceiling), but the 2.4× loss is gone. (int4 keeps `gemm_w4a4`.)

## E — full 6-mode pipeline (standard quantized attention, batch 32)

Speed ([`data/pipeline_speed.csv`](data/pipeline_speed.csv), GPU-busy; wall stdev ≤0.1) + peak memory:

| mode | wall | GPU-busy | vs fp16 | peak MiB |
|---|--:|--:|--:|--:|
| fp32 | 102.76 | 101.83 | 0.54× | 4920 |
| fp16 (standard attn) | 56.11 | 55.05 | 1.00× | 4369 |
| int8 base | 72.32 | 71.61 | **0.77×** | 4906 |
| int8 modiff | 80.53 | 79.52 | 0.69× | 5311 |
| int4 base | 71.08 | 70.18 | **0.78×** | 4514 |
| int4 modiff | 76.51 | 75.30 | 0.73× | 4923 |

**Honest result: the fully-quantized standard attention is slower e2e than fp16 standard (int8/int4 base
0.77–0.78×), even though the attention kernels are 2.2–2.4× faster in isolation.** The profile
([`data/kernel_profile.csv`](data/kernel_profile.csv), int8_baseline) shows why: attention (softmax) 19.0 ms +
QKᵀ/AV+qkv/proj GEMM 20.1 ms + **elementwise 12.9 ms** (the PyTorch per-token Q/K/V quantize) vs fp16's
attention 11.4 + GEMM 13.4 + elementwise 7.6. Three overheads eat the matmul win:
1. **PyTorch Q/K/V quantize** (absmax/round/pad/transpose/pack) — ~+5 ms of elementwise, not fused into CUDA.
2. **fp32 raw scores**: int8 QKᵀ emits fp32 `[BH,T,T]` (int32 overflows fp16) → **doubles** the T×T score
   memory vs fp16's fp16 scores. Analytical IO ([`data/pipeline_io_analytic.csv`](data/pipeline_io_analytic.csv)):
   attention IO **int8 13957 > fp16 11430 MiB** — quantizing the score path *increases* IO.
3. **softmax_requant** (fp32 online softmax over the materialized scores) is memory-bound, ~19 ms.

So the value of this milestone is the **fully-quantized method** (int8 quality-safe at rel 0.015; int4 for
MoDiff temporal compensation), **not** an e2e speedup. Standard attention (fp16 or int) is also far slower than
fp16 flash SDPA (33 ms; REPORT.md). Peak memory rises vs flash (materialized T×T; int8 modiff 5311 MiB).

**To make it e2e-effective would need** (future): a fused CUDA Q/K/V quantize; fp16 (scaled) raw scores instead
of fp32 to halve the T×T IO; and a faster fused softmax — i.e. converging toward a quantized *flash* kernel,
which is the tension the flash-SDPA finding already exposed.

## Status
Complete: A (flash removed) · B0–B3 (int8+int4 QKᵀ/softmax/AV kernels) · C (AWQ-N-pad linear) · D (wiring) ·
E (6-mode speed/IO/profile). Verdict: correct + int8 quality-safe fully-quantized standard attention; kernels
2.2–2.4× vs fp16 standard in isolation but **e2e-negative** (0.77–0.78×) from quantize + fp32-score overhead.

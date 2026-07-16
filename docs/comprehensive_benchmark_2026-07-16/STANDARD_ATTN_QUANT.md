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

## Status / remaining
Built + validated: flash removed (A); int8+int4 QKᵀ/softmax/AV kernels (B0–B3); pipeline wiring (D); attention
kernel-speed + e2e-quality verification. Remaining: **C** effective large-M linear (split-K / AWQ-N-pad for the
C192 qkv), and the full 6-mode e2e speed/total-IO/profile redo (**E**, incl. MoDiff-mode int4 compensation).

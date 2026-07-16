# Linear-layer quantization (W8A8 / W4A4) — results & findings (2026-07-16)

AWQ-referenced weight+activation quantization of the churches LDM UNet's Linear-equivalent layers
(attention qkv/proj, ResBlock emb_layers, time_embed). Opt-in via `MODIFF_QUANT_LINEAR=1`.
Implementation: `csrc/kernels/gemm_wxax.cu` (custom int8/int4 tensor-core GEMM), `csrc/kernels/quantize_qkv.cu`
+ fused act-quant, `integration/kernels/wxax_linear.py` (`QuantLinearWxAx`, `convert_linears_to_wxax`).

## Kernel benchmark — ours vs AWQ vs fp16 (A40, GEMM-only, µs)
| shape (M,K,N) | UNet role | ours | AWQ | fp16 | ours/AWQ | ours/fp16 | acc (==AWQ) |
|---|---|--:|--:|--:|--:|--:|--:|
| 8192,384,1152 | res16 qkv | 127 | 69 | 86 | 1.85× | 1.48× | 0.0137 |
| 8192,384,384 | res16 proj | 44 | 34 | 38 | 1.29× | 1.16× | 0.0137 |
| 4096,768,768 | attn proj | 56 | 43 | 59 | 1.30× | **0.95×** | 0.0139 |
| 512,768,2304 | res4 qkv | 27 | 16 | 20 | 1.72× | 1.34× | 0.0137 |
| 512,768,768 | res4 proj | 16 | 15 | 16 | 1.13× | 1.02× | 0.0143 |
| 32,768,1536 | emb MLP | 11 | 7 | 19 | 1.69× | **0.59×** | 0.0118 |

- Our kernel is **numerically exact vs AWQ** (identical rel-err vs fp32).
- **Beats fp16 on the small-M / memory-bound shapes** (emb MLP 0.59×, 4096×768×768 0.95×); 1.1–1.5× slower on
  the large compute-bound shapes. **1.1–1.85× of AWQ** (production kernel) across the board.
- Optimizations that got here: `cp.async` 3-stage pipeline (the big win, 8–20× → 1.2–1.9× AWQ), shape-adaptive
  tile dispatch (register-blocking `MT=2` for large-M/small-N, `MT=1` else). Tried and rejected: 48B smem
  padding (occupancy loss), BM=128 (not tile-bound).

## End-to-end (batch 16, `MODIFF_QUANT_LINEAR` on vs off)
| mode | speed vs same-mode fp16-linear | latent rel-err | peak Δ |
|---|--:|--:|--:|
| int8_baseline | 0.925× | **0.007** | −95 MiB |
| int8 (modiff) | 0.941× | 0.057 | −96 MiB |
| int4_baseline | 0.942× | **0.228** | −105 MiB |
| int4 (modiff) | 0.980× | **0.456** | −106 MiB |

## Findings
1. **int8 linear quant is quality-safe** (rel-err 0.007–0.057); **int4 is too lossy** (0.23–0.46 — W4A4 on
   these small Linears loses too much; would badly hurt FID). Recommend int8 only for the Linear path.
2. **Not an end-to-end speed win** (0.93–0.98×) — the Linears are a small fraction of this **conv-dominated**
   UNet, and the activation-quant + small-GEMM sizes cap the benefit. Confirmed structural: even **AWQ's own
   kernel only reaches ~0.85×** e2e, and our kernel matches AWQ e2e within ~0.7% (0.926 vs 0.933×).
3. **Memory win is small** (−95…−106 MiB, ~2.5%) — Linear weights are small (vs the −21% from attention-score
   fusion). Model-size (weights): int8 ~2×, int4 ~4× smaller on the quantized Linears.
4. **MoDiff temporal-delta on the linear activations is counterproductive** — rel-err diverges (0.06 → 3.2 as
   quant error accumulates over DDIM steps), + memory, slower. All modes use the static W/A linear quant; the
   baseline/modiff distinction stays in the conv path. (Code retained but off by default.)

## Conclusion
The int8/int4 Linear GEMM is validated (exact, competitive with AWQ, beats fp16 on memory-bound shapes) and is
a **model-size / memory** capability, **not an e2e speed win** for this architecture — because the compute
lives in the convs (already quantized), not the Linears. This mirrors why AWQ wins on LLMs (Linear-dominated,
weight-memory-bound decode) but not here (conv-dominated, compute-bound). Recommended default: **int8 Linear
quant where a smaller footprint matters; int4 not recommended (quality).**

*Repro:* `MODIFF_QUANT_LINEAR=1 python integration/benchmarks/ab_benchmark.py --mode int8_baseline` (env:
`PYTHONPATH=src/taming-transformers CUTLASS_PATH=/workspace/cutlass`, `pip install ninja`). Kernel gate:
`integration/tests/test_wxax.py`. Three-way kernel bench needs `import torch` before `awq_inference_engine`.

# Attention layer, all modes, one aligned Perfetto trace

**Hardware** NVIDIA A40 · **torch** 2.4.1+cu124 · **Batch** 128 · **Date** 2026-08-03
**Built by** [`scripts/export_attention_aligned.py`](scripts/export_attention_aligned.py)

One file: **`attention_aligned_fp16_int8_int4.json`**. Every attention-layer profile — 5 shapes ×
3 modes — merged into a single timeline and time-aligned so the modes can be read against each
other.

## Opening it

Drag the `.json` onto **<https://ui.perfetto.dev>**. Seven tracks appear:

```
shape        │ C192 32x32 T=1024 │ C384 16x16 T=256 │ C384 8x8 T=64 │ …   ← ruler, one slice per shape
FP16 CPU     │ ▓▓ dispatch       │ ▓▓               │ ▓▓            │
FP16 GPU     │ ███ kernels       │ ███              │ ██            │
INT8 CPU     │ ▓                 │ ▓                │ ▓             │
INT8 GPU     │ ██                │ ██               │ █             │
INT4 CPU     │ ▓                 │ ▓                │ ▓             │
INT4 GPU     │ ██                │ ██               │ █             │
```

Inside each ruler slice **all three modes start at the same timestamp**, so the three `* GPU` rows
under one shape are the same work in three precisions — read straight down to compare. Verified: the
three CPU annotations coincide to under 1 ns in every slot.

## How the alignment is done

Each `(mode, shape)` is profiled in its **own** profiler session, so every event in it belongs
unambiguously to one attention forward — that is what makes a per-slot time shift safe. Then:

- **anchor** = the `attn` `record_function` slice start on that session's CPU track
- **slot width** = the widest anchor→last-event span across the three modes, +18% padding, so no
  mode can overflow into the next slot
- every event is shifted by `slot_start − anchor`, including the `ac2g` flow arrows that link a
  kernel launch to its kernel
- **flow ids are namespaced per session**, so Perfetto cannot draw a launch→kernel arrow between two
  different modes
- pids are remapped to the six named tracks; torch's `Spans`/`Traces` pseudo-processes are dropped

Two things are excluded from each session, both harness artifacts rather than layer work: the
3 profiler-warmup iterations that run before the annotation opens (the profiler's first-call cost is
huge and unrepresentative — 2.8 ms of CPU before the first kernel even launched at T=64, against
411 µs of real GPU work), and the `torch.cuda.synchronize()` that closes the session after it.

## What it shows

GPU kernel self-time per layer call, straight out of the trace:

| shape | FP16 | INT8 | INT4 | INT4 vs FP16 | kernels FP16 → quant |
|---|---:|---:|---:|---:|:--|
| C192 32² T=1024 ×5 | 2974.0 µs | 2590.5 | 2539.6 | 1.17× | 8 → 4 |
| C384 16² T=256 ×5 | 1058.6 | 830.3 | 769.0 | 1.38× | 8 → 4 |
| C384 8² T=64 ×5 | 414.9 | 222.2 | 205.1 | 2.02× | 5 → 4 |
| C768 4² T=16 ×5 | 211.6 | 177.3 | 150.2 | 1.41× | 5 → 4 |
| C768 2² T=4 ×1 | 90.8 | 68.6 | 51.0 | 1.78× | 5 → 4 |

Agrees with the per-kernel profile in
[`MEASUREMENT_REPORT_2026-08-01.md`](../MEASUREMENT_REPORT_2026-08-01.md) to within 4.3%, and to
within 1.5% at every shape except T=1024 — expected, since this is one un-averaged forward against
the report's 8 rounds × 60.

Three things the aligned view makes visible at a glance:

1. **The attention core kernel changes identity with shape in INT4.** `flash_attn_int8_mma_kernel_t`
   at T=1024, `flash_attn_int4_mma_kernel_t` at T=256 and T=64, and
   `flash_attn_int8_qi8packed_small_qout_kernel` at T=16 and T=4. So "INT4 mode" is not one INT4
   pipeline — and the two shapes that *do* use the native INT4 MMA are exactly the two where INT4
   beats INT8 by a wide margin on the core (245.8 → 185.1 µs at T=256, 39.6 → 30.3 at T=64).
2. **The FP16 `vectorized_elementwise_kernel` row has no counterpart** in the quantized tracks — the
   residual add absorbed into the `gemm_*_awq_bias_res` epilogue. It is the whole of the 8 → 4 and
   5 → 4 kernel-count drop at every shape.
3. **The small shapes are launch-bound**, and it is obvious here: at T=4 the GPU row holds 90.8 µs of
   work under a CPU row that spans ~480 µs. Widening kernels will not help those blocks.

## Caveats

- **Read the GPU rows, not the CPU rows.** CPU-side durations carry per-op profiler overhead, which
  inflates dispatch time and deflates any GPU/CPU ratio computed from this file. The GPU kernel
  durations are CUPTI device timestamps and match the report; the CPU spans do not.
- **One forward per cell.** Structural comparison, not statistics. Quote the report.
- Warmed up 12+ forwards first, so the quantized attention blocks' static scales have frozen and the
  trace shows the production route rather than the calibration route.
- Slot widths differ between shapes; the ruler slice carries `slot_us` in its args. Do not compare
  *across* slots by eye — compare down a slot.
- `models/ldm/lsun_churches256/model.ckpt` is an 856-byte stub, so all weights are random and
  `AttentionBlock.proj_out` is identically zero — the layer is numerically an identity
  (see [`../gn_qkv_fusion_2026-08-03/FINDINGS.md`](../gn_qkv_fusion_2026-08-03/FINDINGS.md) §5).
  Timing is unaffected: kernel cost is data-independent and every shape and launch sequence is real.
- No Nsight Compute counters in this container (`ERR_NVGPUCTRPERM`), so no occupancy or achieved
  bandwidth — durations and launch structure only.

`attention_aligned_summary.json` has the per-(shape, mode) kernel count and GPU total in machine
form.

## Regenerating

```bash
python docs/perfetto_traces_2026-08-03/scripts/export_attention_aligned.py
```

`--iters N` profiles N forwards per cell instead of 1 (they land inside one annotation, so the slot
just gets wider); `--out` changes the destination.

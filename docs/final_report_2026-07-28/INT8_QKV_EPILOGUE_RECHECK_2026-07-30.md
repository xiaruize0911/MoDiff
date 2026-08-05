# INT8 attention recheck: W8A8 QKV INT8 epilogue

> **Correction, 2026-08-03 — the latent-level evidence below is withdrawn.**
> Every "latent relative L2" and whole-layer "bit-exact" result in this document was vacuous.
> `UNetModel.out[-1]` is a `zero_module` (`ldm/modules/diffusionmodules/openaimodel.py:745`) and
> `AttentionBlock.proj_out` is another (`:345`); this tree's checkpoint is an 856-byte stub whose
> `state_dict` has 0 entries, loaded `strict=False`, so both stayed zero. Consequently every
> attention block was a bit-exact identity on its input and the UNet predicted **identically zero**
> for every input, which makes the sampled latent a function of the initial noise and the DDIM
> schedule alone. A latent relative L2 of `0` was guaranteed for any change, correct or not —
> demonstrated by forcing all 21 attention blocks to return a constant during sampling: `forward`
> fired 420 times and the latent was bit-identical. The five goldens in
> `integration/tests/golden/e2e_*_vacuous.pt` are further evidence: fp16, int8 and int4 are all
> bit-identical to each other.
>
> **What still stands:** the kernel-level correctness results — the nine attention shapes compared
> against an fp32 reference computed from the same quantized codes on synthetic tensors
> (`scripts/qattn_correctness.py`), and the code-difference/relative-L2 numbers measured directly on
> kernel outputs. Those need no checkpoint and do not pass through either zero_module.
> **Also unaffected:** every timing in this document. Kernel cost is data-independent, and the
> shapes and launch sequences were real.
>
> The scripts have been fixed (they now activate the zero-initialised layers and assert
> observability before comparing) so re-running them produces meaningful verdicts. Full account:
> [`docs/gn_qkv_fusion_2026-08-03/FINDINGS.md`](../gn_qkv_fusion_2026-08-03/FINDINGS.md) section 5.


Environment: NVIDIA A40, batch 128, 20 warmups, median of 5 rounds × 60
iterations. FP16 and final INT8 results were measured in the same process.

## Retained dataflow

```text
GroupNorm -> INT8
    |
    v
W8A8 QKV GEMM + bias + per-Q/K/V-scale INT8 epilogue
    |
    +-- Q remains packed token-major and is read directly by Flash
    |
    +-- fused K gather + V transpose
                 |
                 v
        INT8 FlashAttention + INT8 qout
                 |
                 v
        W8A8 projection + residual
```

This replaces FP16 QKV materialization and removes independent Q quantization.
The K gather and V transpose are one kernel instead of two.

## Same-module A/B

| Shape | Previous INT8 | QKV epilogue | Candidate speedup |
|---|---:|---:|---:|
| T1024 / hd24 | 2976.35 us | 2834.05 us | 1.050x |
| T256 / hd48 | 988.58 us | 914.98 us | 1.080x |
| T64 / hd48 | 246.75 us | 232.51 us | 1.061x |

## Final same-process FP16 comparison

| Shape | Instances | FP16 | Final INT8 | Speedup |
|---|---:|---:|---:|---:|
| T1024 / C192 | 5 | 3101.70 us | 2833.57 us | 1.095x |
| T256 / C384 | 5 | 1079.87 us | 921.34 us | 1.172x |
| T64 / C384 | 5 | 411.78 us | 234.14 us | 1.759x |
| T16 / C768 | 5 | 221.89 us | 231.17 us | 0.960x |
| T4 / C768 | 1 | 200.00 us | 200.58 us | 0.997x |
| **21-block weighted** | **21** | **24.276 ms** | **21.302 ms** | **1.140x** |

```text
Weighted attention latency (lower is better)

FP16        24.276 ms |████████████████████████|
Previous    22.233 ms |██████████████████████  |
Final INT8  21.302 ms |█████████████████████   |
1.5x goal   16.184 ms |████████████████        |
```

## Correctness and quality

- All nine attention correctness shapes pass; INT8 relative error is
  0.0076–0.0161 against the FP32 reference.
- General kernel correctness suite passes.
- Fixed-noise, batch-4, 50-step DDIM latent comparison passes at seeds 1234,
  5678, and 9012. Candidate versus previous INT8 latent relative L2 is 0 for
  every seed.
- The QKV epilogue route is auto-enabled only for T1024/hd24, T256/hd48, and
  T64/hd48. `MODIFF_INT8_QKV_EPILOGUE=off` restores the previous route.

## Rejected experiments

- Whole-head persistent Packed fusion at T256/T1024: insufficient CTA
  occupancy and too much serial query work.
- Compact global K+V at hd24: producer became faster, but 8-byte K staging made
  Flash slower.
- Compact V-only at hd24: net 0.5% regression.
- Existing direct packed INT8 Flash at T256: 0.454 ms versus 0.387 ms for fused
  K/V producer plus direct-Q Flash.

## Remaining bottleneck

The final T1024 layer still spends approximately:

| Kernel | Time |
|---|---:|
| INT8 FlashAttention | 1469 us |
| W8A8 QKV INT8-output GEMM | 359 us |
| W8A8 projection GEMM | 351 us |
| fused K gather + V transpose | 311 us |
| GroupNorm + INT8 quantize | 268 us |

The next high-leverage implementation is a shape-specialized GEMM epilogue that
directly emits head-major padded Q/K and transposed V. That removes the remaining
311 us repack kernel. Reaching the weighted 1.5x goal additionally requires
reducing the T1024 Flash/GN/GEMM components; repack removal alone is insufficient.

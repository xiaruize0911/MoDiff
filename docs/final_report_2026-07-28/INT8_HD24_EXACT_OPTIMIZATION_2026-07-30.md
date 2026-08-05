# INT8 T1024/hd24 exact specialization

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


## Outcome

The production `T=1024, hd=24, hd_pad=32, BC=32, WARPS=8` shared-P
FlashAttention route now has a compile-time exact specialization. It fixes
`T=1024`, uses exactly three 8-channel PV/output fragments, reads only the
24 real Q channels from the padded direct-layout Q tensor, and clears the
8-channel shared-memory tail.

`MODIFF_INT8_FLASH_HD24_EXACT` accepts only deterministic boolean values.
It defaults to `1`; setting it to `0` restores the generic `HD_PAD=32`
reference. There is no runtime benchmark or `auto` route.

## Validation

- Batch 1, 4, and 128 candidate qout is bit-exact to the generic kernel.
- Twenty repeated batch-4 launches are deterministic.
- A non-default CUDA stream is bit-exact.
- The complete T1024 attention layer output is bit-exact.
- All nine attention correctness shapes pass; INT8 relative L2 remains
  `0.0076–0.0160`.
- The complete kernel correctness suite passes.
- Three batch-4, 50-step DDIM comparisons pass:
  - seed 1234: latent relative L2 `0`, max absolute difference `0`.
  - seed 5678: latent relative L2 `0`, max absolute difference `0`.
  - seed 9012: latent relative L2 `0`, max absolute difference `0`.

## A40 batch-128 benchmark

Protocol: 20 warmups, 5 rounds × 60 iterations, same-process alternating
reference/candidate. The table reports the median of two complete runs.

| Scope | Generic reference | Exact hd24 | Speedup | Saving |
|---|---:|---:|---:|---:|
| Flash kernel, CUDA events | 1650.50 µs | 1586.31 µs | **1.040×** | 64.18 µs |
| Complete T1024 layer | 2785.21 µs | 2748.57 µs | **1.013×** | 36.64 µs |
| 5× T1024 weighted contribution | 13.926 ms | 13.743 ms | **1.013×** | 0.183 ms |

Applying the measured same-process T1024 saving to the preceding
`20.333 ms` 21-block weighted INT8 result gives approximately `20.150 ms`.
Against the `24.314 ms` FP16 result, the updated weighted speedup is
approximately `1.207×`. The `1.5×` target is `16.209 ms`, leaving about
`3.94 ms`.

PyTorch profiler confirms that only Flash changed materially:

| Kernel | Reference | Candidate |
|---|---:|---:|
| INT8 FlashAttention | 1466–1468 µs | 1418–1427 µs |
| QKV W8A8 epilogue | 630–634 µs | 634–641 µs |
| Projection W8A8 | 349 µs | 350–351 µs |
| GN+SiLU+quantize | 265–266 µs | 266–268 µs |

## Static profile

| Resource/instruction | Generic | Exact |
|---|---:|---:|
| Registers/thread | 64 | **56** |
| Shared memory/CTA | 8192 B | 8192 B |
| Stack/local | 0/0 B | 0/0 B |
| Static occupancy ceiling | 66.7% | 66.7% |
| SASS instructions | 1016 | **888** |
| IMMA instructions | 8 | **7** |
| FMUL instructions | 68 | **48** |
| FFMA instructions | 47 | **43** |
| F2I instructions | 22 | **14** |
| Global stores | 24 | **18** |

The register reduction does not cross the next occupancy boundary: both
kernels remain limited to four 256-thread CTAs per SM. The gain therefore
comes from less PV/output work and simpler fixed-shape addressing, not from
increased occupancy.

Nsight Compute was invoked for both kernels, but the host denied performance
counter access with `ERR_NVGPUCTRPERM`. No hardware-counter claims are made;
the profile above uses CUDA traces, `cuobjdump -res-usage`, and a SASS census.

Reproduction:

- `scripts/int8_hd24_exact_bench.py`
- `scripts/int8_hd24_layer_ab.py`
- `scripts/int8_hd24_exact_quality.py`
- `scripts/int8_hd24_ncu_target.py`
- `data/int8_hd24_exact_optimization.json`

# The GroupNorm family was not at the roofline, and the fix was already in the tree

**A40, LSUN-churches LDM, real 2.7 GB checkpoint, batch 128.** One entry-point swap in
`fused_resblock.py`, measured paired in-process:

| arm | ms/step before | after | recovered | vs fp16 |
|---|--:|--:|--:|---|
| W8A8 PTQ | 71.48 | **64.83** | **+6.645** | 1.441× → **1.589×** |
| W4A4 PTQ | 58.15 | **50.84** | **+7.243** | 1.771× → **2.026×** |

Predicted from the kernel microbenchmark *before* the end-to-end run: +6.907 and +7.562. Measured
+6.645 and +7.243 — 3.8% and 4.2% under, in the direction a kernel-level prediction should err, since
isolated-kernel savings do not fully survive overlap. **W4A4 crosses 2× for the first time.**

## 1. What the standing list got wrong

[OPEN_ITEMS.md](../OPEN_ITEMS.md) C1 read: *"GroupNorm+SiLU family — 32.2% of W4A4 at 1.13× — the next
real lever, full stop. Blocked on: no design has landed."* Two claims, both false:

**It is not at the memory roofline.** The implied reason a normalization kernel would sit at 1.13× is
that quantization only shrinks the *output* write while the fp16 input read is irreducible. That story
predicts the kernels are bandwidth-saturated. They are not:

| shape | kernel | µs | GB/s | % of A40 peak |
|---|---|--:|--:|--:|
| `[128,384,8,8]` | `group_norm_silu_quantize_nhwc` | 183.2 | 86 | **12%** |
| `[128,384,16,16]` | `group_norm_silu_quantize_nhwc` | 297.6 | 211 | 30% |
| `[128,192,32,32]` | `group_norm_silu_quantize_nhwc` | 486.3 | 259 | 37% |
| `[128,192,32,32]` | `group_norm_silu_quantize_nhwc_fast` | **280.2** | **449** | **65%** |

(Traffic counted as a two-pass kernel: read for stats, read + write to normalize. Peak 696 GB/s.)

The last two rows are the *same shape and the same work*, 1.74× apart. A kernel at 12–37% of peak is
not traffic-bound, and the 1.13× was never a property of the problem.

**And the design had landed — just not here.** `..._fast` is not a different kernel. It is the same
`group_norm_silu_quantize_nhwc_impl` with `fast_reduce=true`, which swaps the block-size heuristic:

```
if (fast_reduce) {
    // About six pairs/thread gives the best latency/occupancy balance on A40 [...] The old generic
    // heuristic launched 1024 threads and was 1.27-4.3x slower after warp reductions.
    block_size = 128;
    while ((long)block_size * 12 < group_size && block_size < 512) block_size <<= 1;
}
```

The kernel's own comment records the generic path as 1.27–4.3× slower. The attention paths took the
fast entry point via `getattr(_mc, "..._fast", <plain>)` from the day it was written.
`fused_resblock.py` named the plain entry point directly — and that file owns **62 of the 83 GN
calls/step**. So the family sat at a third of peak with the fix exported, tested and reachable.

## 2. Per shape, at the real call counts

Every GN signature the 2026-08-13 capture recorded, replayed at its captured shape with **production
modulation** (`ms2d`/`sh2d` non-empty — the empty-mod branch is a different dispatch and measuring it
would answer a question nobody asked). `calls/step` is the captured count ÷ `capture_steps`.

| shape | calls/step | plain µs | fast µs | speedup |
|---|--:|--:|--:|--:|
| `[128,576,32,32]` | 1.0 | 1073.3 | 850.7 | 1.26× |
| `[128,384,32,32]` | 3.0 | 772.6 | 552.5 | 1.40× |
| `[128,192,32,32]` | 7.0 | 482.6 | 296.4 | 1.63× |
| `[128,768,16,16]` | 2.0 | 435.6 | 242.5 | 1.80× |
| `[128,576,16,16]` | 1.0 | 389.5 | 199.3 | 1.95× |
| `[128,384,16,16]` | 7.0 | 303.8 | 126.5 | 2.40× |
| `[128,1152,8,8]` | 1.0 | 274.1 | 98.5 | 2.78× |
| `[128,192,16,16]` | 2.0 | 240.3 | 80.5 | 2.98× |
| `[128,768,8,8]` | 3.0 | 223.6 | 68.9 | 3.25× |
| `[128,384,8,8]` | 8.0 | 190.8 | 41.9 | **4.55×** |
| `[128,1536,4,4]` | 2.0 | 183.0 | 41.2 | 4.45× |
| `[128,1152,4,4]` | 1.0 | 180.6 | 36.8 | **4.91×** |
| `[128,768,4,4]` | 7.0 | 60.1 | 26.9 | 2.23× |
| `[128,384,4,4]` | 2.0 | 33.4 | 21.0 | 1.59× |
| `[128,1536,2,2]` | 3.0 | 33.3 | 20.8 | 1.60× |
| `[128,768,2,2]` | 12.0 | 22.1 | 19.5 | 1.12× |
| **weighted** | **62** | **14.507** | **7.600** | **1.91×** |

The int4 sibling (`group_norm_silu_quantize_pack_nhwc`) is the same story at 14.931 → 7.369 = 2.03×.

Two things worth reading off this table. The gain is **largest where the shape is smallest** — 4.5–4.9×
at `8×8` and `4×4`, 1.1–1.3× at `32×32` — because the generic heuristic's 1024 threads are catastrophic
for occupancy exactly when there is not enough work per group to fill them. And the plain column
reproduces the capture: 14.507 ms/step against the capture's 13.97, on different random data.

## 3. Numerics: not bit-identical, and not resolved as a quality difference

`fast_reduce` changes the fp32 reduction order, so a mean/inv_std moves in the last bits and a value
sitting exactly on a quantize code boundary can land either side. Measured, all 16 shapes, both
precisions, production modulation:

| | max \|Δcode\| | fraction of elements that move |
|---|--:|--:|
| int8 | 1 | ≤ 1.6e-5% |
| int4 (packed byte) | 16 | ≤ 1.1e-5% |

The int4 column reads 16 because two codes share a byte: a 1-code change in the high nibble *is* 16.
Both rows are one code, on one element in ten million.

End to end, latent relL2 against a per-seed fp16 reference, PTQ arms, batch 8, 50 steps, 8 seeds:

| arm | OFF (generic) | ON (fast_reduce) | paired diff | ON better on | verdict |
|---|--:|--:|--:|--:|---|
| W8A8 | 0.1194 | 0.1199 | +0.46% ± 0.73% (SEM) | 4/8 seeds | **not resolved** — inside 2×SEM |
| W4A4 | 0.4817 | 0.4860 | **+0.91% ± 0.27%** | 1/8 seeds | **resolved** — 0.91% worse |

So the cost is not zero, and the two arms differ in whether it is even measurable:

**At W8A8 — the shipping configuration — it is not resolved, and the point estimate shrank as *n* grew**:
+1.27% at 4 seeds, +0.46% at 8, with ON ahead on half the seeds. That is the behaviour of an effect near
zero, and it is exactly the reversal-toward-nothing that `docs/act_bits_2026-08-05` warned a 3-seed mean
would miss. Against +6.65 ms/step (9.3%), nothing here argues for leaving 1.9× on the table.

**At W4A4 it IS resolved: 0.91% worse, 0.4817 → 0.4860.** Small, real, and reported rather than buried.
For scale: the two calibration constants in `docs/paper_repro_2026-08-12` moved this same number from
0.8642 to 0.4695. 0.0044 absolute against that 0.39 is inside the noise of what W4A4's problems are —
and W4A4 is not usable at either setting anyway (FID 200 vs PTQ's 278; the dominant error is in the
weights). The 12.5% speedup is worth 0.0044 of relL2 on an arm whose blocker is elsewhere.

`MODIFF_GN_FAST=0` restores the old kernel exactly, which is what makes both rows falsifiable rather
than arguments.

**Two things this measurement had to be fixed to say anything at all**, both worth recording because
both produced a confident wrong answer first:

1. **The first version measured arm-to-arm latent relL2** and got 7.5e-3. That answers the wrong
   question: both arms approximate fp16, and a swap that rounds differently while landing the same
   distance from the truth is free. Scoring each arm against the *same fp16 reference* is the
   instrument `quality_route_b_paired.py` already established.
2. **The first version ran mode `int8` and reported BIT-IDENTICAL.** In MoDiff mode the ResBlock takes
   the GN→delta-quantize fusion, which this swap does not touch — the flag was **inert** and the gate
   was vacuous. It now counts the plain entry point in both arms and fails if the two arms did not run
   different code (12400 calls OFF, 0 ON).

## 4. What shipped

`MODIFF_GN_FAST`, **default ON**, read at call time so an in-process A/B can flip it. Three call sites
in `fused_resblock.py` now route through one helper:

```python
def _gnq(name):
    """The fast entry point when it exists and is enabled, else the one this file always used."""
    if os.environ.get("MODIFF_GN_FAST", "1") == "1":
        f = getattr(modiff_cutlass, name + "_fast", None)
        if f is not None:
            return f
    return getattr(modiff_cutlass, name)
```

covering `group_norm_silu_quantize_nhwc`, `group_norm_silu_quantize_pack_nhwc` and
`group_norm_silu_quantize_pack_nhwc_zp` (the zero-point variant has a `_fast_zp` sibling, so the
asymmetric path is not left behind).

**The counters are what make the timing trustworthy.** `_gnq` declines silently if the symbol is
missing, and the A/B would then time the same code twice and report a believable ~0. Both entry points
are counted per arm, and three invariants asserted: plain is 0 with the swap ON, 62/step with it OFF,
and `plain + fast` is **conserved** (83/step in both arms) — the swap relocates calls, it does not add
or remove any.

## 5. What this does to the standing numbers

Every speed number in [bench_report_2026-08-13_postzp](bench_report_2026-08-13_postzp/REPORT.md) was
measured on the pre-swap tree and is now stale on the PTQ arms. The report has not been re-run; what
changes, and by how much, is the table at the top of this file. The per-conv and per-attention tables
are unaffected — this touches neither.

The lever it retires is the largest one on the list, so the ranking behind it moves up:

| | before | after |
|---|---|---|
| GroupNorm+SiLU family, share of the W4A4 run | 32.2% at 1.13× | **~24% at ~1.9×** |
| next-largest lever | this | the T=1024/hd=24 attention route (15.6 ms/sample at 1.21×, needs a new kernel) |

## 6. Reproduce

```bash
python docs/gn_fast_reduce_2026-08-16/scripts/gn_fast_vs_generic.py
```
```bash
python integration/tests/ab_gn_fast_reduce.py --mode int8_baseline --steps 100
```
```bash
python integration/tests/ab_gn_fast_reduce.py --mode int4_baseline --steps 100
```
```bash
python integration/tests/quality_gn_fast_paired.py --seeds 8 --bits 8
```

The first is CPU-cheap but wants an idle GPU (~2 min). The two A/Bs are ~6 min each. All three assert
their own call counts and fail loudly rather than reporting a believable zero.

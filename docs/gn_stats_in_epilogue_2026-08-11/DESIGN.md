# Folding the GroupNorm statistics into the producing conv's epilogue

2026-08-11. Design and de-risking plan for removing `gn_stats_partials_chanmajor_kernel` from the
MoDiff conv path by having the conv that produces `x` emit the GroupNorm partial sums as an auxiliary
output. Stage A (prototype, de-risk) then Stage C (CUTLASS EVT node), per the agreed order. Stage B
(the `cat2` route) is explicitly deferred, not dropped — see "Ceiling".

Nothing here is measured yet except the ceiling and the call counts. This document exists so the EVT
work starts from a specification rather than from an intuition.

## What is being removed, and what it costs today

`modiff_full_k1`, batch 128, from `docs/profile_kernels_layers_2026-08-11/data/trace_buckets.json`:

| kernel | ms/step | calls/step |
|---|---:|---:|
| `gn_stats_partials_chanmajor_kernel` | **4.75** | 83 |
| `gn_stats_reduce_partials_kernel` | 0.50 | 83 |

The first is a full read of `x` whose only product is `mean`/`inv_std` per `(n, group)`. The second is
a tiny reduction over the partials and is **kept** — the design reuses it verbatim.

This pass exists only on the MoDiff path. The baseline's `group_norm_silu_quantize_nhwc_vec2_kernel`
is block-per-`(n, group)` and accumulates sum/sumsq in registers in the same launch that applies the
norm. The delta path is *flat* (grid-strided over elements) because the delta absmax is a
whole-tensor reduction, so it cannot use the block-per-group shape, and therefore needs mean/rstd
precomputed. That is the whole reason for the extra pass.

## Ceiling: 68 of 83 sites, ≈ 4.3 ms

The call counts bound it without any new measurement:

| | calls/step |
|---|---:|
| `gn_stats_partials_chanmajor` (GN sites) | 83 |
| `cutlass::Kernel<modiff::ImplicitGemmConvolutionEVT…>` (both shapes) | 70 |
| `cat2_channels_last_fp16_kernel` | 15 |

**15 GN sites read a concatenation, not a conv output.** The decoder ResBlocks' GroupNorm sees the
concatenated width (1152/1536), with `cat2_channels_last_fp16` between the conv and the norm. Those
statistics cannot come out of a conv epilogue: at epilogue time the other half of the concatenation
does not exist yet. So the reachable set is ~68 of 83 and the ceiling is **≈ 4.3 ms/step** (68/83 ×
5.25, keeping `gn_stats_reduce_partials`), i.e. ~4% of the 105.42 ms/step step. Stage B — a
concat kernel that also emits partials — would recover the remaining ~0.9 ms and is cheaper work than
this, which is why it was offered first; it is deferred by choice.

One thing already works in our favour, built for another reason: `conv2d_int8_evt_o_hat_residual`
folds the ResBlock skip-add into the conv's accumulate epilogue, and its own comment notes this
"restores a direct producer/consumer relationship between this conv's output and the next
GroupNorm's input, with no intervening op". The adjacency this design needs is largely already there.

## The mapping problem

CUTLASS implicit-GEMM conv, NHWC: `M = N·P·Q` (batch × output spatial), and the GEMM's N dimension is
the output channel count `C`. An epilogue tile covers `[m0, m0+Mt) × [n0, n0+Nt)`.

GroupNorm needs, for each `(n, g)`: the sum and sum-of-squares over all `(p, q)` and over the `CPG =
C/G` channels of group `g`. So a tile contributes a **partial** over the rows it holds, for **each
group its column range touches**. Two indexing facts decide the design:

The tile shape is **verified, not assumed**: `conv2d_evt.cu:85-99` instantiates a single
`cutlass::gemm::GemmShape<128,128,128>` threadblock for both precisions (warp `<64,64,64>` int8,
`<64,64,128>` int4). So `Mt = Nt = 128`, one shape, no family to cover.

**Along N, tiles straddle groups.** `G = 32`, so `CPG = C/32`: 6 at C=192, 12 at C=384, 24 at C=768,
36 at C=1152, 48 at C=1536. `128 / CPG` is 21.3, 10.7, 5.3, 3.6, 2.7 — never integral at the small
widths. **Group boundaries fall inside tiles.** The epilogue must derive `g = (n0 + j) / CPG` per
element rather than per tile.

**Along M, tiles straddle samples.** `n = m / (P·Q)`, and `P·Q` is 1024, 256, 64, 16 down the UNet.
`P·Q % 128 == 0` holds at 1024 and 256 but not at 64 or 16, where **one tile spans 2 and 8 samples**
respectively. The epilogue must derive `n = (m0 + i) / (P·Q)` per element too.

Neither is expensive — both are integer divides by a compile-time-unknown-but-uniform constant, and
can be strength-reduced with a precomputed magic number — but both mean the accumulator cannot be a
single scalar pair per tile. It is a small array indexed by `(n_local, g_local)`.

## Partials buffer

Reuse the shape `gn_launch_group_stats` already allocates and `gn_stats_reduce_partials` already
consumes:

```
part_sum, part_sumsq : float[N * G * NBLK]      // NBLK = number of M-tiles, not min(HW,32)
```

with tile `t` writing slot `[((n * G) + g) * NBLK + t]`. `NBLK` becomes the conv's M-tile count,
which the host knows from the problem size and the chosen tile shape. `gn_stats_reduce_partials` takes
`nblocks` as an argument already, so it needs no change.

**Per-tile slots, never atomics.** `MODIFF_GN_STATS_ALT=2` tried an atomic GN reduction and was
measured 1.7× *slower* than the group-major tree **and nondeterministic** — two replays of one seed
gave latents differing by 1.27e-1 (`group_norm_silu.cu`, `gn_launch_group_stats`). A fixed slot per
tile keeps the summation order a function of the launch geometry alone, which is what makes replays
bit-identical.

## Stage A — prototype, and what it has to prove

A standalone kernel that mimics the epilogue's tile decomposition without touching CUTLASS:
`gn_stats_from_tiles_kernel(x, part_sum, part_sumsq, C, HW, G, Mt, Nt)`, gridded exactly as the conv's
epilogue is, each block accumulating its `(n_local, g_local)` array and writing its slot. Then the
existing `gn_stats_reduce_partials`.

Three things it must show, in this order, because each one can kill the design:

1. **Correctness against the shipped path.** `mean`/`inv_std` from the prototype versus
   `gn_launch_group_stats` on the eight real conv output shapes. Not bit-identical — the summation
   order differs by construction — so the gate is a relative error small enough that the downstream
   int8 codes do not move: `max_code_diff == 0` on the delta quantize, which is the same gate
   `test_gn_resize_fusion.py` uses. A code that moves by 1 is not automatically a failure, but it must
   be *explained* rather than accepted.
2. **Determinism.** Same input, ten launches, `torch.equal` on the partials. If per-tile slots do not
   give this, the design is dead and the reason is worth writing down.
3. **The shared-memory cost of the `(n_local, g_local)` array.** Enumerated over the real
   `(C, P·Q)` pairs at the verified 128×128 tile:

   | C | CPG | P·Q | n/tile | groups/tile | acc pairs | shared |
   |---:|---:|---:|---:|---:|---:|---:|
   | 192 | 6 | 1024 | 1 | 23 | 23 | 184 B |
   | 384 | 12 | 64 | 2 | 12 | 24 | 192 B |
   | 768 | 24 | 16 | **8** | 7 | **56** | **448 B** |
   | 1536 | 48 | 16 | 8 | 4 | 32 | 256 B |

   **Worst case 56 pairs, 448 B of shared per threadblock.** An earlier draft of this document put it
   at ~176 pairs by pairing the widest group count with the deepest sample count; those do not
   co-occur, because a tile spans many samples only where the spatial extent is small, and small
   spatial extent goes with large `C` and therefore large `CPG` and *fewer* groups per tile. The two
   worst cases are anti-correlated. 448 B is negligible against a CUTLASS epilogue's shared budget,
   so **this is no longer the leading risk** — the prototype should still report occupancy, but as a
   check rather than as a gate.

Prototype cost is one kernel in an existing translation unit plus one pybind entry: a two-minute
rebuild, no CUTLASS involvement.

## Stage C — the EVT node

Only after A. A custom EVT node, since no stock CUTLASS node does a grouped reduction along N
combined with a partial along M (`EVTColumnReduction` and friends are single-axis), and
`implicit_gemm_conv_evt.h` currently exposes no auxiliary-output mechanism at all — it has only the
`Epilogue` / `EpilogueOutputOp` typedefs. Order of work:

1. Auxiliary-output plumbing in `implicit_gemm_conv_evt.h` (two pointers + `NBLK` through `Params`).
2. The node itself, sharing the accumulator layout the prototype validated.
3. Wire it on `conv2d_int8_evt_o_hat` / `_residual` only, behind `MODIFF_GN_STATS_IN_EPILOGUE=1`,
   default off, so the fused and unfused paths can be A/B'd in one process — the same discipline
   `MODIFF_UPDOWN_FUSE_REFRESH` exists for, and the reason that measurement was trustworthy.
4. Python: `gn_launch_group_stats` takes the precomputed partials when the producer supplied them and
   skips its own pass. The consumer must **verify** they were supplied rather than assume it; a
   silently skipped stats pass is a wrong-normalisation bug that will not crash.

## What would make me stop

* Shared memory at the worst-case shapes — downgraded from leading risk to a check after the tile
  shape was verified (Stage A item 3). The remaining unknown is not capacity but whether the
  per-thread fragment reduction into that array costs more than the 4.75 ms pass it replaces.
* `max_code_diff > 1` from the reduction-order change, unexplained.
* Any nondeterminism.
* The 68-site figure not holding once the producer of each GN input is enumerated per site rather
  than inferred from kernel call counts. That enumeration has NOT been done — 68 is `83 − 15`, an
  arithmetic bound, not a per-site audit. `fusion_audit.py` is the place to add it.

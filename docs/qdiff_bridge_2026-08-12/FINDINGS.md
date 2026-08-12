# Q-Diffusion calibration: the model was wrong, and now it is less wrong

**The advisor's hypothesis is confirmed.** Replacing the shipped activation scales with
Q-Diffusion-derived ones takes the W8A8 baseline's latent relL2 from **0.2564 to 0.1119 — 2.29×** —
on 3 of 3 seeds, measured on the real CUDA kernels. A fake-quantization harness predicted **2.36×**
before any kernel ran.

All numbers: real LSUN-churches checkpoint, DDIM S=50, batch 8, seeds {1234, 20260805, 777}, latent
relL2 vs a per-seed fp16 reference, at steady state (run 1 discarded).

---

## 0. What the model actually was

Asked directly: *is this model static Q-Diffusion quantized?* No — and on three of four axes it was
not even static.

| axis | static? | Q-Diffusion? | what it actually was |
|---|---|---|---|
| conv activation (baseline, MoDiff t=T) | yes, from `.pt` | **no** | `mean(127/absmax_i)` over calls |
| MoDiff delta (t<T) | **no** — per-call, refresh K=4 | **no** | `delta_absmax_fp16` on device |
| conv/linear weights | yes | **no** | per-channel absmax / MSE. No AdaRound |
| attention Q/K/V | **no** — re-derived each process | **no** | 8-forward observe-then-freeze |

`import qdiff` raised `ModuleNotFoundError`; the package had been deleted in `c9ade7c`. The
calibration file was 70 bare floats with no `delta`/`zero_point`/`alpha`, so structurally not a
Q-Diffusion artifact. And `apply_int8_delta_scales` had **zero call sites** in `integration/`, so the
static delta table was never loaded even when the file existed.

## 1. Why the old scales clipped

The shipped scale is `mean(127 / absmax_i)` over calibration calls
([int8_optimized.py:1620](integration/kernels/int8_optimized.py:1620)) — a **mean of reciprocals**,
so it is dominated by the calls with the *smallest* absmax. A layer whose activation range swings
across timesteps therefore gets a wildly inflated scale, and an inflated scale means
`max|x| · scale > 127`, i.e. clipping.

The worst layer is `middle_block.0.out_conv`: **572.7 → 39.5, 14.5× too large**. It sits deep in the
network where the range swings most. Across all 70 layers the new scales are a median **0.348×** the
old ones; 57 shrink, 13 grow.

> The plan for this work predicted the opposite direction (`ratio ∈ [3, 15]`). That was wrong:
> relieving clipping requires a *smaller* scale, not a larger one.

## 2. The answer is different on the two axes

This is why the decision was left open rather than assumed. Fake-quant A/B, 3 seeds:

| arm | relL2 mean | per seed | clip% | layers clipping |
|---|---:|---|---:|---:|
| baseline / shipped | 0.2558 | .2377 .2685 .2612 | 0.220 | 67.3/70 |
| **baseline / qdiff sym** | **0.1082** | .0939 .1536 .0772 | 0.012 | 57.7/70 |
| baseline / qdiff mse | 0.1135 | .1003 .1573 .0827 | 0.012 | 62.7/70 |
| **modiff / shipped + dynamic** | **0.0069** | .0100 .0060 .0046 | 0.005 | 70/70 |
| modiff / qdiff + dynamic | 0.0088 | .0100 .0121 .0043 | 0.001 | 70/70 |
| modiff / shipped + native table | 0.0175 | .0221 .0079 .0225 | 0.005 | 62/70 |
| modiff / qdiff + qdiff table | 0.0317 | .0267 .0436 .0247 | 0.001 | 20/70 |
| modiff / qdiff + table, flat | 0.0240 | .0161 .0399 .0160 | 0.001 | 20/70 |

**Activation scales: switch.** qdiff wins 3/3 seeds, by 2.5×, 1.75× and 3.4×.

**Delta path: do NOT switch.** Every static delta table loses to today's dynamic path — 0.0317 and
0.0240 against **0.0069**, a 3.5–4.6× regression on all three seeds. Making the model "fully static
Q-Diffusion" would substantially degrade it.

The mechanism is in the same table and it is *not* clipping. The qdiff table cuts clipping hard —
**20/70 layers against 62–70/70** — and is still worse. A static per-step scale has to cover the worst
sample in the calibration batch, so it carries envelope headroom a per-call scale does not; with 255
levels that coarseness costs more than the clipping it prevents. `docs/modiff_correctness_2026-08-03`
reached the same conclusion from the other direction: *"dynamic is required at W4A4 and optional at
W8A8."*

**Symmetric beats asymmetric-MSE** (0.1082 vs 0.1135), so the option that is bit-exactly
integration's own quantizer — and therefore exports losslessly — is also the better one. The two
differ by ≤15% on any layer. The oracle arm the plan reserved for this question was not needed.

## 3. Real kernels agree with the prediction

| arm | relL2 mean | per seed | fake-quant said |
|---|---:|---|---:|
| baseline / shipped | 0.2564 | .2376 .2701 .2616 | 0.2558 |
| **baseline / qdiff** | **0.1119** | .0986 .1526 .0844 | 0.1082 |
| MoDiff / shipped | 0.0567 | .0462 .0877 .0363 | — |
| **MoDiff / qdiff** | **0.0528** | .0409 .0838 .0339 | — |

**2.29× measured against 2.36× predicted.** MoDiff a wash (1.07×), also as predicted.

Baseline absolutes match the harness to ~3% because activation error dominates there. MoDiff
absolutes are 8× apart (0.0567 vs 0.0069) because the harness leaves **weights in fp16**: under MoDiff
the activation error is small, so what remains is mostly weight error the harness does not model.
That is a stated idealisation, not a surprise.

**Do not read `ms/step` from `data/qdiff_ab.json`.** The four arms were built sequentially in one
process, so the first pays cuDNN autotuning the second inherits. It is run order, not a scale effect.

## 4. Why nobody noticed for so long

**MoDiff was masking it.** MoDiff reads the static activation scale only at t=T; for t<T its dynamic
delta path derives its own scale and never touches it. So MoDiff pays the bad scale on 1 step in 50
while the baseline pays it on all 50. Every headline comparison in this project is a MoDiff arm.

## 5. What changed in the tree

* `qdiff/` restored from `c9ade7c^` — 8 files, 2157 lines, no `requirements.txt` change.
* `scripts/sample_diffusion_ldm.py`: `--no_ema`, `--skip_weight_recon`, and a **fix to the
  non-`--modulate` activation path**, which raised `TypeError` on every invocation because it passed
  `min_max`/`out_penalty` to `layer_reconstruction`, which does not accept them. That path had
  evidently never been run, and it is exactly the run this bridge needs.
* `benchmark_ldm.py`: `CALIBRATION_PREFERENCE` replaces a hardcoded default. **The old auto-default
  was the stub-derived file** that FINDINGS measures at relL2 0.882 — "worse than useless". Every
  harness dodged it by passing an explicit path, so nothing caught it. It is now last.
* `measure_utilisation.py`: `MU_CALIB_PATH` (score a file rather than calibrating live), `MU_MODES`.
* `dynamic_delta_ab.py`: `AB_CALIB8`/`AB_CALIB4`, defaults **unchanged on purpose** so its committed
  numbers keep meaning what they meant.

## 6. Traps that would have produced plausible-but-wrong scales

1. **EMA.** `sample_diffusion_ldm.py:527` swaps in EMA weights; `benchmark_ldm.py:152` does not.
   Measured: **0/70 conv weights match, worst 13.8% relative L2.** Gated by
   `scripts/assert_same_network.py`, which asserts 70/70 without EMA *and* 0/70 with it — if EMA had
   been a no-op, `--no_ema` would be dead code.
2. **Two runs, not one.** Under `--modulate` the activation quantizer is never called on `a_T` at all,
   so that run structurally cannot produce an activation scale. The exporter reads `modulate` from the
   run's own config and **refuses** a mismatched `--kind`; tested against the real artifact.
3. **`middle_block` has one index level**, not two. The first name-map regex required `\.\d+\.\d+\.`
   and silently dropped 4 keys — and a dropped key is invisible at runtime, the layer just keeps
   `static_input_scale = 1.0`. Caught by a set-equality assertion.
4. **A stale reference file.** `data/dynamic_delta_ab.json`'s int8-dynamic row reads **10.32** from a
   diverged pre-warm-up capture; FINDINGS carries an explicit correction saying that first version
   "reported the opposite for W8A8". Graded against 10.32, every arm would have looked like a triumph.
5. **`--a_sym` without `--a_min_max`** optimises garbage (the MSE branch computes `zero_point` without
   checking `sym`). The exporter refuses the combination.

## 7. Measured, and it did not work

The `--delta-head` policy — clamp the first H steps to `min(qdiff_scale, act_scale/2)`, provably
non-clipping since `|a_t − â_{t+1}| ≤ 2·act_absmax` — **hurts**: flat `H=0` reads 0.0240 against
`H=2`'s 0.0317. `min()` picks the coarser grid to buy the guarantee, and the coarseness costs more
than the guarantee is worth. Emitting both files so it could be measured is what caught it.

## 8. Open

1. **Clipping is reduced, not eliminated.** On real kernels in MoDiff mode, utilisation medians drop
   5.2×/5.8× (`in_conv` 712.8→138.0, `out_conv` 1733.4→296.7) but both stay above Q=127 and
   `out_conv` still clips on 35/35. The plan's target of ≤5/35 is **not met**.
2. **The utilisation instrument cannot measure `int8_baseline`** — the SiLU and Upsample fusions feed
   those convs pre-quantized int8 codes via `forward_from_int8`, and it records only float inputs. For
   the baseline arm the fake-quant harness is the instrument.
3. **Weights and attention Q/K/V are still not Q-Diffusion.** Weight AdaRound was scoped out
   (day-scale; README:43 says the work targets activations). Attention scales are still re-derived
   live each process and never persisted.
4. **4 of 70 layers have a delta range wider than their activation range** — all high-resolution
   `output_blocks` out_convs — so MoDiff buys nothing there. Median across the other 66 is **1.51 bits
   saved**.
5. **FID exists, but on the OLD scales.** I twice wrote "no FID yet" -- wrong.
   `docs/fid_2026-08-05/data/fid.json` has all five modes at N=10000 against a real LSUN reference,
   and `generate_fid_samples.py --linear` defaults to 0, so its `int8_modiff` row IS the conv-only
   configuration Stage D just made the default:

   | mode | FID vs real | FID vs fp16 |
   |---|---:|---:|
   | fp16 | 7.803 | 0 |
   | W8A8 PTQ baseline | 16.366 | 6.394 |
   | **W8A8 MoDiff (conv-only)** | **7.802** | **0.175** |
   | W4A4 baseline | 277.96 | 277.98 |
   | W4A4 MoDiff | 200.14 | 191.09 |

   That is a stronger endorsement of Stage D than the relL2 evidence it was decided on: the new
   default is FID-indistinguishable from fp16 while the PTQ baseline is 2.1x worse.

   **What is genuinely missing is FID on the Q-DIFFUSION scales** -- that run used
   `int8_calibration_realckpt.pt`. The open question is whether the baseline's 16.366 moves with the
   2.29x relL2 improvement. If it does, W8A8 PTQ -- the fastest arm at 1.453x -- becomes quality
   viable, and this project's headline ("every MoDiff arm is slower than PTQ") gains its missing
   other half. The real-image reference is still on disk at /workspace/fid/real, so only the
   generation has to be redone.
6. **`int4` is untouched.** Everything here is W8A8.

## Reproducing

```bash
python docs/qdiff_bridge_2026-08-12/scripts/assert_same_network.py    # A0 gate, ~2 min
python docs/qdiff_bridge_2026-08-12/scripts/smoke_qdiff.py            # A1 gate, ~5 s, no GPU
bash   docs/qdiff_bridge_2026-08-12/scripts/run_calibration.sh        # A3-A5, ~21 min
python docs/qdiff_bridge_2026-08-12/scripts/export_qdiff_scales.py \
       --run docs/qdiff_bridge_2026-08-12/qdiff_runs/act_sym --kind static \
       --out docs/qdiff_bridge_2026-08-12/data/qdiff_act_sym.pt --dry-run
python docs/qdiff_bridge_2026-08-12/scripts/act_fake_quant.py         # A7, no CUDA kernels
python docs/qdiff_bridge_2026-08-12/scripts/qdiff_ab.py               # A9, the 2x2
```

`.pt` artifacts are gitignored and regenerable; scripts, `data/*.json` and this file are committed.

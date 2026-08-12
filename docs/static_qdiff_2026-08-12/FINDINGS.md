# Static Q-Diffusion, shipped for fidelity — and an 18× unit error found on the way in

The tree now follows the paper's own second reproduction path, README:96:

```
--modulate --quant_mode qdiff --cali_min_max
```

Both the activation scale and the per-step delta table are static and Q-Diffusion-derived.
**This is a fidelity decision, not a quality one**, and the bill is §3. `MODIFF_DELTA_MODE=dynamic`
reverts the delta half in one environment variable.

Three things had to be true before the switch could even be measured, and none of them were:

1. **The static delta path was dead code.** `apply_int8_delta_scales` and `apply_int4_delta_scales`
   had **zero call sites** in `integration/`, so no table was ever loaded and `MODIFF_DELTA_MODE=static`
   ran on an uncalibrated grid. At int4 there was no table to load at all — it silently fell back to
   quantizing the temporal delta on the *full activation* grid, which per Theorem 4.3 leaves the
   error unchanged from baseline, i.e. MoDiff buying only error feedback.
2. **int4 had no delta calibration.** The existing `delta` run is 8-bit. §2 has the new one.
3. **The int4 activation export was wrong by 18.1×** — §1, and it invalidates every W4A4 qdiff
   number in `docs/qdiff_bridge_2026-08-12`.

---

## 1. The unit error, and the four hypotheses it explains

`export_qdiff_scales.py` emitted `127/absmax` for every target. That is exactly right for int8 and
wrong by `127/7 = 18.1×` for int4:

| | int8 | int4 |
|---|---|---|
| `set_static_scale` on load | `scale *= act_q/127` (`int8_optimized.py:1831`) | **no rescale**, stores the float as given |
| native delta table | `127/absmax` | `Q_DELTA/absmax`, **`Q_DELTA = 7.0`** |
| kernel clamp | ±127 | **±7** |

So the shipped int4 qdiff file's median scale of **33.694** was a *correct* 127-based encoding of
absmax 3.769 — and 3.769 is very nearly right, since the true observed range recovered independently
from the shipped calibration has a median of **3.937**. The kernel then read it as 7-based, i.e. as
an assumed absmax of **0.208**, a grid 19× too small. Nearly every activation railed at ±7.

That is the whole of `qdiff_bridge`'s §5c — *"saturated blue and white blobs … codes railing at
±7"* — and it is why §5b's four candidate causes were each refuted by measurement: **none of them was
the cause**. The scales were never the wrong scales. They were the right scales in the wrong units.

Fixed at the source (`TARGET_Q = {"int8": 127.0, "int4": 7.0}`), applied before the shipped-ratio
diagnostics so those compare like with like — reading them in mixed units is what let this survive a
whole session. After the fix the int4 activation export reads a median **1.857**, i.e. an assumed
absmax of 3.77 against the recovered 3.937.

The kernel side was deliberately not touched: the shipped `int4_calibration_realckpt.pt` is already
7-based (`end_calibration` builds it as `7.0/max`), so making `set_static_scale` rescale would have
broken the file that works to fix the one that did not.

## 2. The missing int4 delta calibration

`--modulate --quant_mode qdiff --cali_min_max` at `--weight_bit 4 --act_bit 4`, on the same residual
calibration set as every other arm, 168 layers reconstructed. Appended to `run_calibration.sh` with
its export, so the four shipped artifacts are reproducible from one script rather than from memory.

Both delta exports use `--delta-head 0`. The default head policy clamps the first H steps to
`min(qdiff_scale, act_scale/2)` for a provable non-clipping guarantee, and `qdiff_bridge` §8 measured
that guarantee as a **loss** (flat 0.0240 against H=2's 0.0317): `min()` picks the coarser grid and
the coarseness costs more than the clipping it prevents.

## 3. What the switch costs

Real LSUN-churches checkpoint, DDIM S=50, batch 8, seeds {1234, 20260805, 777}, latent relL2 against
a per-seed fp16 reference, first run per arm discarded, all arms of a table in one process, resolved
through the shipped defaults rather than hand-passed paths.

| | W8A8 | W4A4 |
|---|---:|---:|
| PTQ baseline | **0.1140** | **0.8642** |
| MoDiff, static Q-Diffusion *(shipped)* | **0.0607** | **0.6122** |
| MoDiff, dynamic *(previous default)* | 0.0612 | 0.3577 |
| cost of the switch | **0.99× — a wash** | **1.71× worse** |

**W8A8 static Q-Diffusion costs nothing measurable.** 0.0607 against dynamic's 0.0612 — a wash, on
a seed set where one of the three (20260805) runs hot in both arms. `qdiff_bridge` §2 predicted a
3.5–4.6× regression, but that was the *fake-quant* harness scoring a table the production path could
not even load; on real kernels with the table wired, the prediction does not hold. Paper fidelity is
free here.

**W4A4 static Q-Diffusion costs 1.71×**, and MoDiff does help: 0.6122 against its own PTQ baseline's
0.8642. An earlier version of this table read 1.0469 — *worse* than PTQ, the paper's claim inverted
— which turned out to be a quantize/dequantize desync in the int4 fused path rather than anything
about static calibration. §4a is the debug, and it is worth reading because the first two diagnoses
were both wrong.

The PTQ column moved too, and for a different reason — the delta mode cannot touch it (no modulated
steps), but its activation file is now the qdiff one. W8A8 PTQ 0.2564 → 0.1138 is the 2.25× the
`qdiff_bridge` report already established. W4A4 PTQ 0.8642 supersedes that report's 1.1945, which was
measured through the 18.1× error; it is still worse than the shipped absmax file's 0.7120, so the
qdiff activation scales remain a loss on the W4A4 PTQ axis even with correct units.

## 4. Constancy or staticness? Settled at W8A8, still open at W4A4

qdiff reports **one** `act_quantizer.delta` per layer. `export_qdiff_scales.py` refuses a
per-channel delta and fills all 256 table slots with that single scalar, so the paper's "per-step"
table is not per-step at all. The tree's own `end_delta_calibration` instead observes the delta
absmax at each step index. Measured spread of a layer's table across steps, max/min:

| table | spread |
|---|---:|
| `*_delta_qdiff.pt` (paper) | **1.000** — constant by construction |
| `int4_delta_calibration.pt` (native) | 3.158 |
| `int8_delta_calibration.pt` (native) | 6.464 |

The MoDiff residual shrinks as `t` decreases, so a constant sized for the largest step leaves the
later steps on a grid 3–6× too coarse. That was the hypothesis. It does not survive W8A8.

At W8A8 that prediction is testable and comes out flat — a genuinely per-step table buys nothing
over the constant, and both beat dynamic:

| W8A8 MoDiff static | relL2 |
|---|---:|
| qdiff constant *(shipped)* | 0.0526 |
| native per-step | 0.0534 |
| dynamic | 0.0611 |

So at 8 bits the paper's single-scalar delta is not a compromise at all. 255 levels absorb a 6.5×
step-to-step swing without help.

**At W4A4 the same question could not be answered, because the measurement was standing on a bug.**
§4a.

## 4a. Debugging the W4A4 delta scale — two wrong diagnoses, then the bug

W4A4 MoDiff read **1.0469** against its own PTQ baseline's 0.8642 — MoDiff making the output worse,
the paper's claim inverted — with samples showing accumulating structured scribble rather than the
PTQ arm's fog. Recording the wrong turns, because each was refuted by measurement and the third
answer was not reachable from the first two.

**Wrong diagnosis 1: the table under-sizes the delta and clips every step.** Refuted.
`begin_delta_calibration_int4` records the exact delta absmax the kernel computes:

| | observed delta absmax | table assumes | observed / assumed | layers over range |
|---|---:|---:|---:|---:|
| int8 | 0.7541 | 1.5887 | 0.45× | 0/70 |
| int4 | 1.0217 | 1.8388 | 0.52× | **0/70** |

It over-sizes by ~2×. Nothing clips.

**Wrong diagnosis 2: the table is constant, and 4 bits cannot afford that.** True but not the cause.
The observed delta varies 1.77× across steps, so a constant wastes ~0.95 bit — of four. Yet building
a genuinely per-step table made it *worse*: **18.7152**. Adding headroom appeared to rescue it
(×4 → 0.7583, ×8 → 0.7382, with a cliff between ×2 and ×4), which looked like a fixed-point error:
observe on the dynamic trajectory, deploy on the static one, diverge.

**What it actually was: the fused path quantized and dequantized on different grids.**
`forward_gn_fused_modiff` quantizes with `d_scale` in the pack kernel and dequantizes with `d_alpha`
in the conv epilogue, so they must be reciprocals. They were read from two sources — `d_alpha` from
`_cached_alpha_tensor` (1/activation scale), `d_scale` from the delta table, with the matching alpha
discarded. Consistent while static mode meant "quantize the delta on the activation grid"; silently
wrong the moment a table was loaded, by exactly the delta/activation gain of **2.05×**. The int8 twin
(`int8_optimized.py:932`) always took the pair together, which is why W8A8 was never affected.

It needed a table LOADED to bite, and nothing loaded one until this report wired the static path —
**this work activated a latent bug and then spent two experiments measuring it.** The headroom sweep
was not finding a fixed point; dividing `d_scale` by 8 was partially cancelling a 2.05× dequant
error. Both of those results are void.

After the fix (`ba8b8c9`), with everything else identical:

| W4A4 | before | after |
|---|---:|---:|
| PTQ baseline | 0.8642 | 0.8642 — bit-identical, no modulation to affect |
| MoDiff static *(shipped)* | 1.0469 | **0.6122** |
| MoDiff dynamic | 0.3577 | 0.3577 — bit-identical, overrides both values together |

MoDiff now beats its PTQ baseline at 4 bits, which is the paper's qualitative claim restored on the
static path. The two unaffected arms coming back bit-identical is the control.

**Gated.** `test_int4_gn_fused_delta_table` drives the fused entry point directly and asserts the
invariant that actually holds: with a delta table in force the modulated step works entirely on the
table's grid, so its output must not depend on the activation scale — which enters only through
`a_hat`, fixed by step 1. Perturb the activation scale after seeding and nothing may move. Fixed
reads 0.00e+00; reverted reads 9.81e-01. Two earlier drafts of that gate were vacuous and are
recorded in the commit message.

**What this does NOT resolve.** Constancy versus staticness at W4A4 is still open — every per-step
number above was measured through the desync, so §4's W8A8 answer stands alone. Dynamic is still
1.71× ahead of static at 4 bits.

## 5. Reproducing

```bash
bash   docs/qdiff_bridge_2026-08-12/scripts/run_calibration.sh          # all runs + exports + install
python docs/static_qdiff_2026-08-12/scripts/install_qdiff_defaults.py   # gates + install alone, ~2 s
python docs/static_qdiff_2026-08-12/scripts/static_vs_dynamic_ab.py     # §3, ~15 min
python docs/static_qdiff_2026-08-12/scripts/constant_vs_perstep_ab.py   # §4, ~8 min
```

`.pt` artifacts are gitignored and regenerable; scripts, `data/*.json` and this file are committed.

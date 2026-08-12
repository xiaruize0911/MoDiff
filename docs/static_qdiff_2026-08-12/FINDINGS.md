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
| PTQ baseline | **0.1138** | **0.8642** |
| MoDiff, static Q-Diffusion *(shipped)* | **0.0520** | **1.0469** |
| MoDiff, dynamic *(previous default)* | 0.0611 | 0.3577 |
| cost of the switch | **0.85× — static WINS** | **2.93× worse** |

**W8A8 static Q-Diffusion is not a compromise — it is better.** 0.0520 against dynamic's 0.0611.
`qdiff_bridge` §2 predicted a 3.5–4.6× regression, but that was the *fake-quant* harness scoring a
table the production path could not even load; on real kernels with the table actually wired, the
prediction does not hold. Paper fidelity is free here.

**W4A4 static Q-Diffusion costs 2.93×**, and worse than the ratio: at 1.0469 it is below the PTQ
baseline's 0.8642, meaning **MoDiff actively hurts** under a static 4-bit delta. That is the paper's
central claim inverted, so §4 asks why before the number is quoted anywhere.

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

**At W4A4 the same comparison is confounded and I am not reporting it as a result.** The native
`int4_delta_calibration.pt` reads 14.3789 — 13.8× *worse* than the constant — which is not evidence
that staticness is the cost. `int4_delta_table.py` builds that table against `dynamic_delta_ab`'s
`CALIB`, i.e. the shipped absmax file with **SmoothQuant on**, and this arm applies it on top of the
qdiff file, which has smoothing **off**. MoDiff's `a_hat` cache holds the smoothed activation, so the
delta distribution the table was fitted to is not the one it meets. The arm answers nothing.

Settling it needs the per-step table rebuilt in the shipped configuration, which is one command and
was not run:

```bash
AB_CALIB4=integration/calibration/int4_calibration_qdiff.pt   python docs/modiff_correctness_2026-08-03/scripts/int4_delta_table.py
```

Until that runs, **W4A4's 2.93× is measured but undecomposed** — it is not known how much of it a
per-step qdiff export could recover. The W8A8 result above is the reason to think it might be little:
there, per-step bought nothing.

## 5. Reproducing

```bash
bash   docs/qdiff_bridge_2026-08-12/scripts/run_calibration.sh          # all runs + exports + install
python docs/static_qdiff_2026-08-12/scripts/install_qdiff_defaults.py   # gates + install alone, ~2 s
python docs/static_qdiff_2026-08-12/scripts/static_vs_dynamic_ab.py     # §3, ~15 min
python docs/static_qdiff_2026-08-12/scripts/constant_vs_perstep_ab.py   # §4, ~8 min
```

`.pt` artifacts are gitignored and regenerable; scripts, `data/*.json` and this file are committed.

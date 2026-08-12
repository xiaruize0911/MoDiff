# Aligning W4A4 with the reference: the paper reproduces, and two calibration constants close most of the gap

## What was wrong, and what it took to see it

integration's W4A4 was fog. The paper's own command, run verbatim, is not — so the method was never
the problem. Getting from one to the other took three bug fixes and two constants, and cost two
wrong diagnoses along the way that are recorded here because they were each refuted by measurement.

| | session start | now | |
|---|---:|---:|---|
| W4A4 PTQ | 0.8642 | **0.4695** | 1.84× |
| W4A4 MoDiff | 1.0469 → 0.6122 after `ba8b8c9` | **0.3090** | 2.0× |
| W4A4 MoDiff vs dynamic | 1.71× **worse** | **0.71× better** | — |
| W8A8 PTQ | 0.2564 | 0.1138 | (earlier session) |
| W8A8 MoDiff | 0.0393 | 0.0605 | see the noise floor |

The samples are the real result: W4A4 MoDiff now renders recognisable cathedrals — facades, rose
windows, spires, sky — where it rendered structured scribble at the start
(`docs/state_report_2026-08-12/plots/samples.png`).

## 1. The paper reproduces, and both missing inputs were obtainable

`scripts/sample_diffusion_ldm.py … --modulate --quant_mode qdiff --cali_min_max` (README:96) run
verbatim, with only `-n 50000 → 8` and `-l` changed, produces clean churches — one even reproduces a
shutterstock watermark from the training data (`paper_w4a4_samples.png`). The two inputs the tree
lacked both turned out to be fetchable: `cali_data/church.pt` from the paper's HF dataset (168 MB),
`church_w4a8_ckpt.pth` from Q-Diffusion's Drive folder (2.36 GB, file id
`14ufwzYi90oFcJJKeRVOlmVC9P5yU4Eau`).

**We had never been running the paper's configuration**, in four ways, and two were self-inflicted:

| | paper | what we ran | why |
|---|---|---|---|
| activation quantizer | asymmetric (qdiff default) | `--a_sym` forced symmetric | `apply_static_scales` has no slot for a zero point, so the **kernel format propagated upstream into the calibration command** |
| weights | `--resume_w` from AdaRound W4 ckpt | `--skip_weight_recon`, RTN | the ckpt was not on disk |
| network | EMA | `--no_ema` | integration's loader never swapped EMA |
| calibration data | `cali_data/church.pt` | locally generated residual | not downloaded |

## 2. The two constants that closed most of the gap

Both are the same insight: **at 4 bits, deliberately under-sizing a heavy-tailed range beats covering
it.** Neither needs a kernel change, and each is shared between the live calibration path and
`export_qdiff_scales.py` by import so the two cannot drift.

**`DELTA_CLIP_RATIO = 8.0`** — the MoDiff residual grid. Swept act-only, a clean U:

| ratio | 1 | 2 | 4 | **8** | 16 | 21 | 32 |
|---|---:|---:|---:|---:|---:|---:|---:|
| relL2 | .4945 | .3362 | .1773 | **.1147** | .2193 | .2542 | .3117 |

**`ACT_CLIP_RATIO = 4.5`** — the activation grid. Swept on the **real kernels**, both axes:

| ratio | 1 | 2 | 3 | **4.5** | 6.7 | 10 |
|---|---:|---:|---:|---:|---:|---:|
| PTQ | .8647 | .5482 | .4968 | **.4692** | .5312 | .6373 |
| MoDiff | .3090 | .3176 | .3074 | .3095 | .3121 | .3361 |

MoDiff is insensitive (1.09× across a 10× range) because it reads that grid only at t=T and then
refines `a_hat` with 5 warm-up rounds — so one constant serves both axes.

**A single swept constant beats importing the paper's own per-layer delta values**, which read 0.2452
in our datapath against 0.1147. The optimum follows the trajectory, and ours is not theirs — different
weights, EMA, calibration set and step count. That is why the plan's per-layer MSE search, per-step
histogram and non-fused calibration path all turned out to be unnecessary.

### Two costs, stated rather than buried

* **`ACT_CLIP_RATIO` hurts the dynamic path.** W4A4 dynamic went 0.3577 → 0.4327 (−21%), because
  dynamic also reads the static grid at t=T but recomputes its delta per call and gains nothing from
  the compensation. Static is what ships, so the trade is right — it is not free.
* **The ratio only helps heavy-tailed data.** `test_int4_conv`'s randn/randn fixture (|max|/|min|
  1.26 against the real activations' 19.91) goes 0.221 → 0.340. Golden refreshed with attribution
  proved by `MODIFF_ACT_CLIP_RATIO=1.0` reproducing the old one bit-exactly.

## 3. Two plan items resolved by evidence rather than implemented

Recorded with the numbers, so neither gets re-proposed on intuition.

**Fix #4, AdaRound weight import — deprioritised.** Bit-exact import needs a per-output-channel weight
zero point, which *cannot* fold into the bias: `Σ(w_q − z_w)·a = Σw_q·a − z_w·Σa`, and `Σa` is
per-output-pixel. Measured offline before building anything, over the 70 convs:

| | median | worst |
|---|---:|---:|
| qdiff AdaRound | 0.1506 | 0.3110 |
| **ours, RTN + MSE** | **0.1296** | 0.2588 |
| AdaRound re-quantised on our grid | 0.1581 | 0.3235 |

Ours already wins on that metric and the no-kernel-change shortcut is the worst of the three.
AdaRound optimises block output error rather than ‖W−Q(W)‖, so this does not prove ours is better end
to end — but weights are worth 0.2728 against the activation grid's 0.9060, making this the smallest
lever at the largest cost with the weakest evidence.

**Fix #2, activation zero point — deprioritised, and the reason is methodological.** It was justified
by 1.76× measured *before* the clip ratios, on a grid that had not yet been clipped — and both
constants exploit the same slack. Re-measured after, with 4-bit weights and ranges collected on the
quantized-weight model, it reads 1.27×. But the harness **fails its own self-check**: its symmetric
optimum sits at ratio 6.7 where the real kernels put it at 4.5, so it does not reproduce the shipped
ordering and its asymmetric number cannot decide the question. A zero point cannot be emulated on the
real kernels without the kernel change itself (the offset would have to be applied after SiLU, inside
the fused kernel), so this is genuinely undecidable without building it. The MoDiff axis — the one
that ships — is provably insensitive to this grid, which is the strongest argument against paying for
it.

## 4. Coverage is still not aligned, and here is the measurement

The paper quantizes 168 modules; we calibrate 70.

| paper | count | ours |
|---|---:|---|
| ResBlock conv (in/out) | 70 | quantized + calibrated |
| attention proj | 42 | quantized, MoDiff off (`MODIFF_LINEAR=0`) |
| timestep emb linear | 35 | **not quantized** — and all 37 UNet Linears pass `_eligible` at 4 bits, so this is not a shape constraint |
| skip conv | 17 | converted to int4, never calibrated → per-call dynamic scale |
| in/out conv | 4 | same |

The emb-linear gap is the actionable one: eligibility is not the blocker, so the conversion is simply
not reaching them. `MODIFF_USE_EMA=1` now exists for the EMA half of fix #6 but is **off by default** —
turning it on moves every mode at once, including W8A8, whose noise floor makes the change
unattributable.

## 5. The noise floor, and a retraction

Several conclusions this session rested on ~10% differences, so the floor was measured directly: the
same config, zero changes, run twice.

| arm | run 1 | run 2 | spread |
|---|---:|---:|---:|
| W8A8 PTQ | 0.1149 | 0.1124 | 2.2% |
| W8A8 MoDiff static | 0.0546 | 0.0574 | **5.1%** |
| W8A8 MoDiff dynamic | 0.0606 | 0.0614 | 1.3% |
| W4A4 PTQ | 0.8643 | 0.8647 | 0.05% |
| W4A4 MoDiff static | 0.3099 | 0.3089 | 0.3% |
| W4A4 MoDiff dynamic | 0.3568 | 0.3545 | 0.6% |

**W4A4's results are safe (≤0.6%); anything under ~5% on W8A8 is not resolvable.** Retracting the
earlier attribution of W8A8 0.0520 → 0.0607 to the `_load_delta_table` relocation: that was reasoning
by elimination rather than measurement, and it sits inside this floor.

## Reproducing

```bash
# the paper, verbatim except -n and -l
python scripts/sample_diffusion_ldm.py -r models/ldm/lsun_churches256/model.ckpt \
  --batch_size 8 -c 400 -e 0.0 --seed 42 --ptq --weight_bit 4 --cali_st 20 \
  --cali_batch_size 32 --cali_n 32 --quant_act --act_bit 4 \
  --cali_data_path /workspace/cali_data/church.pt -l <logdir> \
  --cali_ckpt /workspace/quant_models/church_w4a8_ckpt.pth --resume_w \
  --modulate --quant_mode qdiff --cali_min_max -n 8

python docs/paper_repro_2026-08-12/scripts/delta_clip_sweep.py         # the DELTA_CLIP_RATIO sweep
python docs/paper_repro_2026-08-12/scripts/act_clip_sweep_real.py      # the ACT_CLIP_RATIO sweep
python docs/paper_repro_2026-08-12/scripts/paper_params_in_our_path.py # paper params, decomposed
python docs/paper_repro_2026-08-12/scripts/zp_headroom.py              # fix #2, with its self-check
python docs/static_qdiff_2026-08-12/scripts/static_vs_dynamic_ab.py    # the shipped-default A/B
```

`.pt` artifacts are gitignored and regenerable; scripts, `data/*.json` and this file are committed.

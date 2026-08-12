# Aligning W4A4 with the reference: the paper reproduces, and two clip ratios close most of the gap

**The paper's command works.** Run verbatim from the README — only `-n` and `-l` changed —
`--modulate --quant_mode qdiff --cali_min_max` produces clean churches
([`paper_w4a4_samples.png`](paper_w4a4_samples.png)); one sample even reproduces a shutterstock
watermark from the training data. Both inputs the tree lacked turned out to be obtainable:
`cali_data/church.pt` from the paper's HF dataset, `church_w4a8_ckpt.pth` (2.36 GB) from
Q-Diffusion's Drive folder.

**So the method was never the problem.** Two calibration constants closed most of the gap:

| | session start | now | |
|---|---:|---:|---|
| W4A4 PTQ | 0.8642 | **0.4695** | 1.84× |
| W4A4 MoDiff | 0.6122 | **0.3090** | 1.98× |
| W4A4 MoDiff vs its own dynamic arm | 1.71× **worse** | **0.71× better** | — |

Neither needed a kernel change. `DELTA_CLIP_RATIO = 8` and `ACT_CLIP_RATIO = 4.5`, both swept, both
in `int4_optimized.py`, both imported by `export_qdiff_scales.py` rather than duplicated.

---

## 1. What we had been running was not the paper's configuration

Four deviations, and two were self-inflicted:

| README | ours | why |
|---|---|---|
| asymmetric activations (qdiff default) | `--a_sym` | `apply_static_scales` has no slot for a zero point, so the **kernel format propagated upstream into the calibration command** |
| `--cali_ckpt … --resume_w` (AdaRound W4) | `--skip_weight_recon` (RTN) | the checkpoint was not on disk |
| EMA weights | `--no_ema` | integration's loader never swaps EMA |
| `cali_data/church.pt` | locally generated residual data | not downloaded |

The first is the one worth internalising: a limitation in the innermost layer (the `.pt` format) was
propagated all the way out to the reference command, and then the constrained thing was measured and
called "the paper's method".

## 2. Fix #1 — the delta grid was sized to the observed absmax

Sizing to the observed range is right at 8 bits and wrong at 4. The MoDiff residual is heavy-tailed,
so covering it spends 15 codes on a tail nothing lands in. Swept act-only, a clean U:

| ratio | 1 | 2 | 4 | **8** | 16 | 21 | 32 |
|---|---:|---:|---:|---:|---:|---:|---:|
| relL2 | .4945 | .3362 | .1773 | **.1147** | .2193 | .2542 | .3117 |

Real kernels: W4A4 MoDiff **0.6122 → 0.3099**. Both PTQ arms and both dynamic arms unmoved — the
control.

**A single swept constant beats importing the paper's own per-layer delta values**, which read 0.2452
here. The optimum follows the trajectory and ours is not theirs. So the plan's per-layer MSE search,
per-step histogram and non-fused calibration path were all unnecessary.

## 3. Fix #3 — the activation grid, same lever

`silu(gn(x))` is one-sided: measured `|max|/|min| = 19.91×`, only **5 of 15 codes** carrying >0.1% of
the mass, an effective **2.32 bits** of a nominal 3.91. Swept on the **real kernels**, both axes:

| ratio | 1 | 2 | 3 | **4.5** | 6.7 | 10 |
|---|---:|---:|---:|---:|---:|---:|
| PTQ | .8647 | .5482 | .4968 | **.4692** | .5312 | .6373 |
| MoDiff | .3090 | .3176 | .3074 | .3095 | .3121 | .3361 |

MoDiff is flat (1.09× across a 10× range) because it reads this grid only at t=T and then refines
`a_hat` with 5 warm-up rounds. So one constant serves both axes.

**The ratio only helps heavy-tailed data**, and the test suite proves it: `test_int4_conv`'s
randn/randn fixture has `|max|/|min| = 1.26` and gets *worse*, 0.221 → 0.340. Golden refreshed with
the attribution (`MODIFF_ACT_CLIP_RATIO=1.0` reproduces the old one bit-exactly).

**One real cost, stated rather than buried**: the W4A4 *dynamic* arm regressed 0.3577 → 0.4327,
because it also reads the static activation grid at t=T and gains nothing from clipping. The shipped
default is static, so the trade is right — but it is not free for every configuration.

## 4. Two hypotheses tested and refuted

* **"The paper leaves a_T unquantized and we quantize it."** Arms came out **per-seed identical**.
  integration's 5 warm-up rounds already converge `a_hat_T` to fp16 precision.
* **"The zero point matters for the delta."** Symmetric slightly **beat** asymmetric there. The
  residual is zero-mean (zp median 7.50 of 0..15). The zero point matters on the *activation*
  quantizer, a different object.

## 5. Fix #4 deprioritised on evidence, not skipped

Importing AdaRound weights needs a **per-output-channel weight zero point**, and that one cannot fold
into the bias: `Σ(w_q − z_w)·a = Σw_q·a − z_w·Σa`, and `Σa` is per-output-pixel. It needs a shared
reduction over the conv window (cheap in principle — `Σa_q` does not depend on the output channel —
but a new kernel). The paper's weight zero point is genuinely per-channel and spans 1..14, so it
cannot be waved away as centred.

Before building that, weight reconstruction error over the 70 convs, offline:

| | median | worst |
|---|---:|---:|
| qdiff AdaRound | 0.1506 | 0.3110 |
| **ours, RTN + MSE** | **0.1296** | 0.2588 |
| AdaRound re-quantised on our grid | 0.1581 | 0.3235 |

Ours already wins on that metric, and the no-kernel-change shortcut is the worst of the three.
AdaRound optimises block *output* error rather than `‖W−Q(W)‖`, so this does not prove ours is better
end to end — but weights are worth 0.2728 against the activation grid's 0.9060. Smallest lever,
largest cost, weakest evidence.

## 6. Fix #2 scoped but NOT built — because the instrument failed its own check

The zero point on the activation grid is the last quality lever in the plan. It is **not built**, and
the reason is measurement trustworthiness rather than effort.

**Scope, instrumented** (the plan's own first step). Of 53 quantize/pack entry points in
`modiff_cutlass`, the shipped W4A4 path calls 10:

| entry point | calls | needs a zero point |
|---|---:|---|
| `group_norm_silu_quantize_pack_nhwc` / `_fast` / `_resize` | 744 / 147 / 96 | yes — activation grid |
| `scale_quantize_and_pack` | 700 | yes |
| `step1_static_quantize_pack_int4_fprop` | 248 | yes — t=T activation |
| `quantize_act_int4_pack` | 210 | yes |
| `group_norm_silu_delta_quantize_pack_nhwc` / `_resize` | 372 / 80 | **no — the residual is zero-mean** |
| `quantize_attn_qkv_packed` / `quantize_attn_out_int4_pack` | 120 / 63 | different distribution, separate question |

So ~6 CUDA kernels plus the epilogue `Σw_q` fold, the `.pt` format, and an extension rebuild. The
highest-call-count family does **not** need it.

**Why it is not built.** The only instrument that can estimate the benefit without building it is the
fake-quant harness, and it failed a self-check twice:

* on fix #1 it predicted 0.1147 where the kernels delivered **0.3099**;
* after fixing a real bug in it (it collected activation ranges on the fp16 model and quantized the
  weights afterwards, so every arm ran on ranges that did not belong to the model being measured) it
  *still* puts the symmetric optimum at 6.7 where the real kernels put it at **4.5** — wrong
  ordering, not just wrong magnitude.

The script now self-checks against the real-kernel ordering and refuses to issue a verdict. Its
numbers were 1.16× before the fix and 1.27× after; both are unusable. Every matched-ratio comparison
does favour asymmetric (1.04–2.35×), so the lever is probably real — but building 6 CUDA kernels on a
number produced by an instrument that has been wrong twice is the mistake this session kept finding.
**Deciding fix #2 requires implementing it.**

## 7. Noise floor, and a retraction

Several conclusions this session rested on ~10% differences, so a zero-change repeat was run:

| | W8A8 arms | W4A4 arms |
|---|---:|---:|
| run-to-run spread | **1.3–5.1%** | **0.05–0.6%** |

So W4A4's results are safe and anything under ~5% on W8A8 is not resolvable.

**Retracted**: the earlier attribution of W8A8 static 0.0520 → 0.0607 to the `_load_delta_table`
relocation. That was reasoning by elimination, not measurement, and it sits inside this floor.

## 8. Alignment items landed as opt-in flags

`MODIFF_USE_EMA=1` and `CALI_PAPER=1`. Both default **off** deliberately: each moves every mode's
numbers at once, including W8A8 whose noise floor is 5.1%, so enabling them by default would be hard
to attribute and would invalidate every committed number in one step.

## Reproducing

```bash
# the paper, verbatim except -n and -l
python scripts/sample_diffusion_ldm.py -r models/ldm/lsun_churches256/model.ckpt \
  --batch_size 8 -c 400 -e 0.0 --seed 42 --ptq --weight_bit 4 --cali_st 20 \
  --cali_batch_size 32 --cali_n 32 --quant_act --act_bit 4 \
  --cali_data_path /workspace/cali_data/church.pt -l <logdir> \
  --cali_ckpt /workspace/quant_models/church_w4a8_ckpt.pth --resume_w \
  --modulate --quant_mode qdiff --cali_min_max -n 8

python docs/paper_repro_2026-08-12/scripts/paper_params_in_our_path.py   # §4, the refutations
python docs/paper_repro_2026-08-12/scripts/delta_clip_sweep.py           # §2
python docs/paper_repro_2026-08-12/scripts/act_clip_sweep_real.py        # §3, real kernels
python docs/paper_repro_2026-08-12/scripts/zp_headroom.py                # §6, self-checks and refuses
```

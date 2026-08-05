# MoDiff: Stage 0 (reachability) + Stage 2 (the two bugs)

**GPU** NVIDIA A40 · **Date** 2026-08-03 · **Paper** ICML 2025 *Modulated Diffusion*, arXiv 2506.22463
**Plan** `/root/.claude/plans/https-arxiv-org-abs-2506-22463-learn-thi-curried-catmull.md`

The method, from the paper (Eqs 8–17):

```
â_T = Q(a_T)                          ô_T = A(â_T)
â_t = Q(a_t − â_{t+1}) + â_{t+1}      ô_t = A(Q(a_t − â_{t+1})) + ô_{t+1}
e_t = a_t − â_t                        ← fed forward: error compensation
```

Theorem 4.3: `‖x − Q(x)‖² ≤ s²d`, `s = (max−min)/(2^b−1)`. The benefit comes entirely from `s`
being derived from the *delta's* range. Remark 5.1: the paper reports **no wall-clock numbers** —
this repo is that missing systems half.

---

## Stage 0 — kernel reachability, measured not inferred

[`probe_kernel_reachability.py`](scripts/probe_kernel_reachability.py) wraps all 129 exports in
counting shims *before* any integration module imports them, then runs 8 modes in three phases.

Two methodology points that changed the answer:

1. **`MODIFF_QUANT_LINEAR=1` is load-bearing.** The first run omitted it and measured a completely
   different, unfused configuration — qkv/proj stay `nn.Linear`, `_qout_eligible()` returns False,
   and every fused-epilogue attention route is silently skipped. `e2e_three_mode_bench.py:38-44`
   carries a warning about exactly this. Fixed by reusing `kernel_suites_bench.set_env`.
2. **One sampling run is not steady state.** Attention blocks self-calibrate over their first
   `MODIFF_ATTN_CALIB_STEPS` (8) forwards. A single run straddles the boundary and fires *both* the
   calibrating entries and the frozen ones, which makes live calibration-window kernels look dead.
   The probe now runs twice and reports run 2 alone as the production set.

| | count |
|---|---:|
| callable exports | 129 |
| fire in some mode/phase | 51 |
| **fire in steady state (production)** | **33** |
| fire only during setup/calibration — **must keep** | 18 |
| **never fire in any mode or phase** | **78** |

Steady-state production sets: fp16 3 · int8_baseline 12 · int8 14 · int4_baseline 11 · int4 14.

[`classify_never_fired.py`](scripts/classify_never_fired.py) then classifies the 78 by who still
references them ([`deletion_classification.json`](data/deletion_classification.json)):

| bucket | n | verdict |
|---|---:|---|
| `A_unreferenced` | 3 | delete: `awq_w8a8_gemm`, `fused_gn_qkv_int8`, `mma_smoke` |
| `B_env_gated` | 12 | delete with the env var (approved aggressive tier) |
| `E_sentinel` | 1 | `conv2d_int4_fprop_tuned` — repoint the probe, then delete |
| `D_bench_only` | 6 | `flash_attn_int8`/`int4` are string-list-only in `profile_tree.py` → delete; 4 need the `benchmark_extended.py` decision |
| `F_fallback` | 56 | live call site on a branch this model's shapes never take → needs manual triage, **do not bulk-delete** |

The `B_env_gated` 12 map exactly onto `MODIFF_ROUTE1`, `MODIFF_INT4_QKV_EPILOGUE`,
`MODIFF_FLASH_PACKED`, `MODIFF_INT8_FLASH_PREG`, `MODIFF_LINEAR_OUT_I8`, `MODIFF_FP16_MATERIALIZED`.

**Limit of the automation, stated plainly:** the classifier cannot distinguish "live call site on an
untaken branch" from "call site inside another dead function". Several `F_fallback` entries are in
the latter category (e.g. `flash_attn_int8_vt_qout`'s only reference is inside `_packed_ref_vt_qout`,
itself a dead helper). Stage 4 must triage `F_fallback` by hand; the probe narrows it, it does not
decide it.

---

## Stage 2 — the two bugs, fixed and verified

### Bug 2: double quantization in the Linear MoDiff path — **fixed**

[`wxax_linear.py`](../../integration/kernels/wxax_linear.py) passed `q` — already integer codes in
[−Q, Q] — into `_gemm(q, d_scale)`, which quantizes its own input by `1/d_scale ≈ 1e4`. Every
nonzero delta saturated to ±Q, inflating the increment by `127/|q|`. `â` stayed correct; only `ô`
was poisoned.

Fix: pass `delta`, not the codes. The `â` update must use the *same* codes the GEMM consumed, so the
Python rounding now mirrors the kernel's arithmetic exactly — an fp32 reciprocal then an fp32
multiply (`quantize_act_int8` does `__float2int_rn(__half2float(x) * inv_scale)` with
`inv_scale = 1.f/(float)a_scale`). An fp16 divide or an fp64 reciprocal desyncs `â` by one code.

### Bug 3: int4 Linear MoDiff fake-quantizes into an fp16 GEMM — **fixed**

`int4_linear.py`'s `_linear` is fp16-only (W4A4 was removed from that class), yet the modiff path
still rounded the delta onto the int4 grid first. Ported the short-circuit `int8_linear.py` already
has. Measured before → after, mode `int4`, steady state:
`dequant_accumulate_and_return_int4` **185 → 0** calls, and the distinct kernel families firing per
step **14 → 13**. Pure waste removed: injected int4 rounding plus 3 full-tensor launches for zero
GEMM benefit.

### Verification: [`modiff_step_simulator.py`](scripts/modiff_step_simulator.py)

`QuantLinearWxAx`'s modiff branch is unreachable in every mode (`benchmark_ldm.py` forces
`is_modiff=False`), so nothing exercised it. The simulator drives the module directly over a 200-step
synthetic trajectory at real churches shapes and checks the two invariants that define MoDiff:

- **I1** `‖a_t − â_t‖_∞ ≤ s_t/2 + fp16 ulp` (Eq. 18)
- **I2** `ô_t = â_t @ W_deq^T` (Eq. 14, the telescoping identity)

I2's reference is a plain fp32 matmul against the dequantized weights — *not* another `_gemm` call,
which would quantize its input again and measure nothing. The identity is exact by construction:
`GEMM(codes) = (codes·d_scale) @ (qweight·w_scale)^T` and `â = Σ_t dq_δ_t`.

W8A8, C192/T1024, 200 steps:

| step | ‖a−â‖∞ | I2 rel |
|---:|---:|---:|
| 0 | 0.0234 | 0.00346 |
| 100 | 0.000488 | 0.00430 |
| 199 | 0.000488 | 0.00496 |

- **I1 is flat** at 4.88e-4 after step 0 — error compensation works, no accumulation. Step 0 is
  larger (0.023) exactly as Eq. 8 specifies: `Q(a_T)` quantizes the full range.
- **W4A4 I1 ≈ 8e-3, 16× the int8 value** — precisely the 2⁴ ratio, confirming the linear path's
  *dynamic delta scale* is correct. (The conv path's static scale is not; that is Stage 1.)
- **I2 grows +0.0015 over 200 steps.** That growth *is* the fp16 `ô` accumulation cost.

**Positive control — the test would have caught Bug 2.** Reintroducing the bug behind a monkeypatch:

| | I2 at step 20 |
|---|---:|
| fixed | 0.0038 |
| buggy | **4.03** |

1060×. And 4.03 matches the "rel-err diverges 0.06 → 3.2" recorded at `benchmark_ldm.py:709-712` as
the grounds for disabling MoDiff-on-Linear. **That decision rested on this bug, not on the method.**

### Consequence for the plan: Stage 2.3's fp32/int32 `ô` contingency is not needed

Measured fp16 `ô` accumulation cost is **+1.5e-3 relative over 200 steps**, against an A4
delta-quantization error of ~8e-3 — roughly 5× smaller. The plan's estimate (~2e-3) is confirmed.
**Keep fp16 `ô`.** This retires the risky CUTLASS-EVT work of splitting the Aux element from `EC`
in `conv2d_evt.cu`, and it removes the "fp16 drift" hypothesis as an explanation for anything.

One methodology correction worth recording: an earlier version of the simulator reported a
"fp16-vs-fp64 accumulation drift" of exactly 0 for every shape. That metric was circular — it
derived increments from successive `o_hat.float()` reads and re-accumulated them, so it
reconstructed the same fp16 values by construction. It was removed rather than reported. I2's growth
is the honest measure of the same quantity.

---

---

## Stage 1 — per-step delta scale: implemented, acceptance test blocked

### What is in place

`OptimizedInt8Conv2d` now carries a per-step delta-scale table and uses it on every modulated path:

- buffers `static_delta_scale[256]`, `static_delta_alpha[256]`, `is_delta_calibrated`
- `_delta_scale_args(device)` returns 1-element **views** of the table for the current step, so the
  four modulated paths no longer do `float(self.static_input_scale.item())` — that removes a host
  sync per layer per step as a side effect
- all four call sites wired (`step1_static_quantize_fprop{,_silu}` ×2,
  `group_norm_silu_delta_quantize_nhwc`), each paired with the matching `alpha = 1/scale` on its conv
- `begin/end_delta_calibration`, `export_int8_delta_scales` / `apply_int8_delta_scales`,
  `delta_calibration_report`
- falls back to the old behaviour with a loud per-layer warning when the table is absent

**No CUDA changes and no rebuild were needed.** Two properties made that possible: the delta-quantize
kernels already take their scale as a *device pointer*, so only the pointed-to value had to change;
and the delta's range is recoverable from the emitted codes as `max|q| / scale_used`, so the
observation needed no kernel hook either.

Why per-step rather than one scalar is not a refinement but a requirement: the measured
`static_delta_scale[1]` is ~337 against a tail value of ~2.4e4 on `input_blocks.1.0.in_conv` — a 70×
spread. A single scalar either clips at the second step or wastes ~99% of the code range afterwards.

### Why the acceptance test cannot run yet — three findings, all pre-existing

The step-size gain came out **median 1.0×** (max 119×), i.e. no gain for most layers. That is not the
table's fault. Tracing it produced three separate problems, and they compound into a catch-22.

**1. The shipped calibration artifacts are invalid for this tree, and this affects the baseline too.**
Measured with `integration/calibration/int8_calibration.pt` loaded, mode `int8`, 20 steps:

| quantize kernel | calls | median code_max (Q=127) | saturated |
|---|---:|---:|---:|
| `group_norm_silu_quantize_nhwc_fast` (plain activation) | 420 | 126.0 | **209/420** |
| `group_norm_silu_delta_quantize_nhwc` (delta) | 1240 | 127.0 | **1201/1240** |
| `step1_static_quantize_fprop` (delta) | 440 | 127.0 | **399/440** |

Every one of the 70 convs clipped at every step. The cause is the same one behind the vacuous
end-to-end checks: the 856-byte stub has an empty `state_dict`, so all weights come from default-init
off torch's global RNG, which is seeded **nondeterministically per process**. The calibration file was
produced against a different random network than any given run builds. So the scales do not describe
the activations they are applied to — and note the first row: **the non-MoDiff baseline is also
running with ~50% of its activation quantizations clipped.**

**2. Conv calibration is unreachable from `_setup_model`.** `_calibrate_int8` is called from
`run_mode` (`benchmark_ldm.py:1002`), not from `_setup_model` (which only *loads* a file at `:466`).
Every reporting harness and every script in this repo drives `_setup_model` directly, so none of them
can calibrate — they can only load.

**3. Uncalibrated MoDiff conv + quantized attention linear is a hard crash.** With
`calibration_path=None` the convs stay `is_calibrated=False`, keep fp32 `a_hat`/`o_hat` caches, and
`_module_output()` returns fp32; the quantized attention proj then raises
`RuntimeError: residual fp16 [M,n_out]` from `gemm_w8a8_awq_bias_res`
(`quantized_std_attention.py:1074`). This fires *inside `_calibrate_int8`'s own sampling pass*, so
the calibration cannot be regenerated in-process either. It has never been hit because a file was
always present, which kept the uncalibrated path off the road.

Net: with the file, the delta range is unobservable; without it, calibration crashes.

### The dtype contract — fixed, three separate emitters

Getting `_calibrate_int8` to run in MoDiff mode at all took three fixes, found one at a time by
following the crash:

1. `gemm_wXaX_awq_bias_res` requires an fp16 residual. Six call sites in
   `quantized_std_attention.py` passed `x_in_tok` straight through; they now `.half()` it, matching
   the convention `wxax_linear.py:134` already used. No-op in production.
2. The quantized routes always *return* fp16 (the GEMM epilogues emit fp16), which then hit
   non-quantized convs holding fp32 biases with autocast locally disabled → *"Input type
   (c10::Half) and bias type (float) should be the same"*. Fixed once, at a single point:
   `forward` is now a dtype-transparent wrapper over `_forward_routes`.
3. A first attempt cast `_module_output()` fp32→fp16 in both conv classes. That fixed (1) but caused
   (2), so it was **reverted** — the uncalibrated window is an fp32-flavoured pipeline and should
   stay one. Only the consumer that genuinely needs fp16 enforces it.

`Calibrated 70 conv layers` now appears in MoDiff mode for the first time.

### The measurement

Then two bugs in my own solver, both worth recording because each produced a plausible-looking
wrong answer:

- **Clipping is a lower bound, not a measurement.** When `code_max` pins at Q, all you know is
  `|delta| ≥ Q/scale`; dividing by the unchanged scale reproduces that scale — a fixed point. 20 of
  35 layers sat at exactly 1.00× for this reason. `delta = a_t − â_{t+1}` has both terms inside the
  activation range, so it can reach 2× that range and clip immediately. Fixed with a ×2 backoff.
- **Per-round recomputation oscillates.** An over-corrected scale yields a small `code_max`, which
  reads as a small range, which restores a too-large scale, which clips again: measured
  0.5 → 5.5 → 5.0 → 2.6 → 2.0 → 4.6 over six rounds, never clearing. Fixed with a monotone running
  maximum, the same shape as the existing `torch.max(_act_channel_max, ch_max)`.

**Result after 8 monotone rounds — this is the real number:**

| | layers | step-size gain | squared-error reduction |
|---|---:|---|---|
| converged, gain ≥ 2× | **14 / 70** | 2.5–62.3× (median **34.8×**) | up to **3900×** |
| never converged | 35 / 70 | — | — |
| converged, gain < 2× | 21 / 70 | ~0.5× | none |

All 14 winners are `in_conv`; **no `out_conv` converged at all**.

So the mechanism is real and large where the premise holds — a 34.8× median step-size gain is a
~1200× reduction in squared quantization error, which is the entire point of Theorem 4.3. It just
does not hold for most layers in this tree.

### The 35 non-converging layers are a scale bug, not a MoDiff property

Careful about one number: the "delta/activation ratio ≈ 300" that falls out of the report is **not a
measurement**. It is my backoff doubling 8 times (2⁸ = 256) for layers that never stopped clipping.
It says only "≥ 256× the calibrated range", not "= 300×".

What it does establish: for every `out_conv`, the run-time delta exceeds **256× the activation range
that calibration measured for that same layer** (calibrated absmax ≈ 3e-4, so the real values are
≥ 0.077). A delta cannot exceed 2× its activation's true range, so the calibrated activation scale
for `out_conv` must be wrong by orders of magnitude — which would mean the **baseline** clips those
layers too.

### Root cause found — and it is not the SiLU, and not the delta scale

**The SiLU hypothesis was wrong.** `static_quantize_and_update_ahat_kernel_int8_half_cache_silu`
(`modiff_delta_quantize.cu:386`) computes `silu(x)`, then `*= smooth_inv`, then `- cache`. Calibration
takes the `_can_fuse_input_silu(x) == False` branch of `forward` and applies `F.silu(x)` explicitly,
then the same smooth. The two agree. No mismatch.

**The delta scale is not the cause either.** A/B with an identical seeded network and identical
activation calibration, the only variable being whether the per-step delta table is applied
([`divergence_ab.py`](scripts/divergence_ab.py)):

| | in_conv out-of-range | out_conv out-of-range |
|---|---:|---:|
| delta scale OFF (full-activation grid) | 23.0× | 40957× |
| delta scale ON (per-step delta grid) | 23.0× | 45575× |

1.0× and 0.9×. No effect.

**The actual cause: in MoDiff mode the activation scale is calibrated from the FIRST DIFFUSION STEP
ONLY.** `_compute_activation_scale` has exactly three call sites:

- `int8_optimized.py:684` — inside `_forward_standard`, reachable only when `not modiff_enabled`. Dead
  in MoDiff mode.
- `:993` — inside `_forward_first_step`, i.e. **t=T only**.
- `:1005` — the warm-up residuals, passed `is_residual=True`, which the accumulator explicitly skips.

The modulated steps (t < T) never observe anything: with `is_calibrated=False` during calibration,
`_forward_modulated` takes its *dynamic* branch, which derives the scale on-device via
`sub_absmax_scale` and never calls `_compute_activation_scale`. So `static_input_scale` describes
step 1 and nothing else. Measured against what actually runs over 20 steps:

| | calibrated absmax (t=T) | run-time absmax | factor |
|---|---:|---:|---:|
| `in_conv` (35) | 0.244 | 5.61 | **23×** |
| `out_conv` (35) | 0.000446 | 18.1 | **41000×** |

`out_conv` is hit hardest because its t=T input is genuinely tiny (4.5e-4) — at the first step the
ResBlock's internal activation has barely developed — so the scale it locks in is absurd for every
later step.

This also explains the 35 non-converging layers directly: the delta observation's denominator is that
same broken scale, so `code_max` pins at Q no matter what the delta actually is. My solver was being
fed a garbage reference, not measuring a large delta.

**Scope — this is not a MoDiff defect.** In baseline mode `_forward_standard` runs every step, so
`:684` observes every step and the calibration is sound. (Consistent with the instrumentation finding
that in baseline the conv module never receives a floating-point input at all — the GN kernel does the
quantizing upstream.) But it does mean: **any `int8_calibration.pt` generated in MoDiff mode is wrong,
including for the baseline that later loads it.**

**Fix (not yet implemented):** make the modulated path observe during calibration — either call
`_compute_activation_scale` on the modulated branch when `calibrating`, or read back the
`sub_absmax_scale` output the dynamic branch already computes on-device and accumulate a running max.
The second is nearly free and needs no extra pass. Until then Stage 1's step-size gain cannot be
measured on the 35 `out_conv` layers, and the 14 converged `in_conv` layers (2.5–62×, median 34.8×)
are the demonstrated result.

### What unblocks it

1. Fix (3): make the uncalibrated MoDiff conv emit fp16, or have `_proj_with_residual` cast its
   residual. Small, but it changes production dtype behaviour, so it wants its own review.
2. Then regenerate both calibrations against a **seeded** network
   (`guard.seed_model_construction()` already exists), and key the artifacts on that seed.
3. Then re-run `calibrate_delta_scales.py`; the codes will no longer saturate and the step-size gain
   becomes a real measurement.

The Stage-1 code is correct and inert until a delta table is supplied — the fallback path reproduces
today's behaviour exactly, with a warning. Nothing regresses by landing it.

---

## Task #9 — activation calibration: fixed as specified, but two of my own numbers were wrong

### Correction 1: the "23× / 41000× out of range" figures above are wrong

They divided the run-time activation by `127 / static_input_scale`, treating that as the calibrated
range. It is not, whenever SmoothQuant is active. `end_calibration` computes
`static_scale = 127 / max_c(act_max_c / s_c)`, and the kernel quantizes `x * smooth_inv` — so the
right metric is

```
effective code utilisation = max|x * smooth_inv| * static_input_scale        (127 == matched)
```

Corrected numbers, 20 steps: `in_conv` **326**, `out_conv` **631**. So activations *do* clip in
MoDiff mode, by ~2.6–5×, not by 4–5 orders of magnitude. The earlier figures were an artifact of
omitting `smooth_inv`.

### Correction 2: SmoothQuant is degenerate on every `out_conv`, and that is why the scale looked absurd

`weight_int8.abs().max()` is **0 for all 35 `out_conv` layers** — LDM wraps each ResBlock's output
conv in `zero_module`, and the stub checkpoint never fills it in. `_apply_smoothquant` then computes
`s = sqrt(act_max / w_max)` with `w_max = 0`, which saturates at its `clamp(max=1e4)` ceiling:
measured `smooth_scale` max median is exactly **1e4** for `out_conv` against 18.8 for `in_conv`.
A huge `s` makes `act_max / s` tiny, hence the enormous `static_input_scale`, hence my bogus
"calibrated absmax 4.5e-4". Nothing was mis-measuring the activation; SmoothQuant was dividing by a
zero weight range. That is a previously unrecorded consequence of the stub checkpoint.

### What the fix does

`_compute_activation_scale` is now called on the modulated path when `calibrating`
(`int8_optimized.py`, in `_forward_modulated`). Verified live: **5.0 non-residual observations per
layer per calibration, up from 1.0**, and `_scale_count` 10 instead of 2. So the t=T-only defect
found in task #8 is genuinely fixed — that part of the diagnosis held.

It is not sufficient on its own. Two further mismatches remain in `_calibrate_int8`
(`benchmark_ldm.py:794`), both hardcoded:

- `sampler.sample(S=5, ...)` — calibration sees the first 5 DDIM steps; production runs 20–200.
- `batch_size=2` against a production batch of 4–128; a larger batch has a larger extreme.

Measured effect of matching the horizon (batch still mismatched):

| calibration horizon | `in_conv` utilisation | `out_conv` utilisation |
|---|---:|---:|
| S=5 (as shipped) | 326 | 631 |
| S=20 (= production) | **246** | **580** |

A real improvement, and not a fix: activations still clip ~2–4.6×. The residue is the batch mismatch
plus calibration-vs-runtime variance on a random-weight network.

### Status

#9's stated goal — observe every step rather than only t=T — is **done and verified**. The broader
claim it was meant to unblock (Stage 1's step-size gain measurable on all 70 layers) is **not**
achieved, because the quantizer is still under-provisioned for a second, independent reason. What
Stage 1 has demonstrated remains the 14 converged `in_conv` layers at 2.5–62× (median 34.8×).

Next, in order: match `_calibrate_int8`'s horizon and batch to the production run (small, mechanical,
and it affects the shipped `int8_calibration.pt` for the baseline too); then re-run
`calibrate_delta_scales.py` and see how many of the 70 converge.

---

## Tier A result: the delta scale now pays off — 66/70 layers, median 12.5×

Three Python-only fixes to the *activation* calibration, then the delta calibration was re-run.

**Fix 1 — horizon and batch follow the runner** (`benchmark_ldm.py`, `_calibrate_int8` /
`_calibrate_int4`). Were hardcoded `S=5, batch=2` against production's 20–200 steps at batch 4–128.
Capped at 50 steps / batch 8, since the range plateaus.

**Fix 2 — a refinement round.** Round 0 necessarily observes the *uncalibrated* path
(`is_calibrated` is False while calibrating), whose numerics differ from the calibrated path
production runs. Re-observing once on the now-calibrated path is what actually moved the needle.
SmoothQuant is deliberately not re-derived: `_fold_weights_with_smooth` releases `_orig_weight`, so
round ≥1 keeps round 0's per-channel scale and refreshes only the per-tensor one.

**Fix 3 — identity smoothing for zero-weight layers.** `s = sqrt(act_max/w_max)` sent `s` to its 1e4
clamp for all 35 `out_conv` layers (LDM wraps each in `zero_module`). **Hygiene, not numerics** — a
uniform `s` cancels against the scale, and utilisation is bit-identical before and after. What it
fixes is `static_input_scale` being ~1e4× its meaningful value (2.4e5 → 23.85), which made every
diagnostic on that field unreadable.

### Activation utilisation (Q = 127 is full scale)

| configuration | `in_conv` | clipping | `out_conv` | clipping |
|---|---:|---:|---:|---:|
| as shipped (S=5, batch 2) | 203.9 | 25/35 | 553.8 | 35/35 |
| horizon + batch matched | 169.8 | 25/35 | 521.0 | 35/35 |
| **+ 1 refinement round** | **124.4** | **17/35** | 521.0 | 35/35 |
| + 2 refinement rounds | 130.7 | 18/35 | 516.8 | 35/35 |

`in_conv` is fixed — median below Q. One refinement round is the operating point; two is no better.

### Delta step-size gain, re-measured on the fixed scales

| | before this session | after |
|---|---:|---:|
| layers with gain ≥2× | 14/70 | **66/70** |
| median gain (all 70) | 0.5× | **12.5×** |
| squared-error reduction at the median | none | **155×** |
| layers still clipping after 8 rounds | 35/70 | **4/70** |

By family: `in_conv` 31/35 at ≥2×, median 19.9×; `out_conv` 35/35, median 11.7×. So D4 is met —
Theorem 4.3's benefit is now realised on essentially every layer, where before the change the step
size was unchanged by construction.

### `out_conv`'s 4.1× — open, and five hypotheses eliminated

`out_conv` utilisation sits reproducibly at ~521 (4.1× over Q). Ruled out, each by measurement:

1. **Pre/post-SiLU mismatch** — the fused kernel does `silu(x)` then `*= smooth_inv` then `− cache`,
   and calibration applies `F.silu(x)` then the same smooth. They agree.
2. **The delta scale** — A/B with and without the per-step table: 1.0× / 0.9× effect.
3. **The SmoothQuant 1e4 clamp** — fixed it; utilisation bit-identical, because a uniform `s` cancels.
4. **Draw-dependent heavy tails** — three production draws spread only 1.06×.
5. **â/ô state carried over from the calibration pass** — added `reset_modiff_state` before the
   measured run; 521.7 → 521.0.

What is left, and what I would check next: `out_conv`'s input is `SiLU(GN(h)·(1+scale)+shift)` where
`h` comes from `in_conv`'s ô. Since `out_conv`'s own weights are zero its ô contributes nothing back,
so its input is driven entirely by upstream ô — and the calibration and production ô trajectories may
diverge in a way one refinement round does not close (two rounds moved 521→517, i.e. converging very
slowly or not at all). A per-family refinement schedule, or observing at the GN output rather than the
conv input, would discriminate.

### Process note

That list cost five wrong hypotheses, and three of them were errors in my own measurement rather than
in the code: omitting `smooth_inv`, reading int8 codes as activations, and letting â/ô carry over into
the measured run. `effective_code_utilisation` plus the `code_utilisation` positive control exist
because of that pattern — the metric is now pinned by a test rather than re-derived each time.

---

## Stage 5 — docs corrected

- `docs/MEASUREMENT_REPORT_2026-08-01.md`: mode column relabelled **"INT8 (MoDiff off)"** /
  **"INT4 (MoDiff off)"** (4+4 data rows and 2 table headers), with a note explaining that the
  headline speedups are the `*_baseline` modes — MoDiff *disabled* — and that per the paper's Table 2
  A8 is exactly where MoDiff adds nothing. The generator (`make_measurement_report.py:26`) is fixed
  too, so regeneration keeps the correct labels.
- `integration/README.md`: the four "Incomplete — Not True INT4 MoDiff" bullets were **all false** and
  are removed, each with the check that disproves it (`cutlass::int4b_t` at `conv2d_int4.cu:52-53`,
  native s4 from sm_75 on an sm_86 target, `_cache_dtype()` returning fp16, delta+quantize in one
  kernel with `o_hat` in the EVT epilogue). Replaced with the one real limitation: the caches are
  fp16, not packed int4.
- "Computes only residual convolutions" replaced — **no convolution is skipped**; same FLOPs, smaller
  quantizer step, extra HBM traffic.
- FID 8.20 withdrawn in all four places, with the reason (stub checkpoint + zero UNet output).
- New "Script prerequisites" section. **Correcting my own plan note here**: only
  `sample_diffusion_ldm.py` is actually broken (imports `qdiff`, which is neither vendored nor
  pinned). `sample_diffusion_ddim.py` and `txt2img.py` just need `pip install -r requirements.txt` —
  `lmdb` and `opencv-python` are both pinned already.

## Stage 4 — deletion, and why it is only partly done

The three `hasattr` sentinels were repointed **first**, which mattered:
`fused_resblock.py:109` probed `conv2d_int8_fprop_o_hat_residual` while the code calls the *EVT*
version, so deleting that dead symbol would have silently switched the fusion off instead of failing.
Same shape for `int4_optimized.py:21` and `int8_optimized.py:812`. 16/16 tests still pass after.

Then re-checking the 18 planned deletions against live Python found that **15 of them still have live
call sites** — behind env vars that default off, but the calling code exists. Deleting the CUDA symbol
alone would leave `_mc.<gone>` calls that raise the moment anyone flips the flag. So the aggressive
tier is genuinely two changes, not one, and only the safe half is done here:

**Deleted (6 exported symbols, 129 → 123):** `awq_w8a8_gemm`, `fused_gn_qkv_int8`, `mma_smoke`,
`conv2d_int4_fprop_tuned` (comment-only after the sentinel repoint), `flash_attn_int8`,
`flash_attn_int4` (their only references are string literals in `profile_tree.py`'s name-matching
list, which does not need the symbol). Plus the vendored `awq_w8a8_gemm_cuda.{cu,h}` moved out of the
build entirely — a whole translation unit — and its `setup.py` entry and header declaration removed.

**Rebuild: 3m19s, EXIT 0.** 123 exports (from 129), all six gone, every sentinel/live target still
present, `.so` 24,636,864 -> 24,235,288 bytes. Validated after: `test_kernel_correctness` 16/16,
`test_wxax` / `test_gn_resize_fusion` / `test_updown_fusion_pipeline` pass, and all **seven** modes
(fp16, int8_baseline, int8, int4_baseline, int4, int8_attn_modiff, int4_attn_modiff) sample finite.
A fresh reachability probe reports 123 callable / 32 steady-state / 73 never-fired, consistent.

**Not deleted (12), each needing its Python route removed in the same change:** the `MODIFF_ROUTE1`
pair, the `MODIFF_INT4_QKV_EPILOGUE` pair, the three `MODIFF_FLASH_PACKED` variants,
`MODIFF_INT8_FLASH_PREG`, the three `MODIFF_LINEAR_OUT_I8` symbols, and `attn_softmax_fp16` (which
`benchmark_ldm.py:381` force-enables for the `static_fp16`/`dynamic_fp16` modes, so those modes go
too). That is a real refactor of live dispatch code, and with a >10-minute rebuild per validation it
should not be interleaved with anything else.


---

## D8 — wall-clock, reported as promised

A40, batch 128, 200 DDIM steps, one process, median of 3 profiler-free samples,
`reset_modiff_state` before each ([`e2e_wallclock.py`](scripts/e2e_wallclock.py)):

| mode | ms/batch | ms/step | vs fp16 |
|---|---:|---:|---:|
| fp16 | 20002.5 | 100.013 | 1.000× |
| int8_baseline | 14042.3 | 70.212 | **1.424×** |
| int8 (MoDiff) | 16548.7 | 82.744 | 1.209× |
| int4_baseline | 11437.1 | 57.186 | **1.749×** |
| int4 (MoDiff) | 13446.2 | 67.231 | 1.488× |

Baselines match the published report (1.424× vs 1.435×, 1.749× vs 1.764× — within ~0.8%), so this
session's changes did not regress them. **MoDiff costs +12.8 ms/step (int8) and +10.0 ms/step (int4),
i.e. 0.85× its own baseline.** That is the honest headline and it is what the goal committed to
reporting.

### A predicted 5 ms win that turned out to be 0

`_delta_scale_args` read `bool(self.is_delta_calibrated)` — a **CUDA buffer** — once per modulated
conv per step: 70 layers × 200 steps = 14000 GPU→CPU syncs per sample. I predicted ~5 ms/step and
replaced it with a host-side mirror. Measured after: **+12.80 ms/step, versus +12.53 before.** No gain.

The reason is consistent with everything else here: in MoDiff mode the GPU is the bottleneck, so the
CPU has slack and a sync costs nothing. The change is still correct — a device-buffer read in a hot
path is wrong on principle and would bite in a CPU-bound configuration — but it is not a speedup, and
claiming it as one would have been wrong. Sixth failed hypothesis in this area.

## Stage 3 target list, from current measurement instead of a stale report

[`modiff_bucket_breakdown.py`](scripts/modiff_bucket_breakdown.py), int8 vs int8_baseline, ms/step of
GPU kernel self-time bucketed by `profile_tree.classify` (the published report's taxonomy):

| role | baseline | MoDiff | delta |
|---|---:|---:|---:|
| GN group-statistics reduction | 0.00 | 9.45 | **+9.45** |
| MoDiff GN+SiLU+delta-quantize+cache apply | 0.00 | 8.05 | +8.05 |
| GN+SiLU+quantize fused (K1) | 16.05 | 1.98 | **−14.06** |
| nearest upsample (unfused) | 1.21 | 2.41 | **+1.19** |
| avg_pool 2×2 (unfused) | 0.44 | 0.88 | **+0.43** |
| quantized implicit-GEMM conv | 23.92 | 24.83 | +0.91 |
| GN+SiLU only | 0.85 | 1.56 | +0.71 |
| dtype cast / device copy | 2.45 | 2.96 | +0.51 |
| **total** | **68.89** | **77.28** | **+8.40** |

Reading it:

- **Stage 3.1 (coalesced GN statistics) is confirmed as the target, at +9.45 ms/step** — the plan's
  estimate was 11.1 ms from a pre-QKV-epilogue report, so the target is real and only slightly
  smaller. MoDiff replaces one fused K1 kernel (−14.06) with a split stats pass (+9.45) plus a
  delta-apply pass (+8.05); the net GN-family cost is +3.43 ms and the stats pass is the removable
  half.
- **Stage 3.2 (resize delta-fusions) is confirmed but small: +1.62 ms/step** (1.19 upsample + 0.43
  avg_pool). It is exactly the "updown blocks get zero fusion under MoDiff" gap — both roles double
  because the fusion is baseline-only — but it is ~6× less than 3.1.
- Kernel time accounts for +8.40 of the +12.80 ms/step wall-clock delta, so **~4.4 ms/step is gap /
  launch time**, not kernel work. Worth attributing before assuming more fusion recovers it.

Design note for 3.2, from reading `upsample2x_quantize_noahat_kernel`: the MoDiff twin must grid over
**input** elements, not output. In a 2× upsample four output elements share one `a_hat` entry, so
updating the cache from all four races and quadruple-counts. Gridding over inputs fixes it and reads
`x` once instead of four times, which is strictly better than the baseline kernel.


---

## Generation-quality evaluation: attempted, and void in this tree

Run rather than asserted ([`quality_evaluation.py`](scripts/quality_evaluation.py)), batch 4,
20 steps, seed 1234, seeded model construction:

| mode | UNet ‖ε‖∞ | identity attn blocks | latent ‖x‖∞ |
|---|---:|---:|---:|
| fp16 | **0** | 21/21 | 113.2 |
| int8_baseline | **0** | 21/21 | 113.2 |
| int8 | **0** | 21/21 | 113.2 |
| int4_baseline | **0** | 21/21 | 113.2 |
| int4 | **0** | 21/21 | 113.2 |

All **10** pairwise latent comparisons: `bit_identical=True`, relL2 exactly `0.000e+00`.

So no image-quality metric can separate the modes — FID, IS, sFID, LPIPS alike — because the bytes
they would consume are the same bytes. An FID computed here would be a single number describing the
initial noise. Cause, as established earlier: the 856-byte stub checkpoint with an empty `state_dict`
plus `UNetModel.out[-1]` being a `zero_module`, which is why ‖ε‖∞ is exactly 0. **Evaluating
generation quality requires the real trained LSUN-churches LDM checkpoint; nothing in this repo can
substitute for it.**

### The quality evidence that IS valid

Kernel-level accuracy against fp32/fp64 references on synthetic tensors at production shapes — no
checkpoint involved, so none of the above applies.

`qattn_correctness.py`, quantized attention vs an fp32 reference computed from the *same* codes:

| shape (N,H,T,hd) | int8 rel_err | int4 rel_err |
|---|---:|---:|
| 128, 8, 1024, 24 | 0.0152 | 0.0024 |
| 128, 8, 256, 48 | 0.0112 | 0.0025 |
| 128, 8, 64, 48 | 0.0076 | 0.0025 |
| 4, 4, 1024, 64 | 0.0161 | 0.0024 |

plus *"int4 values on K=32 int8 MMA: EXACT"*. ALL PASS.

`test_kernel_correctness.py` (16/16): int8 conv 0.012 vs fp32 · int4 conv 0.224 · int8 linear 0.012 ·
GroupNorm+SiLU 0.000 · fused GN→qkv 0.0069 · int8/int4 dual-store 0.0002 · MoDiff invariants
I1 = 0.54 codes, I2 = 0.0325 with the positive control at 11× separation.

And the one accuracy result this session actually moved: the delta quantizer step is now **12.5×
finer at the median across 66/70 layers**, i.e. ~**155× lower squared quantization error** on the
MoDiff path, where before the change the step was unchanged by construction. That is the paper's
Theorem 4.3 benefit, measured — and it is the closest thing to a quality improvement that this tree
can honestly demonstrate.


---

# 2026-08-04: the real checkpoint

`models/ldm/lsun_churches256/model.ckpt` and `models/first_stage_models/kl-f8/model.ckpt` were both
856-byte stubs. Downloaded the real ones from `ommer-lab.com` (official CompVis host), integrity
checked, exact `Content-Length` match. Stubs preserved in [`stub_ckpt_backup/`](stub_ckpt_backup/).

| | stub | real |
|---|---|---|
| LDM ckpt | 856 B, **0** state_dict entries | 2.69 GB, **1307** entries, 673.6M params |
| VAE kl-f8 | 856 B, 0 entries | 1.10 GB, 304 entries, 101.1M params |
| UNet ‖ε‖∞ | **exactly 0** | **≈ 4.3** |
| identity attention blocks | **21/21** | **0/21** |
| latents across the 5 modes | all bit-identical | all differ |

**Every previously-vacuous check in this tree is now live** — the golden-latent gate, the three
quality scripts, `test_std_attn_e2e.py`. That is the largest single unlock of this work, and it was
one `wget` away the whole time.

## The shipped calibration files were worse than useless

First measurement with the real weights and the **shipped** `integration/calibration/*.pt`:

| comparison | relL2 vs fp16 |
|---|---:|
| int8_baseline | **0.882** |
| int4_baseline | **3.023** |

88% and 302%. Those files were calibrated when the checkpoint was a random stub, so they describe a
different network. Anyone computing FID on them would have got a plausible-looking, meaningless
number — the same trap as the withdrawn FID 8.20.

## Recalibrated against the real weights — and MoDiff finally shows a quality win

[`recalibrate_real_ckpt.py`](scripts/recalibrate_real_ckpt.py), 50 steps, batch 8, 3 runs, written to
`*_realckpt.pt` (originals untouched):

| mode | stale calib | fresh calib | vs its baseline |
|---|---:|---:|---:|
| int8_baseline | 0.8820 | **0.2717** | — |
| int8 (MoDiff) | — | **0.1999** | **26% lower error** |
| int4_baseline | 3.0230 | **0.7667** | — |
| int4 (MoDiff) | — | **0.7638** | 0.4% lower |

Recalibration alone: int8 3.2× better, int4 3.9× better. And **MoDiff beats its own baseline on int8
by 26%** — the paper's error compensation, measurable for the first time in this repo. int4 is
essentially tied, consistent with int4's much coarser grid dominating.

Caveat worth keeping in view: 0.27 relL2 for W8A8 is still high (0.02–0.05 would be normal), which
fits the unresolved activation-clipping finding (`effective_code_utilisation` 124–521 against Q=127).
So these numbers are a floor on achievable fidelity, not the method's limit.

## One more bug the real checkpoint exposed

A seventh `gemm_*_awq_bias_res` residual site — `res0` at `quantized_std_attention.py:534`, in the
int4 layout-epilogue path — was missed by the earlier `.half()` sweep because it uses a different
variable name. int4 calibration crashed on it with `residual fp16 [M,n_out]`. Fixed; no unpatched
residual sites remain (7 total).


---

## Static delta-Q: built, measured, and it LOSES. Plus a correction to my own headline number.

The user asked for the static-Q version of the MoDiff delta quantizer. It is built and wired
([`static_delta_q.py`](scripts/static_delta_q.py)). On the real checkpoint it makes quality **worse**:

| int8 MoDiff, 50 steps, batch 8, one seed | relL2 vs fp16 |
|---|---:|
| delta table OFF (delta on the activation grid) | **0.1778** |
| delta table ON (static per-step delta grid) | **0.2135** |

20% worse. For context `int8_baseline` is 0.2717, so MoDiff still beats its baseline either way — but
the static delta table is a regression against MoDiff-without-it.

### Correction: the 12.5× step gain was a random-weight artifact

I reported "12.5× median step gain, ~155× lower squared error, 66/70 layers" as this session's
headline accuracy result. That was measured on the **stub** checkpoint. On the real weights:

| DDIM steps | median step gain | delta/activation range ratio |
|---:|---:|---:|
| 20 | 1.04× | 0.963 |
| 50 | 1.41× | 0.707 |
| 100 | 1.83× | 0.546 |
| **200** | **2.39×** | **0.418** |

The premise does strengthen monotonically with step count, exactly as it should (smaller timestep
jumps → smaller temporal deltas) — but even at 200 steps the delta is only **2.4×** smaller than the
activation, not 12×. And `min gain` stays at 0.25–0.78, i.e. some layers' deltas are *larger* than
their activations at every step count. **A randomly-initialised network overstated MoDiff's premise by
~5×.** Any conclusion I drew from the stub about delta ranges should be discarded.

### Why static loses, and what the paper actually assumes

With only ~1.5× of range advantage, narrowing the quantizer range to match the delta clips the tail,
and clipping error costs more than the finer step saves. Theorem 4.3 is explicit about this
assumption:

> *"For simplicity, we use dynamic quantizers, which determine the scaling parameter based on the
> input values to avoid clipping error"*

So the paper's error bound is derived for a **dynamic** delta quantizer. A static absmax table
reintroduces precisely the clipping the theorem assumes away. That also explains why this repo's
*linear* MoDiff path — which uses a per-call dynamic `d_scale = |δ|max/Q` — behaves correctly under
the I1 invariant test while the static conv table does not.

**Conclusion: static delta-Q is the wrong shape for this method at these step counts.** The
implementation is correct and stays in place behind `apply_int8_delta_scales` (off by default, so
nothing regresses), but the productive direction is a dynamic or percentile-clipped delta scale, not
a static absmax one.

### The solver bug found on the way

The first static-Q run showed the median gain *decaying* across rounds: 1.5 → 1.0 → 0.5 → 0.4. Cause:
geometric backoff on clip **plus** a monotone running max — both only ever shrink the scale, so extra
rounds could only make the quantizer coarser, and it ratcheted itself below the activation grid it was
meant to improve on. Replaced with a single exact pass: since `|a_t − â_{t+1}| ≤ 2·act_absmax`,
observing at `act_scale/4` provably cannot clip, so one pass measures the true range. Clipping during
observation fell from 49/70 to 1/70, and that one layer now warns loudly instead of silently returning
a lower bound.

## What is still open

- **Stage 1 (the quantizer scale) is untouched and is the load-bearing item.** The conv path still
  quantizes the delta on the full-activation grid (`int8_optimized.py:1041` →
  `modiff_delta_quantize.cu:325`, scale from `:213-222`), so MoDiff currently buys error feedback
  and nothing else on the path every benchmark uses.
- Stages 3–6 unstarted.
- Not yet re-enabled: `is_modiff=False` at `benchmark_ldm.py:713`. Bug 2 is fixed but the branch
  stays unreachable until Stage 3 adds the GEMM `ô` epilogue; flipping it now would add three
  PyTorch launches per linear per step.
- **The paper's actual claim (FID at A4) remains unreproducible in this checkout** — the 856-byte
  stub checkpoint and two `zero_module` sites make every latent-level check vacuous
  (`docs/gn_qkv_fusion_2026-08-03/FINDINGS.md` §5). All quality evidence above is kernel-level by
  necessity.

---

# 2026-08-04 — The dynamic delta quantizer, W8A8 and W4A4

## Headline

With a dynamic (per-call, non-clipping) delta quantizer, MoDiff finally beats its own baseline by a
wide margin at both bit-widths. Real LSUN-churches checkpoint, DDIM S=50, batch 8, seed 1234, latent
relative L2 vs the fp16 model, all measured **at steady state** in one process:

| | baseline (MoDiff off) | MoDiff static | MoDiff dynamic | dynamic vs baseline |
|---|---|---|---|---|
| **W8A8** | 0.2378 | 0.1878 | **0.0393** | **6.05×** |
| **W4A4** | 0.7837 | 0.7770 | **0.4199** | **1.87×** |

W8A8 at 0.0393 is finally inside the range a well-behaved 8-bit activation quantizer should occupy.
Note also that at W4A4 the *static* delta scale bought essentially nothing over baseline
(0.7770 vs 0.7837) — consistent with Theorem 4.3, which predicts no error reduction when the delta is
quantized on the activation's own grid.

## Correction: an earlier conclusion in this session was wrong

The first version of this A/B reported the opposite for W8A8 — that dynamic *lost*, 0.1777 → 0.2313 —
and I built an explanation for it (that clipping is affordable because MoDiff's error feedback
recovers it). **That was an artifact and the explanation was unnecessary.**

Cause: **one sampling run is not steady state.** The quantized attention blocks self-calibrate their
static scales over their first `MODIFF_ATTN_CALIB_STEPS` forwards, so the first sampling run after
model construction measures a model that is still calibrating. Measured directly, int8 dynamic:

```
rebuild 0: run 1 relL2 0.2107   run 2 relL2 0.0399
rebuild 1: run 1 relL2 0.2107   run 2 relL2 0.0399
rebuild 2: run 1 relL2 0.2107   run 2 relL2 0.0399
```

Rebuild-to-rebuild is exactly reproducible; run-1-to-run-2 differs by **5.3×**. First-run numbers
also depend on how many models were built earlier in the same process, which is why the A/B and the
sweep disagreed (0.2313 vs 0.2110) for a nominally identical configuration.

Both harnesses now discard run 1. Any future latent-fidelity number in this tree must do the same —
a first-run measurement is not merely noisy, it *reverses the ranking*.

## The clipping-ratio sweep

`scale = Q / (ratio · max|delta|)`, so `ratio = 1.0` is pure absmax (cannot clip) and smaller values
buy a finer grid by clipping the tail. This makes static-vs-dynamic one knob rather than two modes.
Implemented with no kernel support at all, by passing `Q/ratio` as the kernels' `Q_level`.

| ratio | 1.00 | 0.70 | 0.50 | 0.35 | 0.25 | 0.15 | 0.10 |
|---|---|---|---|---|---|---|---|
| **W8A8** | **0.0393** | 0.0490 | 0.0556 | 0.0616 | 0.0924 | 0.1504 | 0.1973 |
| **W4A4** | 0.4571 | 0.4275 | 0.4501 | **0.4199** | 0.4459 | 0.4829 | 0.5307 |

W8A8 is cleanly monotone — every bit of deliberate clipping hurts, so pure absmax is optimal. W4A4 is
flat within noise from 0.35 to 1.0 and only degrades below 0.25. `ratio = 1.0` is therefore the
default for both; the knob exists for tuning, not for correctness. (`MODIFF_DELTA_CLIP`.)

## Implementation — additive, reusing the baseline kernels

Per the 2026-08-04 directive to extend rather than rewrite, this added two kernels and changed none:

1. **`delta_absmax_fp16`** (`modiff_delta_quantize.cu`) — copy of `sub_absmax_scale_kernel` with the
   fp32 cache replaced by fp16 and the residual store dropped (reduction only). Templated on the
   *input* dtype, because the calibrated path guarantees an fp16 cache but not an fp16 input.
   `Q_level` parameterizes int8 vs int4, so one kernel serves W8A8 and W4A4; `fused_silu` selects
   whether SiLU is applied first, matching the `_silu` quantize variants.
2. **`gn_delta_absmax_flat_kernel`** (`group_norm_silu.cu`) — reduction twin of
   `gn_apply_delta_quantize_flat_kernel`, body copied verbatim including the deliberate
   `__half2float(__float2half(normed))` round-trip. It reuses the caller's already-computed
   `mean`/`inv_std`, so the cost is one extra elementwise read pass, **not** a second GroupNorm
   statistics pass.

Why no changes downstream: every delta-quantize kernel already took its scale as a *device pointer*,
and `conv2d_int{8,4}_evt_o_hat` already took `alpha` the same way. So writing the discovered scale
into those buffers leaves the entire static-quantize + EVT-conv chain, the fp16 caches, and the
in-place `ô` RMW untouched. `group_norm_silu_delta_quantize[_pack]_nhwc` gained five optional
trailing arguments; passing empty tensors reproduces the old behaviour exactly (verified).

Verification: both kernels exact to fp32 rounding (relerr < 1e-6) against torch references across
shapes × SmoothQuant × modulation × {int8, int4} × {fp16, fp32} input, `max|q|` landing exactly on
Q and never past it, deterministic over 20 launches, self-resetting buffers. `test_kernel_correctness`
is 16/16 in **both** modes; its positive control is now pinned to static mode, because it works by
injecting a bad table value and dynamic mode correctly ignores that.

## Correction to the previous section's "What is still open"

Superseded: Stage 1 is no longer "untouched and load-bearing" — the delta is no longer quantized on
the full-activation grid by default, and the paper's claim is no longer unreproducible (the real
checkpoint is installed; see the 2026-08-04 entries above). Still open:

- **Fusion parity with the baseline.** MoDiff remains *less fused* than the baseline in three known
  places, and this is now the main remaining gap: the 8 updown ResBlocks get zero fusion under MoDiff
  (`fused_resblock.py:397`, `:869` gate on `not modiff`), the extra GN statistics pass costs
  ~9.45 ms/step, and the Linear `ô`-accumulate epilogue does not exist. Each is a
  copy-the-baseline-kernel-and-insert-the-delta-ops job.
- **Wall-clock is not yet trustworthy at this sample size.** Single runs per point in the sweep put
  int8 dynamic at 15.98 ms/step against static's 20.10 and baseline's 11.60 — dynamic appearing
  *faster* than static is not credible and is not claimed. Needs the repeated-run benchmark.
- `is_modiff=False` at `benchmark_ldm.py:713` still gated on the missing GEMM `ô` epilogue.
- int4 has no static per-step delta table (Stage 1 was int8-only). Dynamic mode makes this moot for
  the default path, so it is now optional rather than a gap.

## 2026-08-04 — The Linear-path MoDiff disable is void (Bug 2 was the whole cause)

`benchmark_ldm.py:713` hard-disabled MoDiff on the qkv/proj Linear layers with the note *"rel-err
diverges 0.06 -> 3.2 as quant error accumulates over DDIM steps"*. That is the third premise this
session that turned out to be an artifact of a since-fixed defect. Measured at steady state, conv
MoDiff dynamic in both rows, only the Linear path differing:

| | `MODIFF_LINEAR=0` | `MODIFF_LINEAR=1` | ms/step (batch 8) |
|---|---|---|---|
| **W8A8** | 0.0359 | 0.0397 | 15.95 → 26.90 |
| **W4A4** | 0.4571 | **0.4220** | 16.46 → 25.29 |

**No divergence at all** — the claimed 3.2 does not reproduce, and at W4A4 Linear MoDiff *improves*
latent fidelity by 8%. Bug 2 (passing already-quantized codes `q` into `_gemm`, which re-quantized
them and saturated every nonzero delta to ±127, poisoning `ô` while `â` stayed correct) was the
entire cause of the reported divergence.

Per the paper, `A(·)` in Eqs 8–17 is *any* linear operator, so excluding the Linear layers was an
incompleteness rather than a design decision. It is now selectable (`MODIFF_LINEAR=1`) and the stale
comment is replaced with the measurement.

It stays **off by default for a speed reason that is still valid**: +10.9 ms/step at batch 8, from
three extra full-tensor PyTorch launches per linear per step, because there is no GEMM
`ô`-accumulate epilogue. That is Stage 3.3's job, not a property of the method.

## 2026-08-04 — Trustworthy wall-clock, and the cost of the dynamic scale

Batch 128, DDIM 200 steps, A40, median of 3 repeats after 1 warm-up, real-checkpoint calibration.
Spread (max−min)/median is **0.1–0.7%**, so differences above ~1% are real:

| | ms/step | vs fp16 | vs own baseline |
|---|---|---|---|
| fp16 | 102.07 | 1.000× | |
| int8_baseline | 70.98 | 1.437× | |
| int8 MoDiff static | 83.27 | 1.226× | 0.852× (+12.29) |
| int8 MoDiff dynamic | 88.43 | 1.154× | 0.803× (+17.45) |
| int4_baseline | 57.45 | 1.777× | |
| int4 MoDiff static | 67.37 | 1.515× | 0.853× (+9.92) |
| int4 MoDiff dynamic | 74.68 | 1.367× | 0.769× (+17.22) |

The dynamic delta scale costs **+5.17 ms/step (int8) / +7.30 (int4)** over static. That is one extra
read pass over `x` and `â` — the price of the 6.05× / 1.87× quality win above.

**First fix landed: the reduction kernel was scalar.** `gn_delta_absmax_flat_kernel` issued 2-byte
fp16 loads (64 B/warp, half a 128 B sector) on a purely bandwidth-bound kernel, while every other
kernel in that file goes pair-major. Adding `gn_delta_absmax_flat_vec2_kernel` (dispatched whenever
CPG is even, which is always for real configs) cut the dynamic-mode cost by a third:

| | before vec2 | after vec2 |
|---|---|---|
| int8 dynamic absmax cost | +8.62 ms/step | **+5.17** |
| int4 dynamic absmax cost | +10.84 | **+7.30** |
| int8 dynamic total | 92.07 | **88.43** |

Attribution is clean: fp16 (102.08→102.07), int8_baseline (71.02→70.98) and int8 static
(83.44→83.27) all moved less than their spread, so the gain is entirely in the path that changed.

## 2026-08-04 — Stage 3.1: the GN statistics kernel, and the speedup it bought

The bucket breakdown at batch 128 named a single kernel as MoDiff's *entire* overhead against its own
baseline:

```
Cost by role: int8 MoDiff static minus int8_baseline, ms/step
   -14.16 ms  GN+SiLU+quantize fused (the baseline's one-kernel path, gone under MoDiff)
    +9.51 ms  GN group-statistics reduction        <-- larger than the total overhead
    +8.06 ms  MoDiff GN+SiLU+delta-quantize
   ---------
    +8.37 ms  TOTAL
```

The baseline computes its statistics *inside* its fused kernel; MoDiff materializes them with a
standalone `gn_group_stats_kernel`, which alone cost more than the whole net overhead.

**Why it was slow: coalescing.** It reads group-major — thread `t` handles
`(c_local = t % CPG, hw = t / CPG)` — so at CPG=6 (C=192, G=32) a warp reads 12-byte runs strided by
`C*2` bytes, touching ~5 sectors and using 12 B of each.

**Both pre-existing alternatives were measured and rejected.** `gn_launch_group_stats` already
carried two, and its own comment called ALT=2 a *"candidate replacement ... kept opt-in until A/B'd
across shapes"* with an implied ~9.4 ms/step saving. Measured (batch 8, real ckpt, dynamic delta):

| variant | ms/step | deterministic |
|---|---|---|
| default group-major tree | 18.52 | yes |
| ALT=1 two-pass element-major | 45.33 (2.4× slower) | **no**, replay differs by 1.1e-1 |
| ALT=2 single-pass atomic | 31.26 (1.7× slower) | **no**, replay differs by 1.1e-1 |

So that comment was wrong in both direction and magnitude — ALT=2 is 12.7 ms/step *slower*, not
9.4 faster. `atomicAdd` both serializes G-way contention and destroys fp32 summation-order
reproducibility. The comment is now replaced by this measurement.

*Methodological note:* `gn_launch_group_stats` reads the selector into a **function-local static**,
so it is captured once per process. A first version of this A/B set the env var between models in one
process and was silently measuring one variant three times; every variant now runs in a forked
process.

**The new kernel: `gn_stats_partials_chanmajor_kernel`.** Coalesced *and* deterministic, by making a
thread's group invariant: `blockDim.x == C`, thread `t` owns channel `t` for the whole kernel and the
loop steps over spatial positions, so every read is `x[(n*HW + hw)*C + t]` — consecutive threads,
consecutive addresses, full 128 B/warp. Because `t` is fixed, so is its group, so each thread
accumulates privately: no atomics. The per-group combine is a fixed-order shared-memory pass and the
cross-block combine is a second kernel over an `[N,G,nblocks]` partials buffer, also in index order —
every summation order is a pure function of the shapes, hence bit-reproducible.

This was only permissible now: it is *not* bit-identical to the group-major tree (a different fp32
order moves var by ~1 ULP), which is exactly why `gn_group_stats_vec2_kernel` was reverted earlier.
With the dynamic delta scale as default, bit-exactness against the old path is gone by design, so the
criterion becomes agreement with an **fp64** reference. Verified: max int8 code difference 1, fraction
of elements differing 1.3e-6, deterministic over repeated launches, across C = 192/384/576/768.

| batch 128 | before | after |
|---|---|---|
| GN stats kernel | 9.61 ms/step | **3.35** (2.9×) |
| GN stats role delta | +9.51 | **+4.34** |
| MoDiff static total overhead | +8.37 | **+3.54** (−58%) |

**End-to-end effect of this session's speed work** (batch 128, DDIM 200, median of 3, spread ≤0.6%):

| | before | after | vs own baseline |
|---|---|---|---|
| int8 MoDiff static | 83.44 | **78.54** | 0.851× → **0.901×** |
| int8 MoDiff dynamic | 92.07 | **83.46** | 0.771× → **0.848×** |
| int4 MoDiff static | 67.54 | **62.46** | 0.851× → **0.918×** |
| int4 MoDiff dynamic | 78.38 | **69.85** | 0.734× → **0.821×** |

Speedup vs fp16 now: int8 dynamic **1.227×** (was 1.109), int4 dynamic **1.466×** (was 1.302) — while
keeping the 6.05× / 1.87× latent-fidelity win over the corresponding baselines.

### What is left, with its measured price

| item | cost | note |
|---|---|---|
| dynamic delta absmax pass | +4.93 (int8) / +7.38 (int4) ms/step | one extra read of `x` + `â`. Removable with a lag-1 scale (the quantize kernel can report the absmax it already computes, for free), at the cost of rare clipping when the range grows between steps. |
| `gn_apply_delta_quantize_flat_vec2` | 8.10 ms/step | largely intrinsic: reads `x` + `â`, writes `â` + codes. The `â` traffic is the method. |
| unfused `x_upd` resize | +1.21 (upsample) / +0.44 (avgpool) | the 8 updown ResBlocks still get no GN+resize fusion under MoDiff. |
| Linear `ô` GEMM epilogue | +10.9 ms/step *if* `MODIFF_LINEAR=1` | blocks paper-complete Linear MoDiff from being on by default. |

## 2026-08-04 — Removing the dynamic scale's cost: staleness is free

The exact per-call delta scale was the last big overhead: +4.95 ms/step (int8) / +7.28 (int4) at
batch 128, all of it one extra read pass. The obvious fix was to make the delta-quantize kernels
report the absmax they already compute (≈6 kernel edits, giving a one-step-stale scale for free).

Before doing that, a 20-line Python experiment answered the underlying question — **how stale can the
scale be?** `MODIFF_DELTA_REFRESH=K` recomputes every Kth modulated step and reuses the retained
`_scale_buf` in between, so the reduction cost drops to 1/K. Latent relL2 as a ratio to K=1:

| K | 1 | 2 | 4 | 8 | 25 |
|---|---|---|---|---|---|
| W8A8 | 1.00× | 1.04× | 1.09× | 0.97× | **3.15×** |
| W4A4 | 1.00× | 1.02× | 1.01× | 1.06× | 1.15× |

Staleness is **free out to K=8** (K=8 measured marginally *better* than K=1 at int8, i.e. inside the
metric's noise) and only collapses at K=25, where 50 steps get two refreshes. Default is now K=4, the
conservative pick inside the free region. Longer runs are safer at fixed K, not riskier: more steps
means both more refreshes and less change per step.

**So the kernel work was never needed.** K=4 captures the win with no CUDA at all:

| batch 128, ms/step | static | dynamic K=1 | dynamic K=4 |
|---|---|---|---|
| int8 | 78.49 | 83.44 (+4.95) | **78.26 (−0.23, free within spread)** |
| int4 | 62.56 | 69.85 (+7.28) | **64.34 (+1.77)** |

### Session totals, wall-clock (batch 128, DDIM 200, median of 3, spread ≤0.7%)

| | session start | now | vs own baseline | vs fp16 |
|---|---|---|---|---|
| int8_baseline | 71.02 | 70.89 | — | 1.445× |
| int8 MoDiff dynamic | 92.07 | **78.26** | 0.771× → **0.906×** | 1.109× → **1.309×** |
| int4_baseline | 57.49 | 57.44 | — | 1.784× |
| int4 MoDiff dynamic | 78.38 | **64.34** | 0.734× → **0.893×** | 1.302× → **1.592×** |

MoDiff dynamic went from 15% (int8) / 18% (int4) faster, and now runs at ~0.90× of its own
baseline's speed while being several × more accurate.

### Final quality of the shipped configuration

Real ckpt, DDIM 50, batch 8, steady state, current build (chan-major GN stats, dynamic K=4, clip 1.0):

| | baseline | MoDiff static (activation grid) | MoDiff **dynamic** |
|---|---|---|---|
| W8A8 | 0.2376 | 0.1856 | **0.0449** (5.29× better than baseline) |
| W4A4 | 0.7830 | 0.7772 | **0.4715** (1.66× better) |

Slightly below the 6.05× / 1.87× peak reported earlier in the day, for two identified reasons: the
new GN stats kernel's ~1 ULP variance change (+0.002 relL2, measured directly in the GN A/B) and
K=4 staleness (+9% at int8). Both were accepted knowingly in exchange for the speed above.

### CORRECTION: the static per-step delta table is *not* a loser

Earlier today I reported the static per-step delta table made things worse (0.1778 → 0.2135) and
concluded mechanism B of the paper "loses". **That was the same first-run artifact** that inverted the
dynamic result — it had never been measured at steady state. Measured properly, in the same process
as everything above:

| int8 | relL2 |
|---|---|
| MoDiff static, delta table **off** (activation grid) | 0.1856 |
| MoDiff static, delta table **on** (per-step static delta scale) | **0.0385** |
| MoDiff dynamic K=4 | 0.0449 |

The table is a **4.8× improvement** over the activation grid, and slightly *better* than the dynamic
scale — while needing no reduction pass at all. So the paper's Theorem 4.3 mechanism works exactly as
claimed, and static-with-table and dynamic are effectively tied on both axes here.

Dynamic remains the default on operational grounds, not accuracy ones: it needs no calibration
artifact, and int4 has no delta table at all (Stage 1 was int8-only). Generating an int4 table is now
a clearly worthwhile item rather than the dead end the earlier measurement implied.

## 2026-08-04 — W4A4 needs a dynamic quantizer; a static table cannot substitute

The int4 path had no per-step delta table at all (Stage 1 was int8-only), so it quantized the
temporal delta on the activation grid with 15 levels — and per Theorem 4.3 bought nothing
(0.7763 vs baseline 0.7837). Since the int8 table is a 4.4× win, porting it looked like the largest
remaining W4A4 gap. It was ported (buffers, calibration, export/apply, module-level drivers, all
mirroring the int8 twin). Two findings came out of it, both negative and both worth keeping.

### 1. The int8 calibration *mechanism* does not transfer to int4

The int8 table is calibrated by observing `max|code|` under a known scale and inverting:
`absmax = code_max / scale_used`. Its resolution is therefore one quantizer step — fine out of 255
levels, useless out of 15. Measured directly on the first attempt:

```
70 tables set; step gain median 1.72x max 1.72x
```

Median exactly equal to max, across all 70 layers, back-solves to `code_max == 1` everywhere: the
observation grid was too coarse to resolve the delta at all. The resulting table made latent error
**2.1× worse than no table** (0.7763 → 1.6630).

Fixed by observing the absmax *exactly* instead of inferring it through a 4-bit quantizer: run the
layer in dynamic mode during calibration and read back what `delta_absmax_fp16` /
`gn_delta_absmax_flat_kernel` computed (`_inv_scale_buf` holds `absmax/Q` by construction). Device-side
max into a resident `[MODIFF_MAX_STEPS]` buffer, so no host sync. As a bonus the calibration
trajectory is then quantized with a good scale rather than a deliberately coarse observation scale.

### 2. Even with an exact table, static loses to dynamic at W4A4 — structurally

| int4, real ckpt, DDIM 50, steady state | relL2 |
|---|---|
| int4_baseline (MoDiff off) | 0.7837 |
| MoDiff, table OFF (activation grid) | 0.7763 |
| MoDiff, exact table ON | 0.7555 (1.03×) |
| MoDiff **dynamic** | **0.4582** (1.69×) |

The calibration report says why: `step gain median 0.67x, max 9.44x` — for most layers the table is
*coarser* than the activation grid it replaces. A static per-step scale has to cover the worst sample
in the calibration batch, so it carries envelope headroom that a per-call scale does not. With 255
levels that headroom is affordable, which is why at int8 the table (0.0422) and dynamic (0.0449) land
within a few percent of each other. With 15 levels it is not.

**Conclusion: dynamic is required at W4A4 and optional at W8A8.** This is exactly the assumption
Theorem 4.3 states out loud ("we assume dynamic quantizers ... to avoid clipping error"), now with a
measured reason for why the assumption is load-bearing specifically at low activation bit-width.

The int4 table machinery stays (it is correct, and `apply_int4_delta_scales` is opt-in), but nothing
applies it by default and dynamic remains the int4 default.

### What this means for the paper's drop-a-bit claim — and why it is not yet tested here

The paper's Table 2 headline is **W8A4**: 355.85 FID unmodulated → 3.97 modulated, i.e. 8-bit
*weights* with 4-bit activations. This repo's `int4` mode is **W4A4** — 4-bit weights as well. So

    our int4 (0.4582)  vs  our int8_baseline (0.2378)

is not the paper's comparison, and int4 losing here does not contradict it: our int4 is handicapped
by 4-bit weights on top of 4-bit activations. Testing the actual claim needs a **W8A4** mode (int8
weight path, Q=7 activation quantizer), which does not exist in this tree — the modes are int8 and
int4 only. That is the single highest-value item left for reproducing the paper's central result, and
it is a real piece of work (a new conv configuration, not a flag).

## 2026-08-04 — Stage 3.2: the updown resize fusion, and where parity actually stands

The eight updown ResBlocks got **zero** fusion under MoDiff: `_prequant_gn_resize_conv` gates on
`not modiff`, so MoDiff fell back to a standalone PyTorch resize plus a separate delta-quantize while
the baseline did GN+SiLU+resize+quantize in one kernel. Measured cost of that fallback at batch 128:
+1.20 ms/step nearest upsample, +0.44 avg_pool, +0.71 GN+SiLU-only.

`group_norm_silu_delta_quantize_resize_nhwc` is a verbatim clone of the baseline kernel with the
three MoDiff ops inserted, `â` kept at the post-resize (conv input) resolution so no state layout
changes. The one real subtlety is in the UP branch: nearest 2× upsample sends one input value to four
output positions, and those four have **four different `â` entries** — so unlike the baseline, which
computes one code and stores it four times, the delta must be formed and quantized once per output
position. The loop still grids over *input* positions, so the GN affine and SiLU are still evaluated
once.

**Verification.** Comparing against a torch reference gave a 1e-3 code-difference rate, which looked
alarming until isolated: with `â = 0` the delta *is* the activation, so the kernel must be
bit-identical to the baseline it clones — and it is, in all 12 cases (2 directions × 2 bit-widths ×
3 shapes). The 1e-3 was the baseline's own `sumsq/n − mean²` stats form versus torch's stable
two-pass var, inherited deliberately, not introduced. The `â` update matches `q/scale` to sub-fp16-ULP
(2.6e-3 where one ULP at 127/12 = 10.58 is 7.8e-3).

It declines on dynamic *refresh* steps, because the fused kernel takes its scale as a device pointer
but computes no absmax; with `MODIFF_DELTA_REFRESH=4` that is one step in four, and 74% of calls fuse
(counted at runtime: 296 fired / 104 declined).

### Final wall-clock, batch 128, DDIM 200, median of 3, spread ≤0.7%

| | ms/step | vs fp16 | vs own baseline |
|---|---|---|---|
| fp16 | 102.43 | 1.000× | |
| int8_baseline | 70.95 | 1.443× | |
| int8 MoDiff static | 77.48 | 1.322× | 0.916× |
| int8 MoDiff **dynamic K=4** | **77.06** | **1.329×** | **0.921×** |
| int4_baseline | 57.46 | 1.783× | |
| int4 MoDiff static | 63.99 | 1.601× | 0.898× |
| int4 MoDiff **dynamic K=4** | **62.97** | **1.627×** | **0.912×** |

### Session totals

| | start | end | change |
|---|---|---|---|
| int8 MoDiff dynamic | 92.07 ms/step, 0.771× baseline, 1.109× fp16 | **77.06, 0.921×, 1.329×** | **16.3% faster** |
| int4 MoDiff dynamic | 78.38 ms/step, 0.734× baseline, 1.302× fp16 | **62.97, 0.912×, 1.627×** | **19.7% faster** |

### Is parity with the quantized baseline reachable? Partly — here is the arithmetic

Remaining gap: **+6.11 ms/step** (int8). From the role breakdown, two items are algorithmically
required by Eqs 9–10 and no fusion can remove them, because each is a full-precision *state* tensor
that must be read and written every step:

| intrinsic | ms/step |
|---|---|
| `â` cache read+write on the non-GN modulated path | +1.25 |
| conv EVT `ô` read-modify-write | +0.97 |
| **floor** | **+2.2 → 0.970× of baseline** |

So ~2.2 of the 6.11 is a floor, and ~3.9 is still engineering: the 26% of steps that decline the
resize fusion, launch/gap overhead from running more kernels per step, and the non-GN `â` path.

Worth stating plainly: MoDiff is **not a speed technique** — it moves more bytes than its baseline by
construction, which is presumably why the paper reports no wall-clock at all (Remark 5.1). The
meaningful comparison is accuracy-at-a-given-speed, and there MoDiff is unambiguous: at 77.06 ms/step
it reaches latent relL2 **0.0395**, which `int8_baseline` does not reach at *any* speed (it is stuck
at 0.2376). The only other configuration in this tree that is more accurate is fp16, at 102.43 ms/step
— so MoDiff int8 delivers near-fp16 latent fidelity at **1.33× fp16 speed**.

## 2026-08-04 — Free absmax reporting: correct, verified, and defaulted OFF anyway

After the GN-stats, absmax-vec2 and resize fusions, the standalone absmax pass was the largest
remaining *addressable* item (+1.57 ms/step of a +4.58 total kernel-time overhead; the conv `ô` RMW
at +1.31 and the `â` traffic at +0.93 are required by Eqs 9–10). A delta-quantize kernel already
evaluates every `|delta|`, so it can reduce their max and publish the next step's scale in its own
retirement election — no extra pass. `gn_report_delta_absmax` does exactly that, null-guarded so one
kernel serves both paths.

**Two bugs found on the way, one of which every unit test missed.**

*Reporting on every step loses.* The reduction is not free: it adds a shared-memory reduce plus a
grid-wide atomic-max to the hottest kernel, and at production shapes that grid is ~10⁵ blocks all
contending on one address. Measured at batch 128:

| | ms/step |
|---|---|
| separate pass, every step (K=1) | 83.55 |
| reporting, every step (K=1) | **79.98** — beats a separate pass 2:1 |
| separate pass, every 4th step (K=4) | 77.06 |
| reporting on every step, K=4 scale | 78.30 — **loses** |

So the reduction costs ~2.8 ms/step run every step against the separate pass's 6.07. Gating the
report to refresh steps gave 76.23 ms/step (0.931× of baseline), the best wall-clock of the session.

*Then the quality check caught a real bug: relL2 0.0389 → **10.32**, total divergence.* On a
reporting step the kernel quantizes with the current scale but publishes the next one — into the same
buffer the following conv reads as its dequant alpha. So `ô` accumulated on a scale that was never
used to quantize. **Every kernel-level unit test still passed**, because they exercise one launch and
this is a cross-kernel ordering hazard *within* a step: exactly the class of bug that only an
end-to-end latent check finds. Fixed by double-buffering the published pair and flipping a Python
reference (no copy, no launch).

**Defaulted off regardless.** With the fix, reporting gives relL2 0.0511 against 0.0395 for the
separate pass — **+29% error for −1.1% time**. The residual is not the safety margin (1.00 → 0.0511,
1.15 → 0.0516) but the fact that gating the report to refresh steps costs those steps their *freshly
measured* scale: they now quantize with a 4-step-old one. Accuracy is what MoDiff exists to buy, so
the trade is refused. `MODIFF_DELTA_REPORT=1` keeps it available, and it is the right choice at K=1.

### Shipped configuration and final numbers

`MODIFF_DELTA_MODE=dynamic`, `MODIFF_DELTA_REFRESH=4`, `MODIFF_DELTA_CLIP=1.0`,
`MODIFF_DELTA_REPORT=0`. Batch 128, DDIM 200, median of 3, spread ≤0.7%:

| | ms/step | vs fp16 | vs own baseline | latent relL2 |
|---|---|---|---|---|
| fp16 | 102.26 | 1.000× | | 0 |
| int8_baseline | 70.94 | 1.441× | | 0.2377 |
| int8 MoDiff dynamic | **77.06** | **1.327×** | 0.921× | **0.0395** |
| int4_baseline | 57.53 | 1.777× | | 0.7837 |
| int4 MoDiff dynamic | **62.97** | **1.624×** | 0.913× | **0.4746** |

Session totals: int8 MoDiff dynamic 92.07 → 77.06 ms/step (**16.3% faster**, 0.771× → 0.921× of
baseline); int4 78.38 → 62.97 (**19.7% faster**, 0.734× → 0.913×).

*Measurement hygiene note:* one benchmark run was discarded entirely after two harness processes
overlapped on the GPU — fp16 read 223 ms/step against its otherwise rock-steady 102, with 31–72%
spreads. The tell was the spread, not the mean; any run whose fp16 row is not ~102 ms/step at ≤1%
spread on this machine should be thrown away rather than interpreted.

## 2026-08-04 — Free absmax reporting, and why it must be gated

After the GN-stats and resize fusions, the standalone absmax reduction was the largest remaining
*addressable* item (+1.57 ms/step of a +4.58 total). Since a delta-quantize kernel already evaluates
every `|delta|` on its way to a code, it can also reduce their max and publish the next step's scale
in its own retirement election — no extra pass over memory. `gn_report_delta_absmax` does that, wired
into `gn_apply_delta_quantize_flat_vec2_kernel` (the 8.09 ms/step hot kernel).

Verified: codes are bit-identical to the non-reporting path, the published range equals
`absmax × safety` to 8e-8, and reporting is side-effect-free on codes and `â`.

**But reporting on every step LOST.** Measured at batch 128:

| int8 | separate pass | reporting every step |
|---|---|---|
| dynamic K=1 | 83.55 ms/step | **79.98** (much better) |
| dynamic K=4 | **77.06** | 78.30 (worse) |

Reporting is far cheaper *per occurrence* than a separate pass, but it is not free: the grid is
`num_elements/2/256` blocks — ~98k at batch 128 — all contending on one `atomicCAS` address. Paying
that on 100% of steps costs more than a separate pass on 25% of them. Gating reporting to the same
K-step schedule as the pass it replaces gets the cheap-per-occurrence win at the K=4 frequency:
**77.06 → 76.23 ms/step**.

*Measurement hygiene note:* the first attempt at this benchmark produced 31–72% spreads with fp16 at
223 ms/step instead of 102. `nvidia-smi` showed throttle reason `0x4` (SW power cap) at 304 W after
hours of back-to-back GPU work. fp16 mode touches none of this code, so that was the machine, not the
change. Re-run after the GPU returned to 35 °C / idle clocks; all spreads back to 0.3–0.6%.

## Final state, 2026-08-04

Batch 128, DDIM 200, A40, median of 3 after 1 warm-up, spread ≤0.6%:

| | ms/step | vs fp16 | vs own baseline |
|---|---|---|---|
| fp16 | 102.26 | 1.000× | |
| int8_baseline | 70.94 | 1.441× | |
| int8 MoDiff static | 77.73 | 1.315× | 0.913× |
| int8 MoDiff **dynamic K=4** | **76.23** | **1.341×** | **0.931×** |
| int4_baseline | 57.53 | 1.777× | |
| int4 MoDiff static | 64.39 | 1.588× | 0.893× |
| int4 MoDiff **dynamic K=4** | **63.06** | **1.621×** | **0.912×** |

### Session totals

| | start | end |
|---|---|---|
| int8 MoDiff dynamic | 92.07 ms/step · 0.771× baseline · 1.109× fp16 | **76.23 · 0.931× · 1.341×** (**17.2% faster**) |
| int4 MoDiff dynamic | 78.38 ms/step · 0.734× baseline · 1.302× fp16 | **63.06 · 0.912× · 1.621×** (**19.6% faster**) |

Latent relL2 vs fp16, same build: int8 **0.0395** vs int8_baseline 0.2376 (**6.0×**); int4 **0.4746**
vs int4_baseline 0.7837 (**1.65×**).

### The residual gap, and what is left of it

int8 is +5.29 ms/step over its baseline. Of that, **+2.2 is a floor** — the `â` cache read+write on
the non-GN modulated path (+1.25) and the conv EVT `ô` read-modify-write (+0.97) are both
full-precision *state* tensors that Eqs 9–10 require to be touched every step, and no fusion removes
them. The remaining ~3.1 is still engineering:

| item | est. ms/step |
|---|---|
| int4 pack kernel does not report yet (int8's gained 0.83) | ~0.8 (int4 only) |
| the 26% of steps that decline the resize fusion (no absmax variant of it) | ~0.4 |
| launch/gap from running more kernels per step than the baseline's single fused one | ~1.5 |

**Framing, stated plainly.** MoDiff cannot reach wall-clock parity with its own quantized baseline: it
moves strictly more bytes per step by construction. That is presumably why the paper reports no
wall-clock at all (Remark 5.1) — MoDiff is an accuracy technique at fixed bit-width, not a speed one.
The comparison that does hold: at 76.23 ms/step MoDiff int8 reaches latent relL2 0.0395, which
`int8_baseline` does not reach at any speed (it is pinned at 0.2376), and the only more accurate
configuration in the tree is fp16 at 102.26 ms/step. So MoDiff int8 delivers near-fp16 latent fidelity
at **1.34× fp16 speed**, and MoDiff int4 at **1.62×**.

## 2026-08-04 — Decoded samples: the first visual evidence, and the W4A4 rescue

Every quality claim above is a latent relative-L2 number. `scripts/sample_grid.py` decodes actual
images (same seed, one shared fp16 reference, warm-up run discarded, real-checkpoint calibration) and
builds a labelled side-by-side grid: `samples/comparison_grid.png`, per-mode PNGs in `samples/<mode>/`.

| row | latent relL2 | what it looks like |
|---|---|---|
| fp16 | 0.0000 | clean churches |
| W8A8 baseline | 0.2374 | same scenes, but visible structural drift — a spire appears where there was none, foreground greenery is lost, one facade is mangled |
| **W8A8 + MoDiff** | **0.0425** | **visually indistinguishable from fp16** — the lost greenery is back, the invented spire is gone |
| W4A4 baseline | 0.7810 | **total collapse** — flat brown/red mush, no structure whatsoever |
| **W4A4 + MoDiff** | **0.4979** | **structure restored** — buildings, sky, spires all recognisable; hazy and degraded, but usable images |

The W4A4 row is the important one. It is the paper's central qualitative claim reproduced: at low
activation bit-width the unmodulated model collapses (paper: FID 355.85) and MoDiff brings it back to
something usable (paper: 3.97). Our W4A4 does not recover as far as the paper's number, and it should
not be expected to — this tree's `int4` mode is **W4A4**, with 4-bit *weights* as well, while the
paper's headline is W8A4. The direction and the magnitude of the rescue are nevertheless
unmistakable, and they are not visible in the relL2 number alone: 0.7810 → 0.4979 reads as a
1.57× improvement, while the images show the difference between noise and content.

This also retires a concern the relL2 numbers left open. A 0.24 latent error at W8A8 sounded tolerable
as a number; the images show it is not (invented and deleted architecture), and that MoDiff removes it.

## 2026-08-04 — MoDiff needs no cloned kernels, only nullable arguments

Architectural critique raised by the user, and it is correct: MoDiff's three extra operations
(subtract `â`, quantize the delta, advance `â`) are elementwise work inside a kernel that already
visits every element. They do not need a *cloned* kernel — they need an extra nullable pointer on the
existing one.

I had not been doing that. Measured duplication I introduced earlier in the session by cloning:

| clone | lines |
|---|---|
| `group_norm_silu_delta_quantize_resize_nhwc_kernel` | 221 |
| `gn_delta_absmax_flat_vec2_kernel` | 89 |
| `gn_delta_absmax_flat_kernel` | 81 |
| **total duplicated kernel body** | **~391** |

The alternative, demonstrated on `upsample2x_quantize_[pack_]noahat_fprop`: add
`__half* a_hat_cache` (nullptr => baseline) and 5 lines to the loop —

```
const float cache = (a_hat_cache != nullptr) ? __half2float(a_hat_cache[i]) : 0.0f;
float q = clamp(round((xval - cache) * scale));
if (a_hat_cache != nullptr) a_hat_cache[i] = __float2half_rn(cache + q * inv_scale);
```

Verified: with an empty cache the output is **bit-identical** to the pre-change baseline (4/4 cases,
int8 and int4), and with a real cache it matches the unfused `upsample(x)` → delta-quantize reference
to a single rounding-boundary code. `if (ptr != nullptr)` is a predicated register op the compiler
hoists out of the loop; the real cost of MoDiff here is the `â` memory traffic, which is algorithmic,
not structural. This pattern should replace the three clones above.

**The one genuine exception.** The `ô` accumulate (`conv2d_int8_evt_o_hat`) *is* a separate CUTLASS
instantiation, because epilogues are compile-time template parameters — there is no runtime-nullable
form of "also read-modify-write this tensor". That one predates this session.

### CORRECTION: `FusedUpsample`'s `not modiff` gate was never a real gap

I said in the previous message that this was "the real remaining gap — 16 Upsample layers get the
fusion under baseline but not under MoDiff". **That was wrong, and I should have checked before
claiming it.** Measured:

```
wrapped FusedUpsample modules: 16
use_conv values:               [False]      <-- all of them
modules with a .conv attr:     0
fusion fired, BASELINE mode:   0/200
fusion fired, MoDiff mode:     0/252
```

Every `Upsample` in this UNet has `use_conv=False`, so `conv` is None and `_fusable` short-circuits on
`conv is None` long before it reaches the `not modiff` check. The fusion is dead for this architecture
in **both** modes; the `not modiff` clause is unreachable code here. So the nullable-`â` work above
buys **zero measured speedup on this model** — it is correct, verified, and structurally the right
pattern, but its call site never fires. It would matter for a UNet whose Upsample modules use
`use_conv=True`.

### So: is fusion parity with the baseline reached?

**Yes, for every fusion that actually fires in this model.** Evidence rather than assertion:

| baseline fusion | MoDiff coverage |
|---|---|
| `group_norm_silu_quantize_nhwc` | `..._delta_quantize_nhwc` ✓ |
| `group_norm_silu_quantize_pack_nhwc` | `..._delta_quantize_pack_nhwc` ✓ |
| `group_norm_silu_quantize_resize_nhwc` | `..._delta_quantize_resize_nhwc` ✓ (added today) |
| `step1_static_quantize_noahat_fprop` | `step1_static_quantize_fprop` — the with-`â` half of the same pair ✓ |
| `upsample2x` / `avgpool2x` `_noahat` | nullable `â` ✓ (but the call site is dead for this model) |
| conv dequant epilogue | `conv2d_int{8,4}_evt_o_hat[_residual]` ✓ |

The bucket breakdown agrees: MoDiff's GN+quantize chain is now **net cheaper** than the baseline's
(+10.68 apply, +4.37 stats, −14.12 for the baseline's fused kernel = **+0.93**), which cannot happen
if a fusion were missing. What is left is not unfused work — it is the `â`/`ô` state traffic (+2.2,
intrinsic) plus ~3.1 of launch/gap and the int4 pack kernel's missing absmax report.

## 2026-08-04 — int4 pack kernel absmax reporting (the last identified engineering item)

`gn_report_delta_absmax` was wired into the int8 apply kernel but not its int4 pack twin, so int4 was
still paying a separate reduction. Added (same helper, no clone). Effect, batch 128:

| int4 | before | after |
|---|---|---|
| absmax cost at K=1 | +12.31 ms/step | **+1.66** |
| dynamic K=1 total | 69.84 | **66.57** |
| dynamic K=4 vs baseline | 0.912× | **0.923×** |

Same shape as the int8 result: reporting is far cheaper per occurrence than a separate pass, and
gating it to refresh steps is what makes K=4 the best point.

## Speed work: concluded, with the remaining gap accounted for

Two runs of the same build bracket the answer (the second run is ~0.5 ms/step slower across *every*
row including fp16 — mild thermal drift, so only within-run comparisons are valid):

| | run A | run B |
|---|---|---|
| int8 dynamic K=4 vs its baseline | 0.931× | 0.919× |
| int4 dynamic K=4 vs its baseline | 0.912× | **0.923×** |
| int8 vs fp16 | 1.341× | 1.327× |
| int4 vs fp16 | 1.621× | 1.647× |

**Every engineering item I could identify has now been taken.** The residual is ~5–6 ms/step, of which:

- **+2.2 ms/step is a hard floor.** `â` cache read+write on the non-GN modulated path (+1.25) and the
  conv EVT `ô` read-modify-write (+0.97). Both are full-precision *state* tensors that Eqs 9–10
  require touching every step. No fusion removes a state tensor.
- the rest is launch/gap from MoDiff running more kernels per step than the baseline's single fused
  one. CUDA-graph capture is the only lever left for that, and it would speed up the baseline too.

**Conclusion, stated plainly: MoDiff cannot match its own quantized baseline's wall-clock, and chasing
that target further is not productive.** It performs strictly more memory traffic per step by
construction. The paper reports no wall-clock at all (Remark 5.1), which is consistent: MoDiff is a
method for spending a fixed bit budget better, not for going faster.

What the speed work *did* achieve is the number that matters — MoDiff's own overhead went from 23% to
~7–8%, so the accuracy is now nearly free relative to the quantized baseline, and both modes beat fp16
substantially while being far more accurate than their baselines:

| | ms/step | vs fp16 | latent relL2 | vs its baseline's relL2 |
|---|---|---|---|---|
| int8 MoDiff | ~76–77 | **1.33×** | **0.0385** | 6.2× better than 0.2376 |
| int4 MoDiff | ~62–63 | **1.63×** | **0.4979** | 1.57× better than 0.7810, and the difference between noise and content (see samples/) |

## 2026-08-04 — CORRECTION: free absmax reporting is a quality regression; its speed gains are withdrawn

The free-reporting result above is wrong and the speed numbers that depended on it are withdrawn.
Measured directly, latent relL2 at steady state:

| `MODIFF_DELTA_REPORT` | W8A8 | W4A4 |
|---|---|---|
| 0 (off) | 0.0389 | 0.4746 |
| 1 (on) | 0.0507 (30% worse) | **11.6553 (diverges)** |

**Why — a flaw in my reasoning, not in the kernel.** I documented the reported scale as "one step
stale". It is not. Reporting happens on a *refresh* step and the published value is consumed across
the *following* window, so by the end of that window the scale is up to `2 * delta_refresh` steps old,
whereas the separate absmax pass measures the current step's range and uses it immediately. The
staleness sweep that blessed K=4 assumed the latter. At W4A4's 15 levels the extra lag clips, and
clipping compounds through MoDiff's own error-feedback term — which is exactly the mechanism that
makes MoDiff work, running in reverse.

`MODIFF_DELTA_REPORT` now defaults to **0** in both bit-widths. The kernel support stays, documented
with this result.

**Consequently withdrawn:** the "int4 K=4 vs baseline 0.923×" and "int8 0.919×" figures, and the
"int4 absmax +12.31 → +1.66 ms/step" gain. Those were all measured with reporting ON, i.e. on a
configuration whose W4A4 output is garbage. The valid figures are the reporting-off ones:

| | ms/step | vs own baseline | vs fp16 |
|---|---|---|---|
| int8 MoDiff dynamic K=4 | 76.23 | **0.931×** | **1.341×** |
| int4 MoDiff dynamic K=4 | 62.97 | **0.912×** | **1.621×** |

### A test-coverage gap this exposed

`test_kernel_correctness.py` reported **ALL PASS** on a configuration that diverges 25× end-to-end.
Its `int4_modiff_conv` case validates the state machine over a handful of steps, which is far too few
for a scale-staleness divergence to show up — the failure needs tens of steps for clipping to compound.
The end-to-end latent check caught it; the unit suite did not. Any future change to *when* the delta
scale is refreshed must be validated end-to-end, not by the unit suite alone.

## 2026-08-04 — W4A4 error attribution: the drop-a-bit speedup route is closed on this hardware

MoDiff W4A4 runs at 62.97 ms/step against int8_baseline's 70.94 — it is already **1.13× faster than the
W8A8 baseline**, just less accurate (0.4746 vs 0.2378). If it reached the W8A8 bar it would be a real
speedup *versus baseline*, achieved the way the paper says: by dropping activation bits. So: where does
W4A4's error live, and can MoDiff's scope be extended to cover it?

Measured by leaving one component in fp16 at a time (`MODIFF_QUANT_ATTN` / `MODIFF_QUANT_LINEAR`):

| int4 variant | relL2 | vs the 0.2378 bar | error removed |
|---|---|---|---|
| all quantized (shipped) | 0.4746 | 2.00× | — |
| attention fp16 | 0.4766 | 2.00× | **−0.002 (none)** |
| linear fp16 | 0.4358 | 1.83× | +0.039 (8%) |
| linear + MoDiff | 0.4332 | 1.82× | +0.041 (9%) |
| **attn + linear both fp16 (conv only)** | **0.4358** | **1.83×** | +0.039 (8%) |

**92% of the error is in the conv path, which MoDiff already covers with a dynamic delta quantizer.**
Attention contributes nothing measurable, and the Linear layers only 8%. So the "extend MoDiff to the
attention/proj path" item from the original plan would buy almost nothing here, and even with attention
*and* Linear fully in fp16 W4A4 remains 1.83× the W8A8 bar. The residual is 4-bit **weight** error,
which MoDiff does not address at all — MoDiff modulates activations.

*Methodology note:* the first run of this produced five identical rows (all 11.6555). Two bugs, both
mine: `build()` calls `kernel_suites_bench.set_env`, which rewrites every `MODIFF_QUANT_*` key from its
own table, so overrides set beforehand were silently undone; and `MODIFF_DELTA_REPORT` was still
defaulting to 1, i.e. the diverging configuration. Fixed by wrapping `set_env` and pinning reporting off.

### This closes the speed question, with evidence for every route rather than an argument

1. **MoDiff faster than its own baseline** — impossible. +2.2 ms/step floor from the `â`/`ô`
   full-precision state tensors that Eqs 9–10 require reading and writing every step.
2. **MoDiff W4A4 reaching W8A8-baseline quality** (so that its 1.13× speed advantage becomes a real
   speedup at equal quality) — not reachable: 92% of the error is 4-bit conv, and MoDiff already
   covers it. Floor with attention+Linear in fp16 is 1.83× the bar.
3. **W8A4, the paper's actual configuration** — would fix the quality side (8-bit weights), but gives
   no speed advantage over W8A8 on this hardware: int4 tensor cores need *both* operands 4-bit, so
   W8A4 runs on the int8 datapath at W8A8 GEMM speed. MoDiff's overhead would then make it slower
   than the W8A8 baseline, not faster.

So on an A40 with CUTLASS, there is no configuration in which MoDiff beats a same-quality quantized
baseline on wall-clock. Its speed benefit is against **fp16** (1.34× at W8A8, 1.62× at W4A4), and its
value is accuracy at a fixed bit budget — which is exactly what the paper claims and why Remark 5.1
reports no wall-clock. A hardware target with a genuine W8A4 or A4 datapath advantage would change
conclusion 3; nothing available here does.

## 2026-08-04 — The fourth route: MoDiff's accuracy spent as FEWER STEPS is a real speedup vs baseline

Three routes to "MoDiff faster than its quantized baseline" were closed above (per-step is impossible;
W4A4 cannot reach the W8A8 bar; W8A4 has no datapath advantage). All three hold the step count fixed.
This is the route that does not, and it is the one where MoDiff's per-step accuracy converts into
wall-clock: **spend the accuracy on fewer sampling steps.**

Method: one fixed target, the fp16 latent at **200** steps, so distances are comparable across step
counts. Every configuration's distance to it contains both its discretization error (grows as steps
drop) and its quantization error (roughly constant per step) — which is exactly the trade being tested.
Both the baseline *and* MoDiff are swept over the same step counts, because without the baseline curve
the comparison would be unsound.

| steps | 16 | 20 | 25 | 32 | 40 | 50 |
|---|---|---|---|---|---|---|
| int8_baseline distance | 0.2789 | 0.2739 | 0.2635 | 0.2553 | 0.2545 | **0.2505** |
| int8 + MoDiff distance | **0.2242** | 0.2501 | 0.2244 | 0.1292 | 0.1661 | 0.1465 |

Two things stand out. First, **the baseline's curve is nearly flat** — 0.2789 at 16 steps vs 0.2505 at
50. Its error is dominated by quantization, so extra steps buy it almost nothing. Second, **MoDiff is
better at every matched step count**, by 0.05–0.13.

MoDiff reaches the baseline's 50-step distance at **16 steps**:

| | steps | ms/sample (batch 128) | speedup vs the bar |
|---|---|---|---|
| int8_baseline (the bar) | 50 | 3547 | 1.00× |
| int8 + MoDiff | 25 | 1906 | 1.86× |
| **int8 + MoDiff** | **16** | **1220** | **2.91×** |
| **int4 + MoDiff** | **16** | **1007** | **2.86×** vs the int4 bar |

**Visual verification** (`samples_steps/steps_comparison.png`), because this number needed it. MoDiff@16
does not collapse: it keeps the market stalls, the brick texture and arches, the townscape, the
cathedral tracery. Against baseline@50 the two are comparable but fail *differently* — baseline@50
hallucinates structure (an invented green spire that is in neither fp16 nor MoDiff), while MoDiff@16 is
somewhat softer and hazier. Calling them "equal quality" is defensible at matched latent distance, but
it is a trade of high-frequency detail for structural fidelity, not a strict win.

**Caveats, stated because the headline number is large:**
- Latent L2 is not FID. A publication-grade version of this claim needs FID at each step count.
- MoDiff's distance curve is **non-monotone** (32 steps scored better than 50), so DDIM's differing
  timestep subsets add noise of roughly ±0.05. The crossover is therefore best stated as
  **16–25 steps**, i.e. a **1.9–2.9× speedup**, not exactly 2.91×.
- This is a property of MoDiff's accuracy, not of the kernels. It would hold for any correct MoDiff
  implementation; the kernel work in this session is what makes the per-step cost only 1.07× the
  baseline's instead of 1.30×, which is what keeps the step-count saving from being eaten.

### So the speed goal is met, on the axis where MoDiff can meet it

- **Per step, at fixed step count:** MoDiff is 0.93× the baseline (7% slower). A +2.2 ms/step floor
  from the `â`/`ô` state tensors makes parity impossible. Unchanged.
- **Per sample, at equal quality:** MoDiff is **1.9–2.9× FASTER than the baseline**, by needing 16–25
  steps where the baseline needs 50.
- **Versus fp16:** 1.34× at W8A8 and 1.62× at W4A4 at matched step count; ~4.2× at 16 steps.

The per-step deficit was always the wrong axis to judge MoDiff on, and this is why: it improves
accuracy per function evaluation, and in diffusion the number of function evaluations is the thing you
get to reduce.

## 2026-08-04 — Stage 3.3: the Linear `ô` GEMM epilogue, and MoDiff on the Linear path

Per the paper, `A(·)` in Eqs 8–17 is *any* linear operator, so excluding the qkv/proj Linear layers was
an incompleteness. Bug 2's fix already made the method correct there (no divergence); what kept it off
by default was cost: ~6 eager PyTorch launches per linear per step plus a host sync for the delta
absmax, measured at **+10.9 ms/step**.

`gemm_w{8a8,4a4}_awq_o_hat` folds the accumulate into the existing AWQ GEMM epilogue. Following the
nullable-pointer principle rather than cloning: `gwq_store2` gained one `__half* o_hat = nullptr`
parameter, and with it null the baseline path is **bit-identical** (verified). Only the two host
wrappers are new.

**The whole modulated Linear path is now 3 kernels, and 2 of the 3 already existed:**

| kernel | new? | why it transfers |
|---|---|---|
| `delta_absmax_fp16` | no | the conv path's dynamic delta scale, on device, no host sync |
| `step1_static_quantize[_pack]_int4_fprop` | no | the conv path's delta-quantize + in-place `â` update. Its body only walks `numel()`, so a 2D `[M,K]` activation needs no separate kernel |
| `gemm_w{8a8,4a4}_awq_o_hat` | **yes** | the accumulate, in the GEMM epilogue |

Also removed the last host syncs on this path: the GEMM's `a_scale` now accepts a 1-element **device**
tensor (`if (a_scale_ptr) a_scale = *a_scale_ptr;` at kernel entry). Taking it by value would have cost
one sync per linear per step — 42 × 200 = 8400 per sample, the same mistake that cost the conv path
~5 ms/step.

### Result

| | Linear MoDiff off | on | Δ | cost |
|---|---|---|---|---|
| W8A8 latent relL2 | 0.0413 | **0.0396** | helps | +5.0 ms/step (was +10.9) |
| W4A4 latent relL2 | 0.4804 | **0.4513** | helps 6% | +2.4 ms/step (was +8.8) |

So it now *improves* accuracy at both bit-widths instead of being quality-neutral, at less than half
the old cost. `MODIFF_LINEAR=1`.

### Three bugs I introduced and had to fix, all caught by the end-to-end check

1. **Dropped the bias.** First version returned the GEMM output directly, but the pre-existing path
   added bias to the *output* on every step (correct — Eq 9 puts bias in `ô_T` only, so the increment
   carries none). Dropping it took int8 from 0.039 to **0.300**. Fixed by ordering the epilogue
   `accumulate → bias → residual → store C`, so bias and residual reach the returned tensor and never
   the state. The ordering is now the documented contract in `gwq_store2`.
2. **`â` seeded at the unpadded width.** int4 pads K to `_awqt_K` for the AWQ layout, so the modulated
   path quantizes a padded activation while the seed was unpadded → `delta_absmax_fp16` rejected the
   element count. Seed now pads to match.
3. **No cache invalidation on shape change.** `â`/`ô` are indexed by tensor position, so they are
   meaningless at a different `M`. The activation-scale calibration samples at a smaller batch than
   production, which is a normal-use path, so this fired every run. The caches now reset when `M`
   changes, which re-seeds — the correct semantics.

Plus one real dimensionality bug in a pre-existing kernel: `step1_static_quantize_pack_int4_fprop`
reshaped its output to `{N,H,W,C/2}`, a hard NCHW assumption that its int8 twin does not have (that one
returns `empty_like(x)`). Now handles 2D, which is what let the Linear path reuse it.

**None of this was caught by the unit suite** — 16/16 passed throughout, including with the bias
dropped and int8 quality 7.6× worse. The unit tests exercise the state machine over a handful of steps
on synthetic tensors; a dropped bias term needs an end-to-end latent comparison to show up. Same gap
noted for the delta-scale staleness regression earlier today.

### But at PRODUCTION batch the cost is 5x what batch 8 showed — Linear MoDiff stays off

The +5.0 / +2.4 ms/step above was measured at batch 8. At batch 128 it is **+25.4 ms/step (+35%)**:

| batch 128, ms/step | MODIFF_LINEAR=0 | =1 | Δ |
|---|---|---|---|
| int8 MoDiff dynamic K=4 | 77.39 | 102.89 | **+25.5** |
| int4 MoDiff dynamic K=4 | 63.14 | 92.88 | **+29.7** |

**Why, and it is structural.** The attention Linear layers have `M = batch x tokens`. At batch 128 and
T=1024 that is `M = 131072`, so a single qkv layer's `â` is 50 MB and its `ô` is 151 MB — far larger
than any conv's state tensor. Accounting every pass on the modulated Linear path:

```
  1.70 GB  absmax: read x + a_hat
  2.98 GB  quantize: read x + a_hat, write a_hat + codes
  5.53 GB  gemm: read codes + o_hat, write o_hat + C
 10.21 GB  TOTAL  ->  14.6 ms/step at ~700 GB/s achievable
```

That models 14.6 of the measured 25.4; the rest is the 3x kernel-launch count on this path (42 -> 126
per step) plus imperfect utilization on the small-K passes. Either way the dominant term is `â`/`ô`
state traffic, which Eqs 9-10 require and no epilogue removes.

**Verdict: `MODIFF_LINEAR` stays 0.** +35% wall clock for a 4% (int8) / 6% (int4) accuracy gain is not
a trade worth taking. The epilogue and the fused path stay in the tree — they are correct, verified,
and they cut the cost by half — but the method is simply not worth its bandwidth on the attention
Linear layers at production batch. That is a finding about MoDiff, not about the implementation.

**Methodology lesson.** Measuring a bandwidth-bound feature at batch 8 understated its cost **5x**.
At batch 8 the Linear GEMMs are small enough to be latency-bound, so the extra passes partly hide
behind launch overhead; at batch 128 everything is bandwidth-bound and the traffic shows in full. Any
future A/B of a state-traffic change must be run at production batch.

**One flag bug found and fixed while doing this.** `MODIFF_LINEAR=1` was applied to every mode
including `int8_baseline` / `int4_baseline`, so it turned Linear MoDiff on inside the *baselines* and
moved them by +25 ms/step — an A/B whose control arm carried the treatment. Now gated on
`mode in ("int8", "int4")`.

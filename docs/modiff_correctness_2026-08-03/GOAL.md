# Goal: make MoDiff the paper's method, measurably, and leave the tree honest

**Owner** agent · **Written** 2026-08-03 · **Paper** ICML 2025 *Modulated Diffusion* (arXiv 2506.22463)

## The goal, in one sentence

MoDiff in this repo should run the algorithm the paper specifies, with the quantizer step derived
from the temporal delta (Theorem 4.3), be measurable against a tested metric, and report its speed
honestly — with the dead kernel surface removed and the docs no longer claiming things that are not
true.

## Definition of done

A statement is "done" only when a command in this repo prints the evidence.

| # | Done when | Evidence |
|---|---|---|
| D1 | The delta quantizer uses a per-step delta-range scale on every modulated path | ✅ `calibrate_delta_scales.py` sets a table on 70/70 layers |
| D2 | The two MoDiff invariants are tested, with positive controls | ✅ `test_kernel_correctness.py` 16/16 |
| D3 | The activation quantizer is matched: `effective_code_utilisation ≈ Q`, not ≫Q | 🟡 `in_conv` 204→**124** (below Q=127), clipping 25/35→17/35. `out_conv` 554→521, still 35/35 — **cause not found**, five hypotheses eliminated |
| D4 | Stage 1's step-size gain is measured on all 70 layers, not 14 | ✅ **66/70 layers at ≥2×, median 12.5× (155× squared-error reduction)**, up from 14/70 and median 0.5× |
| D5 | Every `not modiff` fusion gate is either MoDiff-compatible or has a recorded reason | ⬜ Stage 3 |
| D6 | No exported kernel is unreachable in all modes without a recorded reason | ⬜ Stage 4 |
| D7 | No doc claims a number this tree cannot produce | ⬜ Stage 5 |
| D8 | MoDiff's wall-clock is reported against a named baseline commit, whatever it says | ⬜ |

## The premise I have to correct

The request assumed "a fully done baseline". It is not. Three things found while doing Stages 0–2:

1. The shipped `integration/calibration/*.pt` do not describe this tree's model — the stub checkpoint
   is re-randomised per process, so a saved scale belongs to a different network. Measured: ~50% of
   *baseline* activation quantizations clip.
2. `_calibrate_int8` observes only 5 DDIM steps at batch 2 while production runs 20–200 at batch
   4–128. Effective code utilisation 246–580 against a ceiling of 127 — the **baseline** clips too.
3. `docs/MEASUREMENT_REPORT_2026-08-01.md`'s headline rows are labelled INT8/INT4 but are the
   MoDiff-**disabled** baselines, i.e. per the paper's Table 2 the regime where MoDiff adds nothing.

So D3 and D7 are baseline defects, not MoDiff work. They come first because every other number
depends on them.

## The constraint that set the schedule — and turned out to be self-inflicted

I first measured a single-file rebuild at **>10 minutes** and scheduled around it. The build log then
showed why:

```
UserWarning: Attempted to use ninja as the BuildExtension backend but we could not
find ninja.. Falling back to using the slow distutils backend.
```

**`ninja` was not installed**, so every rebuild compiled the 14 translation units *serially* on a
96-core box. `pip install ninja` fixes it. The >10-minute figure was not intrinsic to CUTLASS; it was
a missing build dependency, and it is worth adding to `requirements.txt`.

That materially cheapens Tier B and Stage 3: the edit→compile→test loop is the thing that made four
new kernels look expensive. Re-measure before believing any scheduling claim built on the old number.

This splits the remaining work cleanly, and I will do it in this order:

### Tier A — no rebuild required (do now, fully)
- **#10 / D3**: make `_calibrate_int8`'s horizon and batch follow the runner. Python only.
- **D4**: re-run the delta calibration on the fixed scales; report the gain across all 70.
- **Stage 3.5**: remove the remaining per-step `.item()` host syncs in `wxax_linear.py`.
- **Stage 4, Python half**: repoint the three `hasattr` sentinels; delete dead Python
  (`_packed_ref_vt*`, `convert_to_int{8,4}_baseline`, the duplicated `FullPipelineInt{8,4}Wrapper`,
  the four unread `HAS_*` flags).
- **Stage 5 / D7**: correct the docs.
- Re-run every test suite after each step.

### Tier B — needs the build loop (prepare, do not half-land)
- **Stage 4, CUDA half**: remove the 18 high-confidence symbols (3 unreferenced + 12 env-gated + 1
  sentinel + 2 string-only) from `pybind.cpp`, `modiff_kernels_api.h`, the `.cu` bodies, and
  `awq_w8a8_gemm_cuda.cu` from `setup.py`. One rebuild validates all of them together.
- **Stage 3.1–3.4**: the four new kernels. Each needs its own cycle plus a Stage 6 test.

Tier B gets a written, reviewed patch and a single batched rebuild rather than N interleaved ones.
I will not leave a partially-deleted symbol table behind — that breaks the build for everyone.

## Standing rules for myself

- Any metric goes through `effective_code_utilisation` / a test, never ad-hoc instrumentation. I got
  the same measurement wrong twice that way (once by omitting `smooth_inv`, once by reading int8
  codes as activations).
- Every new test carries a positive control; a test that cannot fail does not count as evidence.
- Report wall-clock against a named baseline commit even when MoDiff loses. The bandwidth cost of the
  â/ô caches is intrinsic and the honest framing is quality-at-A4, not speed parity.
- No FID or latent-level claim from this tree. Structurally vacuous.

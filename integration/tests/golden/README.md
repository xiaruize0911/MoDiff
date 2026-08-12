# Golden references

`group_norm_silu_res32.pt`, `int8_conv_res32_3x3.pt`, `int4_conv_res32_3x3.pt`,
`int8_linear_4096.pt` are kernel-level references on synthetic tensors. They are sound: they do
not go through the UNet, so nothing below applies to them.

## `int4_conv_res32_3x3.pt` — refreshed 2026-08-12, and why

Captured 2026-07-27 under the old per-channel **absmax** int4 weight scale. `82af5bc` (2026-08-05)
switched `_int4_weight_scale` to a per-channel MSE clip search — an intentional, measured change
(weight reconstruction error 0.1825 → 0.1254 median) — and left the golden, so the gate read
`FAIL golden rel_err=8.97e-02` at defaults for a week. Its own commit message said so.

Attributed without a bisect: the tree ships `MODIFF_INT4_WSCALE=absmax` as a revert switch, and
under it the old golden matched **bit-exactly**, which is the whole proof that nothing regressed.

    MODIFF_INT4_WSCALE=absmax python integration/tests/test_kernel_correctness.py   # was ALL PASS

Refreshed against the shipped MSE rule (md5 `767a197d…` → `aa3d09f4…`); the absmax original is kept
at `docs/static_qdiff_2026-08-12/data/int4_conv_golden_absmax_2026-07-27.pt`. A red gate detects
nothing, so leaving it red had a real cost: any *new* int4 conv regression in that week was invisible.

## Note: a missing golden currently passes

`check_golden` creates a golden when the file is absent and returns a non-FAIL string, and these
`.pt` files are gitignored (`.gitignore:6`). On a fresh clone the golden tests therefore pass
vacuously on the first run. That is the same shape of hole as the `e2e_*_vacuous` files below, one
level up. Two ways to close it: commit the goldens behind a `.gitignore` negation (~1.5 MB, the
precedent being `!docs/**/plots/*.png`), or make a missing golden FAIL with a "seed it with
UPDATE_GOLDEN=1" message. Neither taken: both change every developer's first run.

## `e2e_*_vacuous.pt` — do not use as references

The five `e2e_*_s50_b4_vacuous.pt` files were captured on 2026-07-27 by
`e2e_output_check.py --capture`, before it was noticed that the check could not fail.

`UNetModel.out[-1]` is a `zero_module` (`ldm/modules/diffusionmodules/openaimodel.py:745`) and
`models/ldm/lsun_churches256/model.ckpt` in this tree is an 856-byte stub whose `state_dict` has
0 entries, loaded with `strict=False`. That layer therefore stayed zero and the UNet predicted
identically zero for every input, so the sampled latent depended only on the initial noise and
the DDIM schedule — not on anything inside the network.

These files are the proof. All five are **bit-identical** to each other:

```
e2e_fp16_s50_b4          absmax 141.61083984375
e2e_int8_s50_b4          identical to fp16
e2e_int8_baseline_s50_b4 identical to fp16
e2e_int4_s50_b4          identical to fp16
e2e_int4_baseline_s50_b4 identical to fp16
```

FP16 and INT4 agreeing to the last bit is not a fidelity result; it is what happens when the
quantity being compared does not depend on the thing under test. Any `--compare` against these
passed unconditionally.

They are kept, renamed rather than deleted, only as evidence. `e2e_output_check.py` now writes
its references without the `_vacuous` suffix and refuses to run unless the UNet output is
observable, so the two sets can never be compared against each other.

Background: `docs/gn_qkv_fusion_2026-08-03/FINDINGS.md` section 5.

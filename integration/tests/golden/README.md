# Golden references

`group_norm_silu_res32.pt`, `int8_conv_res32_3x3.pt`, `int4_conv_res32_3x3.pt`,
`int8_linear_4096.pt` are kernel-level references on synthetic tensors. They are sound: they do
not go through the UNet, so nothing below applies to them.

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

# P3: the double-wrapped conv bug cost 114 MiB, not 1014.6 MiB

**Measured, two independent ways that agree to 0.7%** —
[`orphan_wrapper_memory.py`](scripts/orphan_wrapper_memory.py),
[`data/orphan_wrapper_memory.json`](data/orphan_wrapper_memory.json):

| | MiB |
|---|--:|
| allocator delta (build twice, dedup on vs defeated) | **114.33** |
| orphan tensors that do NOT alias a live storage | **113.59** |
| orphan tensors, counting every reference | 1014.71 |
| — of which aliases a live tensor | 901.12 |

**The tree carried two figures for this bug and they differed by 8.9×.**
`convert_model_to_optimized_int4`'s docstring said *"a 1014.6 MiB leak … 37% of the model's memory"*;
`3d13cf9`'s commit message said *"~113 MiB"*. Both reproduce exactly, because they measure different
things:

`FusedResBlock` aliases one `nn.Conv2d` under two attributes, so the naive walk built **140** wrappers
where **70** was right — confirmed: 70 deduped, 140 naive, 70 orphans identified structurally. But both
wrappers wrap **one** conv, so its fp16 weight is **one allocation with two references**. 901 of the
1014.7 MiB is that aliasing: memory that was already there and was never allocated twice. What the
orphans actually allocated for themselves — packed int4 weights and their own buffers — is 114 MiB.

So **~113 MiB was right**, the docstring overstated by 8.9×, and the docstring is now corrected to
carry the measured number and the reason the larger one is wrong.

## How it was measured without putting the bug back

The walk takes `_memo` as a parameter, so passing a `dict` subclass whose `__contains__` always returns
`False` reproduces the pre-`3d13cf9` behaviour exactly. Editing `int4_optimized.py` to add a debug flag
would have meant restoring a bug in the shipped tree in order to measure it.

Two deep copies of the fp16 UNet are made **before** either conversion, so neither conversion sees the
other's wrappers and the copies' own cost is excluded from both deltas by construction.

## Why the distinction matters beyond this bug

"Sum the tensors each module references" is the obvious way to attribute memory and it over-counts
every time modules share storage — which quantization wrappers do by design (`_orig_weight`, the
pre-fold bias, SmoothQuant buffers). The number a user feels is the allocator delta. Anything else needs
a `data_ptr()` check before it can be called a cost.

# P4: the activation fake-quant harness is retired. It failed a third time, and this time we know the answer

**`zp_headroom.py` is not a decision instrument and cannot be repaired by the fixes already tried.**
Fix #2 has now been decided by real kernels, so for the first time the harness can be scored against a
**known** answer. It fails:

| | says | truth |
|---|--:|--:|
| symmetric clip optimum | **6.7** | 4.5 (real-kernel sweep) |
| zero point worth | **1.28×** | 1.06× on activation reconstruction; **negative** end to end (+82% PTQ, +204% MoDiff) |
| its own verdict branch | *"WORTH IT: implement fix #2"* | do not implement |

Re-run 2026-08-13 with both clip fixes and 4-bit weights in place
([`logs/zp_headroom_revalidate.log`](logs/zp_headroom_revalidate.log)):

```
sym  ratio 1     1.1086     asym ratio 1     0.5895
sym  ratio 3     0.8571     asym ratio 3     0.3643   <- its best asym
sym  ratio 4.5   0.7067     asym ratio 4.5   0.3920
sym  ratio 6.7   0.4655     asym ratio 6.7   0.4466   <- its best sym
self-check: symmetric optimum at 6.7, real kernels say 4.5 -> DISAGREES, verdict NOT usable
```

That is the **third** failure, after 2.7× on fix #1 (predicted 0.1147, kernels delivered 0.3099) and
the earlier ordering failure it was already patched for. And it is the worst of the three, because the
number it produced (1.28×) sits on the far side of its own 1.15× decision bar from the truth: had the
self-check not been there, this harness would have authorised 15 CUDA entry points of work for a lever
that measures *negative*.

**The self-check is the part that worked, and it is the part worth keeping.** It is the reason no
decision was ever made on 1.28×.

## Why it cannot be fixed by more of the same

Both previous repairs were of the same kind — make the emulation more faithful (quantize the weights
first; collect ranges on the quantized-weight model). The remaining error is not faithfulness of the
*model*, it is that the harness emulates a quantization **grid** in PyTorch while the thing it predicts
is a fused CUDA kernel whose rounding, clamping and reduction order it does not reproduce. At 4 bits
there are 15 codes, so a half-code disagreement is ~7% of the range, and it lands differently per
layer. Nothing short of running the kernel resolves that — which is exactly what
`docs/paper_repro_2026-08-12/FINDINGS.md` concluded when it said *"Deciding fix #2 requires
implementing it."*

## What replaces it, and its validated scope

[`zp_activation_error.py`](scripts/zp_activation_error.py) measures the **reconstruction error of the
captured `silu(gn(x))` tensors** — no kernel, no conv, no padding, no sampler, so there is nothing to
emulate incorrectly:

| | best symmetric | best asymmetric | gain |
|---|--:|--:|--:|
| median over 70 convs | 0.2498 (r=3) | 0.2360 (r=3) | **1.06×**, 61/70 convs |

* **It got the fix #2 magnitude right** — 1.06×, under the 1.15× bar, agreeing in direction and roughly
  in size with the real-kernel outcome, where `zp_headroom.py` said 1.28× and pointed the other way.
* **It reproduces the documented distribution statistic**: asymmetry ratio 19.89× against the 19.91×
  recorded by `probe_int4_code_use.py`.
* **It still cannot pick a clip ratio.** It puts the symmetric optimum at r=3 where the real kernels
  put it at 4.5 — adjacent, and far better than 6.7, but wrong. So its scope is *magnitude* questions
  ("is there ≥1.15× headroom in this grid?"), **not** ratio selection.

An analogous instrument for the **weight** axis needs no such caveat and is fully trustworthy, because
weight-only quantization emulates nothing: dequantize the 4-bit weights to fp16 and the model runs the
exact values a deployed W4 kernel would multiply. That is what
[`weight_zp_end_to_end.py`](scripts/weight_zp_end_to_end.py) uses, and it is why fix #4 could be decided
end-to-end today with no kernel at all ([FINDINGS_WEIGHT_ZP.md](FINDINGS_WEIGHT_ZP.md)).

## Action taken

`zp_headroom.py` keeps its self-check and gains a header recording that it has now been wrong three
times, with this file as the reference. It is not deleted: it is the evidence that its own refusal
mechanism works, and a worked example of an instrument that is precise, reproducible, and wrong.

## The general rule

**An instrument that emulates the thing under test must be scored against the thing under test before
its numbers are used.** Both surviving instruments in this area — the activation-reconstruction one and
the weight-only one — earn their trust by reproducing an independently committed number (19.91×,
0.1506, 0.2728). `zp_headroom.py` never reproduced one.

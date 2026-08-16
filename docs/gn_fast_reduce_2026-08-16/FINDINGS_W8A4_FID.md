# W8A4 + MoDiff: the mechanism reproduces, the headline claim does not

**A40, LSUN-churches LDM, real 2.7 GB checkpoint. 10,000 decoded 256² samples per mode, DDIM 50,
`pytorch_fid` 2048-dim, against the same 10k real reference every committed FID row uses.** Same seed
sequence, same `*_realckpt` calibration, one warm-up run discarded per mode — i.e. the protocol of
[fid_2026-08-05](../fid_2026-08-05/FINDINGS.md), so this row drops straight into that table.

This closes [OPEN_ITEMS](../OPEN_ITEMS.md) B2: *"FID for W8A4 + MoDiff — the one row directly comparable
to the paper's table."*

| mode | FID vs real (10k) | × fp16 |
|---|--:|--:|
| fp16 | 7.803 | 1.00 |
| W8A8 PTQ | 16.366 | 2.10 |
| **W8A8 MoDiff** | **7.802** | **1.00** |
| **W8A4 PTQ** | **311.474** | 39.92 |
| **W8A4 MoDiff** | **35.302** | **4.52** |
| W4A4 PTQ | 277.963 | 35.62 |
| W4A4 MoDiff | 200.139 | 25.65 |

## 1. The mechanism reproduces, and strongly

**311.47 → 35.30 is an 8.8× improvement, recovering 90.9% of the distance to fp16.** MoDiff does at A4
exactly what it is supposed to: an activation-only temporal method turns an unusable 4-bit-activation
model into a recognisable one. That is not a small effect and it is not ambiguous.

It also lands where the paper says the value is. At A8 MoDiff buys nothing on our numbers either —
16.366 → 7.802 is a large *relative* gain, but the paper's own A8 row (Q-Diff 4.24 vs +MoDiff 3.85) is
nearly flat, and its stated position is that "its value is at A4 (355.85 → 3.97) and A3 (367.51 → 5.40)".
The unmodulated end reproduces too: our 311.47 against the paper's 355.85, same order, same verdict.

## 2. The headline claim does not reproduce

The paper's A4 claim is not "MoDiff helps" — it is that **dropping a bit becomes free**: 3.97 at A4+MoDiff
is *better* than 4.24 at A8 Q-Diff, a ratio of **0.94**. That comparison is internal to the paper, so it
survives any protocol difference.

The same internal comparison on our side:

| | A4+MoDiff | own A8 baseline | ratio |
|---|--:|--:|--:|
| paper (Table 2, LSUN-Church, W8) | 3.97 | 4.24 | **0.94** |
| **ours (10k)** | **35.30** | **16.37** | **2.16** |

**So on our tree dropping the activation bit costs 2.16×, where the paper reports it saving 6%.** Both
ratios are within-protocol, so the 10k-vs-50k bias below cannot explain the gap — a systematic upward bias
on both numerator and denominator largely cancels in a ratio.

## 3. Why 35.30 must not be compared to 3.97 directly

It is tempting and it is wrong. Our FID is at **10k samples**; the standard protocol, and the paper's, is
**50k**, and 10k is biased upward — stated in
[FID_50K_ESTIMATE](../bench_report_2026-08-13_postzp/FID_50K_ESTIMATE.md) and visible in our own table:
our fp16 reads **7.803** where LDM's published LSUN-churches figure is ~4. The paper's A8 baseline (4.24)
is *better than our fp16*, which is impossible on a shared protocol and is the tell.

So `35.30 / 3.97 = 8.9×` is a unit error, of exactly the kind this session spent the morning removing
(A16: two documents using `ms/sample` for quantities 3× apart). The defensible statements are the
within-protocol ratios in §2, and the honest headline is **2.16 against 0.94**, not 8.9.

## 4. What this means for the list

- **B2 closes.** The row exists, in the committed protocol, and it is a partial reproduction: mechanism
  yes, headline claim no.
- **It sharpens B5** (a weight-side method for W4A4). W8A4 isolates the *activation* axis — weights stay
  at 8 bits — and MoDiff still leaves 4.52× fp16. So the 4-bit activation grid is not fully solved either,
  and B5's framing ("at W4A4 the dominant error is in the weights") is right about W4A4 but should not be
  read as "activations are handled". At W8A4, with weights at 8 bits and MoDiff on, there is still a 4.5×
  gap to close.
- **It gives C7 its missing gate.** C7 wants `MODIFF_WARMUP_STEPS` 5 → 1 (26.5% better latent relL2 at
  W4A4, resolved) but could not move the default without an FID at 1 round. This file establishes the
  W8A4 row at 5 rounds, which is the comparison point that run would need.

## 5. Reproduce

```bash
python docs/fid_2026-08-05/scripts/generate_fid_samples.py --n 10000 --batch 128 --steps 50 --modes w8a4_baseline,w8a4_l0 --linear 0
```
```bash
python docs/fid_2026-08-05/scripts/compute_fid.py --modes w8a4_baseline,w8a4_modiff_l0 --out docs/gn_fast_reduce_2026-08-16/data/fid_w8a4.json
```

Generation is ~25 min for the pair on an idle A40; the FID pass is ~4 min. `pytorch_fid` had to be
installed — the fourth dependency this container lost since 2026-08-13, after matplotlib,
markdown/weasyprint (+libpango), and the omegaconf/einops/pytorch-lightning/torchmetrics set. It failed
on the *last* step of a 25-minute pipeline, which is the argument for a dependency pre-flight before any
multi-stage measurement rather than after it.

# What a 50k-sample FID costs, per mode

**Measured, not modelled.** The generation rate below is the real pipeline
(`docs/fid_2026-08-05/scripts/generate_fid_samples.py`) run at 1280 images per mode on an idle A40,
DDIM 50, batch 128, PNGs written to `/workspace` — so it already includes VAE decode, uint8 conversion,
PNG encoding, the filesystem, the per-batch MoDiff reset and all per-batch overhead. Scaling to 50k is
then a multiplication, because the loop is uniform after the first batch.

## Generation, 50,000 images

| mode | measured ms/img | 50k generation | of which UNet | of which decode+PNG+rest |
|---|--:|--:|--:|--:|
| fp16 | 123.4 | **1.71 h** | 39.8 (32%) | 83.6 (68%) |
| W8A8 PTQ | 104.7 | **1.45 h** | 27.6 (26%) | 77.1 (74%) |
| W8A8 MoDiff¹ | 118.8 | **1.65 h** | 33.5² (28%) | ~85 (72%) |
| W4A4 PTQ | 103.9 | **1.44 h** | 22.4 (22%) | 81.5 (78%) |
| W4A4 MoDiff¹ | 107.8 | **1.50 h** | 27.5² (26%) | ~80 (74%) |

¹ the FID script runs its MoDiff arms with `delta=dynamic`; the shipped default is `static`, which is
slightly faster. ² includes the per-batch warm-up: the script calls `reset(model)` before **every**
batch, so every batch pays `_forward_first_step` (5.2 ms/img at W8A8, 4.8 at W4A4 — 663/615 ms per batch
of 128 spread over its images).

The UNet column is `steady ms/step × 50 / 128` from
[`data/warmup_cost.json`](data/warmup_cost.json); the remainder is the measured total minus it, and it
lands at a consistent 77–85 ms/img across five independent runs, which is the check that the decomposition
is real rather than fitted.

**The headline is the last column: 68–78% of a FID run is not the UNet.** So the 1.78× kernel speedup
buys only 1.19× on total FID wall time (fp16 1.71 h → W4A4 PTQ 1.44 h).

## The rest of the bill

| item | cost | notes |
|---|--:|---|
| generation, one mode | 1.44–1.71 h | above |
| Inception + statistics, generated | ~3–4 min | pytorch_fid, batch 64, ~256 img/s from the 10k run's log |
| Inception + statistics, real | ~3–4 min | once, if cached across modes |
| **building the 50k real reference** | **~0.6 h, one-off** | only 10,000 real images exist today (`/workspace/fid/real`); 40k more must come out of the 4.9 GB LMDB, center-cropped, resized to 256², saved |
| disk | **~7.5 GB per mode** | 50k × ~150 KiB; five modes ≈ 37.5 GB plus 7.5 GB real |

**One mode, end to end, from today's state: ~1.5–1.8 h plus a one-off ~0.6 h for the real reference.
All five modes: ~8 h.**

## The biggest lever is not the precision

PNG writing was measured on both filesystems, same images:

| target | ms/img |
|---|--:|
| `/workspace` mount | **52.6** |
| local scratch (`/tmp/...`) | **12.7** |

4.1× apart, and PNG is the largest single item in the non-UNet 80 ms/img. **Writing to local disk instead
of the workspace mount would take fp16 from 123.4 to ~83 ms/img (1.49×) and W4A4 PTQ from 103.9 to ~65
(1.60×)** — a bigger win on FID wall time than switching fp16→W4A4, and it stacks with it. Better still,
`pytorch_fid` can consume tensors instead of a directory, which removes the 52.6 ms/img encode entirely
and the 7.5 GB per mode with it.

So the ranking of what to fix, if 50k FID runs are going to be routine:

1. **Don't write PNGs to `/workspace`** — ~40 ms/img, free to change.
2. **Skip PNG altogether** and feed Inception from memory — the same 52.6 ms/img, plus the disk.
3. **Then** precision matters: W4A4 PTQ saves a further ~20 ms/img over fp16.

## If the target is the paper's protocol

The LDM paper's LSUN-Churches 4.02 is 50k at **DDIM 200**, not 50. Only the UNet term scales:

| mode | ms/img @ 200 | 50k generation |
|---|--:|--:|
| fp16 | 242.6 | **3.37 h** |
| W8A8 PTQ | 187.5 | **2.60 h** |
| W4A4 PTQ | 171.3 | **2.38 h** |

At 200 steps the UNet is 52–66% of the run, so the precision choice matters more there than at 50 — and
the non-UNet fixes above still apply on top.

## What this does not cover

* Both `fid_*` directories on this machine hold 10k, and the published FID numbers in
  `docs/fid_2026-08-05/FINDINGS.md` are 10k at DDIM 50. Going to 50k changes the absolute values (FID is
  biased upward at small N), so a 50k run cannot be compared against those; it can only be compared
  against other 50k runs.
* The measured rates are single-process. Generation is GPU-bound and PNG encoding is CPU-bound, so
  overlapping them (a writer thread or pool) would recover part of the 52.6 ms/img without changing the
  filesystem. Not measured.

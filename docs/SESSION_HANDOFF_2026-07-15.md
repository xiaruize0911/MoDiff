# Session handoff — custom CUTLASS fused GroupNorm→qkv attention kernel + comprehensive benchmark (2026-07-15)

Follow-on to `SESSION_HANDOFF_2026-07-14b.md`. Focus this session: the **diffusion UNet** (LSUN churches
LDM), specifically the attention block. Prior sessions were ResNet-50 + diffusion conv/quant work.

**All work is COMMITTED AND PUSHED to `origin/main`** (HEAD `dc88997`). An ed25519 SSH key was created at
`~/.ssh/id_ed25519` and registered on the GitHub account this session, so pushes work from this env now.

### Environment / build
- `PYTHONPATH=/workspace/MoDiff/src/taming-transformers:$PYTHONPATH  CUTLASS_PATH=/workspace/cutlass`
- Rebuild the CUDA ext after any `csrc/` change: `python setup.py build_ext --inplace` (distutils, **no
  ninja** → recompiles ALL sources, ~15 min; run in background and poll for the `.so` relink).
- A40 (sm_86) **idles at 210 MHz, boosts to 1740; clocks CANNOT be locked here** (perm denied even as root).
  Benchmarks MUST use heavy sustained warmup or numbers are depressed 2–5 ms/step (see below).
- nsys/ncu unavailable; pynvml not installed (don't call `torch.cuda.clock_rate()`).

---

## What shipped this session

### 1. Custom CUTLASS fused GroupNorm→qkv kernel — SHIPPED, on by default
Files: `csrc/kernels/fused_gn_qkv.cu`, `csrc/kernels/implicit_gemm_fusion_persample.h` (a copy of CUTLASS's
`implicit_gemm_convolution_fusion.h` with one change — see below), wired into
`integration/fused_ops/token_major_attention.py` (`_ensure_fused` / forward), pybind in
`csrc/pybind.cpp` + `csrc/modiff_kernels_api.h`, gate test `test_fused_gn_qkv` in
`integration/tests/test_kernel_correctness.py` (**ALL PASS**, rel 0.0069).

**Mechanism:** the qkv Linear is a 1×1 conv, so it rides CUTLASS fprop mainloop fusion (per-channel
scale+bias applied to activations *inside* the mainloop). GroupNorm's γ folds into the conv weight, GN's β
+ qkv bias into the epilogue bias. Two custom pieces:
- **Per-sample scale iterator** (`ImplicitGemmConvolutionFusionPerSample`): GN stats are per-(sample,group),
  but the stock fusion shares one `[1,C]` scale vector across the batch. The kernel offsets the scale/bias
  pointer by `sample*C` per threadblock. **Valid only when tokens T=H·W is a multiple of the tile kM=128**
  (C192/T1024 and C384/T256 qualify; smaller blocks fall back to GroupNorm+cuBLAS). This is the core edit.
- **ReLU absorption:** the stock fusion does scale+bias+**ReLU**; GN→qkv must not ReLU the sign-bearing
  normalized activations. A constant `SHIFT=16` in the bias keeps the pre-ReLU value ≥0 (normalized acts are
  ~unit variance), and the induced constant `SHIFT·Σ_c Wf` is subtracted back in the (static) epilogue bias.
- Stats: a two-pass CUDA kernel (coalesced per-channel reduce + token-tiled atomics for occupancy).

**Result (with flash attention):** +1.4% end-to-end (correct, rel 0.0016). Per block 1.10× (C192/T1024,
capped by the small-K CUTLASS-vs-cuBLAS gap) / 1.24× (C384/T256). On by default; kill-switch
`MODIFF_FUSE_GN_QKV=0`.

**Dead-end first (documented):** a Triton version of the same fusion was **−11% end-to-end** — Triton can't
match cuBLAS on the small-K (K=192) qkv shape (best Triton GEMM 1.41× off cuBLAS; gap > the whole GN cost).
Feasibility numbers and the CUTLASS win are in `docs/profiling_report_2026-07-14/DIFFUSION_UNET_BOTTLENECK.md`.

### 2. Comprehensive benchmark — `docs/comprehensive_benchmark_2026-07-15/`
Kernel + pipeline **speed / IO / profile** across all 5 modes (fp16, int8 base/modiff, int4 base/modiff),
with plots (7 PNGs, force-added past the `*.png` gitignore), CSVs (`data/`), and reproducible scripts
(`scripts/pipeline.py`, `kernel.py`, `mkplots.py`). REPORT.md ties it together.

### 3. ⚠️ Flash attention REMOVED (current default state)
Per request, `TokenMajorAttentionBlock` now forces the **math (non-flash) SDPA backend**
(`sdpa_kernel(SDPBackend.MATH)`) at both call sites. **This is the current shipping default** and is a
deliberate **~1.6–1.85× slowdown + ~0.9 GiB more memory** (math materializes the full [N,heads,T,T] scores;
SDPA math is 9.1× slower than flash on the T=1024 block). To restore flash, revert the `_SDPA_MATH_CTX`
wrapping in `token_major_attention.py`.

---

## Current measured numbers (A40, batch 32, churches LDM, MATH attention — current default)
Heavy-warmup, low-variance (<0.2% stdev), ms/DDIM step:
```
mode          wall    GPU-busy   vs fp16
fp16          55.93   54.45      1.00x
int8 base     49.99   48.75      1.12x
int8 modiff   58.03   56.63      0.96x
int4 base     47.64   46.65      1.17x   <- fastest
int4 modiff   53.23   51.89      1.05x
```
**With FLASH attention (before the removal — for reference / if reverted):** fp16 32.1, int8 base 27.2,
int4 base 25.8 (1.25×), int8 modiff 36.0, int4 modiff 33.4.

Key facts (dtype-invariant across the Amdahl story):
- **int4 base is the fastest mode** (int4 wins the large-channel 3×3 convs; aggregate conv 7.4 vs int8 9.5 ms).
  Loses only on ≤192-channel / 1×1 convs (weight-unpack overhead) — net win.
- **MoDiff is slower than fp16** — temporal delta machinery (`fprop_o_hat` + `step1` delta-quantize) adds
  ~2.6–3.0 ms + carries a fixed **+634 MiB** `a_hat`/`o_hat` cache. Accuracy mechanism, not speed.
- **Quantization only moves the conv bucket**; GroupNorm (~5.5 ms) + attention are dtype-invariant. With
  flash removed, attention is ~42% of the step, so the quantization speedup shrank (1.25→1.17× for int4 base).

---

## Benchmarking gotcha (cost us a re-run — don't repeat)
The first benchmark draft used only 2 warmup `sample()` calls → measured during clock ramp → **depressed
int4/MoDiff by 2–5 ms/step and inverted the int4-vs-int8 winner**. Fix: ≥6 s sustained warmup + 12
back-to-back timed runs, report median/min/stdev. Also reconciled the older
`DIFFUSION_UNET_BOTTLENECK.md` whose wall-clock row (int4 38.1 ms, "int4 doesn't pay off") was the same
under-warmed/nsys artifact — its GPU-sum numbers were already correct.

---

## Open items / next steps
1. **Decide the flash question.** Current default is math attention (~1.7× slower). If flash removal was for
   a downstream goal (quantizing the attention core — now plain cuBLAS GEMMs you can intercept, unlike the
   opaque flash kernel), that's the natural next task. Otherwise consider reverting to flash.
2. **fused GN→qkv is capped at 1.10× on the dominant C192/T1024 block** by the small-K CUTLASS-vs-cuBLAS gap
   (no kernel beats cuBLAS at K=192). The only lever left there is a bespoke matmul that beats cuBLAS at
   K=192 — not worth it. It does better (1.24×) on C384/T256.
3. **The real remaining bottleneck is GroupNorm + attention** (dtype-invariant, ~half the step) — SDPA on the
   O(T²) 32²/T=1024 block dominates. A faster attention (flash, or lower-res attention) is the biggest lever.
4. MoDiff on this UNet trades speed for temporal accuracy; MoDiff on ResNet diverges (prior session).

## Correctness gate
`python integration/tests/test_kernel_correctness.py` → **ALL PASS** (12 tests incl. `fused_gn_qkv`).

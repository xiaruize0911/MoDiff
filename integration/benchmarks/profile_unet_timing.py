#!/usr/bin/env python3
"""
profile_unet_timing.py – Comprehensive UNet operation timing profiler.

Measures non-overlapping CUDA time for each operation category using
CUDA-event forward hooks, then reports ms/step and % of step time.

Categories (mutually exclusive – no double-counting):
  attention   AttentionBlock total (GroupNorm + QKV proj + softmax + out_proj)
  quant_conv  CUTLASS INT8 or INT4 quantized convolutions
  fp_conv     Standard FP16 nn.Conv2d (input/output proj, skip connections)
  groupnorm   GroupNorm + FusedGroupNormSiLU (outside attention)
  silu        nn.SiLU activations (outside attention / groupnorm)
  linear      nn.Linear (timestep embedding projections)
  upsample    Upsample forward (bilinear + optional conv, full block)
  downsample  Downsample forward (strided conv or avg-pool, full block)
  other       Residual adds, DDIM ops, hook overhead, memory copies

Usage:
    cd /workspace/MoDiff
    python integration/benchmarks/profile_unet_timing.py
"""

import os, sys, time, warnings, json

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, os.getcwd())

import torch
import torch.nn as nn
from collections import defaultdict

warnings.filterwarnings("ignore")
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG     = "configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml"
CKPT       = "models/ldm/lsun_churches256/model.ckpt"
CALIB_INT8 = "integration/calibration/int8_calibration.pt"
CALIB_INT4 = "integration/calibration/int4_calibration.pt"
BATCH      = 42
WARMUP     = 5     # steps before profiling (not counted)
STEPS      = 15    # steps to time
SHAPE      = (4, 32, 32)
MODES      = ["fp16", "int8_baseline", "int4_baseline"]
OUT_JSON   = "integration/results/profile_ops.json"

# ── Optional module imports (handle missing extensions gracefully) ─────────────
try:
    from integration.kernels.int8_optimized import OptimizedInt8Conv2d
    HAS_INT8 = True
except ImportError:
    HAS_INT8 = False
    OptimizedInt8Conv2d = None

try:
    from integration.kernels.int4_optimized import OptimizedInt4Conv2d
    HAS_INT4 = True
except ImportError:
    HAS_INT4 = False
    OptimizedInt4Conv2d = None

try:
    from integration.fused_ops.fused_resblock import FusedGroupNormSiLU
    HAS_FUSED = True
except ImportError:
    HAS_FUSED = False
    FusedGroupNormSiLU = None

from ldm.modules.diffusionmodules.openaimodel import AttentionBlock, Upsample, Downsample


# ── CUDA-event timing hooks ────────────────────────────────────────────────────
class CUDATimer:
    """
    Attach forward hooks to UNet modules for non-overlapping CUDA timing.

    Hierarchy rule: AttentionBlock, Upsample, and Downsample are hooked at
    the block level.  Their internal sub-modules are excluded from independent
    hooks so every GPU cycle is counted exactly once.
    """

    def __init__(self):
        self._handles = []
        self.pairs = defaultdict(list)   # category -> [(start_event, end_event)]

    # ── public API ────────────────────────────────────────────────────────────
    def attach(self, unet: nn.Module):
        """Register timing hooks on all relevant modules in *unet*."""
        # Build exclusion set: children of block-level hooks must not be
        # independently timed to avoid double-counting.
        exclude: set = set()
        for m in unet.modules():
            if isinstance(m, (AttentionBlock, Upsample, Downsample)):
                for sub in m.modules():
                    if id(sub) != id(m):
                        exclude.add(id(sub))

        for m in unet.modules():
            mid = id(m)

            # ── Block-level hooks (capture all children implicitly) ───────────
            if isinstance(m, AttentionBlock):
                self._hook(m, "attention")
            elif isinstance(m, Upsample):
                self._hook(m, "upsample")
            elif isinstance(m, Downsample):
                self._hook(m, "downsample")

            elif mid in exclude:
                continue   # inside a block-level hook → skip

            # ── Leaf-level hooks ─────────────────────────────────────────────
            elif HAS_INT8 and isinstance(m, OptimizedInt8Conv2d):
                self._hook(m, "quant_conv")
            elif HAS_INT4 and isinstance(m, OptimizedInt4Conv2d):
                self._hook(m, "quant_conv")
            elif isinstance(m, nn.Conv2d):
                self._hook(m, "fp_conv")
            elif HAS_FUSED and isinstance(m, FusedGroupNormSiLU):
                self._hook(m, "groupnorm")
            elif isinstance(m, nn.GroupNorm):
                self._hook(m, "groupnorm")
            elif isinstance(m, nn.SiLU):
                self._hook(m, "silu")
            elif isinstance(m, nn.Linear):
                self._hook(m, "linear")

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self.pairs.clear()

    def summarize(self, wall_ms: float, steps: int) -> dict:
        """Synchronize GPU and compute per-category statistics."""
        torch.cuda.synchronize()
        result: dict = {}
        profiled_ms = 0.0

        for cat, ps in self.pairs.items():
            total = sum(s.elapsed_time(e) for s, e in ps)
            result[cat] = {
                "total_ms":    round(total, 4),
                "per_step_ms": round(total / steps, 4),
                "pct":         round(total / wall_ms * 100, 2),
                "calls":       len(ps),
            }
            profiled_ms += total

        other = max(0.0, wall_ms - profiled_ms)
        result["other"] = {
            "total_ms":    round(other, 4),
            "per_step_ms": round(other / steps, 4),
            "pct":         round(other / wall_ms * 100, 2),
            "calls":       0,
        }
        return result

    # ── private ───────────────────────────────────────────────────────────────
    def _hook(self, module: nn.Module, category: str):
        pairs = self.pairs

        def pre_hook(m, inp):
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            m.__prof_start__ = ev

        def post_hook(m, inp, out):
            if not hasattr(m, "__prof_start__"):
                return
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            pairs[category].append((m.__prof_start__, end))
            del m.__prof_start__

        h1 = module.register_forward_pre_hook(pre_hook)
        h2 = module.register_forward_hook(post_hook)
        self._handles += [h1, h2]


# ── Per-mode profiling run ─────────────────────────────────────────────────────
def profile_mode(mode: str) -> tuple:
    """Load model for *mode*, attach profiler, run, return (wall_ms, summary)."""
    from integration.benchmarks.benchmark_ldm import BenchmarkRunner

    calib = (CALIB_INT8 if "int8" in mode
             else CALIB_INT4 if "int4" in mode
             else None)

    runner = BenchmarkRunner(
        CONFIG, CKPT, "/tmp/ldm_profile",
        batch_size=BATCH, steps=STEPS, shape=SHAPE,
        calibration_path=calib,
    )
    model, sampler = runner._setup_model(mode)
    unet = model.model.diffusion_model

    # ── Warmup ────────────────────────────────────────────────────────────────
    print(f"  [warmup  {WARMUP} steps] ...", end="", flush=True)
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        sampler.sample(S=WARMUP, batch_size=BATCH, shape=SHAPE, eta=0.0, verbose=False)
    torch.cuda.synchronize()
    print(" done")

    # ── Attach CUDA-event hooks ────────────────────────────────────────────────
    timer = CUDATimer()
    timer.attach(unet)

    # ── Timed run ─────────────────────────────────────────────────────────────
    print(f"  [profile {STEPS} steps] ...", end="", flush=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        sampler.sample(S=STEPS, batch_size=BATCH, shape=SHAPE, eta=0.0, verbose=False)
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - t0) * 1000
    print(f" done  ({wall_ms:.0f} ms total, {wall_ms/STEPS:.3f} ms/step)")

    summary = timer.summarize(wall_ms, STEPS)
    timer.detach()

    del model, sampler
    torch.cuda.empty_cache()

    return wall_ms, summary


# ── Display helpers ────────────────────────────────────────────────────────────
ORDER = ["attention", "quant_conv", "fp_conv", "groupnorm",
         "silu", "linear", "upsample", "downsample", "other"]


def print_mode_table(mode: str, wall_ms: float, summary: dict):
    print(f"\n{'═'*70}")
    print(f"  {mode.upper():30s}  batch={BATCH}   {STEPS} timed steps")
    print(f"  wall = {wall_ms:.0f} ms   =  {wall_ms/STEPS:.3f} ms/step")
    print(f"{'═'*70}")
    print(f"  {'Category':<20}  {'ms/step':>9}  {'% step':>8}  {'calls/step':>11}")
    print(f"  {'─'*56}")
    for cat in ORDER:
        if cat not in summary:
            continue
        d = summary[cat]
        calls_ps = (d["calls"] / STEPS) if d["calls"] > 0 else 0
        calls_str = f"{calls_ps:.0f}" if calls_ps > 0 else "—"
        print(f"  {cat:<20}  {d['per_step_ms']:>9.3f}  {d['pct']:>8.1f}  {calls_str:>11}")
    print(f"  {'─'*56}")
    print(f"  {'TOTAL (wall)':<20}  {wall_ms/STEPS:>9.3f}  {'100.0':>8}  {'':>11}")


def print_comparison(all_results: dict):
    modes = list(all_results.keys())
    print(f"\n\n{'═'*80}")
    print(f"  CROSS-MODE COMPARISON — ms/step  (batch={BATCH}, {STEPS} timed steps)")
    print(f"{'═'*80}")
    hdr = f"  {'Category':<20}"
    for m in modes:
        hdr += f"  {m:>16}"
    print(hdr)
    print(f"  {'─'*70}")
    for cat in ORDER:
        row = f"  {cat:<20}"
        for m in modes:
            s = all_results[m]["summary"]
            row += f"  {s[cat]['per_step_ms']:>16.3f}" if cat in s else f"  {'---':>16}"
        print(row)
    print(f"  {'─'*70}")
    row = f"  {'TOTAL (wall)':<20}"
    for m in modes:
        row += f"  {all_results[m]['wall_ms']/STEPS:>16.3f}"
    print(row)
    print(f"{'═'*80}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Profiling: {MODES}  |  batch={BATCH}  |  {WARMUP} warmup + {STEPS} timed steps\n")

    all_results: dict = {}
    for mode in MODES:
        print(f"\n{'─'*70}")
        print(f"  MODE: {mode.upper()}")
        wall_ms, summary = profile_mode(mode)
        all_results[mode] = {"wall_ms": wall_ms, "summary": summary}
        print_mode_table(mode, wall_ms, summary)

    print_comparison(all_results)

    # Save JSON for report generation
    os.makedirs("integration/results", exist_ok=True)
    save = {
        mode: {
            "wall_ms":    data["wall_ms"],
            "ms_per_step": round(data["wall_ms"] / STEPS, 4),
            "steps":  STEPS,
            "batch":  BATCH,
            "summary": data["summary"],
        }
        for mode, data in all_results.items()
    }
    with open(OUT_JSON, "w") as f:
        json.dump(save, f, indent=2)
    print(f"\n✓ Saved JSON to {OUT_JSON}")

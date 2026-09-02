"""Regression smoke for the a_hat storage modes the B=32 work touched around.

The B=32 fast path added a template parameter to the GN vec2 kernel, a runtime branch
to the two step1 vec2 kernels and new device helpers shared with the upsample resnap,
so every other a_hat mode recompiled. This checks each still produces a sane latent
(finite, comparable norm, no rainbow blowup) against the fp16 a_hat arm.
"""
from __future__ import annotations
import os, sys
ROOT = "/workspace/MoDiff"
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers")]
os.environ.setdefault("MODIFF_DELTA_MODE", "static")
os.environ["MODIFF_LINEAR"] = "0"
os.environ["MODIFF_CACHE_SKIP_K"] = "1"
os.environ["MODIFF_REPLAY_K"] = "1"
os.environ["MODIFF_AHAT_REFRESH"] = "0"
os.environ["MODIFF_AHAT_BITS"] = "16"
os.environ["MODIFF_IMODE"] = "0"
os.environ["MODIFF_AHAT_BLOCK"] = "0"

from integration.utils.preflight import preflight, MODEL  # noqa: E402
preflight(*MODEL, what="smoke_ahat_modes.py")
import torch  # noqa: E402
import integration.benchmarks.benchmark_ldm as B  # noqa: E402

SHAPE, BATCH, STEPS, SEED = (4, 32, 32), 4, 20, 20260805

# (label, MODIFF_AHAT_BLOCK, MODIFF_AHAT_BITS, MODIFF_IMODE, relL2 ceiling)
# Ceilings are "what this scheme has been observed to produce", not "good":
#  - per-tensor held int8 is deliberately poor (FID 121, cache-schemes report).
#  - I-MoDiff is a different algorithm and its E2E latent is far from the fp16-a_hat
#    arm. Its kernel invariants pass (integration/tests/test_imode.py), and the B=32
#    work moved this number by 0.0003, so the ceiling is a regression tripwire only.
MODES = [
    ("fp16 a_hat (reference)", "0", "16", "0", 0.0),
    ("along-C B=32 int8", "32", "16", "0", 0.35),
    ("along-C B=16 int8", "16", "16", "0", 0.35),
    ("per-tensor int8 (held)", "0", "8", "0", 0.70),
    ("I-MoDiff int16", "0", "16", "1", 3.5),
]


def gen(model, sampler):
    B.reset_modiff_state_int8(model.model.diffusion_model)
    B._reset_wxax_modiff_safe(model)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        out = sampler.sample(S=STEPS, batch_size=BATCH, shape=SHAPE, eta=0.0, verbose=False)
    lat = out[0] if isinstance(out, (tuple, list)) else out
    return lat.detach().float().cpu()


def main():
    runner = B.BenchmarkRunner(
        config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
        ckpt_path="models/ldm/lsun_churches256/model.ckpt",
        output_dir="docs/ahat_blockwise_2026-09-01/tmp_smoke",
        batch_size=BATCH, steps=STEPS, shape=SHAPE,
        calibration_path=B._default_calibration_path("int8"),
        auto_delta_table=True)
    model, sampler = runner._setup_model("int8")
    ref, bad = None, []
    for label, blk, bits, imode, ceil in MODES:
        os.environ["MODIFF_AHAT_BLOCK"] = blk
        os.environ["MODIFF_AHAT_BITS"] = bits
        os.environ["MODIFF_IMODE"] = imode
        lat = gen(model, sampler)
        finite = bool(torch.isfinite(lat).all())
        if ref is None:
            ref, rel = lat.clone(), 0.0
        else:
            rel = float((lat - ref).norm() / ref.norm())
        # rainbow noise (the held-scale failure) showed up as relL2 > 2
        ok = finite and rel <= ceil
        print(f"  {label:24s} relL2 {rel:6.4f} (<= {ceil:.2f})  norm {lat.norm():8.2f}  "
              f"finite {finite}  {'ok' if ok else 'BAD'}", flush=True)
        if not ok:
            bad.append(label)
    print("ALL OK" if not bad else f"PROBLEMS: {bad}", flush=True)
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())

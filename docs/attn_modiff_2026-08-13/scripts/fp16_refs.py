"""One pinned set of fp16 reference latents, on disk, shared by every harness that grades against it.

WHY. relL2 is measured against an fp16 reference that each harness rebuilds for itself. Those
reconstructions are supposed to be identical -- same checkpoint, same seeds, same steps, same batch,
same sampler -- and mostly they are: static_vs_dynamic_ab, arm_position_effect and measure_path_aa all
return a BIT-IDENTICAL [0.3266, 0.2882, 0.3134] for the int4/static arm across separate processes.

But linear_modiff_w4a4_ab could not reproduce that, twice, from what is line-for-line the same
reference construction. The first attempt was 6.9% off (H.AUTO_DELTA_TABLE was assigned inside
measure() instead of before the references, so they were built with it False); fixing that left 1.1%,
still above the 0.6% floor. Four experiments went into establishing that the arms, the delta table, the
calibration file, the arm order and the measure() path are all identical, which leaves the references
and an unidentified source of fp16 nondeterminism -- most plausibly attention kernel selection varying
with GPU state, which changes reduction order and moves the reference at the 1e-3 level.

Rather than keep hunting it, this removes it as a variable. The references are built ONCE and cached,
so cross-harness comparisons are exact by construction rather than by hoping two reconstructions
agree. A harness that loads this file and still cannot reproduce a known arm has a problem in the ARM,
which is a much more useful thing to be told.

This does NOT paper over the nondeterminism: it confines it. If fp16 sampling really does vary between
processes, that is worth knowing on its own, and pinning the reference is what makes it measurable --
regenerate with --rebuild and compare, instead of discovering it through a 1% wobble in an unrelated
A/B. Recorded in FINDINGS as open.

Usage:
    import fp16_refs
    refs = fp16_refs.get(steps=50, batch=8, seeds=[1234, 20260805, 777])

    python docs/attn_modiff_2026-08-13/scripts/fp16_refs.py --rebuild      # force regeneration
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if os.getcwd() != ROOT:
    os.chdir(ROOT)
for p in (ROOT, os.path.join(ROOT, "src/taming-transformers"),
          os.path.join(ROOT, "integration/benchmarks/report"),
          os.path.join(ROOT, "docs/modiff_correctness_2026-08-03/scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch                                                                # noqa: E402

CACHE_DIR = os.path.join(ROOT, "docs/attn_modiff_2026-08-13/data")


def _path(steps, batch, seeds):
    return os.path.join(CACHE_DIR, f"fp16_refs_s{steps}_b{batch}_{'-'.join(map(str, seeds))}.pt")


def build(steps, batch, seeds):
    """Build the references by the path the three agreeing harnesses use, in that exact order.

    The order is load-bearing and was arrived at empirically: H.AUTO_DELTA_TABLE and MODIFF_LINEAR are
    set BEFORE the model is constructed, because both are read at construction time and setting either
    afterwards produced a different reference (that is the 6.9% above).
    """
    import dynamic_delta_ab as H
    H.STEPS, H.BATCH = steps, batch
    H.AUTO_DELTA_TABLE = True
    os.environ["MODIFF_LINEAR"] = "0"
    os.environ["MODIFF_DELTA_MODE"] = "dynamic"
    rf, mf, sf = H.build("fp16", None, "static")
    refs = {}
    for sd in seeds:
        H.SEED = sd
        H.latent(rf, mf, sf)                      # discard, as every arm does
        refs[sd] = H.latent(rf, mf, sf)[0].float()
    del rf, mf, sf
    torch.cuda.empty_cache()
    return refs


def get(steps, batch, seeds, rebuild=False):
    """Cached references. Keyed by (steps, batch, seeds) because a reference for one protocol is
    meaningless for another, and a silently-reused mismatched reference is exactly the failure this
    file exists to prevent."""
    p = _path(steps, batch, seeds)
    if os.path.exists(p) and not rebuild:
        d = torch.load(p, map_location="cpu", weights_only=True)
        return {int(k): v.float() for k, v in d.items()}
    refs = build(steps, batch, seeds)
    os.makedirs(CACHE_DIR, exist_ok=True)
    torch.save({str(k): v for k, v in refs.items()}, p)
    print(f"  fp16_refs: built and cached {p}", flush=True)
    return refs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()
    seeds = [1234, 20260805, 777]
    p = _path(a.steps, a.batch, seeds)
    old = None
    if a.rebuild and os.path.exists(p):
        old = {int(k): v.float() for k, v in
               torch.load(p, map_location="cpu", weights_only=True).items()}
    refs = get(a.steps, a.batch, seeds, rebuild=a.rebuild)
    for sd in seeds:
        print(f"  seed {sd:<10} norm {refs[sd].norm():.6f}  shape {tuple(refs[sd].shape)}")
    if old is not None:
        # The point of --rebuild: quantify whether fp16 sampling reproduces across processes at all.
        print("\n  vs the previous file (this is the fp16 nondeterminism measurement):")
        for sd in seeds:
            d = (refs[sd] - old[sd]).norm() / old[sd].norm()
            print(f"    seed {sd:<10} relative change {d:.2e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

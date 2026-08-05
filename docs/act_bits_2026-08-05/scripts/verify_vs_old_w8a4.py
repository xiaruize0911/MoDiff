"""Why does W8A4+MoDiff measure 0.337 here and 0.127 in docs/fid_2026-08-05/FINDINGS.md?

Both claim to be the same configuration, so one of them is wrong about what it measured. The old
number came from abusing MODIFF_DELTA_CLIP: Q_level = 127/ratio, so ratio = 127/7 put the DELTA
quantizer on a 15-level grid. What it did not touch:

  * the static per-tensor activation grid, which quantizes the t=T warm-up -- and t=T seeds a_hat,
    the reference every later delta is measured against, through the error-feedback term;
  * _forward_modulated's step1_quantize_fprop, which passed a literal 127.0. Any conv layer that
    falls through to the plain modulated path (rather than one of the GN/SiLU-fused variants that
    read _delta_gn_dynamic_args) therefore kept an 8-bit delta grid while the rest went to 4.

MODIFF_ACT_Q covers all three. So this script measures both configurations back to back, and counts
how many conv layers actually take each forward path, which bounds the second effect.

Two arms, same seeds and same fp16 reference as act_bit_sweep.py:
  old   MODIFF_ACT_Q=127, MODIFF_DELTA_CLIP=127/7  -- reproduces FINDINGS' 0.127 if that is the cause
  new   MODIFF_ACT_Q=7,   MODIFF_DELTA_CLIP=1.0    -- the sweep's A4 row, every conv site at 4 bits
"""

import json
import os
import statistics
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers"),
                os.path.join(ROOT, "integration/benchmarks/report"),
                os.path.join(ROOT, "docs/modiff_correctness_2026-08-03/scripts")]

import torch                                                                    # noqa: E402
import dynamic_delta_ab as H                                                    # noqa: E402

SEEDS = [1234, 20260805, 777]
OUT = "docs/act_bits_2026-08-05/data/verify_vs_old_w8a4.json"


def arm(act_q, clip, count_paths=False):
    os.environ["MODIFF_ACT_Q"] = str(act_q)
    os.environ["MODIFF_DELTA_CLIP"] = str(clip)
    r, m, s = H.build("int8", H.CALIB["int8"], "dynamic")
    counts = _instrument(m) if count_paths else None
    H.SEED = SEEDS[0]
    H.latent(r, m, s)                                   # warm-up, discarded
    if counts is not None:
        counts.clear()                                  # count one measured run, not the warm-up
    lat = {}
    for seed in SEEDS:
        H.SEED = seed
        lat[seed], _ = H.latent(r, m, s)
    del m, s, r
    torch.cuda.empty_cache()
    return lat, counts


def _instrument(model):
    """Count calls per delta-quantizing forward path. The fused variants read the Q_level knobs;
    _forward_modulated is the one that used to hardcode 127."""
    from integration.kernels.int8_optimized import OptimizedInt8Conv2d as C
    counts = {}
    for name in ("_forward_modulated", "_forward_modulated_static_fused_silu",
                 "forward_gn_fused_modiff", "forward_modiff_fused_silu_residual",
                 "_forward_first_step"):
        orig = getattr(C, name)

        def wrap(self, *a, _o=orig, _n=name, **k):
            counts[_n] = counts.get(_n, 0) + 1
            counts.setdefault(f"{_n}:layers", set()).add(self.layer_name)
            return _o(self, *a, **k)
        setattr(C, name, wrap)
    return counts


def main():
    os.environ["MODIFF_DELTA_REFRESH"] = "4"
    os.environ["MODIFF_DELTA_REPORT"] = "0"
    os.environ["MODIFF_ACT_Q"] = "127"
    os.environ["MODIFF_DELTA_CLIP"] = "1.0"

    print(f"batch {H.BATCH}, DDIM {H.STEPS}, seeds {SEEDS}\n", flush=True)
    r, m, s = H.build("fp16", None, "static")
    H.SEED = SEEDS[0]
    H.latent(r, m, s)
    refs = {}
    for seed in SEEDS:
        H.SEED = seed
        refs[seed], _ = H.latent(r, m, s)
    del m, s, r
    torch.cuda.empty_cache()

    out = {"seeds": SEEDS, "batch": H.BATCH, "steps": H.STEPS, "arms": {}}
    for label, act_q, clip, probe in (("old_delta_clip_only", 127, 127.0 / 7.0, True),
                                      ("new_act_q", 7, 1.0, False)):
        lat, counts = arm(act_q, clip, probe)
        rel = {k: float((v - refs[k]).norm() / refs[k].norm()) for k, v in lat.items()}
        vals = list(rel.values())
        out["arms"][label] = {"act_q": act_q, "delta_clip": clip, "per_seed": rel,
                              "mean": statistics.mean(vals),
                              "stdev": statistics.stdev(vals) if len(vals) > 1 else 0.0}
        print(f"{label:22s} ACT_Q={act_q:<4} CLIP={clip:<8.4f} relL2 "
              f"{statistics.mean(vals):.4f} +- {statistics.stdev(vals):.4f}  {vals}", flush=True)
        if counts:
            per_step = {k: v for k, v in counts.items() if not k.endswith(":layers")}
            layers = {k.split(":")[0]: sorted(v) for k, v in counts.items() if k.endswith(":layers")}
            out["arms"][label]["path_calls_3_runs"] = per_step
            out["arms"][label]["path_layer_counts"] = {k: len(v) for k, v in layers.items()}
            out["arms"][label]["plain_modulated_layers"] = layers.get("_forward_modulated", [])
            print("  forward-path calls over the measured runs:", flush=True)
            for k, v in sorted(per_step.items(), key=lambda kv: -kv[1]):
                print(f"    {k:44s} {v:7d} calls over {len(layers.get(k, []))} layers", flush=True)

    a, b = out["arms"]["old_delta_clip_only"]["mean"], out["arms"]["new_act_q"]["mean"]
    print(f"\nold {a:.4f} -> new {b:.4f}  ({b / a:.2f}x worse). FINDINGS' W8A4 row claims 0.127.",
          flush=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()

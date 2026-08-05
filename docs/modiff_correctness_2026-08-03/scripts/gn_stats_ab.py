"""A/B the three GN group-statistics reductions, on speed AND on quality AND on determinism.

Why now. The fresh bucket breakdown (2026-08-04, batch 128) shows MoDiff's ENTIRE +8.37 ms/step
overhead against its own baseline is one kernel:

    -14.16 ms  GN+SiLU+quantize fused (the baseline's one-kernel path, gone under MoDiff)
     +9.51 ms  GN group-statistics reduction   <-- this
     +8.06 ms  MoDiff GN+SiLU+delta-quantize

The default `gn_group_stats_kernel` reads group-major: with CPG=4 consecutive threads read 8-byte
chunks strided by C, so a warp touches 8 sectors and uses 8 B of each. `gn_launch_group_stats`
already carries two alternatives behind MODIFF_GN_STATS_ALT, the second of which its own comment
calls a "candidate replacement ... kept opt-in until A/B'd across shapes". This is that A/B.

Three things must be checked together, because the alternatives trade differently on each:
  speed         -- wall-clock ms/step at production batch
  quality       -- latent relL2 vs fp16 (ALT=2 uses fp32 atomicAdd, so its sums are
                   order-dependent; a different mean/var is a different quantization grid)
  determinism   -- same seed twice in one process must give bit-identical latents. atomicAdd
                   ordering is not reproducible, so this is the one that can disqualify ALT=2
                   regardless of how fast it is.
"""

import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers"),
                os.path.join(ROOT, "integration/benchmarks/report"),
                os.path.dirname(os.path.abspath(__file__))]

import torch

from dynamic_delta_ab import CALIB, build, latent

VARIANTS = [("ALT=0 (old group-major tree)", "0"),
            ("default = chan-major partials", None),
            ("ALT=1 (two-pass element-major)", "1"),
            ("ALT=2 (single-pass atomic)", "2")]

# gn_launch_group_stats reads the variant via `static const char* _alt = std::getenv(...)`, so it is
# captured ONCE per process at the first call. Setting the env var between models in one process is
# silently ineffective -- every variant would run whatever was set first. So each variant runs in a
# fresh subprocess, driven by argv, and this script's parent role is only to collect and compare.
#   parent:  gn_stats_ab.py                 (spawns children, prints the table)
#   child:   gn_stats_ab.py <bits> <alt>    (measures one cell, prints one JSON line)


def child(bits, alt):
    if alt == "none":
        os.environ.pop("MODIFF_GN_STATS_ALT", None)
    else:
        os.environ["MODIFF_GN_STATS_ALT"] = alt
    os.environ["MODIFF_DELTA_MODE"] = "dynamic"
    r, m, s = build("fp16", None, "dynamic")
    latent(r, m, s)
    ref, _ = latent(r, m, s)
    del m, s, r
    torch.cuda.empty_cache()
    r, m, s = build(bits, CALIB[bits], "dynamic")
    latent(r, m, s)                      # warm-up
    lat, ms = latent(r, m, s)
    lat2, _ = latent(r, m, s)            # determinism: same seed, same process
    print("RESULT " + json.dumps({
        "rel_l2_vs_fp16": float((lat - ref).norm() / ref.norm()),
        "ms_per_step": ms,
        "deterministic": bool(torch.equal(lat, lat2)),
        "max_abs_diff_replay": float((lat - lat2).abs().max())}), flush=True)


def main():
    import subprocess
    out = {}
    for bits in ("int8", "int4"):
        print(f"{'=' * 78}\n{bits}, conv MoDiff dynamic\n{'=' * 78}", flush=True)
        for label, alt in VARIANTS:
            p = subprocess.run([sys.executable, "-u", os.path.abspath(__file__), bits,
                                "none" if alt is None else alt],
                               capture_output=True, text=True)
            line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
            if not line:
                print(f"  {label:32s} FAILED\n{p.stdout[-600:]}\n{p.stderr[-600:]}", flush=True)
                continue
            v = json.loads(line[-1][len("RESULT "):])
            out[f"{bits}|{label}"] = v
            print(f"  {label:32s} relL2 {v['rel_l2_vs_fp16']:.4f}   "
                  f"{v['ms_per_step']:7.2f} ms/step   deterministic={v['deterministic']}"
                  + ("" if v["deterministic"] else
                     f" (max|d| {v['max_abs_diff_replay']:.2e})"), flush=True)
        print(flush=True)

    print(f"{'=' * 78}\nVerdict\n{'=' * 78}")
    for bits in ("int8", "int4"):
        b = out.get(f"{bits}|ALT=0 (old group-major tree)")
        if not b:
            continue
        for label, _ in VARIANTS[1:]:
            v = out.get(f"{bits}|{label}")
            if not v:
                continue
            flag = "" if v["deterministic"] else "   DISQUALIFIED: nondeterministic"
            print(f"  {bits} {label:32s} {b['ms_per_step'] - v['ms_per_step']:+6.2f} ms/step   "
                  f"relL2 {v['rel_l2_vs_fp16'] - b['rel_l2_vs_fp16']:+.4f}{flag}")

    with open("docs/modiff_correctness_2026-08-03/data/gn_stats_ab.json", "w") as f:
        json.dump({"results": out}, f, indent=2)
    print("\nwrote docs/modiff_correctness_2026-08-03/data/gn_stats_ab.json")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        child(sys.argv[1], sys.argv[2])
    else:
        main()

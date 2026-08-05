"""Where does MoDiff's extra time actually go now? Per-role kernel breakdown, int8 vs int8_baseline.

Stage 3 picks a fusion target, and the plan's target list came from
`docs/benchmark_flash_packed_2026-07-27/REPORT.md` -- measured before the current kernels, the
QKV-epilogue routes, and this session's changes. Committing kernel work to a stale attribution is how
you optimise something that is no longer the bottleneck, so re-measure first.

Method: torch.profiler over a production-length run, kernels bucketed by `profile_tree.classify` (the
same taxonomy the published report uses, so the numbers are comparable). Reports each role's GPU
self-time per step for both modes and the delta -- the MoDiff-only cost, which is what Stage 3 targets.

Do not run this concurrently with a build or another benchmark.
"""

import collections
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers"),
                os.path.join(ROOT, "integration/benchmarks/report")]

import torch
from torch.profiler import ProfilerActivity, profile

import integration.benchmarks.benchmark_ldm as B
import kernel_suites_bench as ks
from profile_tree import classify, short_kernel_name

BATCH = int(os.environ.get("BB_BATCH", "128"))
STEPS = int(os.environ.get("BB_STEPS", "20"))
CALIB = {"int4_baseline": "integration/calibration/int4_calibration_realckpt.pt",
         "int4": "integration/calibration/int4_calibration_realckpt.pt"}
DEFAULT_CALIB = "integration/calibration/int8_calibration_realckpt.pt"


def run(mode, delta_mode="static"):
    ks.set_env(mode)
    os.environ["MODIFF_DELTA_MODE"] = delta_mode
    runner = B.BenchmarkRunner(
        config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
        ckpt_path="models/ldm/lsun_churches256/model.ckpt",
        output_dir="docs/modiff_correctness_2026-08-03/tmp_out",
        batch_size=BATCH, steps=STEPS, shape=(4, 32, 32),
        calibration_path=CALIB.get(mode, DEFAULT_CALIB))
    model, sampler = runner._setup_model(mode)
    cond = runner._cond_kwargs(model, BATCH)

    def sample():
        with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
            sampler.sample(S=STEPS, batch_size=BATCH, shape=runner.shape, eta=0.0,
                           verbose=False, **cond)
        torch.cuda.synchronize()

    sample()                                    # warm + freeze the attention scales
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        sample()

    roles, kernels = collections.Counter(), collections.Counter()
    for evt in prof.key_averages():
        t = evt.self_device_time_total
        if t <= 0:
            continue
        _, role = classify(evt.key)
        roles[role] += t / 1000.0 / STEPS       # ms per step
        kernels[short_kernel_name(evt.key)] += t / 1000.0 / STEPS
    del model, sampler, runner
    torch.cuda.empty_cache()
    return roles, kernels


def main():
    out = {}
    CASES = [("int8_baseline", "int8_baseline", "static"),
             ("int8 dynamic", "int8", "dynamic")]
    for label, mode, dm in CASES:
        r, k = run(mode, dm)
        out[label] = {"roles": dict(r), "kernels": dict(k.most_common(30)),
                      "total_ms_per_step": sum(r.values()), "mode": mode, "delta_mode": dm}
        print(f"\n### {label}: {sum(r.values()):.2f} ms/step of GPU kernel time")
        for role, ms in r.most_common():
            print(f"   {ms:7.2f} ms  {role}")

    for tgt, base in (("int8 dynamic", "int8_baseline"),):
        a, b = out[base]["roles"], out[tgt]["roles"]
        print(f"\n{'=' * 78}\nCost by role: {tgt} minus {base}, ms/step\n{'=' * 78}")
        delta = {r: b.get(r, 0.0) - a.get(r, 0.0) for r in set(a) | set(b)}
        for role, d in sorted(delta.items(), key=lambda kv: -abs(kv[1])):
            if abs(d) > 0.05:
                print(f"   {d:+7.2f} ms  {role}")
        print(f"\n   {out[tgt]['total_ms_per_step'] - out[base]['total_ms_per_step']:+7.2f} ms  TOTAL")

    print(f"\n{'=' * 78}\nTop kernels, int8 dynamic\n{'=' * 78}")
    for kn, ms in list(out["int8 dynamic"]["kernels"].items())[:18]:
        print(f"   {ms:7.2f} ms  {kn}")
    with open("docs/modiff_correctness_2026-08-03/data/bucket_breakdown.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()

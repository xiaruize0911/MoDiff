"""Which of the 129 exported CUDA entry points actually fire, in which mode, and in which phase?

Stage 0 of docs/modiff_correctness_2026-08-03/PLAN notes: the deletion list must rest on evidence,
not on static inference. Static reachability analysis cannot resolve `hasattr` probes, CUTLASS
`can_implement` failures, or route selection that depends on calibration state.

Method: patch every public attribute of `modiff_cutlass` with a counting shim BEFORE any
integration module imports it, so every `import modiff_cutlass as _mc` picks up the patched module
(callers hold the module object, so attribute lookup at call time sees the wrapper). Then for each
mode, count invocations in two phases:

  setup   model construction + calibration. Kernels that fire ONLY here are the calibration-window
          family -- they must be kept even though they never run in steady state.
  steady  a sampling run after setup. This is the production set.

A kernel that fires in neither phase in any mode is a deletion candidate.
"""

import collections
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers")]

import torch

import modiff_cutlass as mc

COUNTS = collections.Counter()
EXPORTS = sorted(n for n in dir(mc) if not n.startswith("_"))


def install_counters():
    for name in EXPORTS:
        fn = getattr(mc, name)
        if not callable(fn):
            continue

        def make(name, fn):
            def wrapper(*a, **k):
                COUNTS[name] += 1
                return fn(*a, **k)
            wrapper.__name__ = name
            return wrapper
        setattr(mc, name, make(name, fn))


install_counters()          # must precede the benchmark_ldm import
sys.path.insert(0, os.path.join(ROOT, "integration/benchmarks/report"))
import integration.benchmarks.benchmark_ldm as B   # noqa: E402
# MODIFF_QUANT_LINEAR=1 and friends are load-bearing: without them qkv/proj stay plain nn.Linear,
# _qout_eligible() returns False, and EVERY fused-epilogue attention route is silently skipped --
# the run still completes and still looks quantized, it just measures a slower configuration.
# kernel_suites_bench.set_env is the same helper the reporting harnesses use.
import kernel_suites_bench as ks                   # noqa: E402

# Internal mode names. _BASEMAP (benchmark_ldm.py:359) only rewrites modes carrying a
# static_/dynamic_ prefix, so these pass through _setup_model unchanged. Note the naming trap:
# internal "int8" is MoDiff ON, internal "int8_baseline" is MoDiff OFF.
MODES = ["fp16", "int8_baseline", "int8", "int4_baseline", "int4",
         "int8_attn_modiff", "int4_attn_modiff", "attn_modiff"]
BATCH = int(os.environ.get("PROBE_BATCH", "4"))
STEPS = int(os.environ.get("PROBE_STEPS", "20"))       # must divide 1000 (DDIM indexes alphas at t+1)
CALIB = {"int4_baseline": "integration/calibration/int4_calibration.pt",
         "int4": "integration/calibration/int4_calibration.pt",
         "int4_attn_modiff": "integration/calibration/int4_calibration.pt"}
DEFAULT_CALIB = "integration/calibration/int8_calibration.pt"


def run_mode(mode):
    """{'setup': {name: n}, 'steady': {name: n}} or {'error': str}."""
    COUNTS.clear()
    try:
        ks.set_env(mode)          # the production env; see the note at the import
        runner = B.BenchmarkRunner(
            config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
            ckpt_path="models/ldm/lsun_churches256/model.ckpt",
            output_dir="docs/modiff_correctness_2026-08-03/tmp_out",
            batch_size=BATCH, steps=STEPS, shape=(4, 32, 32),
            calibration_path=CALIB.get(mode, DEFAULT_CALIB))
        model, sampler = runner._setup_model(mode)
        torch.cuda.synchronize()
        setup = dict(COUNTS)
        COUNTS.clear()

        cond = runner._cond_kwargs(model, BATCH)

        def sample():
            with torch.inference_mode(), torch.amp.autocast("cuda", enabled=(mode != "fp32"),
                                                            dtype=torch.float16):
                sampler.sample(S=STEPS, batch_size=BATCH, shape=runner.shape, eta=0.0,
                               verbose=False, **cond)
            torch.cuda.synchronize()

        # Two sampling runs. The attention blocks self-calibrate over their first
        # MODIFF_ATTN_CALIB_STEPS (default 8) forwards and then freeze, so run 1 straddles the
        # boundary and fires BOTH the calibrating entries (quantize_attn_qkv_packed,
        # flash_attn_intX_vt) and the frozen ones. Run 2 is past the boundary for every block, so
        # it alone is the production set. Conflating them is how a live calibration-window kernel
        # gets mistaken for dead code.
        sample()
        warm = dict(COUNTS)
        COUNTS.clear()
        sample()
        steady = dict(COUNTS)

        del model, sampler, runner
        torch.cuda.empty_cache()
        return {"setup": setup, "warm": warm, "steady": steady}
    except Exception as exc:
        torch.cuda.empty_cache()
        return {"error": f"{type(exc).__name__}: {exc}"}


def main():
    out = {"exports": EXPORTS, "n_exports": len(EXPORTS), "batch": BATCH, "steps": STEPS,
           "gpu": torch.cuda.get_device_name(0), "modes": {}}
    for mode in MODES:
        print(f"\n{'=' * 70}\n{mode}\n{'=' * 70}", flush=True)
        r = run_mode(mode)
        out["modes"][mode] = r
        if "error" in r:
            print(f"  ERROR {r['error']}")
            continue
        calib_only = sorted((set(r["setup"]) | set(r["warm"])) - set(r["steady"]))
        print(f"  setup  entries: {len(r['setup'])}")
        print(f"  warm   entries: {len(r['warm'])}   (run 1, straddles the calib boundary)")
        print(f"  steady entries: {len(r['steady'])}  (run 2, production set)")
        print(f"  setup/calibration-window ONLY ({len(calib_only)}): {', '.join(calib_only)}")

    fired_any, fired_steady = set(), set()
    for r in out["modes"].values():
        if "error" in r:
            continue
        fired_any |= set(r["setup"]) | set(r["warm"]) | set(r["steady"])
        fired_steady |= set(r["steady"])
    callable_exports = {n for n in EXPORTS if callable(getattr(mc, n))}
    out["fired_any"] = sorted(fired_any)
    out["fired_steady_any_mode"] = sorted(fired_steady)
    out["never_fired"] = sorted(callable_exports - fired_any)
    out["setup_only"] = sorted(fired_any - fired_steady)

    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    print(f"callable exports          : {len(callable_exports)}")
    print(f"fired in some mode/phase  : {len(fired_any)}")
    print(f"fired in steady state     : {len(fired_steady)}")
    print(f"setup/calibration only    : {len(out['setup_only'])}")
    print(f"NEVER FIRED               : {len(out['never_fired'])}")
    for n in out["never_fired"]:
        print(f"    {n}")

    path = "docs/modiff_correctness_2026-08-03/data/kernel_reachability.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWROTE {path}")


if __name__ == "__main__":
    main()

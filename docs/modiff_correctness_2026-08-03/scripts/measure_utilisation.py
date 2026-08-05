"""D3: is the static activation quantizer matched to what the layers actually see?

Metric is `OptimizedInt8Conv2d.effective_code_utilisation` -- the one tested definition
(test_kernel_correctness.py::code_utilisation pins it: Q when matched, 8x when 8x under-provisioned).
Deliberately NOT ad-hoc instrumentation; that route produced two wrong answers, once by omitting
`smooth_inv` and once by reading int8 codes as activations.

Reports, per mode and per conv family, the max over a production-length run of
`max|x * smooth_inv| * static_input_scale`. Q = 127 means matched; >Q means clipping.

Instrumentation note: hooks record only FLOATING-POINT inputs. `forward_from_int8` and friends take
int8 codes, and including those pins the answer at a meaningless 127.
"""

import collections
import json
import os
import statistics as st
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers"),
                os.path.join(ROOT, "integration/benchmarks/report")]

import torch

import integration.benchmarks.benchmark_ldm as B
import kernel_suites_bench as ks
from integration.kernels.int8_optimized import OptimizedInt8Conv2d, reset_modiff_state
from integration.utils import attention_identity_guard as guard

STEPS = int(os.environ.get("MU_STEPS", "20"))
BATCH = int(os.environ.get("MU_BATCH", "4"))
UTIL = collections.defaultdict(list)


def install():
    for nm in [n for n in dir(OptimizedInt8Conv2d)
               if n == "forward" or n.startswith(("forward_", "_forward"))]:
        fn = getattr(OptimizedInt8Conv2d, nm)
        if not callable(fn):
            continue

        def mk(fn):
            def w(self, x, *a, **k):
                if torch.is_tensor(x) and x.is_floating_point() and self.is_calibrated:
                    UTIL[self.layer_name].append(
                        self.effective_code_utilisation(x, fused_silu=self.fuse_input_silu))
                return fn(self, x, *a, **k)
            return w
        setattr(OptimizedInt8Conv2d, nm, mk(fn))


def run(mode, calib_steps, calib_batch, refine=1):
    ks.set_env(mode)
    guard.seed_model_construction()
    runner = B.BenchmarkRunner(
        config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
        ckpt_path="models/ldm/lsun_churches256/model.ckpt",
        output_dir="docs/modiff_correctness_2026-08-03/tmp_out",
        batch_size=BATCH, steps=STEPS, shape=(4, 32, 32), calibration_path=None)
    model, sampler = runner._setup_model(mode)
    runner._calibrate_int8(model, sampler, num_runs=2, calib_steps=calib_steps,
                           calib_batch=calib_batch, refine_rounds=refine)
    # Start the measured run from a FRESH MoDiff state, as a real run does (run_mode resets before
    # sampling). Without this the production sample continues from whatever a_hat/o_hat the
    # calibration pass left behind, so o_hat keeps accumulating and the activations read ~4x larger
    # than they are -- which is a harness artifact, not under-observation.
    reset_modiff_state(model)
    UTIL.clear()
    cond = runner._cond_kwargs(model, BATCH)
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        sampler.sample(S=STEPS, batch_size=BATCH, shape=runner.shape, eta=0.0,
                       verbose=False, **cond)
    # SmoothQuant state. On the stub every out_conv had zero weights, so s saturated at its 1e4
    # clamp and smooth_is_identity was trivially True. With real weights it should NOT be identity,
    # so if it still is, SmoothQuant is not engaging and that is worth knowing.
    convs = {c.layer_name: c for c in model.modules() if isinstance(c, OptimizedInt8Conv2d)}
    sq = {}
    for fam in ("in_conv", "out_conv"):
        sel = [c for n, c in convs.items() if n.endswith(fam)]
        if sel:
            sq[fam] = {
                "n": len(sel),
                "identity": sum(1 for c in sel if c._smooth_is_identity),
                "smooth_max_median": st.median(float(c.smooth_scale.abs().max()) for c in sel),
                "wint8_absmax_median": st.median(float(c.weight_int8.abs().max()) for c in sel),
            }
    res = {"_smoothquant": sq}
    for fam in ("in_conv", "out_conv"):
        sel = [n for n in UTIL if n.endswith(fam) and UTIL[n]]
        if sel:
            peaks = [max(UTIL[n]) for n in sel]
            res[fam] = {"n": len(sel), "median": st.median(peaks), "max": max(peaks),
                        "clipping": sum(1 for p in peaks if p > 127.0)}
    del model, sampler, runner
    torch.cuda.empty_cache()
    return res


def main():
    install()
    out = {}
    configs = [("REAL ckpt: S=5 batch=2, no refinement (as shipped)", 5, 2, 0),
               ("REAL ckpt: horizon+batch matched + 1 refinement", None, None, 1)]
    for mode in ("int8",):
        for tag, cs, cb, rf in configs:
            r = run(mode, cs, cb, rf)
            key = f"{mode} | {tag}"
            out[key] = r
            print(f"\n### {key}")
            for fam, d in r.items():
                if fam == "_smoothquant":
                    for f2, q in d.items():
                        print(f"   [smooth] {f2:<9} identity {q['identity']}/{q['n']}  "
                              f"smooth_scale max median {q['smooth_max_median']:.4g}  "
                              f"weight_int8 absmax median {q['wint8_absmax_median']:.0f}")
                    continue
                print(f"   {fam:<9} n={d['n']:2d}  utilisation median {d['median']:7.1f}  "
                      f"max {d['max']:8.1f}  clipping {d['clipping']}/{d['n']}   (Q=127)")
    with open("docs/modiff_correctness_2026-08-03/data/utilisation.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nQ=127 is full scale. >127 means the static activation quantizer clips.")


if __name__ == "__main__":
    main()

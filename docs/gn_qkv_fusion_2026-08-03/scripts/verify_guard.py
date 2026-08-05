"""Does the guard do its job?

For each of fp16 / int8_baseline / int4_baseline:
  1. assert_attention_observable() must RAISE on the untouched model      (structural)
  2. assert_unet_output_observable() must RAISE on the untouched model    (behavioural)
  3. activate_zeroed_modules() must activate every annihilating module
  4. both assertions must then PASS
  5. corrupting the attention outputs must now MOVE the sampled latent -- i.e. the check that
     used to be vacuous has become able to fail
"""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers")]

import torch

import integration.benchmarks.benchmark_ldm as B
from integration.utils.attention_identity_guard import (
    NotObservable, activate_zeroed_modules, assert_attention_observable,
    assert_unet_output_observable, attention_blocks, zeroed_modules)

BATCH, STEPS, SEED = 4, 20, 1234
CONFIG = "configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml"
CKPT = "models/ldm/lsun_churches256/model.ckpt"


def build(mode):
    cal = ("integration/calibration/int4_calibration.pt" if "int4" in mode
           else "integration/calibration/int8_calibration.pt")
    runner = B.BenchmarkRunner(config_path=CONFIG, ckpt_path=CKPT,
                               output_dir="/tmp/guard_out", batch_size=BATCH,
                               steps=STEPS, calibration_path=cal)
    model, sampler = runner._setup_model(mode)
    return runner, model, sampler


def sample(runner, model, sampler):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=True):
        out = sampler.sample(S=STEPS, batch_size=BATCH, shape=runner.shape,
                             eta=0.0, verbose=False)
    lat = out[0] if isinstance(out, (tuple, list)) else out
    return lat.detach().float().cpu()


def expect_raise(fn, label, failures, mode):
    try:
        fn()
    except NotObservable as exc:
        print(f"  ok   {label} raised: {str(exc).splitlines()[0][:110]}")
        return
    print(f"  FAIL {label} did not raise on the untouched model")
    failures.append(f"{mode}: {label} silent before activation")


def main():
    failures = []
    for mode in ("fp16", "int8_baseline", "int4_baseline"):
        print("=" * 72)
        print(mode)
        print("=" * 72)
        runner, model, sampler = build(mode)
        unet = model.model.diffusion_model
        dead_before = len(zeroed_modules(model))

        expect_raise(lambda: assert_attention_observable(model, what="a latent comparison"),
                     "assert_attention_observable", failures, mode)
        expect_raise(lambda: assert_unet_output_observable(unet, what="a latent comparison"),
                     "assert_unet_output_observable", failures, mode)

        n = activate_zeroed_modules(model)
        print(f"  activated {n} of {dead_before} annihilating modules")

        left = zeroed_modules(model)
        if left:
            print(f"  FAIL {len(left)} still annihilating: {[x[0] for x in left[:3]]}")
            failures.append(f"{mode}: {len(left)} still annihilating")
        else:
            assert_attention_observable(model, what="a latent comparison")
            assert_unet_output_observable(unet, what="a latent comparison")
            print("  ok   both assertions pass after activation")

        clean = sample(runner, model, sampler)
        blocks = attention_blocks(model)
        calls = {"n": 0}
        for _, mod in blocks:
            orig = mod.forward

            def corrupt(x, _orig=orig):
                calls["n"] += 1
                return torch.full_like(_orig(x), 3.0)
            mod.forward = corrupt
        dirty = sample(runner, model, sampler)
        d = dirty - clean
        moved = int((d != 0).sum())
        rel = (d.norm() / clean.norm()).item() if float(clean.norm()) > 0 else float("nan")
        print(f"  corrupted {len(blocks)} blocks, forward fired {calls['n']}x: "
              f"differing elements {moved}/{d.numel()}  relL2 {rel:.4g}  "
              f"clean finite={bool(torch.isfinite(clean).all())}")
        if moved == 0:
            print("  FAIL latent still independent of attention")
            failures.append(f"{mode}: latent still vacuous")
        else:
            print("  ok   a wrong attention result would now be caught")

        del runner, model, sampler
        torch.cuda.empty_cache()

    print()
    print("FAILURES:" if failures else "ALL CHECKS PASSED")
    for f in failures:
        print("  -", f)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

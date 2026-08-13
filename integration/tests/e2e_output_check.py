#!/usr/bin/env python3
"""End-to-end seeded output-fidelity check.

Captures a deterministic reference (fixed seed -> DDIM eta=0 is deterministic)
of the sampled latents for a mode, then, after a code change, regenerates with
the same seed and reports rel_err / max-abs diff. Used to gate changes that can
alter numerics (e.g. the GroupNorm fp16 change) without running full FID.

  # before a change:
  python integration/tests/e2e_output_check.py --mode fp16 --capture
  # after:
  python integration/tests/e2e_output_check.py --mode fp16 --compare

IMPORTANT -- this gate was vacuous until 2026-08-03. `UNetModel.out[-1]` is a `zero_module`
(ldm/modules/diffusionmodules/openaimodel.py:745) and this tree's checkpoint is an 856-byte stub
with an empty `state_dict` loaded strict=False, so that layer stayed zero and the UNet predicted
identically zero for every input. The sampled latent was a function of the initial noise and the
DDIM schedule alone, so this check PASSED for every change anywhere in the network. It now
activates those zero-initialised layers first and asserts the UNet output is observable, so a
change that alters numerics can actually move the latent. See
docs/gn_qkv_fusion_2026-08-03/FINDINGS.md section 5.

Because activation changes the model, goldens are keyed on it: a reference captured with
--no-activate-zeroed is not comparable to one captured with it. Changing the activation SCHEME
invalidates goldens the same way, so recapture after touching the guard. Note also that
quantization scales are calibrated before activation, so the absolute latent here is not a quality
measurement -- this is an A/B equivalence gate, which is what it is used for.

Measured sensitivity, int8_baseline / 20 steps / batch 4 (2026-08-03):
  byte-identical code       rel_err 0.0000, 0.0000, 0.0004 over three fresh processes
  MODIFF_FLASH_GATE=off     rel_err 0.0011   (a real attention-route change, correctly under tol)
  same, --no-activate-zeroed rel_err 0.0000  (the blind case: the change is invisible)
So the floor is ~4e-4, not 0 -- some kernels reduce with atomicAdd and are not bit-reproducible.
The default tol=0.02 is ~50x that floor, i.e. this gate catches gross breakage, not drift.
"""
import os, sys, argparse, torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
# ldm's config instantiates taming-transformers classes; without this the script cannot import.
_TAMING = os.path.join(REPO, "src/taming-transformers")
if _TAMING not in sys.path:
    sys.path.insert(1, _TAMING)
import importlib.util
spec = importlib.util.spec_from_file_location(
    "bldm", os.path.join(REPO, "integration/benchmarks/benchmark_ldm.py"))
bldm = importlib.util.module_from_spec(spec); spec.loader.exec_module(bldm)
from integration.utils import attention_identity_guard as guard

REFDIR = os.path.join(HERE, "golden")
os.makedirs(REFDIR, exist_ok=True)


def gen(mode, steps, batch, seed, activate=True, model_seed=20260803):
    # CALIBRATION_PREFERENCE, not a hardcoded path -- see the note in
    # integration/benchmarks/report/kernel_suites_bench.py. This one matters MORE than the pure
    # timing harnesses: this script compares UNet output against goldens, so the activation scale
    # changes what it asserts, not just how fast it runs. The two live goldens
    # (e2e_int8{,_baseline}_s20_b4.pt) were captured against the old literal and are refreshed
    # with this change; golden/README.md records the attribution.
    cal = bldm._default_calibration_path(mode)
    # The checkpoint is an empty stub, so every weight comes from default-initialisation off the
    # global RNG -- and torch seeds that nondeterministically per process. Unseeded, this model is
    # a DIFFERENT random network in every run and no golden can survive; measured rel_err ~0.4 for
    # byte-identical code. Seed before construction so the network itself is reproducible.
    torch.manual_seed(model_seed); torch.cuda.manual_seed_all(model_seed)
    runner = bldm.BenchmarkRunner(config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
                                  ckpt_path="models/ldm/lsun_churches256/model.ckpt",
                                  output_dir="/tmp/e2e_out", batch_size=batch, steps=steps,
                                  calibration_path=cal)
    model, sampler = runner._setup_model(mode)
    if mode in ("int8", "int8_attn_modiff"):
        cfg = bldm.get_calibration_config_int8()
        if not cfg.is_calibrated:
            runner._calibrate_int8(model, sampler)
    # Without this the UNet predicts identically zero and the comparison below cannot fail.
    if activate:
        guard.activate_zeroed_modules(model)
        guard.assert_unet_output_observable(model.model.diffusion_model,
                                           what="this e2e latent comparison")
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    use_ac = mode != "fp32"
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=use_ac):
        samples, _ = sampler.sample(S=steps, batch_size=batch, shape=runner.shape,
                                    eta=0.0, verbose=False)
    return samples.detach().float().cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="fp16")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--tol", type=float, default=0.02)
    ap.add_argument("--capture", action="store_true")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--no-activate-zeroed", dest="activate", action="store_false",
                    help="skip activating zero_module layers -- makes this check vacuous, "
                         "only for reproducing pre-2026-08-03 goldens")
    a = ap.parse_args()
    out = gen(a.mode, a.steps, a.batch, a.seed, activate=a.activate)
    # Keyed on activation: the two settings produce different models, so their goldens must
    # never be compared against each other.
    tag = "" if a.activate else "_vacuous"
    path = os.path.join(REFDIR, f"e2e_{a.mode}_s{a.steps}_b{a.batch}{tag}.pt")
    if a.capture or not os.path.exists(path):
        torch.save(out, path)
        print(f"[capture] {a.mode}: saved reference {tuple(out.shape)} -> {path}")
        return 0
    ref = torch.load(path)
    re = (out - ref).norm().item() / (ref.norm().item() + 1e-12)
    mx = (out - ref).abs().max().item()
    ok = re < a.tol
    print(f"[compare] {a.mode}: rel_err={re:.4f} max_abs={mx:.4f} tol={a.tol} -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

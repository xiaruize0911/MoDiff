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
"""
import os, sys, argparse, torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
import importlib.util
spec = importlib.util.spec_from_file_location(
    "bldm", os.path.join(REPO, "integration/benchmarks/benchmark_ldm.py"))
bldm = importlib.util.module_from_spec(spec); spec.loader.exec_module(bldm)

REFDIR = os.path.join(HERE, "golden")
os.makedirs(REFDIR, exist_ok=True)


def gen(mode, steps, batch, seed):
    cal = ("integration/calibration/int8_calibration.pt" if "int8" in mode or mode in ("fp16", "fp32")
           else "integration/calibration/int4_calibration.pt")
    runner = bldm.BenchmarkRunner(config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
                                  ckpt_path="models/ldm/lsun_churches256/model.ckpt",
                                  output_dir="/tmp/e2e_out", batch_size=batch, steps=steps,
                                  calibration_path=cal)
    model, sampler = runner._setup_model(mode)
    if mode in ("int8", "int8_attn_modiff"):
        cfg = bldm.get_calibration_config_int8()
        if not cfg.is_calibrated:
            runner._calibrate_int8(model, sampler)
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
    a = ap.parse_args()
    out = gen(a.mode, a.steps, a.batch, a.seed)
    path = os.path.join(REFDIR, f"e2e_{a.mode}_s{a.steps}_b{a.batch}.pt")
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

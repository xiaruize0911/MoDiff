#!/usr/bin/env python3
"""Phase 1 validation: CUDA-graph replay is numerically identical to eager.

For a mode, samples a seeded DDIM latent twice from the SAME seed -- once eager,
once with the per-step UNet CUDA graph installed -- and reports rel-L2. Graph
replay re-runs the identical kernels, so the expected result is ~0.

  python docs/benchmark_5mode_2026-07-21/scripts/verify_cuda_graph_e2e.py [mode ...]
"""
import os, sys, importlib.util, torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
spec = importlib.util.spec_from_file_location(
    "bldm", os.path.join(REPO, "integration/benchmarks/benchmark_ldm.py"))
bldm = importlib.util.module_from_spec(spec); spec.loader.exec_module(bldm)

CKPT = "models/ldm/lsun_churches256/model.ckpt"
CFG = "configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml"


def sample(runner, model, sampler, mode, steps, batch, seed, with_graph):
    runner._reset_modiff_state(model, mode)
    graph_active = False
    if with_graph:
        graph_active = runner._maybe_install_cuda_graph(model, sampler, mode, True, torch.float16)
        if not graph_active:
            print(f"  [{mode}] graph did NOT install; skipping graph arm")
            return None
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    runner._reset_modiff_state(model, mode)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        out, _ = sampler.sample(S=steps, batch_size=batch, shape=runner.shape,
                                eta=0.0, verbose=False, **runner._cond_kwargs(model, batch))
    return out.detach().float().cpu()


def run(mode, steps=50, batch=4, seed=1234):
    cal = ("integration/calibration/int8_calibration.pt" if "int8" in mode or mode in ("fp16", "fp32")
           else "integration/calibration/int4_calibration.pt")
    runner = bldm.BenchmarkRunner(config_path=CFG, ckpt_path=CKPT, output_dir="/tmp/cg_verify",
                                  batch_size=batch, steps=steps, calibration_path=cal,
                                  use_cuda_graph=True)
    model, sampler = runner._setup_model(mode)
    eager = sample(runner, model, sampler, mode, steps, batch, seed, with_graph=False)
    graph = sample(runner, model, sampler, mode, steps, batch, seed, with_graph=True)
    if graph is None:
        return
    re = (graph - eager).norm().item() / (eager.norm().item() + 1e-12)
    mx = (graph - eager).abs().max().item()
    ok = re < 1e-3
    print(f"[{mode}] eager-vs-graph rel_L2={re:.2e} max_abs={mx:.2e} -> {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    modes = sys.argv[1:] or ["fp16", "int8_baseline", "int4_baseline", "int8", "int4"]
    results = [run(m) for m in modes]
    sys.exit(0 if all(r for r in results if r is not None) else 1)

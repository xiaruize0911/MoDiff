"""D: e2e latent rel-err of int8/int4 quantized STANDARD attention (+ quantized conv) vs
fp16 standard attention. Confirms wiring and measures NUMERIC AGREEMENT between the routes.

This does NOT measure quality, and until 2026-08-03 it measured nothing at all. Two reasons, both
from the 856-byte stub checkpoint whose state_dict has 0 entries:

  * `UNetModel.out[-1]` is a `zero_module` (ldm/modules/diffusionmodules/openaimodel.py:745) that
    the stub never fills in, so the UNet predicted identically zero and the latent was a function
    of the initial noise alone -- the rel-err below was structurally 0 for every mode. The five
    goldens in integration/tests/golden/e2e_*_vacuous.pt are the evidence: fp16, int8 and int4 are
    bit-identical to each other.
  * every weight comes from default-init off torch's global RNG, which is seeded
    nondeterministically per process, so each mode below was built as a DIFFERENT random network.

Both are handled now (seed the construction, activate the zeroed layers), so the rel-err below is
at least a function of the modes. It is still NOT INTERPRETABLE as accuracy, for a third reason
that activation cannot fix: the static calibration in integration/calibration/*.pt was produced
against the un-activated (all-zero-output) network, so every activation scale is wrong for the
weights this test now runs, and the result is dominated by scale mismatch rather than by
quantization. Measured 2026-08-03, on a SINGLE UNet forward, rel-vs-fp16 was 0.84 for int8 and
0.55 for int4 -- int8 nominally worse than int4, and both far too large to be quantization error.
The ordering is meaningless; do not read these numbers as a precision ranking.

So treat this as a WIRING check: it confirms all three modes build, convert and sample. For actual
attention correctness use the kernel-level tests, which compare against an fp32 reference computed
from the same quantized codes on synthetic tensors and need no checkpoint:
docs/final_report_2026-07-28/scripts/qattn_correctness.py and int4_fused_routes_check.py. To make
this a real cross-mode accuracy test, regenerate the calibration against the activated network
first. See docs/gn_qkv_fusion_2026-08-03/FINDINGS.md section 5."""
import os, sys, importlib.util
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
from integration.utils import attention_identity_guard as guard
spec = importlib.util.spec_from_file_location("abb", "/workspace/MoDiff/integration/benchmarks/ab_benchmark.py")
abb = importlib.util.module_from_spec(spec); spec.loader.exec_module(abb)
import torch
class A: pass
args = A(); args.config = "configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml"
args.ckpt = "models/ldm/lsun_churches256/model.ckpt"; args.batch_size = 8; args.steps = 20
args.linear_backend = "fp16"; args.calibration = None

def latent(mode):
    for k in ("MODIFF_STD_ATTN_BITS",): os.environ.pop(k, None)
    # Same random network in every mode, and a network whose output is observable at all.
    guard.seed_model_construction()
    runner, model, sampler = abb.build(mode, args)
    guard.prepare_for_comparison(model, what=f"the {mode} vs fp16 latent comparison")
    torch.manual_seed(0); cond = runner._cond_kwargs(model, args.batch_size)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=(mode != "fp32")):
        out = sampler.sample(S=args.steps, batch_size=args.batch_size, shape=runner.shape, eta=0.0, verbose=False, **cond)
    lat = (out[0] if isinstance(out, (tuple, list)) else out).float()
    pk = torch.cuda.max_memory_allocated() / 1048576
    del runner, model, sampler; torch.cuda.empty_cache()
    return lat, pk

ref, _ = latent("fp16")
print("REF fp16 standard attention computed", flush=True)
for mode in ["int8_baseline", "int4_baseline"]:
    lat, pk = latent(mode)
    rel = (lat - ref).norm().item() / (ref.norm().item() + 1e-12)
    print(f"RESULT {mode:14s} (int{'8' if '8' in mode else '4'} conv + std attn) latent rel-vs-fp16={rel:.4f} peak={pk:.0f}MiB", flush=True)
print("NOTE   rel-vs-fp16 above is NOT an accuracy ranking: the static calibration was built "
      "against the un-activated network, so these are dominated by scale mismatch. This run only "
      "confirms all three modes build and sample. See the module docstring.", flush=True)

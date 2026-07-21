"""Real-input differential for the o_hat+residual fusion. For the first K eligible
modiff out-conv calls, run BOTH the reference (plain o_hat conv + eager torch.add)
and the fused path on identical snapshotted cache state, and compare:
  - o_hat cache update  (must be BIT-identical: cache write is unchanged)
  - a_hat cache update  (must be BIT-identical: same step1)
  - output              (fp16-ULP: fused fp32-accumulates, eager fp16-adds)
Then continue the real run with the fused result."""
import os, sys
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
os.environ["MODIFF_QUANT_LINEAR"]="1"; os.environ["MODIFF_QUANT_ATTN"]="1"; os.environ["MODIFF_LINEAR_OUT_I8"]="0"
import torch
import integration.benchmarks.benchmark_ldm as B
import integration.fused_ops.fused_resblock as FR

mode = sys.argv[1] if len(sys.argv) > 1 else "int8"
BATCH, S = 8, 20
calib = f"integration/calibration/{mode}_calibration.pt"
r = B.BenchmarkRunner("configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
    "models/ldm/lsun_churches256/model.ckpt", output_dir="integration/results/vr",
    batch_size=BATCH, steps=S, shape=(4,32,32), calibration_path=calib, linear_backend="int_gemm")
model, sampler = r._setup_model(mode); cond = r._cond_kwargs(model, BATCH)

st = {"n": 0, "ohat": 0.0, "ahat": 0.0, "out": 0.0}
orig = FR._modiff_out_conv
def wrap(conv, h, residual_arg):
    eligible = (residual_arg is not None and FR.HAS_O_HAT_RESIDUAL
                and getattr(conv, 'modiff_enabled', False)
                and hasattr(conv, 'forward_modiff_fused_silu_residual')
                and conv._can_fuse_input_silu(h))
    if not eligible or st["n"] >= 30:
        return orig(conv, h, residual_arg)
    with torch.inference_mode():
        a0 = conv.a_hat_cache.clone(); o0 = conv.o_hat_cache.clone()
        sc = int(conv.step_count)
        # reference: plain conv (updates caches) + eager add
        out_ref = conv(h)                       # returns o_hat_cache (aliased)
        out_ref = torch.add(residual_arg.to(out_ref.dtype), out_ref).clone()
        o_ref = conv.o_hat_cache.clone(); a_ref = conv.a_hat_cache.clone()
        # restore, run fused
        conv.a_hat_cache.copy_(a0); conv.o_hat_cache.copy_(o0); conv.step_count = sc
        out_fus = conv.forward_modiff_fused_silu_residual(h, residual_arg)
        o_fus = conv.o_hat_cache.clone(); a_fus = conv.a_hat_cache.clone()
        torch.cuda.synchronize()
        st["ohat"] = max(st["ohat"], (o_ref.float()-o_fus.float()).abs().max().item())
        st["ahat"] = max(st["ahat"], (a_ref.float()-a_fus.float()).abs().max().item())
        st["out"]  = max(st["out"],  float(torch.norm(out_ref.float()-out_fus.float())/(torch.norm(out_ref.float())+1e-12)))
        st["n"] += 1
    return out_fus, True
FR._modiff_out_conv = wrap

torch.manual_seed(1234)
with torch.inference_mode(), torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
    sampler.sample(S=S, batch_size=BATCH, shape=r.shape, eta=0.0, verbose=False, **cond)
print(f"mode={mode}  eligible_calls={st['n']}  max_ohat_diff={st['ohat']:.3e}  "
      f"max_ahat_diff={st['ahat']:.3e}  max_out_relL2={st['out']:.3e}")

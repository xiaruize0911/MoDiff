"""Real-input differential: for the first K modiff GN-fusion calls during a real
sample, run BOTH the reference two-kernel path (group_norm_silu_nhwc +
step1_static_quantize[_pack]_fprop_silu) on a CLONE of the exact inputs AND the
fused kernel, and compare the int codes + the a_hat update. If bit-identical on
real inputs, any full-run cache drift is pure compounding of the (fusion-
independent) nondeterminism, not a per-call fusion error."""
import os, sys
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
os.environ["MODIFF_QUANT_LINEAR"]="1"; os.environ["MODIFF_QUANT_ATTN"]="1"; os.environ["MODIFF_LINEAR_OUT_I8"]="0"
import torch
import integration.benchmarks.benchmark_ldm as B
import integration.fused_ops.fused_resblock as FR
import modiff_cutlass as M

mode = sys.argv[1] if len(sys.argv) > 1 else "int8"
is_int4 = (mode == "int4")
BATCH, S = 8, 20
calib = f"integration/calibration/{mode}_calibration.pt"
r = B.BenchmarkRunner("configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
    "models/ldm/lsun_churches256/model.ckpt", output_dir="integration/results/rid",
    batch_size=BATCH, steps=S, shape=(4,32,32), calibration_path=calib, linear_backend="int_gemm")
model, sampler = r._setup_model(mode); cond = r._cond_kwargs(model, BATCH)

stats = {"calls": 0, "max_code": 0, "max_ahat": 0.0}
orig = FR._prequant_gn_conv_modiff
def wrap(x, gn, conv, mod_scale=None, mod_shift=None, residual=None, x2=None, **kw):
    # x2 (the decoder skip-concat's second half) and **kw added 2026-08-16. _prequant_gn_conv_modiff
    # gained x2= with the cat2 fold on 2026-08-13 and this wrapper's signature did not follow, so this
    # gate has raised TypeError -- i.e. has been UNRUNNABLE -- ever since. It is the gate that a
    # previous reduction change was reverted for failing, so "it failed the gate" and "nobody could run
    # the gate" had become the same observable. **kw so the next added argument degrades to a pass-through
    # instead of breaking the gate again.
    #
    # A cat2-folded call passes x2 and takes a different fused kernel, so it is NOT comparable to the
    # two-kernel reference built below; those calls are forwarded uninstrumented rather than counted.
    # Only instrument the real fused calls (post first-step, eligible).
    if (x2 is None and stats["calls"] < 40 and conv is not None
            and getattr(conv, 'modiff_enabled', False)
            and hasattr(conv, 'can_gn_fuse_modiff') and conv.can_gn_fuse_modiff(x)
            and x.is_contiguous(memory_format=torch.channels_last)):
        ng = gn.num_groups; eps = gn.eps
        w, b = gn._cast_params(x.dtype)
        N, C = x.size(0), x.size(1)
        if mod_scale is not None:
            ms = mod_scale.reshape(N, C).contiguous(); sh = mod_shift.reshape(N, C).contiguous()
        else:
            ms = sh = x.new_empty(0)
        scale = conv.static_input_scale.view(1)
        smooth = conv._smooth_inv_flat if hasattr(conv, '_smooth_inv_flat') else x.new_empty(0, dtype=torch.float32)
        a_ref = conv.a_hat_cache.clone()
        # The 8 trailing dynamic-scale arguments, added to these kernels after this gate was written
        # (the gate passed 11 args to an 18/19-arg kernel and raised TypeError). Taken from the conv's
        # OWN accessor rather than hardcoded, so the next change to that contract cannot silently
        # re-stale the gate -- which is how it came to be unrunnable in two independent ways at once.
        dyn = FR._delta_gn_dynamic_args_any(conv, x.device, is_int4)
        with torch.inference_mode():
            normed = M.group_norm_silu_nhwc(x, w, b, ng, eps, False, ms, sh)
            if is_int4:
                q_ref = M.step1_static_quantize_pack_int4_fprop_silu(normed, a_ref, scale, smooth)
                q_fus = M.group_norm_silu_delta_quantize_pack_nhwc(x, w, b, conv.a_hat_cache.clone(), ng, eps, True, scale, smooth, ms, sh, *dyn[:-1])
            else:
                q_ref = M.step1_static_quantize_fprop_silu(normed, a_ref, scale, smooth)
                q_fus = M.group_norm_silu_delta_quantize_nhwc(x, w, b, conv.a_hat_cache.clone(), ng, eps, True, scale, smooth, ms, sh, *dyn)
            torch.cuda.synchronize()
            cd = (q_ref.int() - q_fus.int()).abs().max().item()
            # a_ref now holds the reference's updated a_hat; recompute fused a_hat on a fresh clone
            a_fus = conv.a_hat_cache.clone()
            if is_int4:
                M.group_norm_silu_delta_quantize_pack_nhwc(x, w, b, a_fus, ng, eps, True, scale, smooth, ms, sh, *dyn[:-1])
            else:
                M.group_norm_silu_delta_quantize_nhwc(x, w, b, a_fus, ng, eps, True, scale, smooth, ms, sh, *dyn)
            torch.cuda.synchronize()
            ad = (a_ref.float() - a_fus.float()).abs().max().item()
        stats["calls"] += 1
        stats["max_code"] = max(stats["max_code"], cd)
        stats["max_ahat"] = max(stats["max_ahat"], ad)
    return orig(x, gn, conv, mod_scale, mod_shift, residual, x2=x2, **kw)
FR._prequant_gn_conv_modiff = wrap

torch.manual_seed(1234)
with torch.inference_mode(), torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
    sampler.sample(S=S, batch_size=BATCH, shape=r.shape, eta=0.0, verbose=False, **cond)
print(f"mode={mode}  instrumented_calls={stats['calls']}  "
      f"max_code_diff={stats['max_code']}  max_ahat_diff={stats['max_ahat']:.3e}")

#!/usr/bin/env python3
"""int4 attention proj fold validation (driver config). Builds one int4 model, freezes attn
self-calibration, then samples the SAME seed twice toggling MODIFF_FUSE_PROJ_I4 (read per-call):
0 = fallback (a.transpose().reshape() copy + proj's quantize_act_int4_pack), 1 = fold
(quantize_attn_out_int4_pack -> proj.forward_from_int4). Reports rel-L2 (expect small/bit-identical)
and proves engagement by counting quantize_attn_out_int4_pack (fold) vs quantize_act_int4_pack (fallback).
"""
import os, sys, importlib.util, torch
os.environ.update(MODIFF_QUANT_LINEAR="1", MODIFF_QUANT_ATTN="1", MODIFF_QUANT_ATTN_STATIC="1", MODIFF_LINEAR_OUT_I8="0")
os.environ.pop("MODIFF_FLASH_ATTN", None)
REPO = "/workspace/MoDiff"; sys.path.insert(0, REPO); sys.path.insert(0, REPO + "/src/taming-transformers"); os.chdir(REPO)
spec = importlib.util.spec_from_file_location("bldm", REPO + "/integration/benchmarks/benchmark_ldm.py")
bldm = importlib.util.module_from_spec(spec); spec.loader.exec_module(bldm)
import modiff_cutlass as _mc
STEPS, BATCH, SEED = 50, 4, 1234

cnt = {"attn_out_i4": 0, "act_i4": 0}
for attr, key in [("quantize_attn_out_int4_pack", "attn_out_i4"), ("quantize_act_int4_pack", "act_i4")]:
    o = getattr(_mc, attr)
    def mk(o, k):
        def w(*a, **kw):
            cnt[k] += 1; return o(*a, **kw)
        return w
    setattr(_mc, attr, mk(o, key))

r = bldm.BenchmarkRunner("configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
                         "models/ldm/lsun_churches256/model.ckpt", output_dir="/tmp/i4proj",
                         batch_size=BATCH, steps=STEPS, calibration_path="integration/calibration/int4_calibration.pt")
model, sampler = r._setup_model("int4")

def sample(fold):
    os.environ["MODIFF_FUSE_PROJ_I4"] = "1" if fold else "0"
    bldm.reset_modiff_state_int4(model.model.diffusion_model)
    if bldm.HAS_INT4_LINEAR: bldm.reset_modiff_state_int4_linear(model.model.diffusion_model)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    cnt.update(attn_out_i4=0, act_i4=0)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        out, _ = sampler.sample(S=STEPS, batch_size=BATCH, shape=r.shape, eta=0.0, verbose=False, **r._cond_kwargs(model, BATCH))
    return out.detach().float().cpu(), dict(cnt)

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):   # warmup: freeze attn self-calib
    sampler.sample(S=STEPS, batch_size=BATCH, shape=r.shape, eta=0.0, verbose=False, **r._cond_kwargs(model, BATCH))

ref, c_off = sample(False)
out, c_on = sample(True)
re = (out - ref).norm().item() / (ref.norm().item() + 1e-12)
print(f"[int4 proj fold] rel_L2_vs_fallback={re:.4e} max_abs={(out-ref).abs().max().item():.4e} "
      f"-> {'PASS' if re < 2e-2 else 'FAIL'}")
print(f"  fallback(off): attn_out_i4={c_off['attn_out_i4']} act_i4={c_off['act_i4']}")
print(f"  fold(on):      attn_out_i4={c_on['attn_out_i4']} act_i4={c_on['act_i4']}   "
      f"(expect attn_out_i4 rises, proj's act_i4 falls)")
sys.exit(0 if re < 2e-2 else 1)

"""Capture the REAL kernel-1 inputs, per layer per step, from a live sampling run.

Monkeypatches modiff_cutlass.group_norm_silu_delta_quantize_nhwc and records every argument
except a_hat -- x, GN weight/bias, num_groups, eps, apply_silu, the per-step delta scale,
smooth_inv, mod_scale/shift -- then calls through. So the replay in measure.py feeds the kernel
byte-identical inputs to what the model actually produced, and the ONLY thing that varies
between arms is a_hat storage.

Only the plain (non-resize, non-cat2) entry point is captured: 62 of the 70 conv layers. The 8
updown ResBlocks go through group_norm_silu_delta_quantize_resize_nhwc and are out of scope here.

Layer identity is the call index within a step, which is deterministic. A fixed set of target
shapes is captured across ALL 50 steps -- the trajectory is the point, so truncating steps would
defeat it, and capturing every layer at every step does not fit in RAM.
argv: int8|int4
"""
import os, sys, json
ROOT="/workspace/MoDiff"; os.chdir(ROOT)
sys.path[:0]=[ROOT, os.path.join(ROOT,"src/taming-transformers")]
PREC = sys.argv[1] if len(sys.argv) > 1 else "int8"
os.environ.update({"MODIFF_LINEAR":"0","MODIFF_CACHE_SKIP_K":"1","MODIFF_REPLAY_K":"1",
    "MODIFF_AHAT_BITS":"16","MODIFF_AHAT_REFRESH":"0","MODIFF_IMODE":"0",
    "MODIFF_DELTA_MODE":"static","MODIFF_CONV_BLOCKK":"0","MODIFF_ACT_BLOCK":"0",
    "MODIFF_AHAT_BLOCK":"0","MODIFF_AHAT_SIM_BITS":"0"})
import torch, modiff_cutlass as mc
import integration.benchmarks.benchmark_ldm as B

BATCH, STEPS, SHAPE = 4, 50, (4, 32, 32)
# span the space: CPG in {12,24,18,6,48}, spatial 4..32, so both vec4-eligible and not
TARGETS = {(384,16,16), (768,16,16), (576,32,32), (192,32,32), (1536,4,4)}
OUT = f"docs/ahat_accuracy_2026-09-02/data/capture_{PREC}.pt"

cap, state = {}, {"call": 0, "step": -1, "keep": {}}
_orig = mc.group_norm_silu_delta_quantize_nhwc

def hook(x, weight, bias, a_hat, num_groups, eps, apply_silu, scale, smooth_inv,
         mod_scale, mod_shift, *rest):
    i = state["call"]
    key = (int(x.size(1)), int(x.size(2)), int(x.size(3)))
    # One representative layer per target shape: the FIRST call index that shape appears at.
    # (The counter is reset per UNet forward by the wrapper below, so the index is stable.)
    if key in TARGETS and state["keep"].setdefault(key, i) == i and state["step"] >= 0:
        e = cap.setdefault((i, key), {"x": [], "s": [], "meta": None})
        if e["meta"] is None:
            e["meta"] = {"num_groups": int(num_groups), "eps": float(eps),
                         "apply_silu": bool(apply_silu),
                         "weight": weight.detach().float().cpu(),
                         "bias": bias.detach().float().cpu(),
                         "smooth_inv": (smooth_inv.detach().float().cpu()
                                        if smooth_inv.numel() else None)}
        e["x"].append(x.detach().to(torch.float16).cpu())
        e["s"].append(float(scale.reshape(-1)[0].item()))
        e.setdefault("mod", []).append(
            None if mod_scale.numel() == 0 else
            (mod_scale.detach().float().cpu(), mod_shift.detach().float().cpu()))
    state["call"] += 1
    return _orig(x, weight, bias, a_hat, num_groups, eps, apply_silu, scale, smooth_inv,
                 mod_scale, mod_shift, *rest)
mc.group_norm_silu_delta_quantize_nhwc = hook

r = B.BenchmarkRunner(config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
    ckpt_path="models/ldm/lsun_churches256/model.ckpt",
    output_dir="docs/ahat_accuracy_2026-09-02/data/tmp",
    batch_size=BATCH, steps=STEPS, shape=SHAPE,
    calibration_path=B._default_calibration_path(PREC), auto_delta_table=True)
m, s = r._setup_model(PREC)

# one clean trajectory: reset, then sample. calls_per_step is learned on the first step.
(B.reset_modiff_state_int8 if PREC == "int8" else B.reset_modiff_state_int4)(m.model.diffusion_model)
B._reset_wxax_modiff_safe(m)
# The call counter must restart every UNet forward -- one forward == one DDIM step -- or the
# running index makes every (layer, step) pair a distinct key. Learned the hard way.
_unet = m.model.diffusion_model
_unet_fwd = _unet.forward
def wrapped_forward(*a, **k):
    state["call"] = 0
    state["step"] += 1
    return _unet_fwd(*a, **k)
_unet.forward = wrapped_forward
state["step"] = -1
torch.manual_seed(1234)
with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
    s.sample(S=STEPS, batch_size=BATCH, shape=SHAPE, eta=0.0, verbose=False)
torch.cuda.synchronize()

# call index is a running counter over the whole run; recover (layer, step) by folding on
# the number of DISTINCT call indices seen for each shape.
out = {}
for (i, key), e in cap.items():
    out[f"{key[0]}x{key[1]}x{key[2]}_call{i}"] = {
        "C": key[0], "H": key[1], "W": key[2], "batch": BATCH,
        "x": torch.stack(e["x"]), "scale": e["s"], "meta": e["meta"],
        "mod": e["mod"]}
torch.save({"prec": PREC, "steps": STEPS, "layers": out}, OUT)
print("CAPJSON:" + json.dumps({"prec": PREC, "file": OUT, "n_layers": len(out),
    "steps_per_layer": {k: v["x"].shape[0] for k, v in out.items()},
    "MB": sum(v["x"].numel()*2 for v in out.values())/2**20}))

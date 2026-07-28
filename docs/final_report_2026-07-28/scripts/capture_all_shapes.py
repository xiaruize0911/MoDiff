"""Capture EVERY real kernel-input shape the UNet actually executes, by hooking the live
model rather than hand-listing shapes (a hand-list silently misses layers).

Emits data/all_shapes.json with, per op family, the deduplicated list of shapes plus how
many times each fires per denoising step -- so the per-kernel benchmark that follows can
cover all of them and weight them by real call count.
"""
import os, sys, json, collections
os.chdir("/workspace/MoDiff")
sys.path.insert(0, "/workspace/MoDiff")
sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
os.environ["MODIFF_QUANT_LINEAR"] = "1"
os.environ["MODIFF_QUANT_ATTN"] = "1"
import torch
import torch.nn as nn
import integration.benchmarks.benchmark_ldm as B

HERE = "docs/final_report_2026-07-28"
BATCH = 128
STEPS = 2   # 2 DDIM steps is enough: every layer fires once per step

rec = collections.defaultdict(collections.Counter)


def hook_conv(name):
    def f(mod, inp, out):
        x = inp[0]
        rec["conv"][(name, tuple(x.shape), mod.in_channels, mod.out_channels,
                     mod.kernel_size[0], mod.stride[0], mod.padding[0])] += 1
    return f


def hook_gn(name):
    def f(mod, inp, out):
        x = inp[0]
        ng = getattr(mod, "num_groups", None)
        rec["groupnorm"][(name, tuple(x.shape), ng)] += 1
    return f


def hook_linear(name):
    def f(mod, inp, out):
        x = inp[0]
        rec["linear"][(name, tuple(x.shape),
                       getattr(mod, "in_features", None),
                       getattr(mod, "out_features", None))] += 1
    return f


def main():
    calib = "integration/calibration/int4_calibration.pt"
    r = B.BenchmarkRunner("configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
                          "models/ldm/lsun_churches256/model.ckpt",
                          output_dir=f"{HERE}/tmp_out", batch_size=BATCH, steps=STEPS,
                          shape=(4, 32, 32), calibration_path=calib,
                          linear_backend="int_gemm")
    model, sampler = r._setup_model("int4_baseline")
    cond = r._cond_kwargs(model, BATCH)

    unet = model.model.diffusion_model
    handles = []
    for name, m in unet.named_modules():
        cls = type(m).__name__
        if isinstance(m, nn.Conv2d) or "Conv2d" in cls:
            if hasattr(m, "in_channels"):
                handles.append(m.register_forward_hook(hook_conv(name)))
        elif isinstance(m, nn.GroupNorm) or cls in ("FusedGroupNormSiLU",):
            handles.append(m.register_forward_hook(hook_gn(name)))
        elif isinstance(m, nn.Linear) or "Linear" in cls:
            if hasattr(m, "in_features") or hasattr(m, "out_features"):
                handles.append(m.register_forward_hook(hook_linear(name)))

    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        sampler.sample(S=STEPS, batch_size=BATCH, shape=r.shape, eta=0.0, verbose=False, **cond)
    for h in handles:
        h.remove()

    out = {}
    # --- conv: dedup by (shape, in, out, k, stride, pad); keep per-step call count
    conv = collections.Counter()
    for (name, shp, cin, cout, k, s, p), n in rec["conv"].items():
        conv[(shp, cin, cout, k, s, p)] += n
    out["conv"] = [
        dict(N=shp[0], C=cin, H=shp[2], W=shp[3], K=cout, k=k, stride=s, pad=p,
             calls_per_step=n / STEPS)
        for (shp, cin, cout, k, s, p), n in sorted(conv.items(), key=lambda kv: -kv[1])
    ]
    gn = collections.Counter()
    for (name, shp, ng), n in rec["groupnorm"].items():
        gn[(shp, ng)] += n
    out["groupnorm"] = [
        dict(N=shp[0], C=shp[1], H=shp[2], W=shp[3], num_groups=ng, calls_per_step=n / STEPS)
        for (shp, ng), n in sorted(gn.items(), key=lambda kv: -kv[1])
    ]
    lin = collections.Counter()
    for (name, shp, inf, outf), n in rec["linear"].items():
        lin[(shp, inf, outf)] += n
    out["linear"] = [
        dict(shape=list(shp), in_features=inf, out_features=outf, calls_per_step=n / STEPS)
        for (shp, inf, outf), n in sorted(lin.items(), key=lambda kv: -kv[1])
    ]

    with open(f"{HERE}/data/all_shapes.json", "w") as f:
        json.dump(out, f, indent=2)
    for fam, items in out.items():
        print(f"{fam}: {len(items)} distinct shapes, "
              f"{sum(i['calls_per_step'] for i in items):.0f} calls/step")
    print(f"WROTE {HERE}/data/all_shapes.json")


if __name__ == "__main__":
    main()

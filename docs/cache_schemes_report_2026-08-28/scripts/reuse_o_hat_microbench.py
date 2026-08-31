"""reuse_o_hat kernel vs full MoDiff conv, conv-layer only.

Same protocol as conv_layer_microbench.py / one_layer_200.py:
  - one layer: 192->192, 32x32, batch 128, 200 CUDA-event steps (no skip-add)
  - conv set: 20 UNet shapes, freq-weighted, independent L=8 chain

reuse_o_hat      = copy stored o_hat (the conv result)
reuse_o_hat_add  = o_hat + live residual (ResBlock replay)
aten_add         = torch.add(o_hat, residual)  (what Python _replay_out does today)

Skip and replay are never mixed. W8A8, fp16 a_hat.

Run: source setup_cuda_env.sh && python docs/cache_schemes_report_2026-08-28/scripts/reuse_o_hat_microbench.py
"""
import json
import os
import statistics
import sys

ROOT = "/workspace/MoDiff"
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "build/lib.linux-x86_64-cpython-311")]

import torch
import modiff_cutlass as mc

torch.manual_seed(0)
N, G, EPS = 128, 32, 1e-5
L, REPS, WARMUP, TRIALS = 8, 30, 4, 3
ONE_STEPS = 200
CL = torch.channels_last
DEV = "cuda"

SHAPES = [
    (768, 768, 2, 2, 12), (384, 384, 8, 8, 8), (192, 192, 32, 32, 7), (384, 384, 16, 16, 7),
    (768, 768, 4, 4, 7), (1536, 768, 2, 2, 3), (1536, 768, 4, 4, 2), (768, 384, 8, 8, 2),
    (768, 384, 16, 16, 2), (384, 192, 32, 32, 2), (192, 192, 16, 16, 1), (192, 384, 16, 16, 1),
    (384, 384, 4, 4, 1), (384, 768, 4, 4, 1), (1152, 768, 4, 4, 1), (768, 768, 8, 8, 1),
    (1152, 384, 8, 8, 1), (576, 384, 16, 16, 1), (384, 384, 32, 32, 1), (576, 192, 32, 32, 1),
]

E32 = torch.empty(0, device=DEV, dtype=torch.float32)
E16 = torch.empty(0, device=DEV, dtype=torch.float16)
EI = torch.empty(0, device=DEV, dtype=torch.int32)


def cl(t):
    return t.contiguous(memory_format=CL)


def make_layer(Cin, Cout, H, W):
    x = cl(torch.randn(N, Cin, H, W, device=DEV, dtype=torch.float16))
    gamma = torch.randn(Cin, device=DEV, dtype=torch.float16).abs() + 0.5
    beta = torch.randn(Cin, device=DEV, dtype=torch.float16) * 0.1
    scale = torch.tensor([16.0], device=DEV, dtype=torch.float32)
    inv_scale = torch.tensor([1.0 / 16.0], device=DEV, dtype=torch.float32)
    w = torch.randint(-8, 8, (Cout, 3, 3, Cin), device=DEV, dtype=torch.int8).contiguous()
    wscale = torch.full((Cout,), 0.02, device=DEV, dtype=torch.float32)
    o_hat = cl(0.1 * torch.randn(N, Cout, H, W, device=DEV, dtype=torch.float16))
    residual = cl(0.1 * torch.randn(N, Cout, H, W, device=DEV, dtype=torch.float16))
    out = cl(torch.empty(N, Cout, H, W, device=DEV, dtype=torch.float16))
    a_hat = cl(0.1 * torch.randn(N, Cin, H, W, device=DEV, dtype=torch.float16))
    ahat_scale = torch.empty(0, device=DEV, dtype=torch.float32)
    return dict(x=x, gamma=gamma, beta=beta, scale=scale, inv_scale=inv_scale,
                a_hat=a_hat, ahat_scale=ahat_scale, w=w, wscale=wscale,
                o_hat=o_hat, residual=residual, out=out)


def gn(lyr, write=True):
    return mc.group_norm_silu_delta_quantize_nhwc(
        lyr["x"], lyr["gamma"], lyr["beta"], lyr["a_hat"], G, EPS, True,
        lyr["scale"], E32, E16, E16, E32, E32, E32, EI, 127.0, False, 1.0,
        False, write, lyr["ahat_scale"])


def call_full(lyr):
    xq = gn(lyr, True)
    mc.conv2d_int8_evt_o_hat(xq, lyr["w"], lyr["inv_scale"], lyr["wscale"],
                             lyr["o_hat"], 1, 1, 1, 1, 1, 1)


def call_reuse(lyr):
    mc.reuse_o_hat(lyr["o_hat"], lyr["out"])


def call_reuse_add(lyr):
    mc.reuse_o_hat_add(lyr["o_hat"], lyr["residual"], lyr["out"])


def call_aten_add(lyr):
    torch.add(lyr["o_hat"], lyr["residual"], out=lyr["out"])


def mix(commit, other, k):
    if k <= 1:
        return commit
    return (commit + (k - 1) * other) / k


def time_chain(Cin, Cout, H, W, fn):
    layers = [make_layer(Cin, Cout, H, W) for _ in range(L)]

    def chain():
        for lyr in layers:
            fn(lyr)

    for _ in range(WARMUP):
        chain()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(REPS):
        chain()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / (L * REPS)


def time_one_layer(fn, steps=ONE_STEPS):
    lyr = make_layer(192, 192, 32, 32)
    for _ in range(WARMUP):
        fn(lyr)
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(steps):
        fn(lyr)
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / steps


def check_correct():
    lyr = make_layer(192, 192, 32, 32)
    mc.reuse_o_hat(lyr["o_hat"], lyr["out"])
    torch.cuda.synchronize()
    copy_ok = torch.equal(lyr["out"], lyr["o_hat"])
    ref = lyr["o_hat"] + lyr["residual"]
    mc.reuse_o_hat_add(lyr["o_hat"], lyr["residual"], lyr["out"])
    torch.cuda.synchronize()
    add_ok = torch.allclose(lyr["out"], ref, rtol=0, atol=0)
    print(f"correctness  copy={copy_ok}  add_vs_aten={add_ok}", flush=True)
    if not copy_ok or not add_ok:
        raise SystemExit("reuse_o_hat correctness failed")


def main():
    print(f"GPU {torch.cuda.get_device_name(0)}  N={N}  reuse_o_hat kernel", flush=True)
    check_correct()

    prims = [
        ("full", call_full),
        ("reuse_o_hat", call_reuse),
        ("reuse_o_hat_add", call_reuse_add),
        ("aten_add", call_aten_add),
    ]

    print("\n===== one layer 192->192 32x32  200 step =====", flush=True)
    one = {}
    for name, fn in prims:
        samples = [time_one_layer(fn) for _ in range(TRIALS)]
        one[name] = statistics.median(samples)
        print(f"  {name:18s} {one[name]:.4f} ms/step  {samples}", flush=True)
    ref1 = one["full"]
    one_arms = []
    for name in ("full", "reuse_o_hat", "reuse_o_hat_add", "aten_add"):
        one_arms.append({"label": name, "ms_step": one[name],
                         "vs_full": ref1 / one[name] if one[name] else 0.0})
    for k in (2, 4, 8):
        for other in ("reuse_o_hat", "reuse_o_hat_add"):
            ms = mix(one["full"], one[other], k)
            one_arms.append({"label": f"K={k} mix {other}", "ms_step": ms,
                             "vs_full": ref1 / ms})
    for a in one_arms:
        print(f"  {a['label']:28s} {a['ms_step']:7.4f}  {a['vs_full']:.3f}x", flush=True)

    print("\n===== conv set 20 shapes L=8 =====", flush=True)
    prim_w = {name: 0.0 for name, _ in prims}
    per_shape = []
    for Cin, Cout, H, W, freq in SHAPES:
        med = {}
        for name, fn in prims:
            samples = [time_chain(Cin, Cout, H, W, fn) for _ in range(TRIALS)]
            med[name] = statistics.median(samples)
            prim_w[name] += med[name] * freq
        per_shape.append({"shape": f"{Cin}->{Cout},{H}x{W}", "freq": freq, "ms": med})
        print(f"{Cin}->{Cout},{H}x{W:>2} f{freq:<3}  "
              f"full {med['full']:.3f}  reuse {med['reuse_o_hat']:.3f}  "
              f"add {med['reuse_o_hat_add']:.3f}",
              flush=True)

    ref = prim_w["full"]
    set_arms = []
    for label, key in (("W8A8 full fp16", "full"),
                       ("reuse_o_hat (copy)", "reuse_o_hat"),
                       ("reuse_o_hat_add", "reuse_o_hat_add"),
                       ("aten add (Python replay)", "aten_add")):
        set_arms.append({"label": label, "ms_step": prim_w[key],
                         "vs_full": ref / prim_w[key] if prim_w[key] else 0.0})
    for k in (2, 4, 8):
        for other, tag in (("reuse_o_hat", "copy"), ("reuse_o_hat_add", "add")):
            ms = mix(prim_w["full"], prim_w[other], k)
            set_arms.append({"label": f"replay-K={k} {tag}", "ms_step": ms,
                             "vs_full": ref / ms})

    print("\n===== conv-set freq-weighted ms/step vs full =====", flush=True)
    for a in set_arms:
        print(f"  {a['label']:32s} {a['ms_step']:7.3f}  {a['vs_full']:.3f}x", flush=True)

    out = {
        "gpu": torch.cuda.get_device_name(0),
        "batch": N,
        "note": "reuse_o_hat kernel copies stored o_hat; add variant is live skip. "
                "one-layer matches one_layer_200.py shape; conv-set matches conv_layer_microbench.py.",
        "one_layer": {"shape": "192->192,32x32", "steps": ONE_STEPS, "ms": one, "arms": one_arms},
        "conv_set": {"primitives_ms_step": prim_w, "arms": set_arms, "per_shape": per_shape},
    }
    path = "docs/cache_schemes_report_2026-08-28/data/reuse_o_hat_microbench.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()

"""Isolated conv-layer microbench for every speed arm in the cache-schemes report.

No DDIM / UNet / attention. Each call is one OptimizedInt8/Int4Conv2d dispatch:
fused GN+SiLU+delta-quantize then the EVT o_hat conv. Mixed-K schedules are
composed from commit vs skip/replay primitives: (1/K)*commit + ((K-1)/K)*other.

Skip and replay are never mixed. Refresh unpacks to fp16, runs the fp16-a_hat
kernels, then packs with a fresh absmax (only on commit).

Run: source setup_cuda_env.sh && python docs/cache_schemes_report_2026-08-28/scripts/conv_layer_microbench.py
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


def pack4(codes):
    c = codes.permute(0, 2, 3, 1).contiguous().to(torch.int64) & 0x0F
    lo, hi = c[..., 0::2], c[..., 1::2]
    v = lo | (hi << 4)
    return (v - 256 * (v > 127)).to(torch.int8).contiguous()


def make_layer(Cin, Cout, H, W, prec, ahat):
    """ahat: 'fp16' | 'i8' | 'i4' (i4 = unpacked int8 bytes, qmax=7)."""
    x = cl(torch.randn(N, Cin, H, W, device=DEV, dtype=torch.float16))
    gamma = torch.randn(Cin, device=DEV, dtype=torch.float16).abs() + 0.5
    beta = torch.randn(Cin, device=DEV, dtype=torch.float16) * 0.1
    qmax = 7.0 if ahat == "i4" else 127.0
    if prec == "W8A8":
        scale = torch.tensor([16.0], device=DEV, dtype=torch.float32)
        inv_scale = torch.tensor([1.0 / 16.0], device=DEV, dtype=torch.float32)
        w = torch.randint(-8, 8, (Cout, 3, 3, Cin), device=DEV, dtype=torch.int8).contiguous()
    else:
        scale = torch.tensor([3.0], device=DEV, dtype=torch.float32)
        inv_scale = torch.tensor([1.0 / 3.0], device=DEV, dtype=torch.float32)
        w_raw = torch.randint(-7, 8, (Cout, Cin, 3, 3), device=DEV, dtype=torch.int64)
        w = pack4(w_raw)
    wscale = torch.full((Cout,), 0.02, device=DEV, dtype=torch.float32)
    o_hat = cl(0.1 * torch.randn(N, Cout, H, W, device=DEV, dtype=torch.float16))
    residual = cl(0.1 * torch.randn(N, Cout, H, W, device=DEV, dtype=torch.float16))
    out = cl(torch.empty(N, Cout, H, W, device=DEV, dtype=torch.float16))
    replay_buf = cl(torch.empty(N, Cout, H, W, device=DEV, dtype=torch.float16))
    if ahat == "fp16":
        a_hat = cl(0.1 * torch.randn(N, Cin, H, W, device=DEV, dtype=torch.float16))
        ahat_scale = torch.empty(0, device=DEV, dtype=torch.float32)
        a_hat_fp16 = a_hat
    elif ahat == "i16":
        a_hat = cl(torch.randint(-1000, 1001, (N, Cin, H, W), device=DEV, dtype=torch.int16))
        ahat_scale = torch.tensor([0.0, 32767.0], device=DEV, dtype=torch.float32)
        a_hat_fp16 = cl(torch.empty(N, Cin, H, W, device=DEV, dtype=torch.float16))
        qmax = 32767.0
    else:
        imode = ahat.endswith("m")  # i8m / i4m
        qmax = 7.0 if "i4" in ahat else 127.0
        a_hat = cl(torch.randint(-int(qmax // 3), int(qmax // 3) + 1,
                                 (N, Cin, H, W), device=DEV, dtype=torch.int8))
        scale0 = 0.0 if imode else 0.02
        ahat_scale = torch.tensor([scale0, qmax], device=DEV, dtype=torch.float32)
        a_hat_fp16 = cl(torch.empty(N, Cin, H, W, device=DEV, dtype=torch.float16))
    return dict(prec=prec, ahat=ahat, qmax=qmax, x=x, gamma=gamma, beta=beta,
                scale=scale, inv_scale=inv_scale, a_hat=a_hat, ahat_scale=ahat_scale,
                a_hat_fp16=a_hat_fp16, w=w, wscale=wscale, o_hat=o_hat,
                residual=residual, out=out, replay_buf=replay_buf)


def gn(lyr, a_hat, write, ahat_scale):
    if lyr["prec"] == "W8A8":
        return mc.group_norm_silu_delta_quantize_nhwc(
            lyr["x"], lyr["gamma"], lyr["beta"], a_hat, G, EPS, True,
            lyr["scale"], E32, E16, E16, E32, E32, E32, EI, 127.0, False, 1.0,
            False, write, ahat_scale)
    q = 7.0
    return mc.group_norm_silu_delta_quantize_pack_nhwc(
        lyr["x"], lyr["gamma"], lyr["beta"], a_hat, G, EPS, True,
        lyr["scale"], E32, E16, E16, E32, E32, E32, EI, q, False, 1.0,
        write, ahat_scale)


def conv_full(lyr, xq):
    fn = mc.conv2d_int8_evt_o_hat if lyr["prec"] == "W8A8" else mc.conv2d_int4_evt_o_hat
    fn(xq, lyr["w"], lyr["inv_scale"], lyr["wscale"], lyr["o_hat"], 1, 1, 1, 1, 1, 1)


def conv_skip(lyr, xq):
    fn = mc.conv2d_int8_evt_o_hat_skip if lyr["prec"] == "W8A8" else mc.conv2d_int4_evt_o_hat_skip
    fn(xq, lyr["w"], lyr["inv_scale"], lyr["wscale"], lyr["o_hat"], lyr["out"], 1, 1, 1, 1, 1, 1)


def unpack(lyr):
    s = lyr["ahat_scale"][0]
    lyr["a_hat_fp16"].copy_((lyr["a_hat"].float() * s).to(torch.float16))


def pack(lyr):
    a = lyr["a_hat_fp16"].float()
    qmax = lyr["qmax"]
    amax = a.abs().amax().clamp_min(1e-6)
    lyr["ahat_scale"][0] = amax / qmax
    lyr["a_hat"].copy_(a.mul(qmax / amax).round_().clamp_(-qmax, qmax).to(torch.int8))


def call_full(lyr):
    conv_full(lyr, gn(lyr, lyr["a_hat"], True, lyr["ahat_scale"]))


def call_skip(lyr):
    conv_skip(lyr, gn(lyr, lyr["a_hat"], False, lyr["ahat_scale"]))


def call_replay(lyr):
    torch.add(lyr["o_hat"], lyr["residual"], out=lyr["replay_buf"])


def call_refresh(lyr):
    unpack(lyr)
    empty = torch.empty(0, device=DEV, dtype=torch.float32)
    conv_full(lyr, gn(lyr, lyr["a_hat_fp16"], True, empty))
    pack(lyr)


# (name, prec, ahat, fn) — timed once per shape; report arms are composed from these.
PRIMITIVES = [
    ("w8_full_fp16", "W8A8", "fp16", call_full),
    ("w8_skip_fp16", "W8A8", "fp16", call_skip),
    ("w8_full_i8", "W8A8", "i8", call_full),
    ("w8_skip_i8", "W8A8", "i8", call_skip),
    ("w8_full_i4", "W8A8", "i4", call_full),
    ("w8_skip_i4", "W8A8", "i4", call_skip),
    ("w8_full_im16", "W8A8", "i16", call_full),
    ("w8_full_im8", "W8A8", "i8m", call_full),
    ("w8_full_im4", "W8A8", "i4m", call_full),
    ("w8_replay", "W8A8", "fp16", call_replay),
    ("w8_refresh_i8", "W8A8", "i8", call_refresh),
    ("w8_refresh_i4", "W8A8", "i4", call_refresh),
    ("w4_full_fp16", "W4A4", "fp16", call_full),
    ("w4_skip_fp16", "W4A4", "fp16", call_skip),
    ("w4_full_i4", "W4A4", "i4", call_full),
    ("w4_skip_i4", "W4A4", "i4", call_skip),
    ("w4_replay", "W4A4", "fp16", call_replay),
    ("w4_refresh_i4", "W4A4", "i4", call_refresh),
]


def mix(commit, other, k):
    if k <= 1:
        return commit
    return (commit + (k - 1) * other) / k


def time_chain(Cin, Cout, H, W, prec, ahat, fn):
    layers = [make_layer(Cin, Cout, H, W, prec, ahat) for _ in range(L)]

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


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--imode", action="store_true",
                    help="Only time W8A8 full fp16 vs I-MoDiff int16/8/4; write imode.json")
    a = ap.parse_args()
    prims = ([p for p in PRIMITIVES if p[0] in
              ("w8_full_fp16", "w8_full_im16", "w8_full_im8", "w8_full_im4")]
             if a.imode else PRIMITIVES)
    print(f"GPU {torch.cuda.get_device_name(0)}  N={N}  L={L} REPS={REPS} TRIALS={TRIALS}"
          f"{'  IMODE-only' if a.imode else ''}",
          flush=True)
    prim_w = {name: 0.0 for name, *_ in prims}
    per_shape = []
    names = [p[0] for p in prims]
    spec = {p[0]: p for p in prims}
    for Cin, Cout, H, W, freq in SHAPES:
        samples = {name: [] for name in names}
        for t in range(TRIALS):
            order = names[t % len(names):] + names[:t % len(names)]
            for name in order:
                _, prec, ahat, fn = spec[name]
                samples[name].append(time_chain(Cin, Cout, H, W, prec, ahat, fn))
        med = {n: statistics.median(v) for n, v in samples.items()}
        for n, v in med.items():
            prim_w[n] += v * freq
        per_shape.append({"shape": f"{Cin}->{Cout},{H}x{W}", "freq": freq, "ms": med})
        if a.imode:
            print(f"{Cin}->{Cout},{H}x{W:>2} f{freq:<3}  "
                  f"fp16 {med['w8_full_fp16']:.3f}  "
                  f"im16 {med['w8_full_im16']:.3f}  "
                  f"im8 {med['w8_full_im8']:.3f}  "
                  f"im4 {med['w8_full_im4']:.3f}",
                  flush=True)
        else:
            print(f"{Cin}->{Cout},{H}x{W:>2} f{freq:<3}  "
                  f"w8 {med['w8_full_fp16']:.3f}/{med['w8_skip_fp16']:.3f}/{med['w8_replay']:.3f}  "
                  f"w4 {med['w4_full_fp16']:.3f}/{med['w4_replay']:.3f}",
                  flush=True)

    p = prim_w
    ref = p["w8_full_fp16"]

    if a.imode:
        rows = []
        for key, label in (("w8_full_fp16", "full_fp16"),
                           ("w8_full_im16", "imode16"),
                           ("w8_full_im8", "imode8"),
                           ("w8_full_im4", "imode4")):
            rows.append({"label": label, "ms_step": p[key],
                         "vs_w8_full": ref / p[key] if p[key] > 0 else 0.0})
            print(f"  {label:16s} {p[key]:7.3f}  {ref / p[key]:.3f}x", flush=True)
        imode_path = "docs/cache_schemes_report_2026-08-28/data/imode.json"
        prev = json.load(open(imode_path)) if os.path.exists(imode_path) else {}
        prev["conv_layer"] = rows
        json.dump(prev, open(imode_path, "w"), indent=1)
        print("wrote", imode_path, "[conv_layer]", flush=True)
        return 0

    def arm(section, label, ms):
        return {"section": section, "label": label, "ms_step": ms,
                "vs_w8_full": ref / ms if ms > 0 else 0.0}

    arms = []
    # --- 1. Skip ---
    for k in (1, 2, 4, 8, 16, 32):
        ms = mix(p["w8_full_fp16"], p["w8_skip_fp16"], k)
        arms.append(arm("skip", f"W8A8 skip-K={k} fp16" if k > 1 else "W8A8 full fp16", ms))
    for k in (1, 2, 4, 8):
        ms = mix(p["w4_full_fp16"], p["w4_skip_fp16"], k)
        arms.append(arm("skip", f"W4A4 skip-K={k} fp16" if k > 1 else "W4A4 full fp16", ms))
    # --- 2. Replay ---
    for k in (1, 2, 4, 8):
        arms.append(arm("replay", f"W8A8 replay-K={k}", mix(p["w8_full_fp16"], p["w8_replay"], k)))
    for k in (1, 2, 4, 8):
        arms.append(arm("replay", f"W4A4 replay-K={k}", mix(p["w4_full_fp16"], p["w4_replay"], k)))
    # --- 3. Quant, full step ---
    arms += [
        arm("quant", "W8A8 a_hat fp16", p["w8_full_fp16"]),
        arm("quant", "W8A8 a_hat int8 held", p["w8_full_i8"]),
        arm("quant", "W8A8 a_hat int8 refresh", p["w8_refresh_i8"]),
        arm("quant", "W8A8 a_hat int4 held", p["w8_full_i4"]),
        arm("quant", "W8A8 a_hat int4 refresh", p["w8_refresh_i4"]),
        arm("quant", "W8A8 I-MoDiff int16", p["w8_full_im16"]),
        arm("quant", "W8A8 I-MoDiff int8", p["w8_full_im8"]),
        arm("quant", "W8A8 I-MoDiff int4", p["w8_full_im4"]),
        arm("quant", "W4A4 a_hat fp16", p["w4_full_fp16"]),
        arm("quant", "W4A4 a_hat int4 held", p["w4_full_i4"]),
        arm("quant", "W4A4 a_hat int4 refresh", p["w4_refresh_i4"]),
    ]
    # --- 4. Combinations ---
    arms += [
        arm("combo", "W8A8 full fp16", p["w8_full_fp16"]),
        arm("combo", "W8A8 full int8 held", p["w8_full_i8"]),
        arm("combo", "W8A8 full int4 held", p["w8_full_i4"]),
        arm("combo", "W8A8 skip-K=4 fp16", mix(p["w8_full_fp16"], p["w8_skip_fp16"], 4)),
        arm("combo", "W8A8 skip-K=4 int8 held", mix(p["w8_full_i8"], p["w8_skip_i8"], 4)),
        arm("combo", "W8A8 skip-K=4 int4 held", mix(p["w8_full_i4"], p["w8_skip_i4"], 4)),
        arm("combo", "W8A8 replay-K=4 fp16", mix(p["w8_full_fp16"], p["w8_replay"], 4)),
        arm("combo", "W8A8 replay-K=4 int8 held", mix(p["w8_full_i8"], p["w8_replay"], 4)),
        arm("combo", "W8A8 replay-K=4 int4 held", mix(p["w8_full_i4"], p["w8_replay"], 4)),
        arm("combo", "W4A4 full fp16 a_hat", p["w4_full_fp16"]),
        arm("combo", "W4A4 full int4 held", p["w4_full_i4"]),
        arm("combo", "W4A4 skip-K=4 fp16", mix(p["w4_full_fp16"], p["w4_skip_fp16"], 4)),
        arm("combo", "W4A4 skip-K=4 int4 held", mix(p["w4_full_i4"], p["w4_skip_i4"], 4)),
        arm("combo", "W4A4 replay-K=4 fp16", mix(p["w4_full_fp16"], p["w4_replay"], 4)),
        arm("combo", "W4A4 replay-K=4 int4 held", mix(p["w4_full_i4"], p["w4_replay"], 4)),
    ]

    out = {
        "gpu": torch.cuda.get_device_name(0),
        "batch": N,
        "method": "independent L=8 chain, CUDA events, median of 3 rotated trials",
        "unit": "freq-weighted ms/step over 20 UNet conv shapes (conv path only)",
        "primitives_ms_step": p,
        "arms": arms,
        "per_shape": per_shape,
    }
    path = "docs/cache_schemes_report_2026-08-28/data/conv_layer_microbench.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    print("\n===== conv-only ms/step vs W8A8 full fp16 =====", flush=True)
    cur = None
    for a in arms:
        if a["section"] != cur:
            cur = a["section"]
            print(f"\n-- {cur} --", flush=True)
        print(f"  {a['label']:36s} {a['ms_step']:7.3f}  {a['vs_w8_full']:.3f}x", flush=True)
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()

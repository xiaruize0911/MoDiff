"""Re-verification of bench_pipeline.py with the CONTROL it was missing.

bench_pipeline.py showed 768,4x4 netting +28% from batch-split 2-stream pipelining -- but it never
pipelined the BASELINE. Without that control the result is unattributable: pipelining might simply be
a general occupancy win that helps every config equally, in which case a_hat is not being "hidden" at
all and MoDiff's overhead ratio stays exactly where it was.

So: 6 configs, and the load-bearing number is whether MoDiff's OVERHEAD RATIO shrinks.

    base_full     modiff_full      -> overhead today       = modiff_full / base_full
    base_split    modiff_split     -> split penalty, both arms
    base_pipe     modiff_pipe      -> overhead pipelined   = modiff_pipe / base_pipe

  a_hat is genuinely hidden  <=>  (modiff_pipe / base_pipe) < (modiff_full / base_full)
  a general occupancy win    <=>  the two ratios are equal and both arms just got faster

5 trials, order rotated each trial to cancel systematic GPU drift (the methodology that caught the
sign-flipping o_hat reading earlier in this session).
"""
import os
import sys
import time
import statistics
import json

ROOT = "/workspace/MoDiff"
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src/taming-transformers"))

import torch
import torch.nn as nn
import modiff_cutlass

torch.manual_seed(0)

N_STAGES = 6
REPS = 20
WARMUP_REPS = 5
N_TRIALS = 5
NUM_GROUPS = 32
EPS = 1e-5

EMPTY_SMOOTH = torch.empty(0, device="cuda", dtype=torch.float32)
EMPTY_MOD = torch.empty(0, device="cuda", dtype=torch.float16)
EMPTY_F32 = torch.empty(0, device="cuda", dtype=torch.float32)
EMPTY_BIAS = torch.empty(0, device="cuda", dtype=torch.float32)
EMPTY_RES = torch.empty(0, device="cuda", dtype=torch.float16)

SHAPES = [
    (128, 768, 2, 2, 12),
    (128, 384, 8, 8, 8),
    (128, 192, 32, 32, 7),
    (128, 384, 16, 16, 7),
    (128, 768, 4, 4, 7),
]


def make_stage(N, C, H, W):
    x = torch.randn(N, C, H, W, device="cuda", dtype=torch.float16).to(
        memory_format=torch.channels_last)
    a_hat = (torch.randn(N, C, H, W, device="cuda", dtype=torch.float16) * 0.1).to(
        memory_format=torch.channels_last)
    w_conv = nn.Conv2d(C, C, 3, padding=1, bias=False).cuda()
    w_data = w_conv.weight.data
    w_flat = w_data.reshape(C, -1)
    ch_scale = torch.clamp(w_flat.abs().max(dim=1).values / 127.0, min=1e-8)
    w_q = (w_flat / ch_scale.unsqueeze(1)).round().clamp(-127, 127).to(torch.int8)
    w_q = w_q.reshape_as(w_data).permute(0, 2, 3, 1).contiguous()
    x_int8 = (x.float() / 8.0).round().clamp(-127, 127).to(torch.int8).contiguous(
        memory_format=torch.channels_last)
    o_hat = torch.randn(N, C, H, W, device="cuda", dtype=torch.float16).to(
        memory_format=torch.channels_last)
    return dict(
        x=x, a_hat=a_hat,
        gw=torch.randn(C, device="cuda", dtype=torch.float16),
        gb=torch.randn(C, device="cuda", dtype=torch.float16),
        scale=torch.tensor([127.0], device="cuda"),
        wq=w_q, ws=ch_scale.contiguous(),
        alpha=torch.tensor([1.0 / 16.0], device="cuda"),
        x_int8=x_int8, o_hat=o_hat, out=torch.empty_like(o_hat),
    )


def gn_base(S):
    modiff_cutlass.group_norm_silu_quantize_nhwc_fast(
        S["x"], S["gw"], S["gb"], NUM_GROUPS, EPS, True, S["scale"],
        EMPTY_SMOOTH, EMPTY_MOD, EMPTY_MOD)


def gn_modiff(S):
    modiff_cutlass.group_norm_silu_delta_quantize_nhwc(
        S["x"], S["gw"], S["gb"], S["a_hat"], NUM_GROUPS, EPS, True, S["scale"],
        EMPTY_SMOOTH, EMPTY_MOD, EMPTY_MOD,
        EMPTY_F32, EMPTY_F32, EMPTY_F32, EMPTY_F32, 127.0, False, 1.0, False)


def conv_base(S):
    modiff_cutlass.conv2d_int8_evt_bias_residual_fp16(
        S["x_int8"], S["wq"], S["alpha"], S["ws"], EMPTY_BIAS, EMPTY_RES, S["out"],
        1, 1, 1, 1, 1, 1)


def conv_modiff(S):
    modiff_cutlass.conv2d_int8_evt_o_hat(
        S["x_int8"], S["wq"], S["alpha"], S["ws"], S["o_hat"], 1, 1, 1, 1, 1, 1)


def run_single(stages, gn, conv):
    for S in stages:
        gn(S)
        conv(S)


def run_split_serial(sa, sb, gn, conv):
    for S in sa:
        gn(S)
        conv(S)
    for S in sb:
        gn(S)
        conv(S)


def run_pipe(sa, sb, gn, conv, s1, s2):
    ev_gate = torch.cuda.Event()
    ev_gate.record()
    s1.wait_event(ev_gate)
    s2.wait_event(ev_gate)
    ev_offset = torch.cuda.Event()
    with torch.cuda.stream(s1):
        gn(sa[0])
        ev_offset.record(s1)
        conv(sa[0])
        for S in sa[1:]:
            gn(S)
            conv(S)
    with torch.cuda.stream(s2):
        s2.wait_event(ev_offset)
        for S in sb:
            gn(S)
            conv(S)
    e1, e2 = torch.cuda.Event(), torch.cuda.Event()
    e1.record(s1)
    e2.record(s2)
    torch.cuda.current_stream().wait_event(e1)
    torch.cuda.current_stream().wait_event(e2)


def timeit(launch):
    for _ in range(WARMUP_REPS):
        launch()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(REPS):
        launch()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3 / REPS


def bench_shape(N, C, H, W):
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
    full = [make_stage(N, C, H, W) for _ in range(N_STAGES)]
    ha = [make_stage(N // 2, C, H, W) for _ in range(N_STAGES)]
    hb = [make_stage(N // 2, C, H, W) for _ in range(N_STAGES)]

    names = ["base_full", "base_split", "base_pipe", "modiff_full", "modiff_split", "modiff_pipe"]
    fns = {
        "base_full":    lambda: run_single(full, gn_base, conv_base),
        "base_split":   lambda: run_split_serial(ha, hb, gn_base, conv_base),
        "base_pipe":    lambda: run_pipe(ha, hb, gn_base, conv_base, s1, s2),
        "modiff_full":  lambda: run_single(full, gn_modiff, conv_modiff),
        "modiff_split": lambda: run_split_serial(ha, hb, gn_modiff, conv_modiff),
        "modiff_pipe":  lambda: run_pipe(ha, hb, gn_modiff, conv_modiff, s1, s2),
    }
    res = {k: [] for k in names}
    for trial in range(N_TRIALS):
        rot = names[trial % len(names):] + names[:trial % len(names)]   # rotate order every trial
        for name in rot:
            res[name].append(timeit(fns[name]))

    del full, ha, hb
    torch.cuda.empty_cache()
    return {k: statistics.median(v) for k, v in res.items()}, {k: statistics.stdev(v) for k, v in res.items()}


print(f"CONTROLLED pipelining test: {N_STAGES} (GN->conv) stages/chain, {REPS} reps/trial, "
      f"{N_TRIALS} trials (order rotated)\n")
hdr = (f"{'shape':14}{'fq':>3} | {'base_full':>10}{'base_pipe':>10}{'mod_full':>10}{'mod_pipe':>10} | "
       f"{'ovh today':>10}{'ovh piped':>10} | {'verdict':>22}")
print(hdr)
print("-" * len(hdr))

out = {}
for (N, C, H, W, freq) in SHAPES:
    r, sd = bench_shape(N, C, H, W)
    bf, bp, mf, mp = r["base_full"], r["base_pipe"], r["modiff_full"], r["modiff_pipe"]
    ovh_today = 100 * (mf - bf) / bf
    ovh_piped = 100 * (mp - bp) / bp
    base_gain = 100 * (bf - bp) / bf
    mod_gain = 100 * (mf - mp) / mf
    if ovh_piped < ovh_today - 2.0:
        v = "a_hat HIDDEN"
    elif abs(ovh_piped - ovh_today) <= 2.0:
        v = "general win, not a_hat"
    else:
        v = "WORSE piped"
    out[f"{C}_{H}x{W}"] = dict(freq=freq, med=r, sd=sd,
                               ovh_today=ovh_today, ovh_piped=ovh_piped,
                               base_gain=base_gain, mod_gain=mod_gain)
    print(f"{f'{C},{H}x{W}':14}{freq:3d} | {bf:10.3f}{bp:10.3f}{mf:10.3f}{mp:10.3f} | "
          f"{ovh_today:9.1f}%{ovh_piped:9.1f}% | {v:>22}")

print("\nper-arm gain from pipelining (how much each arm sped up on its own):")
print(f"{'shape':14}{'base gain':>11}{'modiff gain':>13}{'split penalty (mod)':>21}")
for (N, C, H, W, freq) in SHAPES:
    k = f"{C}_{H}x{W}"
    d = out[k]
    pen = 100 * (d["med"]["modiff_split"] - d["med"]["modiff_full"]) / d["med"]["modiff_full"]
    print(f"{f'{C},{H}x{W}':14}{d['base_gain']:10.1f}%{d['mod_gain']:12.1f}%{pen:20.1f}%")

with open("pipeline_result2.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"\nall times ms per {N_STAGES}-stage chain. wrote pipeline_result2.json")

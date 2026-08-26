"""Does batch-split 2-stream pipelining hide MoDiff's GN (a_hat) behind the conv?

Four configs per shape, all running a chain of N_STAGES (GN -> conv) pairs:

  0 base_full    1 stream, full batch, baseline GN + baseline conv   (no a_hat/o_hat) -- reference
  1 modiff_full  1 stream, full batch, MoDiff GN + MoDiff conv       -- TODAY's behaviour
  2 modiff_split 1 stream, two half batches back to back             -- isolates the SPLIT PENALTY
                                                                        (2x launches, smaller kernels,
                                                                         no overlap possible)
  3 modiff_pipe  2 streams, two half batches, stream2 offset by one  -- split penalty + OVERLAP BENEFIT
                 GN so that conv(A) runs concurrently with GN(B)

  overlap benefit = modiff_split - modiff_pipe      (how much the concurrency actually bought)
  net vs today    = modiff_full  - modiff_pipe      (what ships)

Within one stream CUDA guarantees strict kernel ordering regardless of data dependency, so
independent per-stage buffers reproduce the real execution SCHEDULE (the established
independent-layers-chained methodology from this session). Only A-vs-B independence is load bearing,
and that is real: GroupNorm is per-(sample,group), conv/attention are per-sample, a_hat/o_hat are
[N,C,H,W] per-sample, and the static delta scale is a calibrated constant -- so the split is
numerically exact, not an approximation.
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
N_TRIALS = 3
NUM_GROUPS = 32
EPS = 1e-5

EMPTY_SMOOTH = torch.empty(0, device="cuda", dtype=torch.float32)
EMPTY_MOD = torch.empty(0, device="cuda", dtype=torch.float16)
EMPTY_F32 = torch.empty(0, device="cuda", dtype=torch.float32)
EMPTY_BIAS = torch.empty(0, device="cuda", dtype=torch.float32)
EMPTY_RES = torch.empty(0, device="cuda", dtype=torch.float16)

# Top-5 real conv shapes by call frequency, restricted to Cin == Cout so one stage's output shape
# matches the next stage's input shape (a faithful repeated-ResBlock chain).
SHAPES = [
    (128, 768, 2, 2, 12),
    (128, 384, 8, 8, 8),
    (128, 192, 32, 32, 7),
    (128, 384, 16, 16, 7),
    (128, 768, 4, 4, 7),
]


def make_stage(N, C, H, W):
    """One (GN -> conv) stage's buffers at batch N. Cin == Cout == C."""
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
    out_buf = torch.empty_like(o_hat)

    return dict(
        x=x, a_hat=a_hat,
        gw=torch.randn(C, device="cuda", dtype=torch.float16),
        gb=torch.randn(C, device="cuda", dtype=torch.float16),
        scale=torch.tensor([127.0], device="cuda"),
        wq=w_q, ws=ch_scale.contiguous(),
        alpha=torch.tensor([1.0 / 16.0], device="cuda"),
        x_int8=x_int8, o_hat=o_hat, out=out_buf,
    )


# ---------------- the four kernel pairs ----------------

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


# ---------------- the four configs ----------------

def run_single(stages, gn, conv):
    """1 stream: GN -> conv, GN -> conv, ... over the whole chain."""
    for S in stages:
        gn(S)
        conv(S)


def run_split_serial(stages_a, stages_b, gn, conv):
    """1 stream, two half batches back to back. No overlap is possible -- this measures ONLY the
    penalty of splitting (double the launches, each kernel half as big / less efficient)."""
    for S in stages_a:
        gn(S)
        conv(S)
    for S in stages_b:
        gn(S)
        conv(S)


def run_pipe(stages_a, stages_b, gn, conv, s1, s2):
    """2 streams, half batch each, stream2 offset by one GN so conv(A) overlaps GN(B).

    The offset is created ONCE by an event recorded after A's first GN; after that each stream's own
    ordering keeps the two anti-phase (identical per-stage work in both streams).
    """
    ev_gate = torch.cuda.Event()
    ev_gate.record()               # current stream's prior work
    s1.wait_event(ev_gate)
    s2.wait_event(ev_gate)

    ev_offset = torch.cuda.Event()
    with torch.cuda.stream(s1):
        gn(stages_a[0])
        ev_offset.record(s1)       # <- B may start only once A's first GN is done
        conv(stages_a[0])
        for S in stages_a[1:]:
            gn(S)
            conv(S)

    with torch.cuda.stream(s2):
        s2.wait_event(ev_offset)
        for S in stages_b:
            gn(S)
            conv(S)

    # hand both streams back to the default stream
    ev_end1, ev_end2 = torch.cuda.Event(), torch.cuda.Event()
    ev_end1.record(s1)
    ev_end2.record(s2)
    torch.cuda.current_stream().wait_event(ev_end1)
    torch.cuda.current_stream().wait_event(ev_end2)


def timeit(launch):
    """launch() issues one rep of the whole chain. Wall clock around REPS reps, GPU-synced."""
    for _ in range(WARMUP_REPS):
        launch()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(REPS):
        launch()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3 / REPS   # ms per rep (= per N_STAGES chain)


def bench_shape(N, C, H, W):
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
    half = N // 2

    full = [make_stage(N, C, H, W) for _ in range(N_STAGES)]
    ha = [make_stage(half, C, H, W) for _ in range(N_STAGES)]
    hb = [make_stage(half, C, H, W) for _ in range(N_STAGES)]

    res = {k: [] for k in ("base_full", "modiff_full", "modiff_split", "modiff_pipe")}
    for trial in range(N_TRIALS):
        order = [
            ("base_full", lambda: run_single(full, gn_base, conv_base)),
            ("modiff_full", lambda: run_single(full, gn_modiff, conv_modiff)),
            ("modiff_split", lambda: run_split_serial(ha, hb, gn_modiff, conv_modiff)),
            ("modiff_pipe", lambda: run_pipe(ha, hb, gn_modiff, conv_modiff, s1, s2)),
        ]
        if trial % 2 == 1:
            order = order[::-1]          # alternate to cancel systematic drift
        for name, fn in order:
            res[name].append(timeit(fn))

    del full, ha, hb
    torch.cuda.empty_cache()
    return {k: statistics.median(v) for k, v in res.items()}


print(f"batch-split 2-stream pipelining: {N_STAGES} (GN->conv) stages/chain, {REPS} chain reps/trial, "
      f"{N_TRIALS} trials (order alternated)\n")
print(f"{'shape (C,HxW)':16}{'freq':>5} | {'base':>8}{'modiff':>8}{'split':>8}{'pipe':>8} | "
      f"{'MoDiff ovh':>11}{'pipe ovh':>10} | {'overlap won':>12}{'net vs today':>13}")
print("-" * 108)

out = {}
for (N, C, H, W, freq) in SHAPES:
    r = bench_shape(N, C, H, W)
    b, m, sp, pi = r["base_full"], r["modiff_full"], r["modiff_split"], r["modiff_pipe"]
    ovh_today = 100 * (m - b) / b
    ovh_pipe = 100 * (pi - b) / b
    overlap_won = 100 * (sp - pi) / sp          # how much the concurrency bought vs no-overlap split
    net = 100 * (m - pi) / m                    # what ships, vs today
    out[f"{C}_{H}x{W}"] = dict(freq=freq, **r)
    print(f"{f'{C}, {H}x{W}':16}{freq:5d} | {b:8.3f}{m:8.3f}{sp:8.3f}{pi:8.3f} | "
          f"{ovh_today:10.1f}%{ovh_pipe:9.1f}% | {overlap_won:11.1f}%{net:12.1f}%")

with open("pipeline_result.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nall times ms per {}-stage chain. wrote pipeline_result.json".format(N_STAGES))

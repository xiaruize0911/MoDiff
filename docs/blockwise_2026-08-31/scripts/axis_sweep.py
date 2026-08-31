"""fp16 vs int8 vs blockwise int8, swept one conv axis at a time.

Answers "how does the speedup move with (B, N, H, W)", which a single
frequency-weighted total cannot. Four independent sweeps, each varying one axis
from the same default point as docs/conv_kernel_sweep_2026-08-28:

    B=128, H=16, W=16, N=C=384

Arms, all 3x3 pad=1 stride=1, channels_last, on the same operands:

  fp16              F.conv2d in fp16 (cudnn.benchmark on) -- the torch_conv2d_fp16
                    convention from docs/bench_report_2026-08-13_postzp
  int8 per-tensor   conv2d_int8_evt_o_hat, one scalar alpha (what ships)
  int8 bw G=64/32   channel-block split-K: Cin/G calls, each with its own alpha and
                    its own per-(block, out-channel) weight scales. Exact blockwise
                    (verified in FINDINGS section 2), and the only implementation the
                    current epilogue admits.

Speedup is fp16_ms / arm_ms, so >1 means the arm beats fp16 and <1 means
quantizing that conv makes it slower than leaving it in fp16.

Run: source setup_cuda_env.sh
     python docs/blockwise_2026-08-31/scripts/axis_sweep.py
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from integration.utils.preflight import preflight  # noqa: E402

preflight("torch", what="axis_sweep.py")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import modiff_cutlass  # noqa: E402

#: an under-tuned fp16 baseline would flatter every int8 arm
torch.backends.cudnn.benchmark = True

JSON_OUT = "docs/blockwise_2026-08-31/data/axis_sweep.json"
DEFAULT = {"B": 128, "N": 384, "H": 16, "W": 16}
AXES = {
    "B": (8, 16, 32, 64, 128, 256),
    "N": (128, 192, 256, 384, 512, 768, 1152, 1536),
    "H": (2, 4, 8, 16, 32),
    "W": (2, 4, 8, 16, 32),
}
BLOCKS = (64, 32)


def _time(fn, reps, warmup=8):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    ev0.record()
    for _ in range(reps):
        fn()
    ev1.record()
    torch.cuda.synchronize()
    return ev0.elapsed_time(ev1) / reps


def measure(b, n, h, w, reps, trials):
    """One (B, N, H, W) point. N is both Cin and Cout, matching the committed sweep."""
    cin = cout = n
    dev, st = "cuda", (1, 1, 1, 1, 1, 1)
    x = torch.randint(-127, 127, (b, cin, h, w), device=dev, dtype=torch.int8
                      ).contiguous(memory_format=torch.channels_last)
    wt = torch.randint(-127, 127, (cout, 3, 3, cin), device=dev, dtype=torch.int8).contiguous()
    alpha = torch.full((1,), 1.0 / 127.0, device=dev, dtype=torch.float32)
    wsc = torch.full((cout,), 0.01, device=dev, dtype=torch.float32)
    o = torch.zeros(b, cout, h, w, device=dev, dtype=torch.float16
                    ).contiguous(memory_format=torch.channels_last)
    xh = torch.randn(b, cin, h, w, device=dev, dtype=torch.float16
                     ).contiguous(memory_format=torch.channels_last)
    wh = torch.randn(cout, cin, 3, 3, device=dev, dtype=torch.float16
                     ).contiguous(memory_format=torch.channels_last)
    fn = modiff_cutlass.conv2d_int8_evt_o_hat

    rec = {"B": b, "N": n, "H": h, "W": w}
    rec["fp16_ms"] = min(_time(lambda: F.conv2d(xh, wh, None, 1, 1, 1, 1), reps)
                         for _ in range(trials))
    rec["int8_ms"] = min(_time(lambda: fn(x, wt, alpha, wsc, o, *st), reps)
                         for _ in range(trials))
    rec["int8_vs_fp16"] = rec["fp16_ms"] / rec["int8_ms"]

    for g in BLOCKS:
        if g > cin or cin % g:
            continue
        nb = cin // g
        # Slicing copies are hoisted out of the timed region: a real implementation would
        # have the quantize kernel emit blocks directly, so this is a FLOOR on the cost.
        xs = [x[:, i * g:(i + 1) * g].contiguous(memory_format=torch.channels_last)
              for i in range(nb)]
        ws = [wt[:, :, :, i * g:(i + 1) * g].contiguous() for i in range(nb)]
        als = [alpha.clone() for _ in range(nb)]
        wss = [wsc.clone() for _ in range(nb)]

        def split():
            for i in range(nb):
                fn(xs[i], ws[i], als[i], wss[i], o, *st)

        t = min(_time(split, reps) for _ in range(trials))
        rec[f"bw{g}_ms"] = t
        rec[f"bw{g}_vs_fp16"] = rec["fp16_ms"] / t
        rec[f"bw{g}_blocks"] = nb
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--out", default=JSON_OUT)
    a = ap.parse_args()

    print(f"GPU {torch.cuda.get_device_name(0)}  default "
          f"B={DEFAULT['B']} N={DEFAULT['N']} H={DEFAULT['H']} W={DEFAULT['W']}  "
          f"reps={a.reps} trials={a.trials}", flush=True)

    out = {"gpu": torch.cuda.get_device_name(0), "default": DEFAULT,
           "axes": {k: list(v) for k, v in AXES.items()}, "blocks": list(BLOCKS),
           "method": "3x3 pad=1 stride=1, channels_last. fp16 = F.conv2d with "
                     "cudnn.benchmark; int8 = conv2d_int8_evt_o_hat; blockwise = "
                     "channel-block split-K (slicing copies hoisted, so a floor). "
                     "speedup = fp16_ms / arm_ms; <1 means slower than fp16.",
           "sweeps": {}}

    for axis, vals in AXES.items():
        print(f"\n=== sweep {axis} ===", flush=True)
        hdr = f"  {axis:>5s} {'fp16':>8s} {'int8':>8s} {'int8/fp16':>10s}"
        for g in BLOCKS:
            hdr += f" {'bw' + str(g) + '/fp16':>11s}"
        print(hdr, flush=True)
        rows = []
        for v in vals:
            p = dict(DEFAULT)
            p[axis] = v
            r = measure(p["B"], p["N"], p["H"], p["W"], a.reps, a.trials)
            rows.append(r)
            line = (f"  {v:5d} {r['fp16_ms']:8.3f} {r['int8_ms']:8.3f} "
                    f"{r['int8_vs_fp16']:9.3f}x")
            for g in BLOCKS:
                k = f"bw{g}_vs_fp16"
                line += f" {r[k]:10.3f}x" if k in r else f" {'-':>11s}"
            print(line, flush=True)
        out["sweeps"][axis] = rows

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

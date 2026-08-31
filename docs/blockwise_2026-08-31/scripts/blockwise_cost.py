"""Cost of blockwise on the REAL kernel, via channel-block split-K.

The EVT epilogue computes  o_hat[elem] += acc * alpha * weight_scale[k], with alpha a
scalar broadcast and weight_scale a row broadcast over OUTPUT channels. Neither can
carry a scale that varies along the reduction axis K = Cin*R*S, because the epilogue
only ever sees the finished accumulator. So a per-C-block scale cannot be fixed up
after the GEMM -- it has to enter the reduction.

There is exactly one way to do that with the kernels already in the tree, and it is
exact rather than approximate: the D2 epilogue is a read-modify-write into o_hat, so
calling the conv once per channel block, each call with that block's own alpha and its
own per-(block, out-channel) weight_scales, accumulates the blockwise-dequantized sum.
Block = G input channels x all R*S taps, which is the alignment a fused blockwise
mainloop would also need (weights and activations share the C-block boundaries).

What it costs: the MAC work is unchanged (the K dim is merely partitioned), but every
block re-runs the full epilogue over the whole N*P*Q*K output, so o_hat traffic and
accumulator traffic multiply by Cin/G. This measures that.

Reported per shape: ms for the single fused call vs the Cin/G-call split, at each G.

Run: source setup_cuda_env.sh
     python docs/blockwise_2026-08-31/scripts/blockwise_cost.py
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

preflight("torch", what="blockwise_cost.py")

import torch  # noqa: E402
import modiff_cutlass  # noqa: E402

JSON_OUT = "docs/blockwise_2026-08-31/data/blockwise_cost.json"
BLOCKS = (256, 128, 64, 32, 16)
#: (Cin, Cout, HW, freq) -- the high-frequency UNet ResBlock conv shapes at B=128,
#: taken from docs/conv_kernel_sweep_2026-08-28/FINDINGS.md section 5.
SHAPES = (
    (192, 192, 32, 7),
    (384, 384, 16, 7),
    (384, 384, 8, 8),
    (768, 768, 8, 1),
    (768, 768, 4, 7),
)


def _mk(b, cin, cout, hw, dev="cuda"):
    x = torch.randint(-127, 127, (b, cin, hw, hw), device=dev, dtype=torch.int8
                      ).contiguous(memory_format=torch.channels_last)
    # CUTLASS wants NHWC weights: (K, R, S, C)
    w = torch.randint(-127, 127, (cout, 3, 3, cin), device=dev, dtype=torch.int8).contiguous()
    alpha = torch.full((1,), 1.0 / 127.0, device=dev, dtype=torch.float32)
    ws = torch.full((cout,), 0.01, device=dev, dtype=torch.float32)
    o = torch.zeros(b, cout, hw, hw, device=dev, dtype=torch.float16
                    ).contiguous(memory_format=torch.channels_last)
    return x, w, alpha, ws, o


def _time(fn, reps=30, warmup=8):
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--out", default=JSON_OUT)
    a = ap.parse_args()

    print(f"GPU {torch.cuda.get_device_name(0)}  B={a.batch}  "
          f"reps={a.reps} trials={a.trials}", flush=True)
    fn = modiff_cutlass.conv2d_int8_evt_o_hat
    rows = []

    for cin, cout, hw, freq in SHAPES:
        x, w, alpha, ws, o = _mk(a.batch, cin, cout, hw)
        st = (1, 1, 1, 1, 1, 1)

        def full():
            fn(x, w, alpha, ws, o, *st)

        base = min(_time(full, a.reps) for _ in range(a.trials))
        rec = {"cin": cin, "cout": cout, "hw": hw, "freq": freq, "fused_ms": base,
               "splits": {}}
        print(f"\n{cin}->{cout}, {hw}x{hw} (freq {freq}): fused {base:.3f} ms", flush=True)

        for g in BLOCKS:
            if g > cin:
                continue
            nb = cin // g
            if nb * g != cin:
                continue
            # Contiguous C-slices. .contiguous(channels_last) on the slice gives each
            # block a packed NHWC tensor, which is what the kernel requires; the copy is
            # NOT timed as part of the split (a real implementation would have the
            # quantize kernel emit blocks directly), so this is a floor on the cost.
            xs = [x[:, i * g:(i + 1) * g].contiguous(memory_format=torch.channels_last)
                  for i in range(nb)]
            wsl = [w[:, :, :, i * g:(i + 1) * g].contiguous() for i in range(nb)]
            als = [alpha.clone() for _ in range(nb)]
            wss = [ws.clone() for _ in range(nb)]

            def split():
                for i in range(nb):
                    fn(xs[i], wsl[i], als[i], wss[i], o, *st)

            t = min(_time(split, a.reps) for _ in range(a.trials))
            rec["splits"][str(g)] = {"n_blocks": nb, "ms": t, "vs_fused": base / t}
            print(f"  G={g:4d}  {nb:3d} calls  {t:8.3f} ms  {base / t:.3f}x", flush=True)
        rows.append(rec)

    # frequency-weighted totals over the sampled shapes
    wsum = {"fused": sum(r["fused_ms"] * r["freq"] for r in rows)}
    for g in BLOCKS:
        gs = [r for r in rows if str(g) in r["splits"]]
        if len(gs) == len(rows):
            wsum[f"G={g}"] = sum(r["splits"][str(g)]["ms"] * r["freq"] for r in rows)
    print("\nfreq-weighted over the sampled shapes (ms, conv path only):", flush=True)
    for k, v in wsum.items():
        print(f"  {k:8s} {v:8.2f}  {wsum['fused'] / v:.3f}x vs fused", flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"gpu": torch.cuda.get_device_name(0), "batch": a.batch,
               "reps": a.reps, "trials": a.trials,
               "method": "channel-block split-K on conv2d_int8_evt_o_hat; the D2 epilogue "
                         "RMW into o_hat makes the per-block accumulation exact. Block slicing "
                         "copies are hoisted out of the timed region, so these are a FLOOR.",
               "blocks": list(BLOCKS), "shapes": rows,
               "freq_weighted_ms": wsum}, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

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
#: (Cin, Cout, HW, freq) -- the full 20 UNet ResBlock conv shapes at B=128 with their
#: per-step call counts (62 calls/step), from docs/conv_kernel_sweep_2026-08-28/FINDINGS.md
#: section 5. Cin must be divisible by G for a block split, so only G in {16,32,64} covers
#: every shape (192 and 576 are not divisible by 128 or 256); coarser G are reported
#: per-shape but excluded from the freq-weighted total, and the exclusion is logged.
SHAPES = (
    (768, 768, 2, 12),
    (384, 384, 8, 8),
    (192, 192, 32, 7),
    (384, 384, 16, 7),
    (768, 768, 4, 7),
    (1536, 768, 2, 3),
    (1536, 768, 4, 2),
    (768, 384, 8, 2),
    (768, 384, 16, 2),
    (384, 192, 32, 2),
    (192, 192, 16, 1),
    (192, 384, 16, 1),
    (384, 384, 4, 1),
    (384, 768, 4, 1),
    (1152, 768, 4, 1),
    (768, 768, 8, 1),
    (1152, 384, 8, 1),
    (576, 384, 16, 1),
    (384, 384, 32, 1),
    (576, 192, 32, 1),
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
    missing = [g for g in BLOCKS if f"G={g}" not in wsum]
    if missing:
        print(f"\nexcluded from the weighted total (not every Cin divides G): "
              f"{missing} -- per-shape rows above still stand", flush=True)
    print("\nfreq-weighted over ALL 20 UNet shapes, 62 calls/step "
          "(ms, conv path only):", flush=True)
    for k, v in wsum.items():
        print(f"  {k:8s} {v:8.2f}  {wsum['fused'] / v:.3f}x vs fused", flush=True)

    # ---- where the slowdown comes from ----
    # The per-shape ratios cluster by G rather than by block count, so "the epilogue runs
    # nb times" cannot be the whole story. Split the per-call cost into the epilogue (an
    # o_hat-sized fp16 RMW, which reuse_o_hat_add measures directly with no GEMM at all)
    # and the rest, which is the K-thin GEMM: each call reduces over K = G*R*S instead of
    # Cin*R*S, so it runs far below the efficiency of the fused call's mainloop.
    cin, cout, hw, _f = 768, 768, 8, 1
    g = 32
    nb = cin // g
    x, w, alpha, ws, o = _mk(a.batch, cin, cout, hw)
    st = (1, 1, 1, 1, 1, 1)
    fused = min(_time(lambda: fn(x, w, alpha, ws, o, *st), a.reps) for _ in range(a.trials))
    xg, wg, _a, _w, _o = _mk(a.batch, g, cout, hw)
    one = min(_time(lambda: fn(xg, wg, alpha, ws, o, *st), a.reps) for _ in range(a.trials))
    o2 = torch.zeros_like(o)
    epi = min(_time(lambda: modiff_cutlass.reuse_o_hat_add(o, o2, o2), a.reps)
              for _ in range(a.trials))
    attrib = {"shape": f"{cin}->{cout} {hw}x{hw}", "G": g, "n_blocks": nb,
              "fused_us": fused * 1e3, "ideal_per_call_us": fused * 1e3 / nb,
              "standalone_cin_g_us": one * 1e3, "epilogue_only_us": epi * 1e3,
              "per_call_vs_ideal": one / (fused / nb),
              "nb_x_standalone_vs_fused": one * nb / fused,
              "epilogue_share_of_per_call": epi / one}
    print(f"\nattribution on {cin}->{cout} {hw}x{hw}, G={g}, nb={nb}:", flush=True)
    print(f"  fused (K={cin * 9})                {fused * 1e3:8.1f} us", flush=True)
    print(f"  1/nb of fused (ideal per call)  {fused * 1e3 / nb:8.1f} us", flush=True)
    print(f"  standalone Cin=G conv (K={g * 9})   {one * 1e3:8.1f} us  "
          f"= {one / (fused / nb):.2f}x ideal", flush=True)
    print(f"  epilogue only (o_hat RMW)       {epi * 1e3:8.1f} us  "
          f"= {epi / one * 100:.0f}% of the per-call cost", flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"gpu": torch.cuda.get_device_name(0), "batch": a.batch,
               "reps": a.reps, "trials": a.trials, "attribution": attrib,
               "method": "channel-block split-K on conv2d_int8_evt_o_hat; the D2 epilogue "
                         "RMW into o_hat makes the per-block accumulation exact. Block slicing "
                         "copies are hoisted out of the timed region, so these are a FLOOR.",
               "blocks": list(BLOCKS), "shapes": rows,
               "freq_weighted_ms": wsum}, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

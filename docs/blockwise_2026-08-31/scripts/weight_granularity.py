"""Weight-only: reconstruction error vs quantization block size, on the real checkpoint.

No sampling, no model build -- reads the 72 3x3 conv weights straight out of the
LSUN-Churches state_dict, so this runs in seconds and is the cheap half of the
blockwise question.

Two block axes are compared. "flat" blocks hold G elements of the flattened
[Cout, Cin*kH*kW]; "chan" blocks hold G input channels x all R*S taps, which is the
C-aligned unit the activation blocks use and the only one a channel-block split-K can
implement. Both run along the reduction dim K = Cin*kH*kW, one scale per
(out_channel, block). That is the axis a conv GEMM reduces over, which is why it
is also the axis the CUTLASS epilogue cannot fix up after the fact -- see
FINDINGS.md section 3. Here we only ask what it would buy if it were free.

Rules compared:
  per-channel absmax   scale = absmax/Q over the whole K row      (shipped, int8)
  per-channel mse      13-candidate clip search over the K row    (shipped, int4;
                       mirrors integration/kernels/int4_optimized._int4_weight_scale)
  block-G absmax       scale = absmax/Q per G-element block
  block-G mse          the same clip search, per block

Metric is relative Frobenius error ||W_q - W|| / ||W|| per conv; we report the
median and the worst conv, matching the table in _int4_weight_scale's docstring.

Run: python docs/blockwise_2026-08-31/scripts/weight_granularity.py
"""
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from integration.utils.preflight import preflight  # noqa: E402

preflight("torch", what="weight_granularity.py")

import torch  # noqa: E402

CKPT = "models/ldm/lsun_churches256/model.ckpt"
JSON_OUT = "docs/blockwise_2026-08-31/data/weight_granularity.json"
BLOCKS = (512, 256, 128, 64, 32, 16)
#: C-aligned block sizes, counted in INPUT CHANNELS (each block is G*R*S elements)
CHAN_BLOCKS = (256, 128, 64, 32, 16)
#: the shipped int4 clip ladder, verbatim from _int4_weight_scale
CLIPS = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4)


def _blocks(wf: torch.Tensor, gsize: int):
    """[Cout, K] -> [Cout, nG, G], zero-padded. Padding never changes a block absmax."""
    k = wf.shape[1]
    pad = (gsize - k % gsize) % gsize
    if pad:
        wf = torch.nn.functional.pad(wf, (0, pad))
    return wf.reshape(wf.shape[0], -1, gsize), k


def _blocks_chan(w: torch.Tensor, gchan: int):
    """C-aligned blocks: G input channels x all R*S taps, i.e. G*9 elements.

    This is the implementable alignment -- weights and activations then share the same
    C-block boundaries, which is what a channel-block split-K (or a fused blockwise
    mainloop) requires. `gchan` counts CHANNELS, not elements.
    """
    cout, cin = w.shape[0], w.shape[1]
    wf = w.reshape(cout, cin, -1).float()          # [Cout, Cin, R*S]
    pad = (gchan - cin % gchan) % gchan
    if pad:
        wf = torch.nn.functional.pad(wf, (0, 0, 0, pad))
    ng = wf.shape[1] // gchan
    return wf.reshape(cout, ng, -1), cin           # [Cout, nG, G*R*S]


def _scale_absmax(wg: torch.Tensor, q: float) -> torch.Tensor:
    return (wg.abs().amax(dim=-1, keepdim=True) / q).clamp_min(1e-8)


def _scale_mse(wg: torch.Tensor, q: float) -> torch.Tensor:
    """Clip search minimising sum of squared reconstruction error per block."""
    am = wg.abs().amax(dim=-1, keepdim=True)
    best_err = best_sc = None
    for r in CLIPS:
        sc = (am * r / q).clamp_min(1e-8)
        err = (((wg / sc).round().clamp(-q, q) * sc - wg) ** 2).sum(dim=-1, keepdim=True)
        if best_err is None:
            best_err, best_sc = err, sc
        else:
            m = err < best_err
            best_err = torch.where(m, err, best_err)
            best_sc = torch.where(m, sc, best_sc)
    return best_sc


def rel_err(w: torch.Tensor, gsize, q: float, rule: str, axis: str = "flat") -> float:
    """Relative Frobenius error of the dequantized weight.

    axis 'flat': blocks of `gsize` ELEMENTS along the flattened [Cout, Cin*R*S].
    axis 'chan': blocks of `gsize` CHANNELS (G*R*S elements), C-aligned.
    """
    wf = w.reshape(w.shape[0], -1).float()
    if axis == "chan" and gsize:
        wg, n0 = _blocks_chan(w, gsize)
        keep = n0 * w.shape[2] * w.shape[3]
    else:
        k = wf.shape[1]
        wg, keep = _blocks(wf, gsize if gsize else k)
    sc = _scale_mse(wg, q) if rule == "mse" else _scale_absmax(wg, q)
    wq = (wg / sc).round().clamp(-q, q) * sc
    wq = wq.reshape(wf.shape[0], -1)[:, :keep]
    return float((wq - wf[:, :keep]).norm() / wf.norm().clamp_min(1e-12))


def main() -> int:
    print(f"loading {CKPT}", flush=True)
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"]
    convs = {k: v for k, v in sd.items()
             if k.startswith("model.diffusion_model") and v.ndim == 4 and v.shape[2] == 3}
    del sd
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    convs = {k: v.to(dev) for k, v in convs.items()}
    kdims = sorted({v.shape[1] * 9 for v in convs.values()})
    print(f"{len(convs)} 3x3 convs, K = Cin*9 in {kdims}", flush=True)

    arms = [("per-channel", None, "absmax", "flat"), ("per-channel", None, "mse", "flat")]
    arms += [("block", g, "absmax", "flat") for g in BLOCKS]
    arms += [("block", g, "mse", "flat") for g in BLOCKS]
    # C-aligned blocks: G counts CHANNELS (G*9 elements), the unit the activation
    # blocks use and the only one a channel-block split-K can implement.
    arms += [("chanblock", g, "absmax", "chan") for g in CHAN_BLOCKS]
    arms += [("chanblock", g, "mse", "chan") for g in CHAN_BLOCKS]

    out = {"ckpt": CKPT, "n_convs": len(convs), "k_dims": kdims,
           "metric": "relative Frobenius error of dequantized weight, per conv",
           "blocks": list(BLOCKS), "arms": []}

    for bits in (8, 4):
        q = 127.0 if bits == 8 else 7.0
        print(f"\n--- W{bits} (Q={q:.0f}) ---", flush=True)
        print(f"  {'rule':22s} {'median':>8s} {'worst':>8s} {'scales/wt':>10s}", flush=True)
        for kind, g, rule, axis in arms:
            errs = [rel_err(w, g, q, rule, axis) for w in convs.values()]
            errs.sort()
            med = errs[len(errs) // 2]
            worst = errs[-1]
            # scale metadata vs quantized weight bytes: one fp16 scale per block.
            # flat blocks hold G elements; chan blocks hold G*9.
            elems = (g * 9 if axis == "chan" else g) if g else 0
            per_wt = (2.0 / elems) / (bits / 8.0) if elems else 0.0
            label = f"{kind}-{g} {rule}" if g else f"{kind} {rule}"
            out["arms"].append({"bits": bits, "kind": kind, "block": g, "rule": rule,
                                "axis": axis, "block_elems": elems,
                                "median": med, "worst": worst, "scale_overhead": per_wt})
            print(f"  {label:22s} {med:8.4f} {worst:8.4f} {per_wt * 100:9.1f}%", flush=True)

    os.makedirs(os.path.dirname(JSON_OUT), exist_ok=True)
    json.dump(out, open(JSON_OUT, "w"), indent=1)
    print(f"\nwrote {JSON_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

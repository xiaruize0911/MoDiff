"""Per-BLOCK GPU profile: every UNet block timed at its own module boundary, grouped by block type.

WHY THIS EXISTS ALONGSIDE profile_layers_and_model.py. That harness times LEAF dispatch targets
(conv methods, the attention route, the projection GEMM) and reaches 0.63-0.89 of the step. The 11-37%
it misses is real work -- the ResBlock's own arithmetic, the emb path, the skip connections, the
head and tail of the UNet -- and it is invisible there by construction, because those things are not
any of the four leaf kinds. This harness times the BLOCKS instead, so the residual shrinks to what
sits between blocks rather than to a quarter of the model.

THE TWO INSTRUMENTS ARE NOT NESTED VERSIONS OF EACH OTHER. A conv timed there is inside a ResBlock
timed here; adding a row from one to a row from the other double counts. Read them separately.

METHOD. CUDA events around each block's `forward`, no sync in the hot path, one sync at readout --
the same technique the layer harness uses and for the same reason (the torch profiler's
record_function scopes carry the device time of the kernels inside them and inflated an earlier
attempt 2.2x).

Blocks are SIBLINGS in the UNet, never nested in each other, so unlike the leaf harness this one
needs no subtraction to avoid double counting. The one exception is asserted rather than assumed:
`ResBlock(up=True)` is the last child of the SAME TimestepEmbedSequential that holds the level's
ordinary ResBlock and attention, not its own entry (openaimodel.py:724-737), so it is still a sibling.
`Downsample`/`Upsample` never appear as standalone entries in this config -- they are `h_upd`/`x_upd`
INSIDE the updown ResBlocks, which is why "updown" here means a ResBlock and not a Resample module.

WHAT IS TIMED, and the block types it produces:

  resblock       35 FusedResBlock, split into regular / downsample / upsample by `original.down|up`
  attention      21 QuantizedStandardAttentionBlock, split into 4 shape tiers by (head_dim, T)
  conv_in         1 input_blocks[0], a bare 4->192 conv, NOT quantized (in_channels < 32)
  time_embed      1 Linear(192,768) + SiLU + Linear(768,768), once per step, not per block
  out             1 GroupNorm + SiLU + conv(192->4), also not quantized (path starts with "out.")

Also runs a structural audit of the quantized conv modules, because `fusion_audit.py` has recorded
"70 of 140 conv modules never called" as OPEN since 2026-08-11 and the cause is a graph fact this
harness is already walking the graph to find.

Run: python integration/tests/profile_blocks.py [--batch 128] [--steps 200] [--outdir DIR]
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "integration/benchmarks/report"),
                os.path.join(ROOT, "docs/modiff_correctness_2026-08-03/scripts")]

import torch                                                                # noqa: E402
import dynamic_delta_ab as H                                               # noqa: E402

#: (label, mode, MODIFF_LINEAR, act_bits, K, extra_env) -- the same eight configurations the leaf
#: harness profiles, so the two can be read side by side per configuration.
CONFIGS = [("W8A8 PTQ", "int8_baseline", "0", 8, 4, {}),
           ("W8A8 conv-only", "int8", "0", 8, 4, {}),
           ("W8A8 conv+proj", "int8", "1", 8, 4, {}),
           ("W8A8 conv+proj +projK4", "int8", "1", 8, 4,
            {"MODIFF_LINEAR_DELTA_REFRESH": "4"}),
           ("W8A8 conv+proj +projK4 +routeB", "int8", "1", 8, 4,
            {"MODIFF_LINEAR_DELTA_REFRESH": "4", "MODIFF_FUSE_QKV_I8": "1"}),
           ("W8A4 conv+proj", "int8", "1", 4, 4, {}),
           ("W4A4 conv+proj", "int4", "1", 8, 4, {}),
           ("fp16", "fp16", "0", 8, 4, {})]

#: Cleared before every config unless that config asks for them: profile() loops in ONE process, so a
#: leaked flag silently attributes one configuration's datapath to the next.
OPTIONAL_ENV = ["MODIFF_LINEAR_DELTA_REFRESH", "MODIFF_FUSE_QKV_I8"]


class BlockTimer:
    """CUDA-event timer that also records each block's first-call input shape.

    The shape is what classifies an attention block into its tier, and it is not available before the
    run: `head_dim` is on the module but the token count T depends on the feature map, which depends
    on where in the UNet the block sits. Recording it in the wrapper is exact and costs one branch.
    """

    def __init__(self):
        self.ev = {}
        self.shape = {}

    def wrap(self, obj, name, key):
        fn = getattr(obj, name, None)
        if fn is None:
            return False
        ev = self.ev.setdefault(key, [])

        def inner(*a, __fn=fn, __ev=ev, __key=key, **kw):
            if __key not in self.shape:
                t = next((x for x in a if torch.is_tensor(x)), None)
                if t is not None:
                    self.shape[__key] = list(t.shape)
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            out = __fn(*a, **kw)
            e.record()
            __ev.append((s, e))
            return out
        setattr(obj, name, inner)
        return True

    def totals(self):
        torch.cuda.synchronize()
        return {k: sum(s.elapsed_time(e) for s, e in v) for k, v in self.ev.items() if v}

    def counts(self):
        return {k: len(v) for k, v in self.ev.items() if v}


def classify(unet):
    """Walk the UNet once and return an ordered [(key, module, meta)] of every block to time.

    Order is UNet execution order -- input_blocks, middle, output_blocks -- because that is what the
    x-axis of every per-block figure means. `unet.modules()` would give definition order, which
    happens to agree here, but walking the containers explicitly makes the ordering a fact rather
    than a coincidence and is what lets each block carry its container index.
    """
    import integration.fused_ops.fused_resblock as FR
    out = []

    def visit(container, where, ci):
        for child in container:
            tn = type(child).__name__
            if isinstance(child, FR.FusedResBlock):
                o = child.original
                kind = ("resblock_down" if getattr(o, "down", False) else
                        "resblock_up" if getattr(o, "up", False) else "resblock")
                out.append((f"rb{len(out):03d}", child, "forward",
                            {"type": kind, "where": where, "container": ci,
                             "channels": int(getattr(o, "channels", 0)),
                             "out_channels": int(getattr(o, "out_channels", 0))}))
            elif tn == "QuantizedStandardAttentionBlock":
                out.append((f"at{len(out):03d}", child, "forward",
                            {"type": "attention", "where": where, "container": ci,
                             "channels": int(getattr(child, "channels", 0)),
                             "num_heads": int(getattr(child, "num_heads", 0)),
                             "head_dim": int(getattr(child, "head_dim", 0))}))
            else:
                # input_blocks[0] is a bare conv (in_channels 4 -> 192), below the converter's
                # in_channels >= 32 threshold, so it is an unquantized nn.Conv2d and belongs to no
                # other category. Anything else landing here is new and worth seeing in the output.
                out.append((f"ot{len(out):03d}", child, "forward",
                            {"type": "conv_in" if (where == "in" and ci == 0) else f"other:{tn}",
                             "where": where, "container": ci}))

    for i, blk in enumerate(unet.input_blocks):
        visit(blk, "in", i)
    visit(unet.middle_block, "mid", 0)
    for i, blk in enumerate(unet.output_blocks):
        visit(blk, "out", i)
    return out


def audit_convs(unet):
    """Why 140 quantized conv modules exist when only 70 ever run.

    OPEN in fusion_audit.py since 2026-08-11 as "70 of 140 conv modules never called". The cause is
    structural, not a sampling-path mystery: FusedResBlock keeps `self.original` (fused_resblock.py
    :730) whose `in_layers[-1]` / `out_layers[-1]` ARE the same nn.Conv2d objects it re-exposes as
    `in_conv` / `out_conv`. The int8/int4 converter walks `original` first and setattrs a wrapper into
    the Sequential, then reaches `FusedResBlock.in_conv` -- still the raw conv, because the setattr
    went into the Sequential, not here -- and wraps it a SECOND time. The two wrappers are distinct
    objects holding distinct copies of the quantized weights; only the FusedResBlock one is on the
    live path, because _forward_openai reads self.in_conv.

    Returns the evidence, so the claim is data and not narration.
    """
    import integration.fused_ops.fused_resblock as FR
    from integration.kernels.int8_optimized import OptimizedInt8Conv2d
    try:
        from integration.kernels.int4_optimized import OptimizedInt4Conv2d
        QT = (OptimizedInt8Conv2d, OptimizedInt4Conv2d)
    except Exception:
        QT = (OptimizedInt8Conv2d,)
    allq = list({id(c): c for c in unet.modules() if isinstance(c, QT)}.values())
    live, dead, shadowed = [], [], 0
    for rb in (m for m in unet.modules() if isinstance(m, FR.FusedResBlock)):
        for attr, seq, idx in (("in_conv", "in_layers", -1), ("out_conv", "out_layers", -1)):
            a = getattr(rb, attr, None)
            b = getattr(rb.original, seq, None)
            b = b[idx] if b is not None else None
            if isinstance(a, QT):
                live.append(id(a))
            if isinstance(b, QT) and b is not a:
                dead.append(id(b))
                # same shape and same dtype means the second wrapper re-quantized the same weights
                wa, wb = getattr(a, "weight_int8", None), getattr(b, "weight_int8", None)
                if wa is not None and wb is not None and wa.shape == wb.shape:
                    shadowed += 1
    return {"n_quant_conv_modules": len(allq), "n_live_on_fusedresblock": len(set(live)),
            "n_dead_under_original": len(set(dead)),
            "n_dead_matching_a_live_shape": shadowed,
            "dead_are_disjoint_from_live": len(set(live) & set(dead)) == 0,
            "accounted": len(set(live)) + len(set(dead)) == len(allq)}


def profile(label, mode, lin, act_bits, k, steps):
    for v in OPTIONAL_ENV:
        os.environ.pop(v, None)
    for kk, vv in (EXTRA or {}).items():
        os.environ[kk] = vv
    os.environ["MODIFF_LINEAR"] = lin
    os.environ["MODIFF_ACT_BITS"] = str(act_bits)
    os.environ["MODIFF_DELTA_REFRESH"] = str(k)
    os.environ["MODIFF_DELTA_REPORT"] = "0"
    os.environ.pop("MODIFF_DELTA_CLIP", None)
    H.STEPS = steps
    r, m, s = H.build(mode, None if mode == "fp16" else H.CALIB["int4" if "int4" in mode else "int8"],
                      "dynamic" if mode not in ("fp16", "int8_baseline", "int4_baseline") else "static")
    unet = m.model.diffusion_model

    # Uninstrumented wall clock FIRST, same process same model: the denominator of the error check.
    H.SEED = 1234
    H.latent(r, m, s)                                    # warm-up, discarded
    _, wall_ms = H.latent(r, m, s)

    blocks = classify(unet)
    t = BlockTimer()
    meta = {}
    for key, mod, name, mt in blocks:
        if t.wrap(mod, name, key):
            meta[key] = mt
    # head and tail: not blocks, but they are the rest of the step and the point of this instrument
    for key, obj in (("time_embed", getattr(unet, "time_embed", None)),
                     ("out_tail", getattr(unet, "out", None))):
        if obj is not None and t.wrap(obj, "forward", key):
            meta[key] = {"type": key, "where": "head" if key == "time_embed" else "tail",
                         "container": -1}

    audit = audit_convs(unet) if mode != "fp16" else {}

    H.SEED = 1234
    _, instr_ms = H.latent(r, m, s)
    tot, cnt, shp = t.totals(), t.counts(), dict(t.shape)
    del r, m, s
    torch.cuda.empty_cache()

    per_step = {kk: v / steps for kk, v in tot.items()}
    for kk in meta:
        if kk in shp:
            meta[kk]["in_shape"] = shp[kk]
    # attention tier = (head_dim, T). T comes from the recorded input [B, C, H, W] -> H*W.
    for kk, mt in meta.items():
        if mt.get("type") == "attention" and "in_shape" in mt:
            sh = mt["in_shape"]
            mt["T"] = int(sh[-1] * sh[-2]) if len(sh) == 4 else int(sh[1])
            mt["tier"] = f"hd{mt['head_dim']} T{mt['T']}"
    by_type = {}
    for kk, v in per_step.items():
        by_type.setdefault(meta.get(kk, {}).get("type", "?"), [0.0, 0])
        by_type[meta[kk]["type"]][0] += v
        by_type[meta[kk]["type"]][1] += 1
    return {"config": label, "mode": mode, "modiff_linear": lin, "act_bits": act_bits, "K": k,
            "wall_ms_per_step": wall_ms, "instrumented_ms_per_step": instr_ms,
            "blocks": per_step, "calls": cnt, "meta": meta,
            "by_type": {kk: {"ms_per_step": v[0], "n": v[1]} for kk, v in by_type.items()},
            "sum_over_wall": sum(per_step.values()) / wall_ms if wall_ms else 0.0,
            "conv_audit": audit}


EXTRA = {}


def main():
    global EXTRA
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--outdir", default="docs/current_state_2026-08-12")
    ap.add_argument("--only", default="", help="substring filter on the config label")
    a = ap.parse_args()
    H.BATCH = a.batch
    out = os.path.join(ROOT, a.outdir, "data")
    os.makedirs(out, exist_ok=True)
    rows = []
    for label, mode, lin, bits, k, extra in CONFIGS:
        if a.only and a.only not in label:
            continue
        EXTRA = extra
        print(f"=== {label}   {extra}", flush=True)
        try:
            r = profile(label, mode, lin, bits, k, a.steps)
        except Exception as e:                                  # one bad config must not lose the rest
            print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
            continue
        rows.append(r)
        print(f"  wall {r['wall_ms_per_step']:.2f}  instr {r['instrumented_ms_per_step']:.2f}  "
              f"coverage {r['sum_over_wall']:.3f}", flush=True)
        for tt, v in sorted(r["by_type"].items(), key=lambda kv: -kv[1]["ms_per_step"]):
            print(f"    {tt:16s} {v['ms_per_step']:7.2f} ms/step over {v['n']:3d}", flush=True)
        if r["conv_audit"]:
            print(f"    conv audit: {r['conv_audit']}", flush=True)
        json.dump(rows, open(os.path.join(out, "profile_blocks.json"), "w"), indent=1)
    print(f"wrote {os.path.join(out, 'profile_blocks.json')}  ({len(rows)} configs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

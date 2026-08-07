"""Which fused path does every layer ACTUALLY take, per configuration? Counted, not assumed.

The speedup question this exists to answer: int8 PTQ reaches 1.46x fp16 at batch 128, conv MoDiff
1.38x, and the current default (conv + attention projections) only 0.98x. If a fusion is silently not
engaging, the layers that fall back are where the deficit is -- and this repo has already been bitten
twice by exactly that: MODIFF_QUANT_LINEAR=0 skipping every fused attention epilogue while still
reporting a plausible speedup, and MoDiff-on-projections making all 21 blocks qout-ineligible.

Counts, per config, by wrapping the real methods and tallying calls on one sampling run:

  conv paths      forward_gn_fused_modiff   GN+SiLU+delta-quantize+a_hat in ONE kernel (the good one)
                  _forward_modulated        the unfused sibling: separate GN, separate quantize
                  _forward_first_step       t=T warm-up, runs once per layer per sample
  attention       _qout_eligible            whether the fused int8-output epilogue can engage
  linear          QuantLinearWxAx.modiff    MoDiff on the 42 attention projections
                  _out_i8                   the fused int8-output path on those projections

A layer taking _forward_modulated when it could have taken the GN-fused path is unfused work. The
point of the table is to find out how many there are and whether the count changes with the config.
"""

import json
import os
import sys
from collections import Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers"),
                os.path.join(ROOT, "integration/benchmarks/report"),
                os.path.join(ROOT, "docs/modiff_correctness_2026-08-03/scripts")]

import torch                                                                    # noqa: E402
import dynamic_delta_ab as H                                                    # noqa: E402

OUT = os.environ.get("FA_OUT", "docs/delta_clip_2026-08-06/data/fusion_audit.json")
#: (label, mode, MODIFF_LINEAR)
CONFIGS = [("int8 PTQ (MoDiff off)", "int8_baseline", "0"),
           ("conv MoDiff", "int8", "0"),
           ("conv+proj MoDiff (current default)", "int8", "1")]


def audit(label, mode, lin):
    os.environ["MODIFF_LINEAR"] = lin
    os.environ["MODIFF_ACT_Q"], os.environ["MODIFF_DELTA_CLIP"] = "127", "1.0"
    r, m, s = H.build(mode, None if mode == "fp16" else H.CALIB["int8"], "dynamic")
    unet = m.model.diffusion_model

    from integration.kernels.int8_optimized import OptimizedInt8Conv2d
    calls = Counter()
    keys = ("gn_fused", "modulated", "first_step", "standard", "forward", "mod_static_silu",
            "modiff_silu_resid", "from_int8", "from_int8_dual", "to_int8")
    layers_hit = {k: set() for k in keys}

    convs = [mod for mod in unet.modules() if isinstance(mod, OptimizedInt8Conv2d)]
    for i, mod in enumerate(convs):
        for meth, key in (("forward_gn_fused_modiff", "gn_fused"),
                          ("_forward_modulated", "modulated"),
                          ("_forward_first_step", "first_step"),
                          ("_forward_standard", "standard"),
                          ("forward", "forward"),
                          ("_forward_modulated_static_fused_silu", "mod_static_silu"),
                          ("forward_modiff_fused_silu_residual", "modiff_silu_resid"),
                          ("forward_from_int8", "from_int8"),
                          ("forward_from_int8_dual", "from_int8_dual"),
                          ("forward_to_int8", "to_int8")):
            fn = getattr(mod, meth, None)
            if fn is None:
                continue

            def wrap(fn=fn, key=key, i=i):
                def inner(*a, **kw):
                    calls[key] += 1
                    layers_hit[key].add(i)
                    return fn(*a, **kw)
                return inner
            setattr(mod, meth, wrap())

    # attention + projection state, read statically (no call counting needed)
    attn = [mod for mod in unet.modules()
            if type(mod).__name__ == "QuantizedStandardAttentionBlock"]
    try:
        from integration.kernels.wxax_linear import QuantLinearWxAx
        wx = [mod for mod in unet.modules() if isinstance(mod, QuantLinearWxAx)]
    except Exception:
        wx = []

    H.SEED = 1234
    H.latent(r, m, s)                       # one sampling run; DDIM STEPS forwards per layer
    # AFTER the run, for the reason above.
    qout = sum(bool(b._qout_eligible()) for b in attn) if attn else 0
    qout_why = {}
    if attn:
        b0 = attn[0]
        proj = getattr(b0, "proj", None)
        qout_why = {"fuse_proj_quant": bool(getattr(b0, "_fuse_proj_quant", False)),
                    "fq_frozen2": bool(getattr(b0, "_fq_frozen2", False)),
                    "proj_type": type(proj).__name__,
                    "proj_use_bias_res": bool(getattr(proj, "_use_bias_res", False)),
                    "proj_a_scale_set": getattr(proj, "a_scale", None) is not None,
                    "proj_calib": bool(getattr(proj, "_calib", False)),
                    "proj_bits_match": getattr(proj, "bits", None) == getattr(b0, "bits", None)}

    row = {"config": label, "mode": mode, "modiff_linear": lin,
           "n_quant_convs": len(convs),
           "conv_calls": dict(calls),
           "conv_layers_using": {k: len(v) for k, v in layers_hit.items()},
           "attn_blocks": len(attn), "attn_qout_eligible": qout,
           "wxax_projections": len(wx),
           "wxax_modiff": sum(bool(getattr(x, "modiff", False)) for x in wx),
           "wxax_out_i8": sum(bool(getattr(x, "_out_i8", False)) for x in wx),
           "qout_gate_terms_block0": qout_why}
    del m, s, r
    torch.cuda.empty_cache()
    return row


def main():
    os.environ["MODIFF_DELTA_REPORT"] = "0"
    os.environ["MODIFF_DELTA_REFRESH"] = os.environ.get("MODIFF_DELTA_REFRESH", "4")
    print(f"batch {H.BATCH}, DDIM {H.STEPS} -> each conv is called {H.STEPS} times per run\n",
          flush=True)
    rows = []
    for label, mode, lin in CONFIGS:
        row = audit(label, mode, lin)
        rows.append(row)
        print(f"=== {label} ===", flush=True)
        print(f"  quantized convs: {row['n_quant_convs']}", flush=True)
        for k in ("forward", "gn_fused", "modulated", "first_step", "standard",
                  "mod_static_silu", "modiff_silu_resid", "from_int8", "from_int8_dual", "to_int8"):
            n_layers = row["conv_layers_using"].get(k, 0)
            n_calls = row["conv_calls"].get(k, 0)
            if n_layers or n_calls:
                print(f"    {k:12s} {n_layers:3d} layers, {n_calls:6d} calls", flush=True)
        print(f"  attention: {row['attn_qout_eligible']}/{row['attn_blocks']} qout-eligible "
              f"(fused int8-out epilogue)   gate: {row['qout_gate_terms_block0']}", flush=True)
        print(f"  projections: {row['wxax_projections']} wxax, {row['wxax_modiff']} modiff, "
              f"{row['wxax_out_i8']} with fused int8-out\n", flush=True)
        with open(OUT, "w") as f:
            json.dump(rows, f, indent=2)
    os.environ["MODIFF_LINEAR"] = "0"
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()

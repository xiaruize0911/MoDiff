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

  updown paths    _prequant_gn_resize_conv_modiff / _prequant_gn_resize_conv (added 2026-08-10)

A layer taking _forward_modulated when it could have taken the GN-fused path is unfused work. The
point of the table is to find out how many there are and whether the count changes with the config.

THE UPDOWN ROW EXISTS BECAUSE ITS ABSENCE HID A BUG FOR MONTHS. The eight updown ResBlocks are not
dispatched through any conv METHOD -- FusedResBlock.forward calls the module-level
_prequant_gn_resize_conv_modiff directly, falling back to the baseline twin and then to an unfused
resize+GN+quantize triple. Wrapping conv methods therefore could not see them, and
_prequant_gn_resize_conv_modiff declined on every dynamic refresh step, which at
MODIFF_DELTA_REFRESH=1 is EVERY step: 0/8 fused, with this audit reporting nothing amiss. Those
three functions are module globals resolved at call time, so patching the module attribute catches
them. See docs/updown_refresh_fusion_2026-08-10/.
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
#: (label, mode, MODIFF_LINEAR, act_bits, delta_refresh). K is in the surface because the updown
#: fusion is K-dependent -- that dependence is the bug the updown row was added to make visible.
CONFIGS = [("W8A8 PTQ (MoDiff off)",        "int8_baseline", "0", 8, 4),
           ("W8A8 conv-only  K=4",          "int8",          "0", 8, 4),
           ("W8A8 conv-only  K=1",          "int8",          "0", 8, 1),
           ("W8A8 conv+proj  K=4",          "int8",          "1", 8, 4),
           ("W8A8 conv+proj  K=1",          "int8",          "1", 8, 1),
           ("W8A4 conv+proj  K=1",          "int8",          "1", 4, 1),
           ("W4A4 conv-only  K=4",          "int4",          "0", 8, 4),
           ("W4A4 conv+proj  K=4",          "int4",          "1", 8, 4)]


def audit(label, mode, lin, act_bits=8, refresh=4):
    os.environ["MODIFF_LINEAR"] = lin
    # MODIFF_ACT_Q and MODIFF_DELTA_CLIP were retired 2026-08-10. ACT_Q is simply no longer read
    # (setting it here would have been a silent no-op); CLIP now RAISES on anything but 1.0, so it
    # is not set at all. Activation width comes from MODIFF_ACT_BITS in {8, 4}.
    os.environ["MODIFF_ACT_BITS"] = str(act_bits)
    os.environ.pop("MODIFF_DELTA_CLIP", None)
    os.environ["MODIFF_DELTA_REFRESH"] = str(refresh)
    import integration.fused_ops.fused_resblock as FR
    updown = Counter()
    originals = {}
    for name in ("_prequant_gn_resize_conv_modiff", "_prequant_gn_resize_conv"):
        fn = getattr(FR, name)
        originals[name] = fn

        def wrap(fn=fn, name=name):
            def inner(*a, **kw):
                out = fn(*a, **kw)
                updown[f"{name}:calls"] += 1
                updown[f"{name}:{'fused' if out is not None else 'declined'}"] += 1
                return out
            return inner
        setattr(FR, name, wrap())

    # ---- PER-SITE attribution of the delta kernels, added 2026-08-12 ----
    #
    # Every report so far has attributed these by ARITHMETIC: gn_apply runs 83 times/step, the conv
    # column says 62, so 21 must be the qkv. That inference is right here, but it is an inference, and
    # docs/gn_stats_in_epilogue_2026-08-11 lists exactly this kind of bound ("68 is 83 - 15") as one of
    # its own stop conditions. So attribute by the IMMEDIATE PYTHON CALLER instead: one frame lookup
    # per call, ~4k calls per run, and the answer is observed rather than derived.
    #
    # Wrapping the kernel and not the callers is deliberate: `_qkv_from_gn_modiff_fused` can return
    # None at seven different preconditions before reaching the kernel, so counting method entries
    # would over-count the qkv side by however often it declines.
    import modiff_cutlass as _mcaudit
    delta_sites = Counter()
    delta_originals = {}
    for kname in ("group_norm_silu_delta_quantize_nhwc", "delta_absmax_fp16",
                  "step1_static_quantize_fprop", "group_norm_silu_delta_quantize_resize_nhwc"):
        fn = getattr(_mcaudit, kname, None)
        if fn is None:
            continue
        delta_originals[kname] = fn

        def wrapk(fn=fn, kname=kname):
            def inner(*a, **kw):
                f = sys._getframe(1)
                site = f"{kname} <- {os.path.basename(f.f_code.co_filename)}:{f.f_code.co_name}"
                delta_sites[site] += 1
                return fn(*a, **kw)
            return inner
        setattr(_mcaudit, kname, wrapk())

    r, m, s = H.build(mode, None if mode == "fp16" else H.CALIB["int8"], "dynamic")
    unet = m.model.diffusion_model

    # BOTH conv classes. This scan used to name only OptimizedInt8Conv2d, so every int4 config
    # reported "quantized convs: 0" and an empty fusion column while 70 Int4Conv layers were in fact
    # running -- the same failure mode as the updown blind spot above: an audit that cannot see a
    # path reports nothing wrong with it. Int4Conv carries the same method names.
    # NOT wrapped in try/except. The first attempt at this imported a class named `Int4Conv`, which
    # does not exist -- the class is OptimizedInt4Conv2d -- and a bare except swallowed the
    # ImportError, so int4 configs kept reporting "quantized convs: 0" exactly as before the fix was
    # supposedly applied. In an audit whose whole purpose is to find paths nobody is looking at, a
    # silently skipped import is the bug it is meant to catch. Let it raise.
    from integration.kernels.int8_optimized import OptimizedInt8Conv2d
    from integration.kernels.int4_optimized import OptimizedInt4Conv2d
    conv_types = (OptimizedInt8Conv2d, OptimizedInt4Conv2d)
    calls = Counter()
    keys = ("gn_fused", "modulated", "first_step", "standard", "forward", "mod_static_silu",
            "modiff_silu_resid", "from_int8", "from_int8_dual", "to_int8",
            "from_int4", "from_int4_dual", "to_int4")
    layers_hit = {k: set() for k in keys}

    # Deduplicated by id, because unet.modules() yields a module once per parent that references it.
    # NOTE, and this is measured rather than explained: even after dedup the walk finds 140 quantized
    # conv MODULES while only 70 are ever called (every config's per-layer column reads 70 layers).
    # I first wrote this dedup believing 140 was a double count of 70; it is not, the number did not
    # move. Why 70 of 140 modules are never invoked during sampling is OPEN -- do not read
    # n_conv_modules as the layer count, read n_conv_layers_called.
    convs = list({id(mod): mod for mod in unet.modules()
                  if isinstance(mod, conv_types)}.values())
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
                          ("forward_to_int8", "to_int8"),
                          # Int4Conv's equivalents; getattr returns None on the int8 class and
                          # vice versa, so one table serves both.
                          ("forward_from_int4", "from_int4"),
                          ("forward_from_int4_dual", "from_int4_dual"),
                          ("forward_to_int4", "to_int4")):
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

    for name, fn in originals.items():
        setattr(FR, name, fn)
    for kname, fn in delta_originals.items():
        setattr(_mcaudit, kname, fn)
    # Reported per FORWARD with seeding kept separate, not as one average over all forwards. An
    # average mixes two different things: the steady state, where all 8 blocks either fuse or do
    # not, and the a_hat seeding forwards, where the MoDiff path MUST fall back because there is no
    # cache to subtract yet. Averaging them gave "7.71 of 8 fused", which reads as a partial
    # failure when the truth is "8 of 8, on every forward that is not a seeding forward".
    n_calls = updown.get("_prequant_gn_resize_conv_modiff:calls", 0)
    fwd = n_calls // 8 if n_calls else 0
    mf = updown.get("_prequant_gn_resize_conv_modiff:fused", 0) // 8
    bf = updown.get("_prequant_gn_resize_conv:fused", 0) // 8
    bd = updown.get("_prequant_gn_resize_conv:declined", 0) // 8
    row = {"config": label, "mode": mode, "modiff_linear": lin,
           "act_bits": act_bits, "delta_refresh": refresh,
           "updown_calls": dict(updown),
           "updown": {"forwards": fwd, "blocks": 8,
                      "modiff_fused_forwards": mf, "baseline_fused_forwards": bf,
                      "unfused_forwards": bd,
                      "all8_every_nonseeding_forward": (mf + bf + bd) == fwd and bd <= 2},
           "n_conv_modules": len(convs),
           "n_conv_layers_called": len({i for v in layers_hit.values() for i in v}),
           "conv_calls": dict(calls),
           "conv_layers_using": {k: len(v) for k, v in layers_hit.items()},
           # calls per STEP, so it lines up with the trace tables (which are all per step) rather
           # than with this script's per-run totals.
           "delta_kernel_sites_per_step": {k: v / H.STEPS for k, v in sorted(delta_sites.items())},
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
    # FA_ONLY selects a substring of the labels: each config is a full sampling run, so answering one
    # question ("which caller runs gn_apply?") should not cost eight of them. Unset = all eight, as
    # before. Writing to a different FA_OUT is the caller's job when they filter, since a partial run
    # would otherwise replace the committed eight-config dataset.
    only = os.environ.get("FA_ONLY", "")
    configs = [c for c in CONFIGS if not only or only in c[0]]
    if only and OUT.endswith("docs/delta_clip_2026-08-06/data/fusion_audit.json"):
        sys.exit(f"FA_ONLY={only!r} is a subset run and would overwrite the canonical {OUT}. "
                 f"Set FA_OUT to a path in your own report directory.")
    print(f"batch {H.BATCH}, DDIM {H.STEPS} -> each conv is called {H.STEPS} times per run\n",
          flush=True)
    rows = []
    for label, mode, lin, ab, k in configs:
        row = audit(label, mode, lin, ab, k)
        rows.append(row)
        print(f"=== {label} ===", flush=True)
        print(f"  conv modules: {row['n_conv_modules']}, "
              f"of which CALLED: {row['n_conv_layers_called']}", flush=True)
        for k in ("forward", "gn_fused", "modulated", "first_step", "standard",
                  "mod_static_silu", "modiff_silu_resid", "from_int8", "from_int8_dual", "to_int8",
                  "from_int4", "from_int4_dual", "to_int4"):
            n_layers = row["conv_layers_using"].get(k, 0)
            n_calls = row["conv_calls"].get(k, 0)
            if n_layers or n_calls:
                print(f"    {k:12s} {n_layers:3d} layers, {n_calls:6d} calls", flush=True)
        if row.get("delta_kernel_sites_per_step"):
            print("  delta kernels, per step, BY CALLER (observed, not inferred):", flush=True)
            for site, n in row["delta_kernel_sites_per_step"].items():
                print(f"    {n:6.2f}  {site}", flush=True)
        print(f"  attention: {row['attn_qout_eligible']}/{row['attn_blocks']} qout-eligible "
              f"(fused int8-out epilogue)   gate: {row['qout_gate_terms_block0']}", flush=True)
        u = row["updown"]
        print(f"  updown: 8/8 blocks fused on  {u['modiff_fused_forwards']:3d} modiff + "
              f"{u['baseline_fused_forwards']:3d} baseline  of {u['forwards']} forwards; "
              f"{u['unfused_forwards']} unfused (a_hat seeding)", flush=True)
        print(f"  projections: {row['wxax_projections']} wxax, {row['wxax_modiff']} modiff, "
              f"{row['wxax_out_i8']} with fused int8-out\n", flush=True)
        with open(OUT, "w") as f:
            json.dump(rows, f, indent=2)
    os.environ["MODIFF_LINEAR"] = "0"
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()

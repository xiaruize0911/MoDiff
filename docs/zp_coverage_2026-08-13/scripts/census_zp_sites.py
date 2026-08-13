"""What do the zero-point-guarded quantize sites ACTUALLY quantize? Observed, not reasoned.

docs/zero_point_2026-08-13/FINDINGS.md counted 70 contaminated pairs on the MoDiff arm, 62 of them
via step1_static_quantize_pack_int4_fprop, and labelled that entry point "the t=T activation grid".
That label decides the whole remaining work list: if those 62 really do quantize the activation on
the activation grid while their conv adds a zp-corrected bias, they must learn z. If instead they
quantize a DELTA (x - a_hat) on the delta grid into a BIAS-FREE accumulate, then ignoring z is
correct there and teaching them z would corrupt the a_hat update -- the guard is the bug, not the
kernel.

Reading the code says the second. This script refuses to rely on that reading and records three
facts per call, at the kernel boundary, on the real datapath:

  1. is an a_hat cache passed to the quantizer (and therefore subtracted)?  -> delta vs activation
  2. is the scale tensor the conv's own static_input_scale, or a delta-table slice?
  3. which conv entry point consumes the codes, i.e. does that conv add a bias?

Fact 1 alone settles applicability: z is defined by a = (a_q - z)/s for the ACTIVATION. A difference
of two activations has no z -- it cancels -- so any site that subtracts a_hat before quantizing is
z-free by construction, whatever grid it uses. Facts 2 and 3 are recorded because the failure this
guard exists to prevent is specifically "symmetric codes against a zp-corrected bias", and fact 3 is
the half of that sentence nobody has yet measured.

Run: python docs/zp_coverage_2026-08-13/scripts/census_zp_sites.py     # ~4 min, needs the GPU
Writes docs/zp_coverage_2026-08-13/data/site_census.json
"""
import collections
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(ROOT)
sys.path[:0] = [os.path.join(ROOT, "docs/attn_modiff_2026-08-13/scripts"),
                os.path.join(ROOT, "docs/qdiff_bridge_2026-08-12/scripts"),
                ROOT, os.path.join(ROOT, "src/taming-transformers"),
                os.path.join(ROOT, "integration/benchmarks/report"),
                os.path.join(ROOT, "docs/modiff_correctness_2026-08-03/scripts")]

#: Collect the census rather than raise on the first ignored zero point -- that is what this env var
#: is for (docs/zero_point_2026-08-13/FINDINGS.md, "The guard, and why it is strict by default").
os.environ["MODIFF_ZP_STRICT"] = "0"
#: _refold_zp_bias refuses a non-zero zero point on a PADDED conv (2026-08-13): the fold is
#: per-output-channel while the padding error is per-output-pixel. Every calibrated conv in this model is
#: 3x3 padding=1, so this script -- whose whole subject is the asymmetric grid -- cannot build anything
#: without the override. It does not make the configuration correct; it makes the defect reproducible,
#: which is what a script that MEASURES the defect needs. See docs/zp_coverage_2026-08-13/FINDINGS.md.
os.environ.setdefault("MODIFF_ZP_ALLOW_PADDED", "1")
os.environ["MODIFF_LINEAR"] = "0"

import torch                                                               # noqa: E402
import modiff_cutlass as MC                                                # noqa: E402
import dynamic_delta_ab as H                                               # noqa: E402

D = "docs/zp_coverage_2026-08-13"
ZP_TABLE = "docs/zero_point_2026-08-13/data/int4_calibration_zp_clip4.5.pt"

#: REPRODUCES THE OLD, NAME-BASED CENSUS. With coverage complete and delta sites declaring grid="delta",
#: a run of this script now reports ZERO guard hits -- which is the correct post-fix state and is itself
#: the confirmation that nothing is contaminated. But it means the historical counts that motivated all
#: of this (70 pairs on the MoDiff arm, 62 of them via step1_static_quantize_pack_int4_fprop, 8 on PTQ)
#: are no longer reproducible from a fresh run, and a finding whose evidence cannot be regenerated is a
#: finding on trust. With CENSUS_COUNT_DELTA_AS_GAPS=1 the guard wrapper counts delta sites as gaps
#: exactly as the name-based classification did, so those numbers come back -- alongside the
#: classification that shows they were false positives.
COUNT_DELTA_AS_GAPS = os.environ.get("CENSUS_COUNT_DELTA_AS_GAPS", "0") == "1"
OUT = f"{D}/data/site_census{'_name_based' if COUNT_DELTA_AS_GAPS else ''}.json"

#: entry point -> (index of the scale tensor, index of the a_hat tensor or None)
#: Signatures read off the call sites, not guessed:
#:   step1_static_quantize_pack_int4_fprop(x, a_hat, scale, smooth)          int4_optimized.py:1457
#:   step1_static_quantize_pack_int4_fprop_silu(x, a_hat, scale, smooth)     int4_optimized.py:839
#:   group_norm_silu_quantize_pack_nhwc(x,w,b,ng,eps,silu,scale,smooth,...)  fused_resblock.py:458
#:   group_norm_silu_quantize_resize_nhwc(... scale@6 ..., dir, is_int4)     fused_resblock.py:711
#:   group_norm_silu_delta_quantize_resize_nhwc(... scale@6 ..., ah@13, *dyn) fused_resblock.py:643
#:   group_norm_silu_delta_quantize_pack_nhwc(x,w,b,ah@3,ng,eps,silu,scale@7,...) int4_optimized.py:940
#:   upsample2x_quantize_pack_noahat_fprop(x, scale, smooth, ah)            fused_resblock.py:1248
#:   scale_quantize_and_pack(x, scale)                                      int4_optimized.py:416
WATCH = {
    "step1_static_quantize_pack_int4_fprop":            (2, 1),
    "step1_static_quantize_pack_int4_fprop_silu":       (2, 1),
    "group_norm_silu_quantize_pack_nhwc":               (6, None),
    "group_norm_silu_quantize_pack_nhwc_zp":            (6, None),
    "group_norm_silu_quantize_pack_nhwc_fast":          (6, None),
    "group_norm_silu_quantize_resize_nhwc":             (6, None),
    "group_norm_silu_delta_quantize_resize_nhwc":       (6, 13),
    "group_norm_silu_delta_quantize_pack_nhwc":         (7, 3),
    "group_norm_silu_delta_quantize_pack_cat2_nhwc":    (8, 4),
    "upsample2x_quantize_pack_noahat_fprop":            (1, 3),
    "scale_quantize_and_pack":                          (1, None),
    "scale_quantize_and_pack_zp":                       (1, None),
    "group_norm_silu_quantize_resize_nhwc_zp":          (6, None),
    "upsample2x_quantize_pack_noahat_fprop_zp":         (1, 3),
    "quantize_act_int4_pack":                           (1, None),
    "quantize_and_pack":                                (None, None),
}
#: conv entry points, and whether each ADDS A BIAS inside the kernel. The o_hat family takes no bias
#: argument at all (pybind.cpp:102,118) -- that is the fact fact 3 is after.
CONVS = {
    "conv2d_int4_evt_o_hat": False,
    "conv2d_int4_evt_o_hat_residual": False,
    "conv2d_int4_fprop_o_hat": False,
    "conv2d_int4_fprop_o_hat_residual": False,
    "conv2d_int4_fprop": "arg",          # bias is argument 3; empty tensor == no bias
    "conv2d_int4_evt_d1": "arg",
    "conv2d_int4_dequant_fp16_tuned": False,
    "conv2d_int4_fprop_relu_requant_int4": "arg",
    "conv2d_int4_fprop_bias_residual_dual": "arg",
}

#: (layer, where) -> facts. Capped per key: this runs 70 convs x 50 steps and the interesting content
#: is the SET of behaviours per site, not the count.
CAP = 3
sites = collections.OrderedDict()
guard_hits = collections.OrderedDict()
_ctx = {"conv": None, "where": None}
_last_quant = {"key": None}


def _ptr(t):
    try:
        return int(t.data_ptr())
    except Exception:
        return None


def _caller():
    """File:line of the first frame outside this census, i.e. the site that issued the quantize.

    Recorded because two entry points (quantize_act_int4_pack, group_norm_silu_quantize_pack_nhwc_fast)
    are reached from the ATTENTION families -- QuantLinearWxAx and the quantized AttentionBlocks --
    which carry no static_input_zp and no zp-corrected bias, so they are out of fix #2's scope rather
    than gaps in it. That is a claim about who calls what, and it should be observed like the rest.
    """
    import sys as _s
    f = _s._getframe(1)
    while f is not None:
        fn = f.f_code.co_filename
        if "census_zp_sites" not in fn:
            return f"{os.path.relpath(fn, ROOT)}:{f.f_lineno}"
        f = f.f_back
    return None


def _fact(name, args):
    si, ai = WATCH[name]
    scale_ptr = scale_val = None
    if si is not None and si < len(args) and torch.is_tensor(args[si]):
        scale_ptr = _ptr(args[si])
        if args[si].numel() >= 1:
            scale_val = float(args[si].reshape(-1)[0].item())
    ah = None
    if ai is not None and ai < len(args) and torch.is_tensor(args[ai]):
        ah = int(args[ai].numel())
    conv = _ctx["conv"]
    act_ptrs = set()
    if conv is not None:
        act_ptrs.add(_ptr(conv.static_input_scale))
        cst = getattr(conv, "_cached_scale_tensor", None)
        if cst is not None:
            act_ptrs.add(_ptr(cst))
    out = {
        "quantizer": name,
        "a_hat_numel": ah,
        "subtracts_a_hat": (ah is not None and ah > 0),
        "scale_value": scale_val,
        #: None, not False, when there is no conv context -- an unguarded site tells us nothing about
        #: whose scale this is, and recording False there would read as "delta grid" and be a lie.
        "scale_is_static_input_scale": (
            None if (conv is None or scale_ptr is None) else (scale_ptr in act_ptrs)),
        "conv_static_input_scale": (float(conv.static_input_scale.item()) if conv is not None else None),
        "zp": (float(getattr(conv, "_zp_float", 0.0)) if conv is not None else None),
        "conv_has_bias": (conv is not None and getattr(conv, "bias", None) is not None),
        "consumer_conv": None,
        "consumer_adds_bias": None,
        "caller": _caller(),
    }
    return out


def _wrap_quant(name, fn):
    def w(*args, **kw):
        key = f"{_ctx['where'] or '(unguarded)'}|{name}"
        lst = sites.setdefault(key, [])
        if len(lst) < CAP:
            try:
                lst.append(_fact(name, args))
                _last_quant["key"] = (key, len(lst) - 1)
            except Exception as e:               # never let the census change the datapath
                lst.append({"quantizer": name, "census_error": repr(e)})
                _last_quant["key"] = None
        else:
            _last_quant["key"] = None
        # Context is consumed by exactly one quantize. Leaving it set would attribute the NEXT
        # unguarded quantizer to whichever site last called the guard -- i.e. invent an association.
        _ctx["conv"] = _ctx["where"] = None
        return fn(*args, **kw)
    return w


def _wrap_conv(name, fn, bias_kind):
    def w(*args, **kw):
        k = _last_quant["key"]
        if k is not None:
            rec = sites[k[0]][k[1]]
            if rec.get("consumer_conv") is None:
                rec["consumer_conv"] = name
                if bias_kind == "arg":
                    # bias is the 4th positional in conv2d_int4_fprop's signature; an empty tensor
                    # means the caller adds it in Python instead (int4_optimized.py:438).
                    b = args[3] if len(args) > 3 and torch.is_tensor(args[3]) else None
                    rec["consumer_adds_bias"] = bool(b is not None and b.numel() > 0)
                else:
                    rec["consumer_adds_bias"] = bool(bias_kind)
            _last_quant["key"] = None
        return fn(*args, **kw)
    return w


def install_patches():
    n = 0
    for name in list(WATCH):
        f = getattr(MC, name, None)
        if f is not None:
            setattr(MC, name, _wrap_quant(name, f))
            n += 1
    for name, bk in CONVS.items():
        f = getattr(MC, name, None)
        if f is not None:
            setattr(MC, name, _wrap_conv(name, f, bk))
    # The two guards, patched to RECORD the calling conv and to publish it as context for the
    # quantizer wrapper on the very next line (every guarded site calls the guard immediately
    # before its kernel).
    import integration.kernels.int4_optimized as I4
    import integration.fused_ops.fused_resblock as FR

    orig_m = I4.OptimizedInt4Conv2d._zp_unsupported

    def m_guard(self, where, grid="activation"):
        _ctx["conv"], _ctx["where"] = self, where
        z = float(getattr(self, "_zp_float", 0.0))
        if z != 0.0 and (COUNT_DELTA_AS_GAPS or grid != "delta"):
            guard_hits.setdefault(f"{self.layer_name}|{where}", 0)
            guard_hits[f"{self.layer_name}|{where}"] += 1
        return orig_m(self, where, grid)

    I4.OptimizedInt4Conv2d._zp_unsupported = m_guard
    orig_f = FR._zp_unsupported

    def f_guard(conv, where, grid="activation"):
        _ctx["conv"], _ctx["where"] = conv, where
        z = float(getattr(conv, "_zp_float", 0.0))
        if z != 0.0 and (COUNT_DELTA_AS_GAPS or grid != "delta"):
            k = f"{getattr(conv, 'layer_name', None)}|{where}"
            guard_hits.setdefault(k, 0)
            guard_hits[k] += 1
        return orig_f(conv, where, grid)

    FR._zp_unsupported = f_guard
    return n


def run(mode, table):
    H.AUTO_DELTA_TABLE = True
    os.environ["MODIFF_DELTA_MODE"] = "static"
    r, m, s = H.build(mode, table, "static")
    from integration.kernels.int4_optimized import OptimizedInt4Conv2d
    nz = sum(1 for mo in m.model.diffusion_model.modules()
             if isinstance(mo, OptimizedInt4Conv2d) and float(mo.static_input_zp.item()) != 0.0)
    print(f"  {mode}: {nz} convs carry a non-zero zero point", flush=True)
    if nz == 0:
        print("  GATE FAILED: the table reached nothing -- every census row below would be vacuous")
        del r, m, s
        torch.cuda.empty_cache()
        return None
    H.latent(r, m, s)
    del r, m, s
    torch.cuda.empty_cache()
    return nz


def main():
    H.STEPS, H.BATCH = 6, 2       # a census, not a measurement: only the SET of behaviours matters
    print(f"patched {install_patches()} quantize entry points\n")
    out = {}
    for mode in ("int4_baseline", "int4"):
        sites.clear()
        guard_hits.clear()
        print(f"=== {mode} (asymmetric table loaded, MODIFF_ZP_STRICT=0) ===", flush=True)
        nz = run(mode, ZP_TABLE)
        out[mode] = {"convs_with_zp": nz,
                     "guard_hits": dict(guard_hits),
                     "guard_hit_layers": len(guard_hits),
                     "sites": {k: v for k, v in sites.items()}}
        print(f"  guard fired for {len(guard_hits)} (layer, site) pairs")
        for key, recs in sites.items():
            r0 = recs[0]
            if "census_error" in r0:
                print(f"    {key:70s} CENSUS ERROR {r0['census_error']}")
                continue
            print(f"    {key:70s} a_hat={r0['subtracts_a_hat']!s:5s} "
                  f"act_grid={r0['scale_is_static_input_scale']!s:5s} "
                  f"consumer={r0['consumer_conv']} bias={r0['consumer_adds_bias']}")
        print()

    print("=== COVERAGE ===")
    for mode in out:
        n = out[mode]["guard_hit_layers"]
        print(f"  {mode:14s} {n} (layer, site) pairs still ignore a zero point they should apply"
              + ("   <-- COMPLETE" if n == 0 else "   <-- INCOMPLETE"))
    print()

    os.makedirs(f"{D}/data", exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {OUT}"
          + ("   (delta sites counted as gaps: reproduces the OLD name-based census)"
             if COUNT_DELTA_AS_GAPS else "   (0 guard hits here means coverage is complete)"))

    # The verdict this script exists to reach, stated as a rule rather than a judgement.
    print("\n=== CLASSIFICATION ===")
    for mode in out:
        for key, recs in out[mode]["sites"].items():
            if "census_error" in recs[0]:
                continue
            delta = all(r.get("subtracts_a_hat") for r in recs)
            act = all(not r.get("subtracts_a_hat") for r in recs)
            where, qname = key.split("|", 1)
            guarded = where != "(unguarded)"
            honours_z = qname.endswith("_zp")
            #: a gap needs BOTH an activation-grid quantize AND a consumer whose bias carries the
            #: -z*sum(w_q) correction. No zp-bearing conv in context => a different module family
            #: (attention), which has neither, so it is out of scope rather than contaminated.
            zp_bearing = any(r.get("zp") not in (None, 0.0) for r in recs) or honours_z
            caller = recs[0].get("caller")
            if delta:
                verdict = "DELTA -> z inapplicable" + ("; guard is a FALSE POSITIVE" if guarded else "")
            elif honours_z:
                verdict = "ACTIVATION -> HONOURS z (routed)"
            elif not act:
                verdict = "MIXED -> split by argument, not by entry point name"
            elif not zp_bearing:
                verdict = "ACTIVATION, but no zp-bearing conv -> OUT OF SCOPE (other family)"
            else:
                verdict = "ACTIVATION -> z required; REAL GAP" + ("" if guarded else "; UNGUARDED")
            print(f"  [{mode:14s}] {qname:46s} {verdict}")
            print(f"                   called from {caller}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

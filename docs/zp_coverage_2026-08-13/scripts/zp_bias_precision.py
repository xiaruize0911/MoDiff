"""Is the fp16 STORED bias the reason an asymmetric grid loses, now that coverage is complete?

With a zero point the conv computes

    o[k] = (sum_i w_q[k,i] * a_q[i]) * ws[k]/s  +  bias[k] - z * sum_i w_q[k,i] * ws[k]/s
           \\_________ contains + z*sum(w_q) ________/          \\____ corr[k], cancels it ____/

so TWO LARGE TERMS CANCEL to something of the size of the real output. That is fine in exact
arithmetic and fine in fp32. It is not obviously fine when corr[k] is folded into a bias that is
STORED IN FP16 (int4_optimized.py's bias follows _orig_bias's dtype, and _evt_bias_f32 is built by
widening that already-rounded fp16 value, so the lost bits are not recoverable).

The injected error is |corr| * 2^-11 per channel, systematic, and it does not shrink with the
activation error. test_int4_zero_point.py accepted this rounding with the argument that it is "~2.5e-4
relative against a 4-bit activation error of order 0.3" -- which compares a relative error on corr to a
relative error on the activation, and is only sound if |corr| is of the same order as |o|. This script
measures that ratio instead of assuming it.

MEASURED PER CONV, on the real checkpoint with the real asymmetric table:

    |corr|            the folded correction's magnitude
    |bias_eff|        what actually gets stored
    fp16 error        |bias_eff_fp16 - bias_eff_fp64|, the absolute error the storage injects
    |o|               the conv output's own scale, measured from a real activation
    error / |o|       THE NUMBER THIS SCRIPT EXISTS FOR

If error/|o| is on the order of percent, the fp16 store is a real defect and the asymmetric arm has
been measured through it -- in which case the fix is to keep the folded bias in fp32 rather than to
conclude anything about zero points.

Run: python docs/zp_coverage_2026-08-13/scripts/zp_bias_precision.py    # ~2 min, needs the GPU
"""
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(ROOT)
sys.path[:0] = [os.path.join(ROOT, "docs/qdiff_bridge_2026-08-12/scripts"),
                ROOT, os.path.join(ROOT, "src/taming-transformers"),
                os.path.join(ROOT, "integration/benchmarks/report"),
                os.path.join(ROOT, "docs/modiff_correctness_2026-08-03/scripts")]
os.environ["MODIFF_ZP_STRICT"] = "0"
#: _refold_zp_bias refuses a non-zero zero point on a PADDED conv (2026-08-13): the fold is
#: per-output-channel while the padding error is per-output-pixel. Every calibrated conv in this model is
#: 3x3 padding=1, so this script -- whose whole subject is the asymmetric grid -- cannot build anything
#: without the override. It does not make the configuration correct; it makes the defect reproducible,
#: which is what a script that MEASURES the defect needs. See docs/zp_coverage_2026-08-13/FINDINGS.md.
os.environ.setdefault("MODIFF_ZP_ALLOW_PADDED", "1")
os.environ["MODIFF_LINEAR"] = "0"

import torch                                                              # noqa: E402
import dynamic_delta_ab as H                                             # noqa: E402

D = "docs/zp_coverage_2026-08-13"
ZP_TABLE = "docs/zero_point_2026-08-13/data/int4_calibration_zp_clip4.5.pt"


def main():
    H.STEPS, H.BATCH = 4, 2
    H.AUTO_DELTA_TABLE = True
    os.environ["MODIFF_DELTA_MODE"] = "static"
    from integration.kernels.int4_optimized import OptimizedInt4Conv2d

    # int4_baseline: the arm the zero point was predicted to help, and the one that got worse.
    r, m, s = H.build("int4_baseline", ZP_TABLE, "static")

    convs = {}
    for name, mo in m.model.diffusion_model.named_modules():
        if isinstance(mo, OptimizedInt4Conv2d) and float(mo.static_input_zp.item()) != 0.0:
            convs[name] = mo
    print(f"{len(convs)} convs carry a non-zero zero point", flush=True)

    # Output scale per conv, from a real forward.
    #
    # NOT A forward_hook. On the fused path the conv is entered as conv._conv_from_int4(...) /
    # forward_from_int4(...), a direct METHOD call, so nn.Module hooks never fire -- the first version
    # of this script used one and every |o| came back nan. That is the identical trap recorded as
    # mistake #3 in docs/zero_point_2026-08-13/FINDINGS.md, and the nan is why it was caught here
    # rather than believed. Wrapping the methods observes every entry, fused or not.
    scales = {}
    METHODS = ("_conv_from_int4", "_conv_from_int4_o_hat", "forward_from_int4", "forward")
    originals = {}

    def wrap(mo, name, meth):
        fn = getattr(mo, meth, None)
        if fn is None:
            return
        originals[(name, meth)] = fn

        def w(*a, **kw):
            out = fn(*a, **kw)
            o = out[0] if isinstance(out, (tuple, list)) else out
            if torch.is_tensor(o):
                v = float(o.detach().float().abs().mean())
                scales[name] = max(scales.get(name, 0.0), v)
            return out
        setattr(mo, meth, w)

    for name, mo in convs.items():
        for meth in METHODS:
            wrap(mo, name, meth)
    H.SEED = 1234
    H.latent(r, m, s)
    for (name, meth), fn in originals.items():
        try:
            delattr(convs[name], meth)      # drop the instance attribute, restoring the class method
        except AttributeError:
            setattr(convs[name], meth, fn)
    missing = [n for n in convs if n not in scales]
    print(f"output scale observed for {len(scales)}/{len(convs)} convs", flush=True)
    if missing:
        print(f"  NOT OBSERVED ({len(missing)}): {missing[:4]}{' ...' if len(missing) > 4 else ''}")
        print("  err/|o| is left null for those rather than filled with a guess.")

    rows = []
    for name, mo in convs.items():
        z = float(mo._zp_float)
        sc = float(mo.static_input_scale.item())
        wq = mo.weight_sum_q.double().view(-1)
        ws = mo.weight_scale_channel.double().view(-1)
        corr = -(z / sc) * wq * ws
        base = mo._orig_bias
        exact = corr if base is None else (base.double().view(-1) + corr)
        stored = mo.bias.double().view(-1)                 # what the epilogue actually reads
        err = (stored - exact).abs()
        o = scales.get(name, float("nan"))
        rows.append({
            "layer": name, "z": z, "s": sc,
            "corr_absmax": float(corr.abs().max()),
            "corr_absmed": float(corr.abs().median()),
            "bias_dtype": str(mo.bias.dtype),
            "fp16_err_max": float(err.max()),
            "fp16_err_med": float(err.median()),
            "out_absmean": o,
            "err_over_out": float(err.max()) / o if o and o == o else None,
        })

    rows.sort(key=lambda r_: -(r_["err_over_out"] or 0))
    print(f"\n{'layer':44s}{'z':>4}{'|corr|max':>11}{'fp16 err':>10}{'|o|':>9}{'err/|o|':>10}")
    for r_ in rows[:15]:
        print(f"{r_['layer'][:44]:44s}{r_['z']:4.0f}{r_['corr_absmax']:11.2f}"
              f"{r_['fp16_err_max']:10.4f}{r_['out_absmean']:9.4f}"
              f"{100 * (r_['err_over_out'] or 0):9.2f}%")

    ratios = [r_["err_over_out"] for r_ in rows if r_["err_over_out"] is not None]
    corrs = [r_["corr_absmax"] for r_ in rows]
    outs = [r_["out_absmean"] for r_ in rows if r_["out_absmean"] == r_["out_absmean"]]
    dtypes = sorted({r_["bias_dtype"] for r_ in rows})
    print(f"\nstored bias dtype(s)        {dtypes}")
    print(f"median |corr|max            {statistics.median(corrs):.2f}")
    if outs:
        print(f"median |o| (abs mean)       {statistics.median(outs):.4f}")
        print(f"|corr| / |o|, median        "
              f"{statistics.median(corrs) / statistics.median(outs):.0f}x")
    os.makedirs(f"{D}/data", exist_ok=True)
    json.dump({"convs": len(convs), "bias_dtypes": dtypes, "rows": rows,
               "median_err_over_out": (statistics.median(ratios) if ratios else None),
               "worst_err_over_out": (max(ratios) if ratios else None)},
              open(f"{D}/data/zp_bias_precision.json", "w"), indent=1)
    print(f"wrote {D}/data/zp_bias_precision.json")
    if not ratios:
        print("\nNO OUTPUT SCALES OBSERVED -- err/|o| cannot be computed, so this script reaches NO "
              "verdict. Fix the observation before reading anything into the columns above.")
        return 1
    print(f"stored-bias error / |o|:    median {100 * statistics.median(ratios):.3f}%   "
          f"worst {100 * max(ratios):.3f}%")

    # The verdict rule, fixed before the numbers were seen. The W4A4 run-to-run floor is 0.6%, so an
    # injected per-channel error at or above that scale can move a relL2 measurably and the
    # asymmetric arm cannot be graded until it is gone.
    med = 100 * statistics.median(ratios)
    if med >= 0.6:
        print(f"\nTHE FP16 STORED BIAS IS A DEFECT, NOT A DETAIL: it injects {med:.2f}% of the "
              f"output scale per channel, at or above the 0.6% noise floor. The asymmetric arm was "
              f"measured through this. Fix the storage (fp32 folded bias) and re-measure BEFORE "
              f"drawing any conclusion about whether a zero point helps.")
    else:
        print(f"\nfp16 storage injects {med:.2f}% of the output scale, below the 0.6% floor -- it is "
              f"not what makes the asymmetric arm lose. Look elsewhere.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

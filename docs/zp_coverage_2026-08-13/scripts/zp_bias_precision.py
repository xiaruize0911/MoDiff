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

    # Output scale per conv, from a real forward. Hooked on the MODULE OUTPUT, because that is the
    # quantity the injected bias error competes with -- not the input, and not a synthetic tensor.
    scales = {}

    def hook(name):
        def f(mod, args, out):
            o = out[0] if isinstance(out, (tuple, list)) else out
            if torch.is_tensor(o):
                v = float(o.detach().float().abs().mean())
                scales[name] = max(scales.get(name, 0.0), v)
        return f

    hs = []
    convs = {}
    for name, mo in m.model.diffusion_model.named_modules():
        if isinstance(mo, OptimizedInt4Conv2d) and float(mo.static_input_zp.item()) != 0.0:
            convs[name] = mo
            hs.append(mo.register_forward_hook(hook(name)))
    print(f"{len(convs)} convs carry a non-zero zero point", flush=True)
    H.SEED = 1234
    H.latent(r, m, s)
    for h in hs:
        h.remove()

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
    print(f"\nmedian |corr|max            {statistics.median(corrs):.2f}")
    print(f"median |o| (abs mean)       {statistics.median(outs):.4f}")
    print(f"|corr| / |o|, median        {statistics.median(corrs) / statistics.median(outs):.0f}x")
    print(f"fp16 bias error / |o|:      median {100 * statistics.median(ratios):.2f}%   "
          f"worst {100 * max(ratios):.2f}%")

    os.makedirs(f"{D}/data", exist_ok=True)
    json.dump({"convs": len(convs), "rows": rows,
               "median_err_over_out": statistics.median(ratios),
               "worst_err_over_out": max(ratios)},
              open(f"{D}/data/zp_bias_precision.json", "w"), indent=1)
    print(f"\nwrote {D}/data/zp_bias_precision.json")

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

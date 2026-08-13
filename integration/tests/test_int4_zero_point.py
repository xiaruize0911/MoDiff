"""Gates for the activation zero point's bias fold, before any CUDA kernel learns about it.

An asymmetric activation grid stores a_q = round(a*s) + z, so a = (a_q - z)/s and

    sum_i w_q[k,i] * a[i] = (sum_i w_q[k,i]*a_q[i] - z * sum_i w_q[k,i]) * ws[k] / s

The second term is constant per output channel, so it folds into the bias and neither the GEMM nor the
EVT epilogue ever sees the zero point. This file gates that fold ON ITS OWN: nothing here depends on a
rebuild, so if the fold is wrong it is wrong here rather than three CUDA kernels later.

THE KERNELS ARE GATED SEPARATELY, in test_int4_zero_point_kernels.py -- which is the file to read for
which quantize entry points emit `+ z` and why the delta ones deliberately do not.

FOUR GATES:

  1. z = 0 IS BIT-IDENTICAL. The bias must come back byte-for-byte from _orig_bias and the conv output
     must be exactly what it was before this state existed. An asymmetric-capable layer that is
     configured symmetrically has to be indistinguishable from the symmetric one, or every committed
     number silently moves the day the field is added. `torch.equal`, not allclose.

  2. THE CORRECTION IS EXACT at z != 0. The reference is computed in fp64 and then ROUNDED TO THE
     STORAGE DTYPE, and exact equality is required. The bias is fp16 here (so is
     weight_scale_channel), which puts one unavoidable rounding on the folded value -- ~2.5e-4
     relative, i.e. fp16 epsilon. Demanding 1e-6 against an unrounded fp64 reference fails on that
     rounding and says nothing about the arithmetic, which is how the first version of this file
     "failed". Rounding the reference the same way keeps the gate exact rather than loosening it to a
     tolerance that would also accept a wrong scale factor.

     fp16 STORAGE IS DELIBERATE. An fp32 bias would promote the eager `out + bias` and change the
     conv's output dtype for every downstream consumer. The correction's own fp16 rounding is ~2.5e-4
     relative against a 4-bit activation error of order 0.3, so it is not the term that matters.

  3. RE-CALIBRATION DOES NOT COMPOUND. Setting the scale twice must give the same bias as setting it
     once: the correction scales as 1/s, so refolding from an already-corrected bias would square the
     error. This is why _orig_bias exists, and this gate is why I trust it.

  4. A BIAS-FREE CONV GAINS ONE, and gives it back. Most convs here have no bias; the correction is a
     real per-channel offset with nowhere else to live, so the buffer must appear at z != 0 and return
     to None at z = 0.

WHY NO ACCURACY CLAIM HERE. Whether an asymmetric grid HELPS is a separate question this file does not
touch, and deliberately: the only instrument able to price it (the fake-quant harness) failed its own
self-check twice, which is why the plan's fix #2 says deciding it requires implementing it. This gates
the mechanism. The benefit gets measured on real kernels, afterwards, or not claimed.

Run: python integration/tests/test_int4_zero_point.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import torch                                                             # noqa: E402
import torch.nn as nn                                                    # noqa: E402
from integration.kernels.int4_optimized import OptimizedInt4Conv2d       # noqa: E402


def make(cin=64, cout=32, k=3, bias=True, seed=20260813):
    torch.manual_seed(seed)
    c = nn.Conv2d(cin, cout, k, padding=k // 2, bias=bias).cuda().half()
    return OptimizedInt4Conv2d(c, layer_name="probe").cuda()


def ref_correction(m, z):
    """-(z/s) * sum_i w_q[k,i] * ws[k], in fp64, from the module's own buffers."""
    s = float(m.static_input_scale.item())
    wq = m.weight_sum_q.double().view(-1)
    ws = m.weight_scale_channel.double().view(-1)
    return -(z / s) * wq * ws


def ref_bias(m, z, base):
    """The fp64 reference rounded to the dtype the module actually stores, so the comparison can be
    EXACT. `base` is the pre-fold bias (None for a bias-free conv)."""
    corr = ref_correction(m, z)
    dt = base.dtype if base is not None else m.weight_scale_channel.dtype
    full = corr if base is None else (base.double().view(-1) + corr)
    return full.to(dt)


def main():
    fails = []

    # ---- 1. z = 0 is bit-identical -------------------------------------------------------------
    a = make(bias=True)
    b = make(bias=True)
    a.set_static_scale(8.0)                              # symmetric, the shipped route
    b.set_static_calibration(8.0, None, 0.0)             # asymmetric API, z = 0
    ok = torch.equal(a.bias, b.bias) and float(b.static_input_zp.item()) == 0.0
    print(f"1. z=0 bias bit-identical to the symmetric path : {'ok' if ok else 'FAIL'}")
    fails += [] if ok else ["z=0 bias"]

    x = torch.randn(2, 64, 16, 16, device="cuda", dtype=torch.float16
                    ).to(memory_format=torch.channels_last)
    with torch.no_grad():
        ya, yb = a(x), b(x)
    ok = torch.equal(ya, yb)
    print(f"   z=0 conv output bit-identical                : {'ok' if ok else 'FAIL'}")
    fails += [] if ok else ["z=0 output"]

    # ---- 2. the correction is exact ------------------------------------------------------------
    for z in (1.0, -3.0, 7.0):
        m = make(bias=True)
        base = m.bias.clone()
        m.set_static_calibration(8.0, None, z)
        want = ref_bias(m, z, base)
        ok = torch.equal(m.bias.view(-1), want)
        mag = float(ref_correction(m, z).abs().max())
        print(f"2. z={z:+5.1f} folded bias EXACTLY equals fp64->fp16 : "
              f"{'ok' if ok else 'FAIL'}   |corr|max {mag:.4f}")
        fails += [] if ok else [f"correction z={z}"]

    # ---- 3. re-calibration does not compound ---------------------------------------------------
    m = make(bias=True)
    m.set_static_calibration(8.0, None, 5.0)
    once = m.bias.clone()
    m.set_static_calibration(8.0, None, 5.0)             # again, same values
    ok = torch.equal(once, m.bias)
    print(f"3. setting the same calibration twice is idempotent: {'ok' if ok else 'FAIL'}")
    fails += [] if ok else ["idempotent"]

    m.set_static_calibration(4.0, None, 5.0)             # different scale -> different correction
    ok = torch.equal(m.bias.view(-1), ref_bias(m, 5.0, m._orig_bias))
    print(f"   rescaling refolds from the ORIGINAL bias        : {'ok' if ok else 'FAIL'}")
    fails += [] if ok else ["rescale"]

    # ---- 4. a bias-free conv gains one, and gives it back --------------------------------------
    m = make(bias=False)
    ok0 = m.bias is None
    m.set_static_calibration(8.0, None, 2.0)
    ok1 = (m.bias is not None and m.bias.numel() == m.weight_sum_q.numel()
           and torch.equal(m.bias.view(-1), ref_bias(m, 2.0, None)))
    m.set_static_calibration(8.0, None, 0.0)
    ok2 = m.bias is None
    ok = ok0 and ok1 and ok2
    print(f"4. bias-free conv: None -> correction -> None      : {'ok' if ok else 'FAIL'}"
          f"   ({ok0}/{ok1}/{ok2})")
    fails += [] if ok else ["bias-free"]

    print()
    if fails:
        print(f"FAILED: {', '.join(fails)}")
        return 1
    print("ALL PASS: the zero point's bias fold is exact, idempotent, and provably inert at z=0.\n"
          "NOTE: this gates the HOST-SIDE FOLD only. The kernels are gated in\n"
          "test_int4_zero_point_kernels.py, and no accuracy claim follows from either file.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

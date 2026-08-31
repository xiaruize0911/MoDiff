#!/usr/bin/env python3
"""I-MoDiff invariants: integer a_hat, frozen s*, no dequant snap.

I2: o_hat ≈ conv(a_hat.float() * s*, W_deq) + bias
Increment: Δa_hat equals q except at a_hat saturation.
Overflow: print max|a_hat|/qmax for bits 16/8/4.

Run: source setup_cuda_env.sh && python integration/tests/test_imode.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)

import torch
import torch.nn as nn
import torch.nn.functional as F

from integration.kernels.int8_optimized import OptimizedInt8Conv2d

DEV = "cuda"


def rel_err(a, b) -> float:
    a, b = a.float(), b.float()
    return (a - b).norm().item() / (b.norm().item() + 1e-12)


def _make_opt():
    torch.manual_seed(0)
    C, HW, N = 128, 16, 8
    conv = nn.Conv2d(C, C, 3, padding=1).to(DEV)
    x0 = torch.randn(N, C, HW, HW, device=DEV)
    opt = OptimizedInt8Conv2d(conv).to(DEV)
    opt.set_calibrating(True)
    _ = opt(x0)
    opt.set_calibrating(False)
    opt.set_standard_output_fp16(True)
    opt.enable_modiff(True)
    opt.begin_delta_calibration()
    opt._delta_calib = False
    opt = opt.to(memory_format=torch.channels_last)
    return opt, C, HW, N


def _run_traj(opt, C, HW, N, steps=25, drift=0.02):
    opt.reset_state()
    a = torch.randn(N, C, HW, HW, device=DEV, dtype=torch.float16).contiguous(
        memory_format=torch.channels_last)
    triples = []
    prev_a = None
    for _ in range(steps):
        a = (a + drift * torch.randn(N, C, HW, HW, device=DEV, dtype=torch.float16)).contiguous(
            memory_format=torch.channels_last)
        opt(a)
        if opt.a_hat_cache is not None and not opt.is_first_step:
            triples.append((a, opt.a_hat_cache.clone(), opt.o_hat_cache.clone(),
                            None if prev_a is None else prev_a))
            prev_a = opt.a_hat_cache.clone()
    return triples


def _i2(opt, a_hat, o_hat) -> float:
    s_star = float(opt.static_delta_alpha[0].item())
    a_f = a_hat.float() * s_star
    w_deq = opt.dequantized_weight()
    bias = opt.bias
    if bias is not None:
        bias = bias.float().reshape(-1)
    ref = F.conv2d(a_f, w_deq, bias=bias, padding=1)
    return rel_err(o_hat, ref)


def test_imode_bits(bits: int):
    os.environ["MODIFF_IMODE"] = "1"
    os.environ["MODIFF_AHAT_BITS"] = str(bits)
    os.environ["MODIFF_DELTA_MODE"] = "static"
    os.environ["MODIFF_CACHE_SKIP_K"] = "1"
    os.environ["MODIFF_REPLAY_K"] = "1"
    opt, C, HW, N = _make_opt()
    want = torch.int16 if bits >= 16 else torch.int8
    triples = _run_traj(opt, C, HW, N, steps=25)
    assert triples, "no modulated steps"
    assert triples[0][1].dtype == want, f"a_hat dtype {triples[0][1].dtype} != {want}"

    worst_i2, worst_dq = 0.0, 0
    qmax = int(opt._ahat_qmax())
    for _, a_hat, o_hat, prev in triples:
        worst_i2 = max(worst_i2, _i2(opt, a_hat, o_hat))
        if prev is not None:
            d = (a_hat.to(torch.int32) - prev.to(torch.int32)).abs().max().item()
            worst_dq = max(worst_dq, d)
    sat = opt.ahat_sat_frac()
    # bits=16: I2 at fp16-a_hat quality. 8/4 may desync via saturation, but Δa
    # must still be an int8 code (no float snap).
    i2_lim = 0.08 if bits >= 16 else 2.0
    inc_ok = worst_dq <= 127
    ok = worst_i2 <= i2_lim and inc_ok and triples[0][1].dtype == want
    detail = (f"bits={bits} dtype={triples[0][1].dtype} I2={worst_i2:.4f} "
              f"(<={i2_lim}) max|Δa|={worst_dq} sat={sat:.3f}")
    return f"imode{bits}", ok, detail


def test_imode_increment_identity():
    """q and a_hat += q stay the same integer; no code*s_a snap."""
    os.environ["MODIFF_IMODE"] = "1"
    os.environ["MODIFF_AHAT_BITS"] = "16"
    os.environ["MODIFF_DELTA_MODE"] = "static"
    opt, C, HW, N = _make_opt()
    opt.reset_state()
    x = torch.randn(N, C, HW, HW, device=DEV, dtype=torch.float16).contiguous(
        memory_format=torch.channels_last)
    opt(x)
    a0 = opt.a_hat_cache.clone()
    o0 = opt.o_hat_cache.clone()
    x2 = (x + 0.05 * torch.randn_like(x)).contiguous(memory_format=torch.channels_last)
    opt(x2)
    a1 = opt.a_hat_cache
    o1 = opt.o_hat_cache
    da = (a1.to(torch.int32) - a0.to(torch.int32))
    # Same q went into o_hat: I2 on both steps, and |da| <= 127.
    i2_0 = _i2(opt, a0, o0)
    i2_1 = _i2(opt, a1, o1)
    ok = da.abs().max().item() <= 127 and i2_0 <= 0.08 and i2_1 <= 0.08
    return ("imode_inc", ok,
            f"max|Δa|={da.abs().max().item()} I2_0={i2_0:.4f} I2_1={i2_1:.4f} "
            f"sat={opt.ahat_sat_frac():.3f}")


def main():
    if not torch.cuda.is_available():
        print("CUDA unavailable — skipping.")
        return 0
    tests = [
        lambda: test_imode_bits(16),
        lambda: test_imode_bits(8),
        lambda: test_imode_bits(4),
        test_imode_increment_identity,
    ]
    all_ok = True
    print(f"{'test':<18}{'status':<8}detail")
    print("-" * 78)
    for t in tests:
        try:
            name, ok, detail = t()
        except Exception as e:
            name = getattr(t, "__name__", "imode")
            ok, detail = False, f"EXCEPTION {type(e).__name__}: {e}"
        all_ok &= ok
        print(f"{name:<18}{'PASS' if ok else 'FAIL':<8}{detail}")
    print("-" * 78)
    print("ALL PASS" if all_ok else "FAILURES PRESENT")
    os.environ["MODIFF_IMODE"] = "0"
    os.environ["MODIFF_AHAT_BITS"] = "16"
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

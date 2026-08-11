"""Is the int8 GATHER path worth taking? The pre-check that decides route (b).

Route (b) folds the three `aq_*` re-quantize kernels (4.60 ms/step) into the qkv GEMM's epilogue by
emitting int8 straight into `flash_attn_int8_packed_vt`. That removes the quantize -- but it also
CHANGES THE SCORE KERNEL, from the mma kernel that reads pre-transposed qi/ki/vt to the packed
kernel that gathers per-token bytes with cp.async. Nobody has measured the gather path on its own.

Route (a) fed that SAME entry point fp16 and was 18.0 ms/step SLOWER, because the kHalf branch
quantizes on every k/v re-read. That refutation says nothing about the int8 branch, which gathers
instead -- so this file exists to keep the next decision from inheriting the last one's evidence.

WHAT IS TIMED, and why the two arms are not symmetric:

  arm U (production, frozen scales -- mirrors _packed_ref_vt in quantized_std_attention.py:261)
      quantize_attn_qkv_packed_static (the three aq_* kernels)  +  flash_attn_int8_vt_static
  arm P (route (b))
      flash_attn_int8_packed_vt on an int8 qkv                   <- quantize NOT timed

  P excludes the quantize on purpose: in production the int8 comes out of
  `gemm_w8a8_awq_o_hat_out_i8`'s epilogue, which runs anyway. Timing it here would charge route (b)
  for work its whole point is to make free. U therefore carries the quantize and P does not, and the
  difference of the totals is the honest per-block delta.

BREAK-EVEN, computed from the committed trace before running this (docs/profile_kernels_layers_
2026-08-11/data/trace_buckets.json, arm modiff_full_k1, batch 128) and stated here so the result
cannot be read to taste:

    hd48 group (10 blocks): aq_* ~1.47 ms + flash ~1.48 ms = ~2.95 ms/step today
      -> route (b) nets a gain IFF the packed kernel is under ~2.0x the current flash time.
    hd24 group (5 blocks):  aq_* ~3.13 ms + flash ~7.10 ms = ~10.2 ms/step today
      -> an 8-byte-loader variant would net a gain IFF under ~1.44x. (hd24 cannot run today: the
         int8 gather needs hd%16==0 for its 16 B cp.async and hd=24 is 24 B/token.)

Run: python integration/tests/bench_flash_packed_vs_unpacked.py [--batch 128] [--iters 20]
"""
import argparse
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import torch                                                             # noqa: E402
import modiff_cutlass as mc                                              # noqa: E402

DEV = "cuda"

#: (C, T, label) at nh=8 -- the churches UNet's attention shapes. hd = C/nh.
#: 192/1024 x5 blocks, 384/256 x5, 384/64 x5 are the 15 flash-eligible ones; 768/16 (hd=96) is
#: structurally ineligible (FA_MMA_MAXHD=64, T%64) and is not a candidate for either route.
SHAPES = [(192, 1024), (384, 256), (384, 64)]

#: Frozen flash scales. Their VALUES do not matter for timing and only have to be shared between the
#: arms for the output comparison; production reads them from _fq_sqc / _fq_skc / _fq_svv.
SQ_C, SK_C = 0.031, 0.027


def make_inputs(b, C, T, nh):
    """qkv plus the per-channel V scale at the width the quantize kernel demands.

    `quantize_attn_qkv_packed_static` requires sv_vec to be [hp_av] = hd_pad, not [hd] -- production
    passes `_fq_svv`, which is already padded. The tail beyond hd is never read for correctness
    (flash's `svr` zeroes d >= hd) but it has to be THERE, so it is filled with 1.0 rather than left
    to whatever the allocator held.
    """
    hd = C // nh
    hd_pad = ((hd + 31) // 32) * 32
    torch.manual_seed(7)
    qkv = (torch.randn(b, T, nh, 3, hd, device=DEV, dtype=torch.float16) * 0.5).contiguous()
    sv = torch.ones(hd_pad, device=DEV, dtype=torch.float32)
    sv[:hd] = torch.rand(hd, device=DEV, dtype=torch.float32) * 0.02 + 0.01
    return qkv, sv.contiguous(), hd


def quantize_packed(qkv, sv, hd, sq_c=SQ_C, sk_c=SK_C):
    """fp16 [b,T,nh,3,hd] -> int8 at the per-COLUMN scale route (b)'s GEMM epilogue applies.

    Column c of the interleaved (nh, 3, hd) layout belongs to q/k/v by (c//hd)%3, and within v to
    head-channel c%hd -- the same mapping _qkv_inv_out_scale builds, kept here as an independent
    expression of it rather than an import, so a change to one does not silently pass this.
    """
    inv = torch.empty(3, hd, device=qkv.device, dtype=torch.float32)
    inv[0] = 1.0 / sq_c
    inv[1] = 1.0 / sk_c
    inv[2] = 1.0 / sv[:hd]
    scaled = qkv.float() * inv.view(1, 1, 1, 3, hd)
    return torch.clamp(torch.round(scaled), -127, 127).to(torch.int8).contiguous()


def arm_u(qkv, sv, b, nh, T, hd, hd_pad, softmax_scale):
    """Production: three aq_* kernels then the mma kernel. Mirrors _packed_ref_vt."""
    qi, ki, vt, sq, sk, svo = mc.quantize_attn_qkv_packed_static(
        qkv, nh, T, hd, hd_pad, hd_pad, 8, SQ_C, SK_C, sv)
    qi = qi.view(b, nh, T, hd_pad)
    ki = ki.view(b, nh, T, hd_pad)
    vt = vt.view(b, nh, hd_pad, T)
    svo = svo[..., :hd].contiguous().view(b, nh, hd)
    return mc.flash_attn_int8_vt_static(qi, ki, vt, svo, SQ_C, SK_C, softmax_scale)


def arm_u_quant_only(qkv, sv, nh, T, hd, hd_pad):
    """The aq_* half of arm U alone, so the two halves can be reported separately."""
    return mc.quantize_attn_qkv_packed_static(qkv, nh, T, hd, hd_pad, hd_pad, 8, SQ_C, SK_C, sv)


def arm_p(qkv_i8, sv, hd, hd_pad, softmax_scale):
    """Route (b): the int8 gather path. No quantize -- the GEMM epilogue produced this int8."""
    return mc.flash_attn_int8_packed_vt(qkv_i8, sv[:hd].contiguous(), hd_pad,
                                        SQ_C, SK_C, softmax_scale)


def ref_fp32(qkv_i8, sv, hd, scale, chunk=16):
    """Attention in fp32 from the SAME int8 codes both arms consume. The gate's ground truth.

    Why this exists rather than comparing the arms to each other: at hd48/T64 the two arms differ by
    2.57e-3 while each is 3.7-3.9e-3 from fp32, i.e. their disagreement is SMALLER than their common
    distance from the truth. A U-vs-P threshold there is measuring fp16 accumulation noise and would
    have failed a correct kernel. Against fp32, the question becomes the one that matters: is P worse
    than U? (Measured: 3.869e-3 vs 3.735e-3 -- 3.6% relative, on codes that are bit-identical.)

    Chunked over the batch because the scores are [chunk, nh, T, T] fp32: at b=128/T=1024 the dense
    form is 4.3 GB and would OOM the reference, not the kernel under test.
    """
    b, T, nh = qkv_i8.shape[0], qkv_i8.shape[1], qkv_i8.shape[2]
    out = torch.empty(b, nh, T, hd, device=qkv_i8.device, dtype=torch.float32)
    svf = sv[:hd].view(1, 1, 1, hd)
    for i in range(0, b, chunk):
        x = qkv_i8[i:i + chunk].float()
        q = (x[:, :, :, 0, :] * SQ_C).transpose(1, 2)
        k = (x[:, :, :, 1, :] * SK_C).transpose(1, 2)
        v = (x[:, :, :, 2, :] * svf).transpose(1, 2)
        a = torch.softmax((q @ k.transpose(-1, -2)) * scale, dim=-1)
        out[i:i + chunk] = a @ v
    return out


def time_ms(fn, iters, warmups=3):
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return statistics.median(times)


def rel_l2(a, b):
    d = (a.float() - b.float()).pow(2).sum().sqrt()
    return (d / b.float().pow(2).sum().sqrt().clamp_min(1e-12)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--nh", type=int, default=8)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    print(f"{torch.cuda.get_device_name(0)}, batch {args.batch}, nh {args.nh}, "
          f"median of {args.iters}\n")
    print("| C | T | hd | arm U aq_* | arm U flash | arm U total | arm P packed | P/U_flash | "
          "net (U-P) | U vs fp32 | P vs fp32 | U vs P | det |")
    print("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|")

    rows = []
    for C, T in SHAPES:
        b, nh = args.batch, args.nh
        qkv, sv, hd = make_inputs(b, C, T, nh)
        hd_pad = ((hd + 31) // 32) * 32
        softmax_scale = hd ** -0.5
        qkv_i8 = quantize_packed(qkv, sv, hd)

        with torch.inference_mode():
            u_out = arm_u(qkv, sv, b, nh, T, hd, hd_pad, softmax_scale)
            u_total = time_ms(lambda: arm_u(qkv, sv, b, nh, T, hd, hd_pad, softmax_scale),
                              args.iters)
            u_quant = time_ms(lambda: arm_u_quant_only(qkv, sv, nh, T, hd, hd_pad), args.iters)

            # Route (b) may be structurally rejected here (hd24: per-token bytes; hd96: mma
            # eligibility). Report that as the verdict rather than crashing the whole sweep --
            # which shapes are eligible is exactly what this table is for.
            try:
                p_out = arm_p(qkv_i8, sv, hd, hd_pad, softmax_scale)
                p_ms = time_ms(lambda: arm_p(qkv_i8, sv, hd, hd_pad, softmax_scale), args.iters)
                det = all(torch.equal(p_out, arm_p(qkv_i8, sv, hd, hd_pad, softmax_scale))
                          for _ in range(10))
                ref = ref_fp32(qkv_i8, sv, hd, softmax_scale)
                u_err, p_err = rel_l2(u_out, ref), rel_l2(p_out, ref)
                up_err = rel_l2(p_out, u_out)
                del ref
                verdict = None
            except RuntimeError as exc:
                p_ms = det = u_err = p_err = up_err = None
                verdict = str(exc).split("\n")[0][:58]

        u_flash = u_total - u_quant
        if verdict is not None:
            print(f"| {C} | {T} | {hd} | {u_quant:.3f} | {u_flash:.3f} | {u_total:.3f} | "
                  f"REJECTED | | | | | | {verdict} |")
            rows.append(dict(C=C, T=T, hd=hd, u_quant_ms=u_quant, u_flash_ms=u_flash,
                             u_total_ms=u_total, p_ms=None, rejected=verdict))
            continue
        ratio = p_ms / u_flash if u_flash > 0 else float("inf")
        print(f"| {C} | {T} | {hd} | {u_quant:.3f} | {u_flash:.3f} | {u_total:.3f} | "
              f"{p_ms:.3f} | {ratio:.2f}x | {u_total - p_ms:+.3f} | {u_err:.2e} | {p_err:.2e} | "
              f"{up_err:.2e} | {'ok' if det else 'FAIL'} |")
        rows.append(dict(C=C, T=T, hd=hd, u_quant_ms=u_quant, u_flash_ms=u_flash,
                         u_total_ms=u_total, p_ms=p_ms, p_over_u_flash=ratio,
                         net_ms=u_total - p_ms, u_vs_fp32=u_err, p_vs_fp32=p_err,
                         u_vs_p=up_err, deterministic=bool(det)))

    print("\nPer-STEP totals, weighting each shape by its block count (5 blocks per shape):")
    per_step_u = per_step_p = 0.0
    for r in rows:
        if r.get("p_ms") is None:
            continue
        per_step_u += 5 * r["u_total_ms"]
        per_step_p += 5 * r["p_ms"]
    if per_step_p:
        print(f"  eligible shapes only: U {per_step_u:.2f} ms -> P {per_step_p:.2f} ms "
              f"({per_step_u - per_step_p:+.2f} ms/step)")
    else:
        print("  no shape took the gather path -- nothing to total")

    # GATE: P must not be measurably less accurate than U against fp32, and must be deterministic.
    # 1.10 is a margin, not a tolerance -- the arms consume bit-identical codes, so a real defect in
    # the gather path would show as a ratio far above 1, not at 1.04.
    bad = []
    for r in rows:
        if r.get("p_ms") is None:
            continue
        if r["deterministic"] is False:
            bad.append(f"{r['C']}x{r['T']}: nondeterministic")
        if r["p_vs_fp32"] > r["u_vs_fp32"] * 1.10:
            bad.append(f"{r['C']}x{r['T']}: P {r['p_vs_fp32']:.3e} vs U {r['u_vs_fp32']:.3e} "
                       f"against fp32 ({r['p_vs_fp32'] / r['u_vs_fp32']:.2f}x worse)")
    if bad:
        print("\nGATE NOT MET -- the timing above is not a comparison of equivalent computations:")
        for line in bad:
            print("  -", line)
    else:
        print("\nGate met: every eligible shape is deterministic and no less accurate than "
              "production against fp32.")

    if args.json:
        import json
        with open(args.json, "w") as f:
            json.dump(dict(gpu=torch.cuda.get_device_name(0), batch=args.batch, nh=args.nh,
                           iters=args.iters, sq_c=SQ_C, sk_c=SK_C, rows=rows), f, indent=1)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

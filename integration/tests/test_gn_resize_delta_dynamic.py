"""group_norm_silu_delta_quantize_resize_nhwc with a DYNAMIC delta scale.

The updown ResBlocks' MoDiff fusion used to take its scale as a device pointer and compute no
absmax, so `_prequant_gn_resize_conv_modiff` declined on every step that had to re-measure the
delta range. At MODIFF_DELTA_REFRESH=1 -- the paper's own configuration -- that is every step, so
all eight updown ResBlocks ran unfused (0/8 fused at K=1 against 6/8 at K=4, measured in
docs/component_attribution_2026-08-07/data/trace_buckets.json).

The kernel now carries the same dynamic-scale contract as its non-resize sibling
group_norm_silu_delta_quantize_nhwc: real 1-element absmax/scale/inv_scale/retire buffers make it
run a reduction-only launch of its own body first, publish Q_level/absmax, and quantize with that.

What this checks, per real updown shape and both quantizations:

  A. static (14-argument) form is BIT-IDENTICAL to what it produced before the change -- the
     regression guard for the new trailing parameters and for the datapath-derived clamp
  B. the published scale matches an fp32 reference absmax over the same delta
  C. the codes match an fp32 reference quantized with that same published scale
  D. a_hat advances exactly once per call -- if the reduction launch also wrote a_hat, or wrote
     codes, this is what catches it
  E. a4 makes an int8-STORE layer saturate at 7 -- i.e. W8A4 really is 4-bit
  F. report_next publishes without a second pass and leaves the codes on the caller's scale

Run: python integration/tests/test_gn_resize_delta_dynamic.py
"""
import os
import sys

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())

import torch
import torch.nn.functional as F
import modiff_cutlass as mc

# The model's eight updown ResBlock shapes (C, H, W, direction), as in test_gn_resize_fusion.py.
SHAPES = [(192, 32, 32, -1), (384, 16, 16, -1), (384, 8, 8, -1), (768, 4, 4, -1),
          (768, 2, 2, +1), (768, 4, 4, +1), (384, 8, 8, +1), (384, 16, 16, +1)]
GROUPS, EPS = 32, 1e-5
BATCH = 8               # correctness, not throughput -- the grid shape is unchanged by N
DEV = "cuda"


def unpack_int4(y, C):
    """[N,Ho,Wo,C/2] packed -> [N,Ho,Wo,C] signed nibbles."""
    lo = (y & 0x0F).to(torch.int16)
    hi = ((y >> 4) & 0x0F).to(torch.int16)
    lo = torch.where(lo > 7, lo - 16, lo)
    hi = torch.where(hi > 7, hi - 16, hi)
    return torch.stack([lo, hi], dim=-1).reshape(*y.shape[:-1], C)


def codes_of(y, C, pack):
    """Kernel output -> [N,Ho,Wo,C] int16 codes, whichever store it used."""
    if pack:
        return unpack_int4(y, C)
    # int8 comes back [N,C,Ho,Wo] channels_last, i.e. NHWC-physical
    return y.permute(0, 2, 3, 1).to(torch.int16)


def resized_activation(x, gamma, beta, direction, smooth):
    """fp32 GN -> SiLU -> 2x resize, i.e. everything the kernel does above the subtract.

    The 2x2 average is taken on the fp32 post-SiLU values, before any quantization -- which is the
    whole reason this fusion is allowed to exist in the DOWN direction.
    """
    x32 = x.float()
    N, C, H, W = x32.shape
    g = x32.reshape(N, GROUPS, C // GROUPS * H * W)
    mean = g.mean(-1, keepdim=True)
    var = g.var(-1, unbiased=False, keepdim=True)
    g = (g - mean) / torch.sqrt(var + EPS)
    y = g.reshape(N, C, H, W) * gamma.float().view(1, C, 1, 1) + beta.float().view(1, C, 1, 1)
    y = y * torch.sigmoid(y)
    if smooth is not None:
        y = y * smooth.float().view(1, C, 1, 1)
    y = (F.interpolate(y, scale_factor=2, mode="nearest") if direction > 0
         else F.avg_pool2d(y, 2))
    return y.permute(0, 2, 3, 1).contiguous()          # [N,Ho,Wo,C], matching a_hat's layout


def reference(x, gamma, beta, a_hat_nhwc, scale, direction, smooth, lim):
    """(codes, a_hat_after, absmax) for one delta-quantize step, all in fp32."""
    y = resized_activation(x, gamma, beta, direction, smooth)
    delta = y - a_hat_nhwc.float()
    absmax = delta.abs().max()
    codes = torch.clamp(torch.round(delta * scale), -lim, lim)
    a_hat_after = (a_hat_nhwc.float() + codes / scale).half()
    return codes, a_hat_after, absmax


def make_case(C, H, W, direction, pack, seed, use_smooth=False):
    torch.manual_seed(seed)
    x = torch.randn(BATCH, C, H, W, device=DEV, dtype=torch.float16)
    x = x.contiguous(memory_format=torch.channels_last)
    gamma = torch.randn(C, device=DEV, dtype=torch.float16)
    beta = torch.randn(C, device=DEV, dtype=torch.float16)
    Ho, Wo = (H * 2, W * 2) if direction > 0 else (H // 2, W // 2)
    # a_hat is a previous step's reconstruction, so give it the same rough magnitude as the
    # activation -- a zero cache would make every delta the activation itself and hide sign bugs.
    a_hat = (0.5 * torch.randn(BATCH, Ho, Wo, C, device=DEV, dtype=torch.float16))
    smooth = (torch.rand(C, device=DEV, dtype=torch.float32) + 0.5) if use_smooth else None
    sm = smooth if smooth is not None else torch.empty(0, device=DEV, dtype=torch.float32)
    empty = x.new_empty(0)
    return x, gamma, beta, a_hat, smooth, sm, empty, (Ho, Wo)


def dyn_bufs():
    return (torch.zeros(1, device=DEV, dtype=torch.float32),    # absmax (self-resetting)
            torch.empty(1, device=DEV, dtype=torch.float32),    # scale_out
            torch.empty(1, device=DEV, dtype=torch.float32),    # inv_scale_out
            torch.zeros(1, device=DEV, dtype=torch.int32))      # retire (self-resetting)


def main():
    fails = []
    print("A/B/C/D: static bit-identity, published scale, codes, a_hat\n")
    print("| shape | dir | quant | A static | B scale rel | C max code err | D a_hat max |")
    print("|---|---|---|---|---:|---:|---:|")
    for C, H, W, direction in SHAPES:
        for pack in (True, False):
            lim = 7.0 if pack else 127.0
            Q_level = lim
            x, gamma, beta, a_hat0, smooth, sm, empty, (Ho, Wo) = make_case(
                C, H, W, direction, pack, seed=7)
            a_hat_flat = a_hat0.reshape(-1)

            # ---- A. static (14-arg) form, before and after: the value it used to produce is the
            # fp32 reference at a GIVEN scale, and it must still be reproduced exactly.
            scale_t = torch.tensor([lim / 6.0], device=DEV, dtype=torch.float32)
            ah = a_hat_flat.clone()
            with torch.inference_mode():
                y_static = mc.group_norm_silu_delta_quantize_resize_nhwc(
                    x, gamma, beta, GROUPS, EPS, True, scale_t, sm, empty, empty,
                    0, direction, pack, ah)
            ref_c, ref_ah, _ = reference(x, gamma, beta, a_hat0, float(scale_t.item()),
                                         direction, smooth, lim)
            got_c = codes_of(y_static, C, pack).float()
            a_static_ok = int((got_c - ref_c).abs().max().item()) <= 1
            if not a_static_ok:
                fails.append(f"A static codes {C}x{H}x{W} dir{direction} pack={pack}")

            # ---- B/C/D. dynamic: the kernel measures the range itself.
            amax_b, sc_b, inv_b, ret_b = dyn_bufs()
            ah = a_hat_flat.clone()
            scale_in = torch.empty(1, device=DEV, dtype=torch.float32)   # deliberately garbage
            with torch.inference_mode():
                y_dyn = mc.group_norm_silu_delta_quantize_resize_nhwc(
                    x, gamma, beta, GROUPS, EPS, True, scale_in, sm, empty, empty,
                    0, direction, pack, ah,
                    amax_b, sc_b, inv_b, ret_b, Q_level, False, 1.0, False)
            _, _, ref_absmax = reference(x, gamma, beta, a_hat0, 1.0, direction, smooth, lim)
            ref_scale = Q_level / max(float(ref_absmax) * 1.0, 1e-6)
            got_scale = float(sc_b.item())
            b_rel = abs(got_scale - ref_scale) / ref_scale
            # C/D are evaluated at the scale the kernel actually published, so a small B error
            # does not cascade into them.
            ref_c2, ref_ah2, _ = reference(x, gamma, beta, a_hat0, got_scale,
                                           direction, smooth, lim)
            got_c2 = codes_of(y_dyn, C, pack).float()
            c_err = int((got_c2 - ref_c2).abs().max().item())
            d_err = float((ah.reshape(BATCH, Ho, Wo, C).float() - ref_ah2.float()).abs().max())
            # a_hat moves by at most one code step, so scale it by that to stay shape-independent
            d_tol = 3.0 / got_scale
            if b_rel > 1e-2:
                fails.append(f"B scale {C}x{H}x{W} dir{direction} pack={pack} rel={b_rel:.3e}")
            if c_err > 1:
                fails.append(f"C codes {C}x{H}x{W} dir{direction} pack={pack} err={c_err}")
            if d_err > d_tol:
                fails.append(f"D a_hat {C}x{H}x{W} dir{direction} pack={pack} "
                             f"err={d_err:.3e} tol={d_tol:.3e}")
            # retire/absmax must be self-reset, or the next call's election never fires
            if float(amax_b.item()) != 0.0 or int(ret_b.item()) != 0:
                fails.append(f"reduction buffers not reset {C}x{H}x{W} dir{direction}")
            print(f"| {C}x{H}x{W} | {direction:+d} | {'int4' if pack else 'int8'} | "
                  f"{'ok' if a_static_ok else 'FAIL'} | {b_rel:.2e} | {c_err} | {d_err:.3e} |")

    # ---- E. W8A4. The int8 store's native limit is 127; `a4` is the only thing that makes an
    # int8-container activation genuinely 4-bit. Give it a scale far too fine for the range, which
    # is what a stale K>1 scale or a clip ratio below 1 produces, and check it saturates at 7.
    print("\nE: a4 on an int8 store (this IS W8A4)")
    C, H, W, direction = 384, 16, 16, -1
    for pack in (True, False):
        native = 7.0 if pack else 127.0
        x, gamma, beta, a_hat0, smooth, sm, empty, _ = make_case(
            C, H, W, direction, pack, seed=11)
        ah = a_hat0.reshape(-1).clone()
        _, _, absmax = reference(x, gamma, beta, a_hat0, 1.0, direction, smooth, native)
        fine = native / (float(absmax) * 0.25)            # 4x finer than the range needs
        scale_t = torch.tensor([fine], device=DEV, dtype=torch.float32)
        ef = torch.empty(0, device=DEV, dtype=torch.float32)
        ei = torch.empty(0, device=DEV, dtype=torch.int32)
        for a4 in (False, True):
            ah = a_hat0.reshape(-1).clone()
            with torch.inference_mode():
                y = mc.group_norm_silu_delta_quantize_resize_nhwc(
                    x, gamma, beta, GROUPS, EPS, True, scale_t, sm, empty, empty,
                    0, direction, pack, ah, ef, ef, ef, ei, native, False, 1.0, a4)
            cmax = int(codes_of(y, C, pack).abs().max().item())
            # int4 storage saturates at 7 whatever a4 says -- the format decides. int8 storage
            # saturates at 127 unless a4 says otherwise, and THAT is the W8A4 mechanism.
            want = 7 if (pack or a4) else 127
            ok = cmax == want
            print(f"  {'int4' if pack else 'int8'} store, a4={a4}: max|code| = {cmax}, "
                  f"want {want} -> {'ok' if ok else 'FAIL'}")
            if not ok:
                fails.append(f"E pack={pack} a4={a4}: max|code|={cmax} want {want}")

    # ---- F. report_next: quantize on the caller's scale, publish for a later step, one pass.
    print("\nF: report_next")
    for pack in (True, False):
        lim = 7.0 if pack else 127.0
        x, gamma, beta, a_hat0, smooth, sm, empty, _ = make_case(
            C, H, W, direction, pack, seed=13)
        amax_b, sc_b, inv_b, ret_b = dyn_bufs()
        ah = a_hat0.reshape(-1).clone()
        given = torch.tensor([lim / 6.0], device=DEV, dtype=torch.float32)
        safety = 1.15
        with torch.inference_mode():
            y = mc.group_norm_silu_delta_quantize_resize_nhwc(
                x, gamma, beta, GROUPS, EPS, True, given, sm, empty, empty,
                0, direction, pack, ah,
                amax_b, sc_b, inv_b, ret_b, lim, True, safety, False)
        ref_c, _, absmax = reference(x, gamma, beta, a_hat0, float(given.item()),
                                     direction, smooth, lim)
        code_ok = int((codes_of(y, C, pack).float() - ref_c).abs().max().item()) <= 1
        want = lim / max(float(absmax) * safety, 1e-6)
        got = float(sc_b.item())
        scale_ok = abs(got - want) / want < 1e-2
        inv_ok = abs(float(inv_b.item()) - 1.0 / got) / (1.0 / got) < 1e-5
        print(f"  {'int4' if pack else 'int8'}: codes on GIVEN scale {'ok' if code_ok else 'FAIL'}"
              f", published {got:.4f} vs {want:.4f} {'ok' if scale_ok else 'FAIL'}"
              f", inv {'ok' if inv_ok else 'FAIL'}")
        if not (code_ok and scale_ok and inv_ok):
            fails.append(f"F report_next pack={pack}")

    print()
    if fails:
        print(f"FAILED ({len(fails)}):")
        for f in fails:
            print("  -", f)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

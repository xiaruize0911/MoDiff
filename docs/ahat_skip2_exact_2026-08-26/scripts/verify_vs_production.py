"""Strongest possible validation: compare MY stats+apply pipeline (stats_launch + probe_standard)
against the REAL production modiff_cutlass.group_norm_silu_delta_quantize_nhwc, on identical
random inputs, WITH real modulation (mod_scale/mod_shift) and smooth_inv active -- not just probe
vs probe. If this passes, both my stats kernel and my apply kernel are faithful to production, and
the K=2 correctness check (probe vs probe) inherits that fidelity instead of only proving internal
self-consistency.

Then repeats the K=2 skip/catchup correctness check with modulation active (the previous check
used mod_scale=mod_shift=smooth_inv=empty).

Run: python docs/ahat_skip2_exact_2026-08-26/scripts/verify_vs_production.py
"""
import os
import sys

os.chdir("/workspace/MoDiff")
sys.path.insert(0, "src/taming-transformers")
sys.path.insert(0, ".")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/build")

import torch  # noqa: E402
import ahat_skip2_probe as probe  # noqa: E402
import modiff_cutlass as mc  # noqa: E402

torch.manual_seed(0)
N, G = 16, 32
SHAPES = [(192, 32, 32), (384, 16, 16), (384, 32, 32), (576, 32, 32), (768, 16, 16), (384, 8, 8)]


def make(C, H, W, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(N, C, H, W, device="cuda", dtype=torch.float16, generator=g).to(memory_format=torch.channels_last)
    gamma = (0.5 + torch.rand(C, device="cuda", dtype=torch.float16, generator=g))
    beta = (0.1 * torch.randn(C, device="cuda", dtype=torch.float16, generator=g))
    mod_scale = (0.2 * torch.randn(N, C, device="cuda", dtype=torch.float16, generator=g))
    mod_shift = (0.1 * torch.randn(N, C, device="cuda", dtype=torch.float16, generator=g))
    smooth_inv = (0.7 + 0.6 * torch.rand(C, device="cuda", dtype=torch.float16, generator=g))
    return x, gamma, beta, mod_scale, mod_shift, smooth_inv


print("=== Part 1: MY stats+apply pipeline vs the REAL production kernel ===")
empty16 = torch.empty(0, device="cuda", dtype=torch.float16)
empty32 = torch.empty(0, device="cuda", dtype=torch.float32)
empty_i32 = torch.empty(0, device="cuda", dtype=torch.int32)
all_ok = True
for C, H, W in SHAPES:
    for use_mod, use_smooth, tag in [(False, False, "plain"), (True, True, "mod+smooth")]:
        x, gamma, beta, mod_scale, mod_shift, smooth_inv = make(C, H, W, seed=1)
        a0 = (0.1 * torch.randn(N, C, H, W, device="cuda", dtype=torch.float16)).to(memory_format=torch.channels_last)
        scale = torch.tensor([53.0], device="cuda", dtype=torch.float32)
        ms = mod_scale if use_mod else empty16
        sh = mod_shift if use_mod else empty16
        si = smooth_inv if use_smooth else empty32  # production wants float32 smooth_inv (per int8_optimized.py's _smooth_inv_flat)

        # --- production ---
        a_prod = a0.clone()
        # production's smooth_inv is float32 (see _smooth_inv_flat in int8_optimized.py); mod_scale/shift
        # match x's dtype (fp16) per the TORCH_CHECK in the host function.
        si_f32 = smooth_inv.float() if use_smooth else empty32
        x_int8_prod = mc.group_norm_silu_delta_quantize_nhwc(
            x, gamma, beta, a_prod, G, 1e-5, True, scale, si_f32, ms, sh,
            empty32, empty32, empty32, empty_i32, 127.0, False, 1.0)

        # --- mine: stats_launch + probe_standard ---
        a_mine = a0.clone()
        yq_mine = torch.empty(N, C, H, W, device="cuda", dtype=torch.int8).to(memory_format=torch.channels_last)
        mean = torch.empty(N * G, device="cuda", dtype=torch.float32)
        inv_std = torch.empty(N * G, device="cuda", dtype=torch.float32)
        probe.stats_launch(x, mean, inv_std, C, G, H * W, 1e-5)
        # my kernel wants fp16 smooth_inv (matches gn_load2's dtype expectation)
        si_mine = smooth_inv if use_smooth else empty16
        probe.probe_standard_launch(x, a_mine, yq_mine, gamma, beta, ms, sh, si_mine,
                                    mean, inv_std, scale, C, G, C * H * W, N * C * H * W, True, False)

        a_exact = torch.equal(a_prod, a_mine)
        yq_exact = torch.equal(x_int8_prod, yq_mine)
        ok = a_exact and yq_exact
        all_ok = all_ok and ok
        adiff = (a_prod.float() - a_mine.float()).abs().max().item()
        print(f"  C={C:>4} {H}x{W:<3} [{tag:>11}]  a_hat exact={a_exact}  code exact={yq_exact}  "
              f"max|a_hat diff|={adiff:.3g}")

print(f"\nMY PIPELINE MATCHES PRODUCTION BIT-FOR-BIT: {'YES' if all_ok else 'NO -- fix before trusting anything downstream'}")
if not all_ok:
    sys.exit(1)

print("\n=== Part 2: K=2 skip/catchup vs production, WITH modulation active ===")
all_ok2 = True
for C, H, W in SHAPES:
    x1, gamma1, beta1, ms1, sh1, si1 = make(C, H, W, seed=11)
    x2, gamma2, beta2, ms2, sh2, si2 = make(C, H, W, seed=22)
    a0 = (0.1 * torch.randn(N, C, H, W, device="cuda", dtype=torch.float16)).to(memory_format=torch.channels_last)
    scale1 = torch.tensor([53.0], device="cuda", dtype=torch.float32)
    scale2 = torch.tensor([71.0], device="cuda", dtype=torch.float32)
    prev_inv_scale = 1.0 / scale1

    # --- production, two real steps ---
    a_prod = a0.clone()
    mc.group_norm_silu_delta_quantize_nhwc(x1, gamma1, beta1, a_prod, G, 1e-5, True, scale1,
                                           si1.float(), ms1, sh1, empty32, empty32, empty32, empty_i32,
                                           127.0, False, 1.0)
    yq_prod2 = mc.group_norm_silu_delta_quantize_nhwc(x2, gamma2, beta2, a_prod, G, 1e-5, True, scale2,
                                                      si2.float(), ms2, sh2, empty32, empty32, empty32,
                                                      empty_i32, 127.0, False, 1.0)

    # --- skip2 ---
    a_s2 = a0.clone()
    yq_s2 = torch.empty(N, C, H, W, device="cuda", dtype=torch.int8).to(memory_format=torch.channels_last)
    mean1, inv_std1 = torch.empty(N * G, device="cuda", dtype=torch.float32), torch.empty(N * G, device="cuda", dtype=torch.float32)
    mean2, inv_std2 = torch.empty(N * G, device="cuda", dtype=torch.float32), torch.empty(N * G, device="cuda", dtype=torch.float32)
    probe.stats_launch(x1, mean1, inv_std1, C, G, H * W, 1e-5)
    probe.stats_launch(x2, mean2, inv_std2, C, G, H * W, 1e-5)
    probe.probe_skip_launch(x1, a_s2, yq_s2, gamma1, beta1, ms1, sh1, si1, mean1, inv_std1, scale1,
                            C, G, C * H * W, N * C * H * W, True, False)
    probe.probe_catchup_launch(x2, a_s2, yq_s2, gamma2, beta2, ms2, sh2, si2, mean2, inv_std2, scale2,
                               prev_inv_scale, C, G, C * H * W, N * C * H * W, True, False)

    a_exact = torch.equal(a_prod, a_s2)
    yq_exact = torch.equal(yq_prod2, yq_s2)
    ok = a_exact and yq_exact
    all_ok2 = all_ok2 and ok
    print(f"  C={C:>4} {H}x{W:<3}  a_hat exact={a_exact}  code exact={yq_exact}")

print(f"\nK=2 SCHEME MATCHES PRODUCTION BIT-FOR-BIT WITH MODULATION: {'YES' if all_ok2 else 'NO'}")

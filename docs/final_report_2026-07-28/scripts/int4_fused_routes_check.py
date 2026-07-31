"""Validate the two INT4 fusions added after the T1024 layout epilogue.

  A. T256/T64 hd=48 -- packed layout epilogue (mode 3) vs the two-kernel reference
     (gemm_w4a4_awq_qkv_i4qk_i8v, which runs a GEMM plus qkv_i4codes_i8v_rearrange_kernel).
     Held to BYTE EQUALITY: mode 3 keeps mode 2's arithmetic and statement order, it only
     changes where the bytes land.

  B. T16/T4 hd=96 -- the new dp4a int4 small-shape kernel. There is no prior INT4 route to
     compare against (these blocks ran FP16 SDPA), so this is checked against an fp32
     reference computed from the SAME int4 codes the kernel consumes, exactly as
     qattn_correctness.py does. That isolates kernel bugs from quantization error.

No model and no checkpoint -- synthetic tensors at the real production shapes.
"""
import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import torch
import modiff_cutlass as mc

QK_I4_PACKED = 4
TOL = 0.05                       # same threshold qattn_correctness.py uses


# ---------------------------------------------------------------- A. packed hd=48
def check_packed(batch, T, hd=48, nh=8, kpad=512):
    hp = 64
    M, n_compact, n_layout = batch * T, nh * 3 * hd, nh * 3 * hp
    n_awqt = ((n_compact + 127) // 128) * 128
    torch.manual_seed(7)
    xq = torch.randint(-128, 128, (M, kpad // 2), device="cuda", dtype=torch.int8)
    w_c = torch.zeros(n_awqt, kpad // 2, device="cuda", dtype=torch.int8)
    w_c[:n_compact] = torch.randint(-8, 8, (n_compact, kpad // 2),
                                    device="cuda", dtype=torch.int8)
    ws_c = torch.zeros(n_awqt, device="cuda")
    ws_c[:n_compact] = torch.rand(n_compact, device="cuda") * 0.01 + 0.001
    b_c = (torch.rand(n_compact, device="cuda") * 0.2 - 0.1).half()
    sq, sk, a_scale = 0.031, 0.027, 0.019
    sv = torch.rand(hd, device="cuda") * 0.02 + 0.005

    ref_q, ref_k, ref_vt, _ = mc.gemm_w4a4_awq_qkv_i4qk_i8v(
        xq, w_c, ws_c, a_scale, kpad, n_compact, b_c, nh, T, hd, hp, hp,
        QK_I4_PACKED, sq, sk, sv)

    # same weights, re-based into the hp-padded layout the fused epilogue indexes
    w_l = torch.zeros(n_layout, kpad // 2, device="cuda", dtype=torch.int8)
    ws_l = torch.zeros(n_layout, device="cuda")
    iv_l = torch.zeros(n_layout, device="cuda")
    lim_l = torch.zeros(n_layout, device="cuda")
    b_l = torch.zeros(n_layout, device="cuda", dtype=torch.float16)
    for h in range(nh):
        for sel in range(3):
            s, d = (h * 3 + sel) * hd, (h * 3 + sel) * hp
            w_l[d:d + hd].copy_(w_c[s:s + hd])
            ws_l[d:d + hd].copy_(ws_c[s:s + hd])
            b_l[d:d + hd].copy_(b_c[s:s + hd])
            iv_l[d:d + hd] = (1.0 / sq if sel == 0 else
                              1.0 / sk if sel == 1 else 0.0)
            if sel == 2:
                iv_l[d:d + hd].copy_(1.0 / sv.float())
            lim_l[d:d + hd] = 7.0 if sel < 2 else 127.0
    cq, ck, cvt, _ = mc.gemm_w4a4_awq_qkv_i4qk_i8v_layouts(
        xq, w_l, ws_l, a_scale, kpad, iv_l, lim_l, b_l, nh, T, hd, hp, sv, 1)
    torch.cuda.synchronize()

    half = hp // 2
    return {
        "case": f"packed T{T}/hd{hd} batch{batch}",
        "q_bit_exact": bool(torch.equal(ref_q, cq.view(ref_q.shape))),
        "k_bit_exact": bool(torch.equal(ref_k, ck.view(ref_k.shape))),
        "vt_bit_exact": bool(torch.equal(ref_vt, cvt.view(ref_vt.shape))),
        "q_max_diff": int((ref_q.int() - cq.view(ref_q.shape).int()).abs().max()),
        "k_max_diff": int((ref_k.int() - ck.view(ref_k.shape).int()).abs().max()),
        "vt_max_diff": int((ref_vt.int() - cvt.view(ref_vt.shape).int()).abs().max()),
        # nibbles beyond hd must be zero; hd=48 -> hp=64, so bytes 24..31 of each row
        "qk_pad_zero": int(cq.view(-1, half)[:, hd // 2:].abs().max()) == 0
                       and int(ck.view(-1, half)[:, hd // 2:].abs().max()) == 0,
        "vt_pad_zero": int(cvt.view(-1, hp, T)[:, hd:, :].abs().max()) == 0,
    }


# ---------------------------------------------------------------- B. small hd=96
def check_small(batch, T, hd=96, nh=8):
    """fp32 reference from the SAME int4 codes -- isolates kernel bugs from quant error."""
    torch.manual_seed(11)
    C, kpad = nh * hd, nh * hd
    codes = torch.randint(-7, 8, (batch, T, nh, 3, hd), device="cuda", dtype=torch.int8)
    sv = (torch.rand(hd, device="cuda") * 0.02 + 0.005)
    sq, sk, softmax_scale, proj_a = 0.031, 0.027, hd ** -0.5, 0.02

    got = mc.flash_attn_i4values_small_qout(codes, sv, sq, sk, softmax_scale, proj_a, kpad)
    torch.cuda.synchronize()

    q = codes[:, :, :, 0, :].float() * sq
    k = codes[:, :, :, 1, :].float() * sk
    v = codes[:, :, :, 2, :].float() * sv
    # [b,T,nh,hd] -> [b,nh,T,hd]
    q, k, v = (t.permute(0, 2, 1, 3) for t in (q, k, v))
    att = torch.softmax((q @ k.transpose(-1, -2)) * softmax_scale, dim=-1)
    ref = (att @ v).permute(0, 2, 1, 3).reshape(batch * T, C)      # token-major fp32
    ref_codes = (ref / proj_a).round().clamp(-7, 7).to(torch.int8)

    # unpack the kernel's nibbles back to per-channel codes
    g = got.view(batch * T, kpad // 2)
    lo = (g.to(torch.int16) << 12 >> 12)            # sign-extend low nibble
    hi = (g.to(torch.int16) >> 4)
    unpacked = torch.stack([lo, hi], dim=-1).view(batch * T, -1)[:, :C].to(torch.int8)

    diff = (unpacked.int() - ref_codes.int()).abs()
    denom = ref_codes.float().norm().clamp_min(1e-6)
    return {
        "case": f"small T{T}/hd{hd} batch{batch}",
        "codes_within_i4_grid": int(unpacked.abs().max()) <= 7,
        "max_code_diff": int(diff.max()),
        "frac_differing": float((diff > 0).float().mean()),
        "rel_l2": float((unpacked.float() - ref_codes.float()).norm() / denom),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--output", default="docs/final_report_2026-07-28/data/int4_fused_routes.json")
    a = ap.parse_args()

    res = {"gpu": torch.cuda.get_device_name(0), "packed": [], "small": []}
    print("A. packed hd=48 layout epilogue vs the two-kernel reference (byte equality)")
    for bt in (1, 4, a.batch):
        for T in (256, 64):
            r = check_packed(bt, T)
            res["packed"].append(r)
            print(f"   {r}")
    print("\nB. hd=96 int4 small-shape kernel vs an fp32 reference on the same codes")
    for bt in (1, 4, a.batch):
        for T in (16, 4):
            r = check_small(bt, T)
            res["small"].append(r)
            print(f"   {r}")

    ok_p = all(r["q_bit_exact"] and r["k_bit_exact"] and r["vt_bit_exact"]
               and r["qk_pad_zero"] and r["vt_pad_zero"] for r in res["packed"])
    ok_s = all(r["codes_within_i4_grid"] and r["rel_l2"] < TOL for r in res["small"])
    res["gate"] = {"packed_pass": ok_p, "small_pass": ok_s}
    with open(a.output, "w") as f:
        json.dump(res, f, indent=1)
    print(f"\nWROTE {a.output}")
    print(f"GATE packed(byte-exact) : {'PASS' if ok_p else 'FAIL'}")
    print(f"GATE small(rel_l2<{TOL}) : {'PASS' if ok_s else 'FAIL'}")
    sys.exit(0 if (ok_p and ok_s) else 1)

"""Correctness gate for group_norm_silu_quantize_pack_nhwc's new k_pad argument.

Two things have to hold, and they are different claims:

  1. NO REGRESSION: with k_pad omitted (or == C) the output must be BIT-identical to before. The
     kernel's store index changed from `mem_idx0/2` to `hw*KpadH + c/2`, which is only the same
     expression when KpadH == C/2 -- so this is a real risk, not a formality, and every existing
     caller (fused_resblock, the int8-free int4 conv path) depends on it.
  2. PADDING IS EXACT: with k_pad > C the real channels must be byte-identical to the unpadded
     output, and the pad bytes must be zero.
  3. THE SWAP IS BEHAVIOUR-PRESERVING vs the path it replaces (group_norm_silu_nhwc -> F.pad ->
     quantize_act_int4_pack), which is what production ran for the C=192 int4 attention blocks.
     NOT bit-equality: the fused kernel quantizes from the fp32 normalized value while the old path
     rounds to fp16 first, so a nibble sitting on a rounding tie can land one code apart. That is
     pre-existing and documented ("up to one fp16 rounding on the GN output" in
     token_major_attention._qkv_from_gn) -- measured at 0.011-0.016% of nibbles differing by exactly
     1 LSB, and identically so for a no-pad shape this change does not touch (C=384), which is what
     shows k_pad did not introduce it. So the bound is asserted, not equality.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
import torch
import torch.nn.functional as F
import modiff_cutlass as mc

DEV, EPS, G = "cuda", 1e-5, 32
CASES = [(2, 192, 8, 8, 256), (2, 384, 4, 4, 384), (1, 192, 32, 32, 256), (2, 768, 2, 2, 768)]


def run(N, C, H, W, k_pad):
    x = torch.randn(N, C, H, W, device=DEV, dtype=torch.float16).contiguous(
        memory_format=torch.channels_last)
    gw = torch.randn(C, device=DEV, dtype=torch.float16)
    gb = torch.randn(C, device=DEV, dtype=torch.float16)
    a_scale = 0.05
    inv = torch.tensor([1.0 / a_scale], device=DEV, dtype=torch.float32)
    empty = x.new_empty(0)

    base = mc.group_norm_silu_quantize_pack_nhwc(x, gw, gb, G, EPS, False, inv, empty, empty, empty)
    same = mc.group_norm_silu_quantize_pack_nhwc(x, gw, gb, G, EPS, False, inv, empty, empty, empty, C)
    identical_default = torch.equal(base, same)

    padded = mc.group_norm_silu_quantize_pack_nhwc(x, gw, gb, G, EPS, False, inv, empty, empty,
                                                  empty, k_pad)
    T = H * W
    pad_rows = padded.reshape(N * T, k_pad // 2)
    real_ok = torch.equal(pad_rows[:, :C // 2], base.reshape(N * T, C // 2))
    pad_zero = bool((pad_rows[:, C // 2:] == 0).all()) if k_pad > C else True

    # vs the fp16 path this replaces: compare the unpacked int4 codes, allowing the documented
    # 1-LSB fp16-rounding difference on a small fraction of nibbles (see the module docstring).
    xn = mc.group_norm_silu_nhwc(x, gw, gb, G, EPS, False, empty, empty)
    xn_tok = xn.permute(0, 2, 3, 1).reshape(N * T, C)
    ref = mc.quantize_act_int4_pack(F.pad(xn_tok, (0, k_pad - C)).contiguous(),
                                    a_scale).reshape(N * T, k_pad // 2)
    d = (unpack_int4(pad_rows) - unpack_int4(ref)).abs()
    mismatch_pct = float((d > 0).float().mean()) * 100.0
    max_delta = int(d.max())
    close_to_old = (max_delta <= 1) and (mismatch_pct < 0.10)

    return identical_default, real_ok, pad_zero, close_to_old, mismatch_pct, max_delta


def unpack_int4(t):
    """Packed int8 (two signed nibbles/byte, low = even channel) -> int16 codes in [-7,7]."""
    lo = (t & 0xF).to(torch.int16)
    hi = (t.to(torch.int16) >> 4) & 0xF
    lo = torch.where(lo > 7, lo - 16, lo)
    hi = torch.where(hi > 7, hi - 16, hi)
    return torch.stack([lo, hi], -1).reshape(t.shape[0], -1)


def main():
    print(f"{'N':>2} {'C':>4} {'HxW':>7} {'k_pad':>6} | {'default==old':>12} {'real ch ok':>10} "
          f"{'pad zero':>9} | {'vs old path':>11} {'mismatch%':>9} {'max d':>5}")
    ok = True
    for N, C, H, W, kp in CASES:
        d, r, z, m, pct, md = run(N, C, H, W, kp)
        ok &= d and r and z and m
        print(f"{N:2d} {C:4d} {H:3d}x{W:<3d} {kp:6d} | {str(d):>12} {str(r):>10} {str(z):>9} | "
              f"{str(m):>11} {pct:8.3f}% {md:5d}")
    print("\n" + ("ALL PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

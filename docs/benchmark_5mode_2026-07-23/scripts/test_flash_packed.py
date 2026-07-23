"""Bit-exact gate for the packed-input flash kernel (flash_attn_int8_packed_vt / _qout).
Asserts torch.equal vs the current two-kernel path it replaces, over the churches flash shapes
(hd24/T1024, hd48/T256, hd48/T64). Stages:
  fp16 : flash_attn_int8_packed_vt(qkv_fp16)      == flash_attn_int8_vt(*quantize_attn_qkv_packed_static)
  int8 : flash_attn_int8_packed_vt(qkv_int8)      == flash_attn_int8_vt(*quantize_attn_qkv_from_i8)
  qout : flash_attn_int8_packed_vt_qout(qkv_fp16) == flash_attn_int8_vt_qout(*quantize_attn_qkv_packed_static)
"""
import os, sys
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff")
import torch
import modiff_cutlass as mc

SHAPES = [(2, 8, 24, 1024), (2, 8, 48, 256), (2, 8, 48, 64)]   # (b, nh, hd, T)


def rel(a, b):
    return (a.float() - b.float()).norm().item() / (b.float().norm().item() + 1e-12)


def static_scales(qkv, hd):
    hd_pad = ((hd + 31) // 32) * 32
    q, k, v = qkv[:, :, :, 0, :], qkv[:, :, :, 1, :], qkv[:, :, :, 2, :]
    sqc = q.abs().max().item() / 127.0
    skc = k.abs().max().item() / 127.0
    avc = v.abs().amax(dim=(0, 1, 2)).float()
    svv = torch.ones(hd_pad, device=qkv.device)
    svv[:hd] = (avc / 127.0).clamp_min(1e-8)
    return hd_pad, sqc, skc, svv.contiguous()


def ref_flash_vt(qkv, nh, T, hd, hd_pad, sqc, skc, svv, scale):
    b = qkv.size(0)
    qi, ki, vt, sq, sk, sv = mc.quantize_attn_qkv_packed_static(qkv, nh, T, hd, hd_pad, hd_pad, 8, sqc, skc, svv)
    qi = qi.view(b, nh, T, hd_pad); ki = ki.view(b, nh, T, hd_pad); vt = vt.view(b, nh, hd_pad, T)
    sq = sq.view(b, nh, T).contiguous(); sk = sk.view(b, nh, T).contiguous()
    sv = sv[..., :hd].contiguous().view(b, nh, hd)
    return mc.flash_attn_int8_vt(qi, ki, vt, sq, sk, sv, scale)


def main():
    allok = True
    print("=== Stage 1: fp16-input packed flash ===")
    for (b, nh, hd, T) in SHAPES:
        torch.manual_seed(0)
        qkv = torch.randn(b, T, nh, 3, hd, device="cuda", dtype=torch.float16)
        hd_pad, sqc, skc, svv = static_scales(qkv, hd)
        scale = 1.0 / (hd ** 0.5)
        ref = ref_flash_vt(qkv, nh, T, hd, hd_pad, sqc, skc, svv, scale)
        out = mc.flash_attn_int8_packed_vt(qkv, svv[:hd].contiguous(), hd_pad, sqc, skc, scale)
        eq = torch.equal(out, ref)
        allok &= eq
        print(f"  b{b} nh{nh} hd{hd} T{T}: equal={eq}  relL2={rel(out, ref):.2e}")

    # Stage 2: int8-input (Route-1) mode -- feed the SAME int8 tensor to from_i8 (ref) and packed (new).
    if hasattr(mc, "quantize_attn_qkv_from_i8"):
        print("=== Stage 2: int8-input (Route-1) packed flash ===")
        for (b, nh, hd, T) in SHAPES:
            if (hd % 16) != 0:
                print(f"  b{b} nh{nh} hd{hd} T{T}: SKIP (int8 needs hd%16==0; Python falls back)")
                continue
            torch.manual_seed(1)
            hd_pad = ((hd + 31) // 32) * 32
            scale = 1.0 / (hd ** 0.5)
            # int8 packed qkv already scaled+folded upstream (Route-1); simulate with random int8.
            qkv_i8 = torch.randint(-127, 128, (b, T, nh, 3, hd), device="cuda", dtype=torch.int8).contiguous()
            svv = (0.5 + torch.rand(hd_pad, device="cuda")).contiguous()   # arbitrary per-channel dequant
            sqc, skc = 0.013, 0.011
            qi, ki, vt = mc.quantize_attn_qkv_from_i8(qkv_i8, nh, T, hd, hd_pad, hd_pad)
            qi = qi.view(b, nh, T, hd_pad); ki = ki.view(b, nh, T, hd_pad); vt = vt.view(b, nh, hd_pad, T)
            sq = torch.full((b, nh, T), sqc, device="cuda", dtype=torch.float32)
            sk = torch.full((b, nh, T), skc, device="cuda", dtype=torch.float32)
            sv = svv[:hd].view(1, 1, hd).expand(b, nh, hd).contiguous()
            ref = mc.flash_attn_int8_vt(qi, ki, vt, sq, sk, sv, scale)
            out = mc.flash_attn_int8_packed_vt(qkv_i8, svv[:hd].contiguous(), hd_pad, sqc, skc, scale)
            eq = torch.equal(out, ref)
            allok &= eq
            print(f"  b{b} nh{nh} hd{hd} T{T}: equal={eq}  relL2={rel(out, ref):.2e}")

    # Stage 3: _qout (fp16 in -> proj-quantized int8 out)
    if hasattr(mc, "flash_attn_int8_packed_vt_qout") and hasattr(mc, "flash_attn_int8_vt_qout"):
        print("=== Stage 3: _qout (proj-quantized int8 store) ===")
        for (b, nh, hd, T) in SHAPES:
            torch.manual_seed(2)
            qkv = torch.randn(b, T, nh, 3, hd, device="cuda", dtype=torch.float16)
            hd_pad, sqc, skc, svv = static_scales(qkv, hd)
            scale = 1.0 / (hd ** 0.5)
            proj_a = 0.02
            qi, ki, vt, sq, sk, sv = mc.quantize_attn_qkv_packed_static(qkv, nh, T, hd, hd_pad, hd_pad, 8, sqc, skc, svv)
            qi = qi.view(b, nh, T, hd_pad); ki = ki.view(b, nh, T, hd_pad); vt = vt.view(b, nh, hd_pad, T)
            sq = sq.view(b, nh, T).contiguous(); sk = sk.view(b, nh, T).contiguous()
            sv = sv[..., :hd].contiguous().view(b, nh, hd)
            ref = mc.flash_attn_int8_vt_qout(qi, ki, vt, sq, sk, sv, scale, proj_a)
            out = mc.flash_attn_int8_packed_vt_qout(qkv, svv[:hd].contiguous(), hd_pad, sqc, skc, scale, proj_a)
            eq = torch.equal(out, ref)
            allok &= eq
            print(f"  b{b} nh{nh} hd{hd} T{T}: equal={eq}  relL2={rel(out, ref):.2e}")

    print("ALL_EQUAL" if allok else "MISMATCH")


if __name__ == "__main__":
    main()

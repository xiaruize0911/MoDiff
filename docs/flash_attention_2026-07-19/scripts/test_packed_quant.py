"""Verify packed-qkv quantize == current (transpose+contiguous) quantize, bit-for-bit where possible.
Covers int8 and int4-QK/int8-V, dynamic and static, over the churches attention shapes."""
import os, sys
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff")
import torch
import modiff_cutlass as mc

def rel(a, b):
    a = a.float(); b = b.float()
    return (a - b).norm().item() / (b.norm().item() + 1e-12)

def check(b, nh, hd, T, bits):
    torch.manual_seed(0)
    qkv = torch.randn(b, T, nh, 3, hd, device="cuda", dtype=torch.float16)
    BH = b * nh
    hd_pad = ((hd + 31) // 32) * 32
    hp_qk = 64 if bits == 4 else hd_pad
    # --- current path: unbind -> transpose -> contiguous ---
    q = qkv[:, :, :, 0, :].transpose(1, 2).reshape(BH, T, hd).contiguous()
    k = qkv[:, :, :, 1, :].transpose(1, 2).reshape(BH, T, hd).contiguous()
    v = qkv[:, :, :, 2, :].transpose(1, 2).reshape(BH, T, hd).contiguous()
    if bits == 8:
        qi0, ki0, vt0, sq0, sk0, sv0 = mc.quantize_attn_qkv(q, k, v, hd_pad, hd_pad, 8)
    else:
        qi0, ki0, vt0, sq0, sk0, sv0 = mc.quantize_attn_qkv_i4qk_i8v(q, k, v, hp_qk, hd_pad)
    # --- packed path ---
    qi1, ki1, vt1, sq1, sk1, sv1 = mc.quantize_attn_qkv_packed(qkv, nh, T, hd, hp_qk, hd_pad, bits)
    tag = f"b{b} nh{nh} hd{hd} T{T} int{bits}"
    ok = (torch.equal(qi0, qi1) and torch.equal(ki0, ki1) and torch.equal(vt0, vt1))
    print(f"  {tag:26s} DYN  qi/ki/vt equal={ok}  sq rel={rel(sq1,sq0):.1e} sv rel={rel(sv1,sv0):.1e}")
    # --- static ---
    sqc = q.abs().max().item() / (127.0 if bits == 8 else 7.0)
    skc = k.abs().max().item() / (127.0 if bits == 8 else 7.0)
    avc = v.abs().amax(dim=(0, 1)).float()
    svv = torch.ones(hd_pad, device="cuda"); svv[:hd] = (avc / 127.0).clamp_min(1e-8)
    if bits == 8:
        a0 = mc.quantize_attn_qkv_static(q, k, v, hd_pad, hd_pad, 8, sqc, skc, svv)
    else:
        a0 = mc.quantize_attn_qkv_i4qk_i8v_static(q, k, v, hp_qk, hd_pad, sqc, skc, svv)
    a1 = mc.quantize_attn_qkv_packed_static(qkv, nh, T, hd, hp_qk, hd_pad, bits, sqc, skc, svv)
    ok = torch.equal(a0[0], a1[0]) and torch.equal(a0[1], a1[1]) and torch.equal(a0[2], a1[2])
    print(f"  {tag:26s} STAT qi/ki/vt equal={ok}")

# churches flash-eligible shapes (hd<=48, T%64==0): hd24/T1024, hd48/T256, hd48/T64
for bits in (8, 4):
    print(f"===== int{bits} =====")
    check(2, 8, 24, 1024, bits)
    check(2, 8, 48, 256, bits)
    check(2, 8, 48, 64, bits)
print("done")

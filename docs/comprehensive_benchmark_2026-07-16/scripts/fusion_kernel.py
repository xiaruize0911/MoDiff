"""F0/F1 correctness: gemm_w8a8_out_int8 / gemm_w4a4_out_int8 vs round(fp16_gemm*oscale),
and transpose_qkv_int8 vs a torch head-major gather reference."""
import os, sys
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff")
import torch, modiff_cutlass as mc
torch.manual_seed(0)

def pack4(t):
    lo = t[..., 0::2] & 0xF; hi = t[..., 1::2] & 0xF
    return (lo | (hi << 4)).to(torch.int8).contiguous()

print("=== F0: int8-output GEMM ===")
for (M, K, N) in [(32768, 192, 576), (8192, 384, 1152), (2048, 768, 2304)]:
    A16 = torch.randn(M, K, device="cuda").half(); W16 = torch.randn(N, K, device="cuda").half()
    A8 = (A16.float() * 4).round().clamp(-127, 127).to(torch.int8)
    B8 = (W16.float() * 4).round().clamp(-127, 127).to(torch.int8)
    ws = torch.rand(N, device="cuda") * 0.01 + 0.001
    a_scale = 0.02
    nobias = torch.empty(0, device="cuda", dtype=torch.float32)
    bias = torch.randn(N, device="cuda", dtype=torch.float32) * 0.1
    C16 = mc.gemm_w8a8(A8, B8, ws, a_scale)                      # fp16 ref
    oscale = (127.0 / (C16.float().abs().amax(0).clamp_min(1e-6))).contiguous()  # per-col 127/absmax
    ref = torch.round(C16.float() * oscale).clamp(-127, 127).to(torch.int8)
    got = mc.gemm_w8a8_out_int8(A8, B8, ws, a_scale, oscale, nobias)
    diff = (ref.int() - got.int()).abs()
    refb = torch.round((C16.float() + bias) * oscale).clamp(-127, 127).to(torch.int8)
    gotb = mc.gemm_w8a8_out_int8(A8, B8, ws, a_scale, oscale, bias)
    db = (refb.int() - gotb.int()).abs()
    print(f"  w8a8 M{M} K{K} N{N}: max|Δ|={diff.max().item()} within1={100*(diff<=1).float().mean():.2f}%  +bias max|Δ|={db.max().item()} within1={100*(db<=1).float().mean():.2f}%")
    # int4
    A4 = pack4((A16.float() * 2).round().clamp(-7, 7).to(torch.int8))
    B4 = pack4((W16.float() * 2).round().clamp(-7, 7).to(torch.int8))
    C16b = mc.gemm_w4a4(A4, B4, ws, a_scale, K)
    osc4 = (127.0 / (C16b.float().abs().amax(0).clamp_min(1e-6))).contiguous()
    ref4 = torch.round(C16b.float() * osc4).clamp(-127, 127).to(torch.int8)
    got4 = mc.gemm_w4a4_out_int8(A4, B4, ws, a_scale, K, osc4, nobias)
    d4 = (ref4.int() - got4.int()).abs()
    print(f"  w4a4 M{M} K{K} N{N}: max|Δ|={d4.max().item()} within1={100*(d4<=1).float().mean():.2f}%")

print("=== F1: transpose_qkv_int8 ===")
for (B, T, nh, hd) in [(32, 1024, 8, 24), (32, 256, 8, 48)]:
    hd_pad = (hd + 31) // 32 * 32
    qkv = torch.randint(-127, 128, (B, T, nh, 3, hd), device="cuda", dtype=torch.int8)
    qi, ki, vi = mc.transpose_qkv_int8(qkv, nh, hd_pad)          # [B,nh,T,hd_pad]
    # torch reference: head-major gather + zero-pad
    def ref_comp(c):
        r = qkv[:, :, :, c, :].permute(0, 2, 1, 3).contiguous()  # [B,nh,T,hd]
        out = torch.zeros(B, nh, T, hd_pad, device="cuda", dtype=torch.int8)
        out[..., :hd] = r
        return out
    ok = all(torch.equal(g, ref_comp(c)) for g, c in [(qi, 0), (ki, 1), (vi, 2)])
    print(f"  B{B} T{T} nh{nh} hd{hd}->hd_pad{hd_pad}: exact match = {ok}")
print("DONE")

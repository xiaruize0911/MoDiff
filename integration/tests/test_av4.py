"""B3: full int4 (W4A4) standard attention vs fp16 reference. int4 scores lossy — reported."""
import os, sys, math
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff")
import torch, torch.nn.functional as F, modiff_cutlass as mc
torch.manual_seed(0)
def pack(t): lo = t[..., 0::2] & 0xF; hi = t[..., 1::2] & 0xF; return (lo | (hi << 4)).to(torch.int8).contiguous()
def qtok4(x, hd_pad):
    sc = x.abs().amax(-1, keepdim=True).clamp_min(1e-8) / 7.0
    xi = F.pad(torch.round(x / sc).clamp(-7, 7).to(torch.int8), (0, hd_pad - x.shape[-1])).contiguous()
    return pack(xi), sc.squeeze(-1).float().contiguous()
for (BH, T, hd) in [(32, 1024, 24), (32, 256, 48)]:
    hp = 64 if hd <= 64 else 128; scale = 1.0 / math.sqrt(hd)          # hd_pad %64 for int4
    Q = torch.randn(BH, T, hd, device="cuda", dtype=torch.float16)
    K = torch.randn(BH, T, hd, device="cuda", dtype=torch.float16)
    V = torch.randn(BH, T, hd, device="cuda", dtype=torch.float16)
    qi, sq = qtok4(Q, hp); ki, sk = qtok4(K, hp)
    S = mc.attn_qk_int4(qi, ki, hp)
    P, sp = mc.attn_softmax_requant4(S, sq, sk, scale)                 # packed int4 P [BH,T,T/2]
    svc = V.abs().amax(1, keepdim=True).clamp_min(1e-8) / 7.0          # [BH,1,hd]
    Vi = torch.round(V / svc).clamp(-7, 7).to(torch.int8)             # [BH,T,hd]
    Vt = F.pad(Vi.transpose(1, 2).contiguous(), (0, 0, 0, hp - hd)).contiguous()   # [BH,hp,T]
    Vtp = pack(Vt)                                                     # [BH,hp,T/2]
    sv = F.pad(svc.squeeze(1), (0, hp - hd)).float().contiguous()
    O = mc.attn_av_int4(P, Vtp, sp, sv, T)[:, :, :hd]
    ref = torch.bmm(F.softmax(torch.bmm(Q.float(), K.float().transpose(1, 2)) * scale, -1), V.float())
    rel = (O.float() - ref).norm() / ref.norm()
    print(f"BH{BH} T{T} hd{hd}: full int4 (W4A4) attention rel-vs-fp16 = {rel:.4f}")
print("DONE")

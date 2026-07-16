"""B2: full int8 standard attention (QKᵀ→softmax→AV) vs fp16 reference softmax(QKᵀ/√d)·V."""
import os, sys, math
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff")
import torch, torch.nn.functional as F, modiff_cutlass as mc
torch.manual_seed(0)
def qtok(x, hd_pad):
    sc = x.abs().amax(-1, keepdim=True).clamp_min(1e-8) / 127.0
    xi = F.pad(torch.round(x / sc).clamp(-127, 127).to(torch.int8), (0, hd_pad - x.shape[-1])).contiguous()
    return xi, sc.squeeze(-1).float().contiguous()
for (BH, T, hd) in [(32, 1024, 24), (32, 256, 48), (16, 64, 96)]:
    hp_qk = (hd + 31) // 32 * 32; hp_av = (hd + 63) // 64 * 64; scale = 1.0 / math.sqrt(hd)
    Q = torch.randn(BH, T, hd, device="cuda", dtype=torch.float16)
    K = torch.randn(BH, T, hd, device="cuda", dtype=torch.float16)
    V = torch.randn(BH, T, hd, device="cuda", dtype=torch.float16)
    qi, sq = qtok(Q, hp_qk); ki, sk = qtok(K, hp_qk)
    S = mc.attn_qk_int8(qi, ki)
    P, sp = mc.attn_softmax_requant(S, sq, sk, scale)
    # V per-channel-over-T int8, transposed to [BH,hd,T] then padded to [BH,hp_av,T]
    svc = V.abs().amax(1, keepdim=True).clamp_min(1e-8) / 127.0          # [BH,1,hd]
    Vi = torch.round(V / svc).clamp(-127, 127).to(torch.int8)            # [BH,T,hd]
    Vt = F.pad(Vi.transpose(1, 2).contiguous(), (0, 0, 0, hp_av - hd)).contiguous()  # [BH,hp_av,T]
    sv = F.pad(svc.squeeze(1), (0, hp_av - hd)).float().contiguous()     # [BH,hp_av]
    O = mc.attn_av_int8(P, Vt, sp, sv)[:, :, :hd]                        # [BH,T,hd]
    ref = torch.bmm(F.softmax(torch.bmm(Q.float(), K.float().transpose(1, 2)) * scale, -1), V.float())
    rel = (O.float() - ref).norm() / ref.norm()
    print(f"BH{BH} T{T} hd{hd}: full int8 attention rel-vs-fp16 = {rel:.4f}")
print("DONE")

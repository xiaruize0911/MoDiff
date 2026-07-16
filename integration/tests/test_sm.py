"""B1: softmax+requant vs F.softmax; row-sum≈1."""
import os, sys, math
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff")
import torch, torch.nn.functional as F, modiff_cutlass as mc
torch.manual_seed(0)
def qtok(x, hd_pad):
    sc = x.abs().amax(-1, keepdim=True).clamp_min(1e-8) / 127.0
    xi = F.pad(torch.round(x / sc).clamp(-127, 127).to(torch.int8), (0, hd_pad - x.shape[-1])).contiguous()
    return xi, sc.squeeze(-1).float().contiguous()
for (BH, T, hd) in [(32, 1024, 24), (32, 256, 48)]:
    hd_pad = (hd + 31) // 32 * 32; scale = 1.0 / math.sqrt(hd)
    Q = torch.randn(BH, T, hd, device="cuda", dtype=torch.float16)
    K = torch.randn(BH, T, hd, device="cuda", dtype=torch.float16)
    qi, sq = qtok(Q, hd_pad); ki, sk = qtok(K, hd_pad)
    S = mc.attn_qk_int8(qi, ki)
    P, sp = mc.attn_softmax_requant(S, sq, sk, scale)     # P int8[BH,T,T] in [0,127], sp[BH,T]
    p_deq = P.float() * sp.unsqueeze(2)                    # dequantized probs
    S_deq = S * sq.unsqueeze(2) * sk.unsqueeze(1) * scale
    ref = F.softmax(S_deq, dim=-1)
    rel = (p_deq - ref).norm() / ref.norm()
    rowsum = p_deq.sum(-1)                                  # ≈ 1
    print(f"BH{BH} T{T} hd{hd}: softmax rel-vs-F.softmax={rel:.4f}  rowsum mean={rowsum.mean():.4f} min={rowsum.min():.4f} max={rowsum.max():.4f}  Prange=[{P.min().item()},{P.max().item()}]")
print("DONE")

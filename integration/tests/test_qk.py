"""B0: batched int8 QKᵀ vs torch reference (raw int matmul + dequantized scores)."""
import os, sys, math
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff")
import torch, torch.nn.functional as F, modiff_cutlass as mc
torch.manual_seed(0)

def qtok(x, hd_pad):                       # per-token int8, pad hd->hd_pad
    amax = x.abs().amax(-1, keepdim=True).clamp_min(1e-8)
    sc = amax / 127.0
    xi = torch.round(x / sc).clamp(-127, 127).to(torch.int8)
    xi = F.pad(xi, (0, hd_pad - x.shape[-1])).contiguous()
    return xi, sc.squeeze(-1).float().contiguous()   # [BH,T]

for (BH, T, hd) in [(32, 1024, 24), (32, 256, 48), (16, 64, 96)]:
    hd_pad = (hd + 31) // 32 * 32
    Q = torch.randn(BH, T, hd, device="cuda", dtype=torch.float16)
    K = torch.randn(BH, T, hd, device="cuda", dtype=torch.float16)
    qi, sq = qtok(Q, hd_pad); ki, sk = qtok(K, hd_pad)
    S = mc.attn_qk_int8(qi, ki)                        # [BH,T,T] fp32 raw
    # raw int reference (exact): int8 matmul over hd_pad
    S_ref = torch.bmm(qi.float(), ki.float().transpose(1, 2))
    rel_raw = (S - S_ref).norm() / (S_ref.norm() + 1e-8)
    # dequantized scores vs fp16 QKᵀ/sqrt(hd)
    S_deq = S * sq.unsqueeze(2) * sk.unsqueeze(1) * (1.0 / math.sqrt(hd))
    S_fp = torch.bmm(Q.float(), K.float().transpose(1, 2)) * (1.0 / math.sqrt(hd))
    rel_deq = (S_deq - S_fp).norm() / (S_fp.norm() + 1e-8)
    print(f"BH{BH} T{T} hd{hd}->hd_pad{hd_pad}: raw rel={rel_raw:.2e} (exact int)  deq rel-vs-fp16={rel_deq:.4f}")
print("DONE")

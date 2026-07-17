"""Prototype: int8-output qkv-linear -> attention-quantize fusion.

Path A (current): int8 qkv-linear -> fp16 out -> reshape -> quantize_attn_qkv (re-quantize per-head).
Path B (fused):   int8 qkv-linear -> INT8 out (gemm_w8a8_out_int8) -> quantize_attn_qkv_from_i8
                  (dequant-on-the-fly, no fp16 round-trip, no reshape copy).
Both feed the same QKᵀ/softmax/AV. Reports correctness (vs fp32) and the linear+quantize step speedup.
Real C192 attention block: C=192, nh=8, hd=24, T=1024, batch 32."""
import os, sys, math, csv
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff")
import torch, torch.nn.functional as F, modiff_cutlass as mc
OUT = "/workspace/MoDiff/docs/quant_speedup_vs_fp16_2026-07-16/data"

def bench(fn, it=50, warm=20):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / it * 1e3

rows = []
for (b, T, C, nh) in [(32, 1024, 192, 8), (32, 256, 384, 8), (32, 64, 768, 8)]:
    hd = C // nh; M = b * T; C3 = 3 * C; scale = 1.0 / math.sqrt(hd)
    hp_qk, hp_av = (hd + 31) // 32 * 32, (hd + 63) // 64 * 64
    torch.manual_seed(0)
    x = torch.randn(M, C, device="cuda", dtype=torch.float16) * 0.5           # GN output (qkv-linear input)
    W = (torch.randn(C3, C, device="cuda", dtype=torch.float16) * 0.1)        # qkv weight [3C, C]
    # int8 linear operands (static per-tensor a_scale, per-row w_scale)
    a_scale = x.abs().max().item() / 127.0
    w_scale = (W.float().abs().amax(1).clamp_min(1e-8) / 127.0)
    Wq = torch.round(W.float() / w_scale.unsqueeze(1)).clamp(-127, 127).to(torch.int8).contiguous()
    xq = mc.quantize_act_int8(x, a_scale)

    # fp16 reference qkv (what the linear approximates) for oscale + fp32 attention ref
    qkv_fp16 = F.linear(x, W)                                                 # [M, 3C]
    oscale = (127.0 / qkv_fp16.float().abs().amax(0).clamp_min(1e-6)).contiguous()

    def to_heads(qkv):   # [M,3C] -> q,k,v [BH,T,hd]
        z = qkv.view(b, T, nh, 3, hd)
        q, k, v = z.unbind(3)
        return (q.transpose(1, 2).reshape(b * nh, T, hd).contiguous(),
                k.transpose(1, 2).reshape(b * nh, T, hd).contiguous(),
                v.transpose(1, 2).reshape(b * nh, T, hd).contiguous())

    # fp32 reference attention (from the true fp16 qkv)
    qf, kf, vf = to_heads(qkv_fp16)
    ref = torch.bmm(F.softmax(torch.bmm(qf.float(), kf.float().transpose(1, 2)) * scale, -1), vf.float())

    def attn_from_quant(qi, ki, vt, sq, sk, sv):
        S = mc.attn_qk_int8(qi, ki, sq, sk, scale); P, sp = mc.attn_softmax_requant(S)
        return mc.attn_av_int8(P, vt, sp, sv)[:, :, :hd]

    # ---- Path A: int8 linear -> fp16 -> quantize_attn_qkv ----
    def pathA_quant():
        qkv = mc.gemm_w8a8(xq, Wq, w_scale, a_scale)                          # fp16 [M,3C]
        q, k, v = to_heads(qkv)
        return mc.quantize_attn_qkv(q, k, v, hp_qk, hp_av, 8)
    qiA, kiA, vtA, sqA, skA, svA = pathA_quant()
    relA = ((attn_from_quant(qiA, kiA, vtA, sqA, skA, svA).float() - ref).norm() / ref.norm()).item()

    # ---- Path B: int8 linear -> INT8 out -> quantize_attn_qkv_from_i8 ----
    bias0 = torch.empty(0, device="cuda", dtype=torch.float32)
    def pathB_quant():
        qkv_i8 = mc.gemm_w8a8_out_int8(xq, Wq, w_scale, a_scale, oscale, bias0)   # int8 [M,3C]
        return mc.quantize_attn_qkv_from_i8(qkv_i8, oscale, nh, T, hp_qk, hp_av)
    qiB, kiB, vtB, sqB, skB, svB = pathB_quant()
    relB = ((attn_from_quant(qiB, kiB, vtB, sqB, skB, svB).float() - ref).norm() / ref.norm()).item()

    tA = bench(pathA_quant); tB = bench(pathB_quant)
    print(f"C{C} T{T}: linear+quant  A(fp16 rt)={tA:7.1f}us  B(int8 fused)={tB:7.1f}us  {tA/tB:.2f}x | "
          f"rel A={relA:.4f} B={relB:.4f}")
    rows.append(dict(block=f"C{C}", T=T, pathA_us=round(tA, 1), pathB_us=round(tB, 1),
                     fusion_speedup=round(tA / tB, 3), relA=round(relA, 4), relB=round(relB, 4)))
with open(f"{OUT}/fusion_qkv_attn.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("WROTE fusion_qkv_attn.csv")

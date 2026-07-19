"""Before/after the int8-score fix: measure full attention (QKᵀ+softmax+AV) at the real churches
shapes for three paths, with quality vs fp32:
  fp16      : torch bmm QKᵀ -> softmax -> bmm AV
  int8-oldS : attn_qk_int8 (fp16 S) -> attn_softmax_requant (dyn) -> attn_av_int8   [T×T scores fp16]
  int8-newS : attn_qk_int8_s8out (int8 S) -> attn_softmax_requant_s8_dyn (dyn) -> attn_av_int8  [T×T int8]
Writes data/attn_fix_b64.csv."""
import os, sys, csv, math
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff")
import torch, torch.nn.functional as F, modiff_cutlass as mc


def bench(fn, it=100, warm=30, reps=5):
    ts = []
    for _ in range(reps):
        for _ in range(warm): fn()
        torch.cuda.synchronize(); s = torch.cuda.Event(True); e = torch.cuda.Event(True); s.record()
        for _ in range(it): fn()
        e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e) / it * 1e3)
    ts.sort(); return ts[len(ts) // 2]


def rel(a, b):
    return (a.float() - b.float()).norm().item() / (b.float().norm().item() + 1e-12)


BH = 512
SHAPES = [("32² T1024 hd24", 1024, 24, 5), ("16² T256 hd48", 256, 48, 5), ("8² T64 hd48", 64, 48, 5)]
rows = []
torch.manual_seed(0)
print(f"{'shape':16s} | {'fp16':>7} {'i8-oldS':>8} {'i8-newS':>8} | {'newS/old':>8} {'newS/fp16':>9} | {'relOld':>7} {'relNew':>7}")
for (nm, T, hd, cnt) in SHAPES:
    scale = 1.0 / math.sqrt(hd); hp = (hd + 31) // 32 * 32; hpa = (hd + 63) // 64 * 64
    Q = torch.randn(BH, T, hd, device="cuda", dtype=torch.float16)
    K = torch.randn(BH, T, hd, device="cuda", dtype=torch.float16)
    V = torch.randn(BH, T, hd, device="cuda", dtype=torch.float16)
    ref = torch.einsum("nij,njd->nid", torch.softmax(torch.einsum("nid,njd->nij", Q.float(), K.float()) * scale, -1), V.float())

    def full_fp16():
        S = torch.bmm(Q, K.transpose(1, 2)) * scale
        return torch.bmm(F.softmax(S, -1), V)
    qi, ki, vt, sq, sk, sv = mc.quantize_attn_qkv(Q, K, V, hp, hpa, 8)
    # sS = per-tensor score scale (calibrated absmax/127) for the int8-out QKᵀ
    Sf = mc.attn_qk_int8(qi, ki, sq, sk, scale)
    sS = Sf.float().abs().max().item() / 127.0

    def full_old():   # fp16 scores + dynamic requant
        S = mc.attn_qk_int8(qi, ki, sq, sk, scale)
        P, sp = mc.attn_softmax_requant(S)
        return mc.attn_av_int8(P, vt, sp, sv)

    def full_new():   # int8 scores + DYNAMIC int8-score requant
        S = mc.attn_qk_int8_s8out(qi, ki, sq, sk, scale, sS)
        P, sp = mc.attn_softmax_requant_s8_dyn(S, sS)
        return mc.attn_av_int8(P, vt, sp, sv)

    t16 = bench(full_fp16); told = bench(full_old); tnew = bench(full_new)
    relold = rel(full_old()[:, :, :hd], ref); relnew = rel(full_new()[:, :, :hd], ref)
    print(f"{nm:16s} | {t16:7.1f} {told:8.1f} {tnew:8.1f} | {told/tnew:8.2f} {t16/tnew:9.2f} | {relold:7.4f} {relnew:7.4f}")
    rows.append(dict(shape=nm, T=T, hd=hd, count=cnt, fp16_us=round(t16, 2), int8_oldS_us=round(told, 2),
                     int8_newS_us=round(tnew, 2), newS_vs_oldS=round(told / tnew, 3),
                     newS_vs_fp16=round(t16 / tnew, 3), rel_oldS=round(relold, 4), rel_newS=round(relnew, 4)))

with open("docs/layer_roofline_2026-07-19/data/attn_fix_b64.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("\nWROTE attn_fix_b64.csv")

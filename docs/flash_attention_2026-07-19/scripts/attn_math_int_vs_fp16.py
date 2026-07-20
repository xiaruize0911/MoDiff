"""Kernel benchmark: MATERIALIZED int8/int4 attention (MATH-style: full [BH,T,T] scores,
quantize_attn_qkv -> attn_qk_int{8,4} -> softmax_requant -> attn_av_int{8,4}) vs fp16 MATH SDPA,
across every churches attention shape @ b128. Flash is NOT involved. Components broken out.
Writes data/attn_math_int_vs_fp16_b128.csv."""
import os, sys, math, csv
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff")
import torch, torch.nn.functional as F, modiff_cutlass as mc
from torch.nn.attention import sdpa_kernel, SDPBackend
B = int(sys.argv[1]) if len(sys.argv) > 1 else 128
torch.manual_seed(0); dev = "cuda"

def bench(fn, it=100, warm=30, reps=7):
    ts = []
    for _ in range(reps):
        for _ in range(warm): fn()
        torch.cuda.synchronize(); s = torch.cuda.Event(True); e = torch.cuda.Event(True); s.record()
        for _ in range(it): fn()
        e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e) / it * 1e3)
    ts.sort(); return ts[len(ts) // 2]
def relL2(a, b): return (a.float() - b.float()).norm().item() / (b.float().norm().item() + 1e-9)
_bn = torch.randn(4096, 4096, device=dev, dtype=torch.float16)
for _ in range(50): _bn = _bn @ _bn * 1e-4 + 1.0
torch.cuda.synchronize()

# (C, nh, hd, T, count)
SHAPES = [(192, 8, 24, 1024, 5), (384, 8, 48, 256, 5), (384, 8, 48, 64, 5), (768, 8, 96, 16, 5), (768, 8, 96, 4, 1)]
rows = []; tot = {"fp16": 0.0, "i8": 0.0, "i4": 0.0, "i8c": 0.0, "i4c": 0.0}
print(f"MATERIALIZED int8/int4 attention vs fp16 MATH @ b{B}. 'core' = qk+softmax+av (EXCLUDES the")
print(f"quantize_attn_qkv amax/norm+quant, which fp16 has no equivalent for); 'tot' = core+quantize.")
print(f"{'hd/T':>9} {'cnt':>3} | {'fp16':>7} | {'i8 core':>7} {'vsFP':>5} | {'i8 tot':>7} {'vsFP':>5} {'rel':>5} | "
      f"{'i4 core':>7} {'vsFP':>5} | {'i4 tot':>7} {'vsFP':>5} {'rel':>5}")
for (C, H, hd, T, cnt) in SHAPES:
    N = B; BH = N * H; sc = 1.0 / math.sqrt(hd)
    q = torch.randn(N, H, T, hd, device=dev, dtype=torch.float16); k = torch.randn_like(q); v = torch.randn_like(q)
    ref = torch.einsum("nhij,nhjd->nhid", torch.softmax(torch.einsum("nhid,nhjd->nhij", q.float(), k.float()) * sc, -1), v.float())
    q4v = q.reshape(1, BH, T, hd); k4v = k.reshape(1, BH, T, hd); v4v = v.reshape(1, BH, T, hd)
    with sdpa_kernel(SDPBackend.MATH):
        tM = bench(lambda: F.scaled_dot_product_attention(q4v, k4v, v4v, scale=sc))
    qm = q.reshape(BH, T, hd).contiguous(); km = k.reshape(BH, T, hd).contiguous(); vm = v.reshape(BH, T, hd).contiguous()
    hp = (hd + 31) // 32 * 32; hpa = (hd + 63) // 64 * 64; hp4 = (hd + 63) // 64 * 64
    r8 = r4 = None; i8_tot = i4_tot = None; i8_q = i8_core = i4_tot_c = 0.0
    try:
        qi, ki, vt, sq, sk, sv = mc.quantize_attn_qkv(qm, km, vm, hp, hpa, 8)
        def q8(): return mc.quantize_attn_qkv(qm, km, vm, hp, hpa, 8)
        def core8():
            S = mc.attn_qk_int8(qi, ki, sq, sk, sc); P, sp = mc.attn_softmax_requant(S); return mc.attn_av_int8(P, vt, sp, sv)
        r8 = relL2(core8().reshape(BH, T, -1)[..., :hd].reshape(N, H, T, hd), ref)
        i8_q = bench(q8); i8_core = bench(core8); i8_tot = i8_q + i8_core
    except Exception as ex:
        i8_err = type(ex).__name__
    i4_q = i4_core = 0.0
    try:
        qi4, ki4, vt4, sq4, sk4, sv4 = mc.quantize_attn_qkv(qm, km, vm, hp4, hpa, 4)
        def q4f(): return mc.quantize_attn_qkv(qm, km, vm, hp4, hpa, 4)
        def core4():
            S = mc.attn_qk_int4(qi4, ki4, hp4, sq4, sk4, sc); P, sp = mc.attn_softmax_requant4(S); return mc.attn_av_int4(P, vt4, sp, sv4, T)
        r4 = relL2(core4().reshape(BH, T, -1)[..., :hd].reshape(N, H, T, hd), ref)
        i4_q = bench(q4f); i4_core = bench(core4); i4_tot = i4_q + i4_core
    except Exception:
        pass
    def n(x, w=7): return (f"%{w}.1f" % x) if x else " " * (w - 3) + "N/A"
    def rx(x): return f"{x:5.3f}" if x is not None else "  N/A"
    def vx(x): return f"{x:.2f}x" if x else " N/A"
    v8 = (tM / i8_tot) if i8_tot else None; v4 = (tM / i4_tot) if i4_tot else None
    v8c = (tM / i8_core) if i8_core else None; v4c = (tM / i4_core) if i4_core else None
    print(f"{f'{hd}/{T}':>9} {cnt:3d} | {tM:7.1f} | {n(i8_core)} {vx(v8c):>5} | {n(i8_tot)} {vx(v8):>5} {rx(r8)} | "
          f"{n(i4_core)} {vx(v4c):>5} | {n(i4_tot)} {vx(v4):>5} {rx(r4)}")
    rows.append(dict(C=C, hd=hd, T=T, count=cnt, fp16_math_us=round(tM, 1),
                     i8_quant_us=round(i8_q, 1), i8_core_us=round(i8_core, 1), i8_core_vs_fp16=round(v8c, 3) if v8c else "",
                     i8_total_us=round(i8_tot, 1) if i8_tot else "", i8_total_vs_fp16=round(v8, 3) if v8 else "", i8_relL2=round(r8, 4) if r8 is not None else "",
                     i4_quant_us=round(i4_q, 1), i4_core_us=round(i4_core, 1), i4_core_vs_fp16=round(v4c, 3) if v4c else "",
                     i4_total_us=round(i4_tot, 1) if i4_tot else "", i4_total_vs_fp16=round(v4, 3) if v4 else "", i4_relL2=round(r4, 4) if r4 is not None else ""))
    tot["fp16"] += cnt * tM
    tot["i8"] += cnt * (i8_tot if i8_tot else tM)      # ineligible (hd=96) -> stays fp16 MATH
    tot["i4"] += cnt * (i4_tot if i4_tot else tM)
    tot["i8c"] += cnt * (i8_core if i8_core else tM)   # core = excludes the quantize/norm
    tot["i4c"] += cnt * (i4_core if i4_core else tM)

print(f"\n=== weighted total attention / forward (21 blocks) @ b{B} ===")
print(f"fp16 MATH everywhere                         {tot['fp16']:9.1f} us   1.00x")
print(f"int8 CORE (no quantize/norm) elig + fp16      {tot['i8c']:9.1f} us   {tot['fp16']/tot['i8c']:.3f}x   <- fair kernel-only")
print(f"int4 CORE (no quantize/norm) elig + fp16      {tot['i4c']:9.1f} us   {tot['fp16']/tot['i4c']:.3f}x   <- fair kernel-only")
print(f"int8 total (core+quantize) elig + fp16        {tot['i8']:9.1f} us   {tot['fp16']/tot['i8']:.3f}x")
print(f"int4 total (core+quantize) elig + fp16        {tot['i4']:9.1f} us   {tot['fp16']/tot['i4']:.3f}x")
print("(hd=96 blocks T=16/4: materialized int GEMM needs T%64==0 -> stay fp16 MATH)")
with open(f"docs/flash_attention_2026-07-19/data/attn_math_int_vs_fp16_b{B}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"WROTE data/attn_math_int_vs_fp16_b{B}.csv")

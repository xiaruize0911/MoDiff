"""Kernel-level attention benchmark across EVERY shape the churches UNet runs.

For each of the 5 unique attention shapes (from capture_attn_shapes.py), at the
benchmark batch (default b128 -> BH=b*nh), time every applicable attention path:
  - fp16 MATH SDPA        (the old default; quantizable-attention baseline)
  - fp16 FlashAttention-2 (the SHIPPED default = "the real")
  - int8 flash (ours)     kernel-only + quantize + total   [eligible shapes]
  - int4 flash (ours)     kernel-only + quantize + total   [eligible shapes]
  - int8 materialized     QKᵀ+softmax+AV                    [eligible shapes]
  - int8-score materialized (int8 QKᵀ + dyn softmax + AV)  [eligible shapes]
Records rel-L2 vs fp32 reference for each quantized path.

Then weights every path by the per-forward block count of its shape and reports
the EXPECTED total-attention time per forward under each policy, and the speedup
vs the real (all-fp16-flash) baseline.

Writes data/attn_allshapes_kernel_b<batch>.csv and data/attn_policy_b<batch>.csv
Usage: python attn_allshapes_bench.py [batch]   (default 128)
"""
import os, sys, csv, math
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff")
import torch, torch.nn.functional as F, modiff_cutlass as mc
from torch.nn.attention import sdpa_kernel, SDPBackend

B = int(sys.argv[1]) if len(sys.argv) > 1 else 128
FLASH_ORDER = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
torch.manual_seed(0); dev = "cuda"

# (C, nh, hd, T, count) — from capture_attn_shapes.py at any batch (counts are per-forward)
SHAPES = [
    (192, 8, 24, 1024, 5),
    (384, 8, 48,  256, 5),
    (384, 8, 48,   64, 5),
    (768, 8, 96,   16, 5),
    (768, 8, 96,    4, 1),
]

def bench(fn, it=50, warm=20, reps=5):
    ts = []
    for _ in range(reps):
        for _ in range(warm): fn()
        torch.cuda.synchronize(); s = torch.cuda.Event(True); e = torch.cuda.Event(True); s.record()
        for _ in range(it): fn()
        e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e) / it * 1e3)  # us
    ts.sort(); return ts[len(ts) // 2]

def relL2(a, b): return (a.float() - b.float()).norm().item() / (b.float().norm().item() + 1e-9)

# GPU clock burn-in so the FIRST timed path doesn't eat a cold-clock penalty.
_burn = torch.randn(4096, 4096, device=dev, dtype=torch.float16)
for _ in range(50): _burn = _burn @ _burn * 1e-4 + 1.0
torch.cuda.synchronize()

def pack_i4(qi, hdp4):
    hd = qi.shape[-1]; qi = F.pad(qi, (0, hdp4 - hd))
    lo = (qi[..., 0::2].int() & 0xF); hi = (qi[..., 1::2].int() & 0xF)
    return (lo | (hi << 4)).to(torch.uint8).view(torch.int8).contiguous()

rows = []
per_shape = {}   # (C,nh,hd,T) -> dict of path->us  (for policy aggregation)
print(f"Kernel-level attention across all churches shapes @ batch={B}  (BH=b*nh)")
print(f"{'shape (C/nh/hd/T)':>20} {'path':30} {'kernel_us':>9} {'quant_us':>8} {'total_us':>8} {'vs_real':>7} {'relL2':>7}")

for (C, nh, hd, T, count) in SHAPES:
    N = B; H = nh; BH = N * H
    sc = 1.0 / math.sqrt(hd)
    q = torch.randn(N, H, T, hd, device=dev, dtype=torch.float16); k = torch.randn_like(q); v = torch.randn_like(q)
    q4v = q.reshape(1, BH, T, hd); k4v = k.reshape(1, BH, T, hd); v4v = v.reshape(1, BH, T, hd)
    # fp32 reference
    S = torch.einsum("nhid,nhjd->nhij", q.float(), k.float()) * sc
    ref = torch.einsum("nhij,nhjd->nhid", torch.softmax(S, -1), v.float())
    eligible = (T % 64 == 0) and (hd <= 48)
    tag = f"{C}/{nh}/{hd}/{T}"
    shp = {}

    # fp16 flash (THE REAL) — measure first so it's the vs_real denominator
    with sdpa_kernel(FLASH_ORDER):
        t_flash = bench(lambda: F.scaled_dot_product_attention(q4v, k4v, v4v, scale=sc))
    shp["fp16 flash (real)"] = t_flash
    # fp16 MATH (old default)
    with sdpa_kernel(SDPBackend.MATH):
        t_math = bench(lambda: F.scaled_dot_product_attention(q4v, k4v, v4v, scale=sc))
    shp["fp16 MATH (old)"] = t_math

    def emit(path, kern_us, quant_us, total_us, rel):
        vs = t_flash / total_us if total_us else float("nan")
        print(f"{tag:>20} {path:30} {kern_us:9.1f} {quant_us:8.1f} {total_us:8.1f} {vs:6.2f}x {rel:>7}")
        rows.append(dict(C=C, nh=nh, hd=hd, T=T, batch=B, BH=BH, count=count, path=path,
                         kernel_us=round(kern_us, 1), quant_us=round(quant_us, 1),
                         total_us=round(total_us, 1), vs_real=round(vs, 3), relL2=rel))

    emit("fp16 flash (real)", t_flash, 0.0, t_flash, "")
    emit("fp16 MATH (old)", t_math, 0.0, t_math, "")

    if eligible:
        # ---- int8 flash (ours) ----
        hd_pad = ((hd + 31) // 32) * 32
        def quant8():
            sq = (q.abs().amax(-1).clamp_min(1e-8) / 127.0).float()
            sk = (k.abs().amax(-1).clamp_min(1e-8) / 127.0).float()
            sv = (v.abs().amax(2).clamp_min(1e-8) / 127.0).float()
            qi = F.pad(torch.round(q / sq.unsqueeze(-1)).clamp(-127, 127).to(torch.int8), (0, hd_pad - hd)).contiguous()
            ki = F.pad(torch.round(k / sk.unsqueeze(-1)).clamp(-127, 127).to(torch.int8), (0, hd_pad - hd)).contiguous()
            vi = F.pad(torch.round(v / sv.unsqueeze(2)).clamp(-127, 127).to(torch.int8), (0, hd_pad - hd)).contiguous()
            return qi, ki, vi, sq, sk, sv
        qi, ki, vi, sq, sk, sv = quant8()
        out8 = mc.flash_attn_int8(qi, ki, vi, sq, sk, sv, sc)
        r8 = relL2(out8[..., :hd], ref)
        tq8 = bench(quant8); tk8 = bench(lambda: mc.flash_attn_int8(qi, ki, vi, sq, sk, sv, sc))
        emit("int8 flash (ours)", tk8, tq8, tk8 + tq8, round(r8, 4))
        shp["int8 flash (ours)"] = tk8 + tq8
        shp["int8 flash kernel-only"] = tk8

        # ---- int4 flash (ours) ----
        hdp4 = 64; hdp_v = ((hd + 31) // 32) * 32
        def quant4():
            sq = (q.abs().amax(-1).clamp_min(1e-8) / 7.0).float()
            sk = (k.abs().amax(-1).clamp_min(1e-8) / 7.0).float()
            sv = (v.abs().amax(2).clamp_min(1e-8) / 127.0).float()
            qi = torch.round(q / sq.unsqueeze(-1)).clamp(-8, 7).to(torch.int8)
            ki = torch.round(k / sk.unsqueeze(-1)).clamp(-8, 7).to(torch.int8)
            vi = F.pad(torch.round(v / sv.unsqueeze(2)).clamp(-127, 127).to(torch.int8), (0, hdp_v - hd)).contiguous()
            return pack_i4(qi, hdp4), pack_i4(ki, hdp4), vi, sq, sk, sv
        q4, k4, vi4, sq4, sk4, sv4 = quant4()
        out4 = mc.flash_attn_int4(q4, k4, vi4, sq4, sk4, sv4, hdp4, sc)
        r4 = relL2(out4[..., :hd], ref)
        tq4 = bench(quant4); tk4 = bench(lambda: mc.flash_attn_int4(q4, k4, vi4, sq4, sk4, sv4, hdp4, sc))
        emit("int4 flash (ours)", tk4, tq4, tk4 + tq4, round(r4, 4))
        shp["int4 flash (ours)"] = tk4 + tq4
        shp["int4 flash kernel-only"] = tk4

        # ---- int8 materialized (older non-flash path) ----
        try:
            hp = ((hd + 31) // 32) * 32; hpa = hp
            qm = q.reshape(BH, T, hd).contiguous(); km = k.reshape(BH, T, hd).contiguous(); vm = v.reshape(BH, T, hd).contiguous()
            qi2, ki2, vt2, sq2, sk2, sv2 = mc.quantize_attn_qkv(qm, km, vm, hp, hpa, 8)
            def quant8m(): return mc.quantize_attn_qkv(qm, km, vm, hp, hpa, 8)
            def int8_mat():
                Sm = mc.attn_qk_int8(qi2, ki2, sq2, sk2, sc); P, sp = mc.attn_softmax_requant(Sm); return mc.attn_av_int8(P, vt2, sp, sv2)
            om = int8_mat().reshape(BH, T, -1)[..., :hd].reshape(N, H, T, hd)
            rm = relL2(om, ref)
            tqm = bench(quant8m); tkm = bench(int8_mat)
            emit("int8 materialized", tkm, tqm, tkm + tqm, round(rm, 4))
        except Exception as ex:
            print(f"{tag:>20} {'int8 materialized':30}  N/A ({type(ex).__name__})")
    else:
        # ineligible for flash-quant (hd>48 or T%64!=0): only fp16 paths run in-model
        shp["int8 flash (ours)"] = None
        shp["int4 flash (ours)"] = None
        print(f"{tag:>20} {'(flash-quant not eligible: hd>48 or T%64!=0 -> stays fp16 flash)':30}")

    per_shape[(C, nh, hd, T, count)] = shp

# ---------- policy aggregation (per forward, all 21 blocks) ----------
def policy_total(pick):
    tot = 0.0; miss = False
    for key, shp in per_shape.items():
        C, nh, hd, T, count = key
        us = pick(key, shp)
        if us is None:  # fall back to fp16 flash where the chosen path can't run
            us = shp["fp16 flash (real)"]
        tot += count * us
    return tot

pol_real  = policy_total(lambda k, s: s["fp16 flash (real)"])
pol_math  = policy_total(lambda k, s: s["fp16 MATH (old)"])
pol_int8  = policy_total(lambda k, s: s.get("int8 flash (ours)"))
pol_int4  = policy_total(lambda k, s: s.get("int4 flash (ours)"))
pol_int8k = policy_total(lambda k, s: s.get("int8 flash kernel-only"))
pol_int4k = policy_total(lambda k, s: s.get("int4 flash kernel-only"))

print("\n===== EXPECTED total-attention time per forward (sum over all 21 blocks) =====")
print(f"{'policy':46} {'us/forward':>11} {'vs real':>8}")
def prow(name, us): print(f"{name:46} {us:11.1f} {pol_real/us:7.2f}x")
prow("fp16 MATH everywhere (old default)", pol_math)
prow("fp16 flash everywhere (THE REAL, shipped)", pol_real)
prow("int8 flash where eligible, else fp16 flash", pol_int8)
prow("int4 flash where eligible, else fp16 flash", pol_int4)
print("  -- if quantize were FREE (fused prologue) — kernel-only ceiling --")
prow("int8 flash kernel-only where eligible", pol_int8k)
prow("int4 flash kernel-only where eligible", pol_int4k)

with open(f"docs/flash_attention_2026-07-19/data/attn_allshapes_kernel_b{B}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
prows = [
    dict(policy="fp16 MATH (old)", us_per_forward=round(pol_math, 1), speedup_vs_real=round(pol_real / pol_math, 3)),
    dict(policy="fp16 flash (real)", us_per_forward=round(pol_real, 1), speedup_vs_real=1.0),
    dict(policy="int8 flash where eligible", us_per_forward=round(pol_int8, 1), speedup_vs_real=round(pol_real / pol_int8, 3)),
    dict(policy="int4 flash where eligible", us_per_forward=round(pol_int4, 1), speedup_vs_real=round(pol_real / pol_int4, 3)),
    dict(policy="int8 flash kernel-only (quant fused)", us_per_forward=round(pol_int8k, 1), speedup_vs_real=round(pol_real / pol_int8k, 3)),
    dict(policy="int4 flash kernel-only (quant fused)", us_per_forward=round(pol_int4k, 1), speedup_vs_real=round(pol_real / pol_int4k, 3)),
]
with open(f"docs/flash_attention_2026-07-19/data/attn_policy_b{B}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(prows[0].keys())); w.writeheader(); w.writerows(prows)
print(f"\nWROTE data/attn_allshapes_kernel_b{B}.csv + data/attn_policy_b{B}.csv")

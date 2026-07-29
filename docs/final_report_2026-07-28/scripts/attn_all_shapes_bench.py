"""Benchmark + per-kernel profile for EVERY attention input shape in this model.

Two halves, because "speedup" means different things at the two levels and both are needed to
read the result honestly:

  A. score kernel only -- our int8/int4 flash vs PyTorch fp16 SDPA on the same (N,H,T,hd).
     This is the number the kernel work moves. It excludes the quantize, the qkv projection and
     the output projection, so it is an upper bound on what the layer can gain.

  B. whole attention layer, per kernel -- GroupNorm+qkv, activation quantize, the score kernel,
     the output transpose, the output projection, the residual. Measured for the fp16 path and
     for the int8/int4 paths so each kernel can be lined up against its fp16 counterpart. This
     is what actually decides the end-to-end effect.

Shapes come from the model config (configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml:
model_channels=192, channel_mult=[1,2,2,4,4], num_heads=8, latent 32x32), so hd = C/8 and
T = H*W at each resolution. All 21 blocks are covered, including the six hd=96 blocks that our
flash kernel cannot serve (hd > FA_MMA_MAXHD/…): those are reported as ineligible rather than
silently dropped, because they are 6/21 of the layer's work.
"""
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import torch
import torch.nn.functional as F

import modiff_cutlass as mc

DEV = "cuda"
BATCH = 128                     # matches the e2e benchmark batch
HEADS = 8

# (C, H, W, count) for every AttentionBlock in the UNet. hd = C // HEADS, T = H*W.
BLOCKS = [
    (192, 32, 32, 5),
    (384, 16, 16, 5),
    (384,  8,  8, 5),
    (768,  4,  4, 5),
    (768,  2,  2, 1),
]

# Structural eligibility of our flash kernels, from flash_attn_int8.cu's host checks:
#   hd_pad <= FA_MMA_MAXHD (64)  and  T % (FA_MMA_WARPS * FA_MMA_BR) == 64 == 0  and  hd % 8 == 0
FA_MMA_MAXHD, FA_TILE = 64, 64


def eligible(T, hd):
    hd_pad = ((hd + 31) // 32) * 32
    return hd_pad <= FA_MMA_MAXHD and T % FA_TILE == 0 and hd % 8 == 0


def bench(fn, it=25, reps=5):
    """Median of `reps` windows of `it` iterations. Median, not min: min flatters a kernel whose
    variance comes from cache state, and these shapes differ by 250x in footprint."""
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(it):
            fn()
        e.record()
        torch.cuda.synchronize()
        out.append(s.elapsed_time(e) / it * 1e3)          # us per call
    return statistics.median(out)


# ---------------------------------------------------------------- A: score kernel only
def bench_score(N, H, T, hd):
    hp = ((hd + 31) // 32) * 32
    sc = 1.0 / (hd ** 0.5)
    r = {"eligible": eligible(T, hd)}

    q = torch.randn(N, H, T, hd, device=DEV, dtype=torch.float16)
    k = torch.randn(N, H, T, hd, device=DEV, dtype=torch.float16)
    v = torch.randn(N, H, T, hd, device=DEV, dtype=torch.float16)
    r["pt_fp16_us"] = bench(lambda: F.scaled_dot_product_attention(q, k, v, scale=sc))
    del q, k, v
    torch.cuda.empty_cache()

    if not r["eligible"]:
        r["int8_us"] = r["int4_us"] = None
        return r

    qi = torch.randint(-127, 127, (N, H, T, hp), device=DEV, dtype=torch.int8)
    ki = torch.randint(-127, 127, (N, H, T, hp), device=DEV, dtype=torch.int8)
    vt = torch.randint(-127, 127, (N, H, hp, T), device=DEV, dtype=torch.int8).contiguous()
    sq = torch.full((N, H, T), 0.01, device=DEV)
    sk = torch.full((N, H, T), 0.01, device=DEV)
    sv = torch.full((N, H, hd), 0.01, device=DEV)
    r["int8_us"] = bench(lambda: mc.flash_attn_int8_vt(qi, ki, vt, sq, sk, sv, sc))
    del qi, ki
    torch.cuda.empty_cache()

    q4 = torch.randint(-127, 127, (N, H, T, 32), device=DEV, dtype=torch.int8)   # hdp4=64 packed
    k4 = torch.randint(-127, 127, (N, H, T, 32), device=DEV, dtype=torch.int8)
    r["int4_us"] = bench(lambda: mc.flash_attn_int4_vt(q4, k4, vt, sq, sk, sv, 64, sc))
    del q4, k4, vt, sq, sk, sv
    torch.cuda.empty_cache()
    return r


# ---------------------------------------------------------------- B: whole layer, per kernel
def bench_layer(N, C, HW, hd):
    """Per-kernel timing of one AttentionBlock at this shape, for fp16 / int8 / int4.

    The fp16 path is what the repo actually runs today (fused GroupNorm+qkv where T%128==0,
    PyTorch flash, fp16 proj). The quantized paths add the activation quantize and swap the score
    kernel. Kernel names in the output match the SASS/profiler names so the numbers can be
    cross-checked against the e2e profile.
    """
    T = HW * HW
    sc = 1.0 / (hd ** 0.5)
    out = {}

    x = torch.randn(N, T, C, device=DEV, dtype=torch.float16).contiguous()
    gw = torch.randn(C, device=DEV, dtype=torch.float16)
    gb = torch.randn(C, device=DEV, dtype=torch.float16)
    qkvw = torch.randn(3 * C, C, device=DEV, dtype=torch.float16)
    qkvb = torch.randn(3 * C, device=DEV, dtype=torch.float16)
    pw = torch.randn(C, C, device=DEV, dtype=torch.float16)
    pb = torch.randn(C, device=DEV, dtype=torch.float16)

    # S1+S2: GroupNorm(32) + qkv projection.
    # The repo fuses these into one CUTLASS kernel only when T % 128 == 0 (the fused conv's
    # threadblock tile is kM=128 and the per-sample scale/bias offset is only valid inside one
    # sample). Time whichever path this shape actually takes, and say which.
    gn = torch.nn.GroupNorm(32, C).to(DEV).half()

    def gn_qkv_unfused():
        xn = gn(x.transpose(1, 2).reshape(N, C, HW, HW)).reshape(N, C, T).transpose(1, 2)
        return F.linear(xn, qkvw, qkvb)

    out["gn+qkv (unfused: GN + fp16 GEMM)"] = bench(gn_qkv_unfused, it=15)
    out["_gn_qkv_fusable"] = (T % 128 == 0)

    qkv = torch.randn(N, T, HEADS, 3, hd, device=DEV, dtype=torch.float16).contiguous()

    # S4 score kernel, all three precisions
    q = qkv[:, :, :, 0].permute(0, 2, 1, 3).contiguous()
    k = qkv[:, :, :, 1].permute(0, 2, 1, 3).contiguous()
    v = qkv[:, :, :, 2].permute(0, 2, 1, 3).contiguous()
    out["score: PyTorch fp16 flash"] = bench(lambda: F.scaled_dot_product_attention(q, k, v, scale=sc))

    if eligible(T, hd):
        hp = ((hd + 31) // 32) * 32
        sq = torch.full((N, HEADS, T), 0.01, device=DEV)
        sk = torch.full((N, HEADS, T), 0.01, device=DEV)
        sv = torch.full((N, HEADS, hd), 0.01, device=DEV)
        # S3: activation quantize. int8 can fold it into the packed flash kernel; int4 cannot
        # (the packed path is gated on bits==8), so int4 always pays this as its own kernel.
        # sv_vec is indexed by the PADDED av dim, not hd (kernel checks sv_vec.numel()==hp_av)
        svp = torch.full((hp,), 0.01, device=DEV)
        out["quantize q/k/v (int8, standalone)"] = bench(
            lambda: mc.quantize_attn_qkv_packed_static(qkv, HEADS, T, hd, hp, hp, 8,
                                                       0.02, 0.02, svp), it=15)
        qi = torch.randint(-127, 127, (N, HEADS, T, hp), device=DEV, dtype=torch.int8)
        ki = torch.randint(-127, 127, (N, HEADS, T, hp), device=DEV, dtype=torch.int8)
        vt = torch.randint(-127, 127, (N, HEADS, hp, T), device=DEV, dtype=torch.int8).contiguous()
        out["score: ours int8 flash"] = bench(lambda: mc.flash_attn_int8_vt(qi, ki, vt, sq, sk, sv, sc))
        q4 = torch.randint(-127, 127, (N, HEADS, T, 32), device=DEV, dtype=torch.int8)
        k4 = torch.randint(-127, 127, (N, HEADS, T, 32), device=DEV, dtype=torch.int8)
        out["score: ours int4 flash"] = bench(
            lambda: mc.flash_attn_int4_vt(q4, k4, vt, sq, sk, sv, 64, sc))
        # packed int8 variant: quantize folded into flash's own smem staging
        try:
            out["score: ours int8 packed (quantize folded in)"] = bench(
                lambda: mc.flash_attn_int8_packed_vt(qkv, torch.full((hd,), 0.01, device=DEV),
                                                     hp, 0.02, 0.02, sc))   # this one wants [hd]
        except Exception as ex:
            out["score: ours int8 packed (quantize folded in)"] = None
            out["_packed_err"] = str(ex)[:70]
        del qi, ki, vt, q4, k4, sq, sk, sv
    else:
        out["_flash_ineligible"] = f"hd={hd} > 48 or T={T} % 64 != 0"

    # S5: head-major -> token-major transpose (a real copy), then proj GEMM + bias, then residual
    a = torch.randn(N, HEADS, T, hd, device=DEV, dtype=torch.float16)
    out["transpose attn out (head->token major)"] = bench(
        lambda: a.transpose(1, 2).reshape(N, T, C).contiguous())
    at = a.transpose(1, 2).reshape(N, T, C).contiguous()
    out["proj GEMM + bias (fp16)"] = bench(lambda: F.linear(at, pw, pb))
    pr = F.linear(at, pw, pb)
    out["residual add"] = bench(lambda: pr + x)

    del x, qkv, q, k, v, a, at, pr, gw, gb, qkvw, qkvb, pw, pb
    torch.cuda.empty_cache()
    return out


def main():
    # settle clocks
    bn = torch.randn(4096, 4096, device=DEV, dtype=torch.float16)
    for _ in range(60):
        bn = bn @ bn * 1e-4 + 1.0
    torch.cuda.synchronize()
    del bn
    torch.cuda.empty_cache()

    res = {"batch": BATCH, "heads": HEADS, "blocks": [], "layer": {}}

    print("=" * 108)
    print("A. 分数 kernel（仅 QK^T+softmax+AV），batch=128 heads=8")
    print("=" * 108)
    print(f"{'C':>5s} {'HxW':>7s} {'T':>5s} {'hd':>3s} {'块数':>4s} | {'PT fp16':>9s} "
          f"{'int8':>9s} {'int4':>9s} | {'i8/PT':>7s} {'i4/PT':>7s}")
    tot = {"pt": 0.0, "i8": 0.0, "i4": 0.0, "pt_all": 0.0}
    for C, Hs, Ws, cnt in BLOCKS:
        T, hd = Hs * Ws, C // HEADS
        r = bench_score(BATCH, HEADS, T, hd)
        r.update(C=C, HW=f"{Hs}x{Ws}", T=T, hd=hd, count=cnt)
        res["blocks"].append(r)
        tot["pt_all"] += r["pt_fp16_us"] * cnt
        if r["int8_us"]:
            tot["pt"] += r["pt_fp16_us"] * cnt
            tot["i8"] += r["int8_us"] * cnt
            tot["i4"] += r["int4_us"] * cnt
            s8 = f"{r['pt_fp16_us']/r['int8_us']:6.2f}x"
            s4 = f"{r['pt_fp16_us']/r['int4_us']:6.2f}x"
            c8 = f"{r['int8_us']:9.1f}"
            c4 = f"{r['int4_us']:9.1f}"
        else:
            s8 = s4 = "  不合格"
            c8 = c4 = f"{'--':>9s}"
        print(f"{C:5d} {r['HW']:>7s} {T:5d} {hd:3d} {cnt:4d} | {r['pt_fp16_us']:9.1f} "
              f"{c8} {c4} | {s8:>7s} {s4:>7s}")
    print("-" * 108)
    print(f"{'合格块合计 (x块数)':<28s} | {tot['pt']:9.1f} {tot['i8']:9.1f} {tot['i4']:9.1f} | "
          f"{tot['pt']/tot['i8']:6.2f}x {tot['pt']/tot['i4']:6.2f}x")
    print(f"{'全部 21 块合计 (PT fp16)':<28s} | {tot['pt_all']:9.1f}   "
          f"其中不合格块占 {(tot['pt_all']-tot['pt'])/tot['pt_all']*100:.1f}%")
    res["totals_score_kernel"] = tot

    print()
    print("=" * 108)
    print("B. 整层逐 kernel 耗时 (us/块, batch=128)")
    print("=" * 108)
    for C, Hs, Ws, cnt in BLOCKS:
        T, hd = Hs * Ws, C // HEADS
        lay = bench_layer(BATCH, C, Hs, hd)
        res["layer"][f"C{C}_{Hs}x{Ws}"] = lay
        print(f"\n--- C={C}  {Hs}x{Ws}  T={T}  hd={hd}  ({cnt} 块) ---")
        for kk, vv in lay.items():
            if kk.startswith("_"):
                print(f"    [{kk[1:]}] {vv}")
            elif vv is None:
                print(f"    {kk:46s}      --")
            else:
                print(f"    {kk:46s} {vv:9.1f} us")

    out = "docs/final_report_2026-07-28/data/attn_all_shapes_bench.json"
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nWROTE {out}")


if __name__ == "__main__":
    main()

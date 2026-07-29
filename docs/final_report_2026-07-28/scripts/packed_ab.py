"""Decompose why flash_attn_int8_packed_vt loses to quantize+flash, by A/B rather than by reading.

The packed kernel exists to fold the Q/K/V quantize + V-transpose into flash's own smem staging,
which would remove the separate quantize pass (724 us on the C192/T1024 block). It is 1.94x SLOWER
than quantize+flash even after templating killed its register spill, so something inside it costs
more than the pass it replaces. Three candidates, and they imply different fixes:

  (a) the fp16->int8 quantize ARITHMETIC done per key tile,
  (b) the smem->smem staging/transpose itself (independent of dtype),
  (c) something else entirely (occupancy from the extra dynamic smem, pipeline depth, ...).

`mfp_stage` is a pure copy for an int8 input, and the host entry accepts int8 qkv, so running the
SAME kernel on int8 input isolates (a): it keeps the staging but removes the quantize. Hence:

  A  non-packed flash alone                 reads already-quantized, already-transposed K/V
  B  quantize + non-packed flash            the production alternative, end to end
  C  packed, fp16 input                     production packed  = staging + quantize
  D  packed, int8 input                     staging only, quantize removed

  C - D  = the quantize arithmetic inside the kernel        -> candidate (a)
  D - A  = the staging/transpose vs reading it pre-made     -> candidate (b)
  B - C  = what the packed path actually wins or loses today

int8 input needs hd % 16 == 0 for the 16-byte cp.async, so hd=24 (the dominant block) cannot be
run as int8. hd=32/48/64 cover both HD_PAD instantiations, and hd=32 stands in for hd=24's HD_PAD=32
template.
"""
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
import torch
import modiff_cutlass as mc

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "packed_ab.json")
DEV = "cuda"
N, H = 128, 8
SQC = SKC = 0.02
SVC = 0.01
# (T, hd). hd % 16 == 0 so the int8-input variant is legal; hd=24 is covered by hd=32's template.
CASES = [(1024, 32), (1024, 48), (256, 48), (256, 32), (64, 48), (1024, 64)]


def bench(fn, it=25, reps=5):
    try:
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
    except Exception as ex:
        return None, f"{type(ex).__name__}: {str(ex)[:70]}"
    o = []
    for _ in range(reps):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(it):
            fn()
        e.record()
        torch.cuda.synchronize()
        o.append(s.elapsed_time(e) / it * 1e3)
    return statistics.median(o), None


def run(T, hd):
    hp = ((hd + 31) // 32) * 32
    sc = 1.0 / (hd ** 0.5)
    r = {"T": T, "hd": hd, "hd_pad": hp}

    qkv16 = torch.randn(N, T, H, 3, hd, device=DEV, dtype=torch.float16).contiguous()
    # the int8 qkv the D variant consumes: exactly what the fp16 one would quantize to, so D and C
    # do the same mma work and differ only by the quantize arithmetic.
    qkv8 = torch.empty(N, T, H, 3, hd, device=DEV, dtype=torch.int8)
    qkv8[:, :, :, 0] = (qkv16[:, :, :, 0].float() / SQC).round().clamp(-127, 127).to(torch.int8)
    qkv8[:, :, :, 1] = (qkv16[:, :, :, 1].float() / SKC).round().clamp(-127, 127).to(torch.int8)
    qkv8[:, :, :, 2] = (qkv16[:, :, :, 2].float() / SVC).round().clamp(-127, 127).to(torch.int8)
    svv = torch.full((hp,), SVC, device=DEV)
    sv_hd = torch.full((hd,), SVC, device=DEV)

    qi, ki, vt, sq, sk, sv = mc.quantize_attn_qkv_packed_static(qkv16, H, T, hd, hp, hp, 8,
                                                                SQC, SKC, svv)
    qi = qi.view(N, H, T, hp); ki = ki.view(N, H, T, hp); vt = vt.view(N, H, hp, T)
    sq = sq.view(N, H, T).contiguous(); sk = sk.view(N, H, T).contiguous()
    sv = sv[..., :hd].contiguous().view(N, H, hd)

    r["A_flash_only"], _ = bench(lambda: mc.flash_attn_int8_vt(qi, ki, vt, sq, sk, sv, sc))
    r["quantize_only"], _ = bench(lambda: mc.quantize_attn_qkv_packed_static(
        qkv16, H, T, hd, hp, hp, 8, SQC, SKC, svv))

    def b_total():
        a, b, c, d, e, f = mc.quantize_attn_qkv_packed_static(qkv16, H, T, hd, hp, hp, 8,
                                                              SQC, SKC, svv)
        return mc.flash_attn_int8_vt(a.view(N, H, T, hp), b.view(N, H, T, hp), c.view(N, H, hp, T),
                                     d.view(N, H, T).contiguous(), e.view(N, H, T).contiguous(),
                                     f[..., :hd].contiguous().view(N, H, hd), sc)
    r["B_quant_plus_flash"], _ = bench(b_total)
    r["C_packed_fp16"], r["C_err"] = bench(
        lambda: mc.flash_attn_int8_packed_vt(qkv16, sv_hd, hp, SQC, SKC, sc))
    r["D_packed_int8"], r["D_err"] = bench(
        lambda: mc.flash_attn_int8_packed_vt(qkv8, sv_hd, hp, SQC, SKC, sc))

    C, D, A, B = r["C_packed_fp16"], r["D_packed_int8"], r["A_flash_only"], r["B_quant_plus_flash"]
    if C and D:
        r["quantize_arith_in_kernel"] = round(C - D, 1)
    if D and A:
        r["staging_overhead"] = round(D - A, 1)
    if B and C:
        r["packed_vs_alternative"] = round(B / C, 3)
    # HBM bytes each path must move, to say whether the loser is even traffic-bound
    r["bytes_C_packed"] = 3 * N * T * H * hd * 2 + N * H * T * hd * 2          # read fp16 qkv, write fp16 out
    r["bytes_B_alt"] = (3 * N * T * H * hd * 2                                  # quantize reads fp16 qkv
                        + 3 * N * H * T * hp                                    # quantize writes int8 q,k,vt
                        + 3 * N * H * T * hp                                    # flash reads them back
                        + N * H * T * hd * 2)                                   # flash writes fp16 out
    del qkv16, qkv8, qi, ki, vt, sq, sk, sv, svv, sv_hd
    torch.cuda.empty_cache()
    return r


def main():
    bn = torch.randn(4096, 4096, device=DEV, dtype=torch.float16)
    for _ in range(60):
        bn = bn @ bn * 1e-4 + 1.0
    torch.cuda.synchronize(); del bn; torch.cuda.empty_cache()

    rows = [run(T, hd) for T, hd in CASES]
    hdr = (f"{'T':>5} {'hd':>3} {'hdp':>4} | {'A flash':>9} {'quant':>8} {'B=q+f':>9} "
           f"{'C pk16':>9} {'D pk8':>9} | {'C-D quant':>10} {'D-A stage':>10} {'B/C':>6}")
    print(hdr)
    for r in rows:
        f = lambda k: (f"{r[k]:9.1f}" if r.get(k) is not None else f"{'ERR':>9}")
        print(f"{r['T']:5d} {r['hd']:3d} {r['hd_pad']:4d} | {f('A_flash_only')} "
              f"{r['quantize_only']:8.1f} {f('B_quant_plus_flash')} {f('C_packed_fp16')} "
              f"{f('D_packed_int8')} | {r.get('quantize_arith_in_kernel', float('nan')):10.1f} "
              f"{r.get('staging_overhead', float('nan')):10.1f} "
              f"{r.get('packed_vs_alternative', float('nan')):6.2f}")
    for r in rows:
        if r.get("C_err") or r.get("D_err"):
            print(f"  T={r['T']} hd={r['hd']}: C_err={r.get('C_err')} D_err={r.get('D_err')}")
    with open(OUT, "w") as fh:
        json.dump({"batch": N, "heads": H, "rows": rows}, fh, indent=2)
    print(f"\nWROTE {OUT}")


if __name__ == "__main__":
    main()

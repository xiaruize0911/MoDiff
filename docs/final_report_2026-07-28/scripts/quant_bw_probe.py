"""Where do quantize_attn_qkv_packed_static's 724 us go? Bandwidth accounting, by A/B on hd.

This pass is pure quantization overhead -- fp16 pays none of it -- and runs at ~348 GB/s against the
~590 GB/s a single-pass streaming kernel sustains on this card, so roughly half of it is recoverable
in principle. Two candidate causes, with different fixes:

  (a) PADDING. hd is padded to hd_pad = ceil(hd/32)*32, so hd=24 writes 32 bytes per token per
      head for q, k and vt: 25% of every write is zeros. Fixing this means teaching the flash
      kernel to read unpadded rows and pad in smem -- invasive.
  (b) THE V TRANSPOSE. vt is written channel-major [hp_av, T] while the source is token-major, and
      the repo already measured that exact penalty elsewhere (quantize_attn_out: 306 GB/s
      transposed vs 582 GB/s contiguous). Fixing this means a better tiling, which is local.

The A/B that separates them: compare hd=24 (padded, hd_pad=32) against hd=32 (NOT padded, same
hd_pad=32). Both move the same bytes to/from the padded buffers, but hd=32 has 33% more REAL
payload for the same write footprint. If padding were the dominant cost, hd=32 would cost about the
same wall-clock as hd=24 while doing more work (i.e. effective GB/s on real bytes jumps); if the
transpose dominates, both sit at the same GB/s on TOTAL bytes.

Reports both denominators, since they answer different questions:
  bw_total = every byte the kernel actually touches, padding included -> how close to the 590 ceiling
  bw_real  = only the non-padding payload                            -> how much useful work per second
"""
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
import torch
import modiff_cutlass as mc

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "quant_bw_probe.json")
DEV = "cuda"
N, H = 128, 8
SQC = SKC = 0.02
STREAM_CEILING_GBS = 590.0        # measured elsewhere in this repo for a single-pass NHWC stream
CASES = [(1024, 24), (1024, 32), (1024, 48), (1024, 64), (256, 24), (256, 48), (64, 48)]


def bench(fn, it=25, reps=5):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    o = []
    for _ in range(reps):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(it):
            fn()
        e.record()
        torch.cuda.synchronize()
        o.append(s.elapsed_time(e) / it * 1e3)
    return statistics.median(o)


def run(T, hd):
    hp = ((hd + 31) // 32) * 32
    qkv = torch.randn(N, T, H, 3, hd, device=DEV, dtype=torch.float16).contiguous()
    svv = torch.full((hp,), 0.01, device=DEV)
    us = bench(lambda: mc.quantize_attn_qkv_packed_static(qkv, H, T, hd, hp, hp, 8, SQC, SKC, svv))

    BH = N * H
    read_fp16 = 3 * BH * T * hd * 2                      # q, k, v read from the interleaved tensor
    write_qk = 2 * BH * T * hp                           # qi, ki int8, PADDED width
    write_vt = BH * hp * T                               # vt int8, PADDED height, transposed
    write_scales = 2 * BH * T * 4                        # sq, sk f32
    sv_expand = 2 * BH * hp * 4                          # sv_vec.expand().contiguous(): read+write
    total = read_fp16 + write_qk + write_vt + write_scales + sv_expand
    real = read_fp16 + (2 * BH * T * hd) + (BH * hd * T) + write_scales
    pad_bytes = (write_qk + write_vt) - (2 * BH * T * hd + BH * hd * T)

    r = dict(T=T, hd=hd, hd_pad=hp, us=round(us, 1),
             bw_total_gbs=round(total / (us * 1e-6) / 1e9, 1),
             bw_real_gbs=round(real / (us * 1e-6) / 1e9, 1),
             pct_of_ceiling=round(total / (us * 1e-6) / 1e9 / STREAM_CEILING_GBS * 100, 1),
             mb_total=round(total / 1e6, 1), mb_pad=round(pad_bytes / 1e6, 1),
             pad_pct=round(pad_bytes / total * 100, 1),
             us_at_ceiling=round(total / (STREAM_CEILING_GBS * 1e9) * 1e6, 1),
             us_at_ceiling_no_pad=round((total - pad_bytes) / (STREAM_CEILING_GBS * 1e9) * 1e6, 1))
    del qkv, svv
    torch.cuda.empty_cache()
    return r


def main():
    bn = torch.randn(4096, 4096, device=DEV, dtype=torch.float16)
    for _ in range(60):
        bn = bn @ bn * 1e-4 + 1.0
    torch.cuda.synchronize(); del bn; torch.cuda.empty_cache()

    rows = [run(T, hd) for T, hd in CASES]
    print(f"{'T':>5} {'hd':>3} {'hdp':>4} {'us':>8} {'MB':>7} {'pad%':>5} | "
          f"{'bw_tot':>7} {'bw_real':>8} {'%ceil':>6} | {'us@590':>7} {'us@590,nopad':>12}")
    for r in rows:
        print(f"{r['T']:5d} {r['hd']:3d} {r['hd_pad']:4d} {r['us']:8.1f} {r['mb_total']:7.1f} "
              f"{r['pad_pct']:5.1f} | {r['bw_total_gbs']:7.1f} {r['bw_real_gbs']:8.1f} "
              f"{r['pct_of_ceiling']:5.1f}% | {r['us_at_ceiling']:7.1f} "
              f"{r['us_at_ceiling_no_pad']:12.1f}")
    with open(OUT, "w") as f:
        json.dump({"batch": N, "heads": H, "ceiling_gbs": STREAM_CEILING_GBS, "rows": rows}, f,
                  indent=2)
    print(f"\nWROTE {OUT}")


if __name__ == "__main__":
    main()

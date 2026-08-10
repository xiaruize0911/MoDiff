"""What routing W8A4's activation through int4 STORAGE costs, and whether it changes any number.

W8A4 has no hardware path: both int4 tensor-core MMAs take BOTH operands at 4 bits and no
mainstream ISA has a mixed s8xs4 MMA, so a 4-bit activation cannot be fed to an int8-weight GEMM as
nibbles. Making the 4-bitness a property of the FORMAT rather than of a clamp parameter therefore
means: quantize to packed int4, widen back to int8, run the existing int8 conv. That retires
`clamp_code` -- a nibble physically cannot hold a code above 7 -- at the price of one extra pass
over every conv activation.

This measures both halves of that trade:

  A. EQUIVALENCE. int4-pack + widen vs int8-quantize with code_ceiling=7, on the same input and the
     same scale. If these are not bit-identical the refactor is not a refactor.
  B. COST. the widening pass, per conv shape and summed over the UNet's real activation shapes, as
     ms/step to be added to the ~105 ms/step the configuration currently runs at.

Run: python integration/tests/bench_unpack_int4_widen.py [batch]
"""
import os
import sys

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())

import torch
import modiff_cutlass as mc

# The UNet's quantized-conv activation shapes (C, H, W) with how many conv layers see each, taken
# from the LSUN-churches 32x32-latent model. Weighted so the total is a per-step figure.
SHAPES = [(192, 32, 32, 14), (384, 32, 32, 4), (384, 16, 16, 16), (768, 16, 16, 4),
          (384, 8, 8, 12), (768, 8, 8, 6), (768, 4, 4, 10), (1152, 8, 8, 2),
          (1536, 4, 4, 2)]
DEV = "cuda"


def bench(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def equivalence():
    """int4-pack + widen == int8-quantize with code_ceiling=7, bit for bit?"""
    print("A. equivalence: pack+widen vs int8 quantize at code_ceiling=7\n")
    print("| shape | scale regime | max|code| | differing elements |")
    print("|---|---|---:|---:|")
    ok = True
    torch.manual_seed(0)
    ef = torch.empty(0, device=DEV, dtype=torch.float32)
    for C, H, W, _ in SHAPES[:5]:
        x = torch.randn(8, C, H, W, device=DEV, dtype=torch.float16)
        x = x.contiguous(memory_format=torch.channels_last)
        a_hat = (0.5 * torch.randn_like(x)).contiguous(memory_format=torch.channels_last)
        absmax = float((x.float() - a_hat.float()).abs().max())
        # Two regimes. At Q/absmax no code can exceed 7 either way, so the two paths agree
        # trivially. The interesting one is a scale deliberately too fine -- what a stale K>1 scale
        # or a clip ratio below 1 produces -- where the ceiling is the only thing bounding the code.
        for label, scale in (("exact 7/absmax", 7.0 / absmax),
                             ("4x too fine (stale/clip)", 7.0 / (absmax * 0.25))):
            st = torch.tensor([scale], device=DEV, dtype=torch.float32)
            with torch.inference_mode():
                ah_a = a_hat.clone()
                i8 = mc.step1_static_quantize_fprop(x, ah_a, st, ef, 7.0)
                ah_b = a_hat.clone()
                packed = mc.step1_static_quantize_pack_int4_fprop(x, ah_b, st, ef)
                widened = mc.unpack_int4_to_int8_cl(packed, x.shape[0], C, H, W)
            a = i8.permute(0, 2, 3, 1).reshape(-1).to(torch.int16)
            b = widened.permute(0, 2, 3, 1).reshape(-1).to(torch.int16)
            ndiff = int((a != b).sum())
            print(f"| {C}x{H}x{W} | {label} | {int(a.abs().max())} | {ndiff} |")
            ok = ok and ndiff == 0
    print(f"\n{'bit-identical on every case' if ok else 'DIVERGES -- not a pure refactor'}\n")
    return ok


def cost(batch):
    print(f"B. cost of the widening pass, batch {batch}\n")
    print("| shape | layers | per call µs | per step ms |")
    print("|---|---:|---:|---:|")
    total = 0.0
    for C, H, W, n in SHAPES:
        packed = torch.randint(-128, 127, (batch, H, W, C // 2),
                               device=DEV, dtype=torch.int8).contiguous()
        with torch.inference_mode():
            t = bench(lambda: mc.unpack_int4_to_int8_cl(packed, batch, C, H, W))
        total += t * n
        print(f"| {C}x{H}x{W} | {n} | {t * 1000:.1f} | {t * n:.3f} |")
    print(f"| **all {sum(n for *_, n in SHAPES)}** | | | **{total:.3f}** |")
    print(f"\nadded per step: {total:.2f} ms on top of the ~105 ms/step W8A4 currently runs at "
          f"({total / 105.0 * 100:.1f}%)")
    return total


if __name__ == "__main__":
    b = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    same = equivalence()
    cost(b)
    if same:
        print("\nThe two routes produce the same codes, so this is a pure re-encoding of the same "
              "rule: 4-bitness moves from a clamp parameter into the storage format. The ms above "
              "is what that costs.")

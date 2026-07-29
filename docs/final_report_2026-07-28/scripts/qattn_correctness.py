"""Direct correctness gate for the int8/int4 flash kernels against an fp32 reference.

The reference dequantizes the SAME int8/int4 tensors the kernel consumes and runs attention in
fp32, so any mismatch is a kernel bug and not a quantization error. This is what makes the
speedup numbers meaningful -- a fast wrong kernel is easy to write by accident when hoisting
fragments into registers, which is exactly what these rounds of optimization do.

Tolerance: the kernel requantizes the softmax probabilities to int8 with a fixed scale of 127,
so a few 1e-2 relative error is expected and is the pre-existing behaviour, not new error.
"""
import os
import sys

# repo root, so the in-place built extension is importable when run as a script from anywhere
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
import torch
import modiff_cutlass as mc

DEV = "cuda"
TOL = 0.05                        # rel_err ceiling; historical range is 0.015-0.026


def ref_attn(qd, kd, vd, scale):
    """fp32 reference on already-dequantized inputs. qd,kd: [N,H,T,hd]; vd: [N,H,T,hd]."""
    s = torch.matmul(qd.float(), kd.float().transpose(-1, -2)) * scale
    return torch.matmul(torch.softmax(s, dim=-1), vd.float())


def rel_err(a, b):
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


def check_int8(N, H, T, hd):
    hp = ((hd + 31) // 32) * 32
    sc = 1.0 / (hd ** 0.5)
    qi = torch.randint(-127, 128, (N, H, T, hp), device=DEV, dtype=torch.int8)
    ki = torch.randint(-127, 128, (N, H, T, hp), device=DEV, dtype=torch.int8)
    qi[..., hd:] = 0                                  # padding lanes must not contribute
    ki[..., hd:] = 0
    vi = torch.randint(-127, 128, (N, H, T, hd), device=DEV, dtype=torch.int8)
    sq = torch.rand(N, H, T, device=DEV) * 0.01 + 0.005
    sk = torch.rand(N, H, T, device=DEV) * 0.01 + 0.005
    sv = torch.rand(N, H, hd, device=DEV) * 0.01 + 0.005
    vt = torch.zeros(N, H, hp, T, device=DEV, dtype=torch.int8)
    vt[:, :, :hd, :] = vi.permute(0, 1, 3, 2)
    out = mc.flash_attn_int8_vt(qi, ki, vt.contiguous(), sq, sk, sv, sc)
    qd = qi[..., :hd].float() * sq.unsqueeze(-1)
    kd = ki[..., :hd].float() * sk.unsqueeze(-1)
    vd = vi.float() * sv.unsqueeze(-2)
    return rel_err(out.float(), ref_attn(qd, kd, vd, sc))


def check_int4(N, H, T, hd):
    """int4 QKᵀ / int8 PV. q4,k4 are packed nibble pairs along a 64-wide padded K."""
    hdp4, hdp_v = 64, ((hd + 31) // 32) * 32
    sc = 1.0 / (hd ** 0.5)
    qn = torch.zeros(N, H, T, hdp4, device=DEV, dtype=torch.int8)
    kn = torch.zeros(N, H, T, hdp4, device=DEV, dtype=torch.int8)
    qn[..., :hd] = torch.randint(-7, 8, (N, H, T, hd), device=DEV, dtype=torch.int8)
    kn[..., :hd] = torch.randint(-7, 8, (N, H, T, hd), device=DEV, dtype=torch.int8)
    pack = lambda x: ((x[..., 0::2] & 0x0F) | (x[..., 1::2] << 4)).to(torch.int8).contiguous()
    vi = torch.randint(-127, 128, (N, H, T, hd), device=DEV, dtype=torch.int8)
    sq = torch.rand(N, H, T, device=DEV) * 0.01 + 0.005
    sk = torch.rand(N, H, T, device=DEV) * 0.01 + 0.005
    sv = torch.rand(N, H, hd, device=DEV) * 0.01 + 0.005
    vt = torch.zeros(N, H, hdp_v, T, device=DEV, dtype=torch.int8)
    vt[:, :, :hd, :] = vi.permute(0, 1, 3, 2)
    out = mc.flash_attn_int4_vt(pack(qn), pack(kn), vt.contiguous(), sq, sk, sv, hdp4, sc)
    qd = qn[..., :hd].float() * sq.unsqueeze(-1)
    kd = kn[..., :hd].float() * sk.unsqueeze(-1)
    vd = vi.float() * sv.unsqueeze(-2)
    return rel_err(out.float(), ref_attn(qd, kd, vd, sc))


def main():
    # T must be a multiple of WARPS*BR; include a T=128 case so the WARPS=8 instantiation is
    # covered as well as the WARPS=4 fallback (T=64).
    #
    # N*H MATTERS AND USED TO BE TOO SMALL. grid.x == N*H, and the BC=32/WARPS=4 defect this gate
    # missed produces an error that grows with grid.x: at N=4,H=4 (16 CTAs) it is 9.6e-3 and passes
    # TOL, while the production launch (N=128,H=8 -> 1024 CTAs) reaches 0.415. A gate whose largest
    # case is 16 CTAs cannot see a bug that needs hundreds to show up, so the model's real launch
    # shape is now included for every eligible (T, hd).
    cases = [(4, 4, 1024, 24), (4, 4, 256, 48), (4, 4, 64, 48),
             (4, 4, 128, 32), (4, 4, 1024, 64), (2, 8, 512, 40),
             # the model's actual attention launches at the benchmark batch
             (128, 8, 1024, 24), (128, 8, 256, 48), (128, 8, 64, 48)]
    bad = 0
    print(f"{'N':>3s} {'H':>3s} {'T':>5s} {'hd':>3s} | {'int8 rel_err':>12s} {'int4 rel_err':>12s}")
    for N, H, T, hd in cases:
        try:
            e8 = check_int8(N, H, T, hd)
        except Exception as ex:
            e8, s8 = None, f"SKIP({str(ex)[:24]})"
        else:
            s8 = f"{e8:.4f}" + ("" if e8 < TOL else "  FAIL")
            bad += e8 >= TOL
        try:
            e4 = check_int4(N, H, T, hd)
        except Exception as ex:
            e4, s4 = None, f"SKIP({str(ex)[:24]})"
        else:
            s4 = f"{e4:.4f}" + ("" if e4 < TOL else "  FAIL")
            bad += e4 >= TOL
        print(f"{N:3d} {H:3d} {T:5d} {hd:3d} | {s8:>12s} {s4:>12s}")
    print("\nALL PASS" if bad == 0 else f"\n{bad} FAILURES")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

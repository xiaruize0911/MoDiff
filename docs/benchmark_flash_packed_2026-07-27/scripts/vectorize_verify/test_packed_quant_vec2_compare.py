"""Capture-then-compare regression check for quantize_attn_qkv_packed_static
(Cycle 2's aq_qtok_packed_static_qk_kernel / aq_vquant_trans_packed_tiled_kernel).

  python test_packed_quant_vec2_compare.py --capture   # before any .cu change
  python test_packed_quant_vec2_compare.py --compare   # after a rebuild

Test matrix: the 3 real churches flash shapes (hd24/T1024, hd48/T256, hd48/T64)
x {int8, int4}, plus a synthetic ragged shape (hd48/T97) to exercise
aq_vquant_trans_packed_tiled_vec2_kernel's tt%4!=0 scalar-tail branch (T=97 with
VQ_TILE_T=64 gives tiles tt=64,33 -- 33%4=1, genuinely ragged; an earlier T=100
draft was rejected here because both its tiles (64,36) happen to be multiples of
4 and would have silently NOT exercised the tail branch at all).

Also checks that hd=25 (odd) is now cleanly rejected by the TORCH_CHECK(hd%2==0)
added alongside the vec2 kernels, rather than silently corrupting output.
"""
import os, sys, argparse
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff")
import torch
import modiff_cutlass as mc

HERE = os.path.dirname(os.path.abspath(__file__))
CAPTURE_PATH = os.path.join(HERE, "packed_quant_vec2_capture.pt")

# (nh, hd, T) -- b fixed at 2. hd48/T97 is synthetic/ragged (not a real churches shape).
SHAPES = [(8, 24, 1024), (8, 48, 256), (8, 48, 64), (8, 48, 97)]
B = 2


def run_case(nh, hd, T, bits, seed):
    torch.manual_seed(seed)
    qkv = torch.randn(B, T, nh, 3, hd, device="cuda", dtype=torch.float16)
    hd_pad = ((hd + 31) // 32) * 32
    hp_qk = 64 if bits == 4 else hd_pad
    q = qkv[:, :, :, 0, :].transpose(1, 2).reshape(B * nh, T, hd).contiguous()
    k = qkv[:, :, :, 1, :].transpose(1, 2).reshape(B * nh, T, hd).contiguous()
    v = qkv[:, :, :, 2, :].transpose(1, 2).reshape(B * nh, T, hd).contiguous()
    sqc = q.abs().max().item() / (127.0 if bits == 8 else 7.0)
    skc = k.abs().max().item() / (127.0 if bits == 8 else 7.0)
    avc = v.abs().amax(dim=(0, 1)).float()
    svv = torch.ones(hd_pad, device="cuda"); svv[:hd] = (avc / 127.0).clamp_min(1e-8)
    out = mc.quantize_attn_qkv_packed_static(qkv, nh, T, hd, hp_qk, hd_pad, bits, sqc, skc, svv)
    return tuple(t.clone() for t in out)


def all_cases():
    for nh, hd, T in SHAPES:
        for bits in (8, 4):
            yield nh, hd, T, bits


def check_odd_hd_rejected():
    """Post-rebuild only: hd=25 must now raise (TORCH_CHECK), not silently corrupt."""
    try:
        run_case(8, 25, 64, 8, seed=999)
        print("[FAIL] hd=25 (odd) did NOT raise -- TORCH_CHECK(hd%2==0) missing or not rebuilt")
        return False
    except RuntimeError as e:
        print(f"[PASS] hd=25 (odd) cleanly rejected: {str(e).splitlines()[-1][:80]}")
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", action="store_true")
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args()

    if a.capture or not os.path.exists(CAPTURE_PATH):
        results = {}
        for i, (nh, hd, T, bits) in enumerate(all_cases()):
            key = f"nh{nh}_hd{hd}_T{T}_int{bits}"
            results[key] = run_case(nh, hd, T, bits, seed=i)
        torch.save(results, CAPTURE_PATH)
        print(f"[capture] saved {len(results)} cases -> {CAPTURE_PATH}")
        return 0

    ref = torch.load(CAPTURE_PATH)
    all_ok = True
    for i, (nh, hd, T, bits) in enumerate(all_cases()):
        key = f"nh{nh}_hd{hd}_T{T}_int{bits}"
        out = run_case(nh, hd, T, bits, seed=i)
        ok = all(torch.equal(o, r) for o, r in zip(out, ref[key]))
        all_ok &= ok
        print(f"[{'PASS' if ok else 'FAIL'}] {key:26s} qi/ki/vt/sq/sk/sv equal={ok}")
    all_ok &= check_odd_hd_rejected()
    print("ALL PASS" if all_ok else "SOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

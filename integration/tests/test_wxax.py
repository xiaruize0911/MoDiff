"""Gates for the W8A8/W4A4 linear kernels + QuantLinearWxAx module.
Run: PYTHONPATH=src/taming-transformers python integration/tests/test_wxax.py
"""
import os, sys
import torch, torch.nn as nn
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff")
import modiff_cutlass as mc
from integration.kernels.wxax_linear import QuantLinearWxAx, _pack4

torch.manual_seed(0); dev = "cuda"
def relerr(a, b): return (a.float()-b.float()).norm().item()/(b.float().norm().item()+1e-12)
SHAPES = [(32768,192,576),(8192,384,1152),(512,768,2304),(32,768,1536),(4096,768,768),(32,192,768)]


def test_kernels():
    print("[1] gemm_w8a8 / gemm_w4a4 vs fp32 (rel_vs_deq = kernel exactness)")
    ok = True
    for (M,K,N) in SHAPES:
        A = torch.randn(M,K,device=dev)*0.5; B = torch.randn(N,K,device=dev)*0.5
        # W8A8
        sa8 = A.abs().max().item()/127.0
        Aq8 = torch.round(A/sa8).clamp(-127,127).to(torch.int8)
        sb8 = (B.abs().amax(1).clamp_min(1e-8)/127.0)
        Bq8 = torch.round(B/sb8.unsqueeze(1)).clamp(-127,127).to(torch.int8)
        C8 = mc.gemm_w8a8(Aq8, Bq8, sb8.float().contiguous(), sa8)
        deq8 = (Aq8.float()*sa8)@(Bq8.float()*sb8.unsqueeze(1)).t()
        e8 = relerr(C8, deq8)
        # W4A4
        sa4 = A.abs().max().item()/7.0
        Aq4 = torch.round(A/sa4).clamp(-7,7).to(torch.int8)
        sb4 = (B.abs().amax(1).clamp_min(1e-8)/7.0)
        Bq4 = torch.round(B/sb4.unsqueeze(1)).clamp(-7,7).to(torch.int8)
        C4 = mc.gemm_w4a4(_pack4(Aq4), _pack4(Bq4), sb4.float().contiguous(), sa4, K)
        deq4 = (Aq4.float()*sa4)@(Bq4.float()*sb4.unsqueeze(1)).t()
        e4 = relerr(C4, deq4)
        p = e8 < 1e-3 and e4 < 1e-3; ok &= p
        print(f"    (M{M},K{K},N{N}) w8a8 kdiff={e8:.5f} w4a4 kdiff={e4:.5f}  {'PASS' if p else 'FAIL'}")
    return ok


def test_module():
    print("[2] QuantLinearWxAx vs fp16 nn.Linear (e2e rel-err)")
    ok = True
    for bits in (8, 4):
        for (K,N) in [(768,1536),(384,1152),(768,768),(192,576)]:
            lin = nn.Linear(K, N).to(dev).half()
            x = torch.randn(64, K, device=dev, dtype=torch.float16)
            ref = lin(x)
            q = QuantLinearWxAx(lin, bits).to(dev)
            q.set_a_scale(x.abs().max().item()/q.Q)   # static (calibrated on this input)
            out = q(x)
            e = relerr(out, ref)
            thr = 0.05 if bits == 8 else 0.35
            p = e < thr; ok &= p
            print(f"    bits{bits} {K}->{N}: rel_vs_fp16={e:.4f}  {'PASS' if p else 'FAIL'}")
    return ok


if __name__ == "__main__":
    a = test_kernels(); b = test_module()
    print("\nALL PASS" if (a and b) else "\nSOME FAILED")

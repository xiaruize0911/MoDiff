#!/usr/bin/env python3
"""Correctness gate for MoDiff quantized kernels and fused ops.

There is no pre-existing test suite in this repo, so this provides a hard
pass/fail regression gate used by the optimization plan
(/root/.claude/plans/make-a-plan-for-tingly-russell.md).

Two independent checks per module:

1. GOLDEN regression (tight): the module's output on fixed seeded inputs is
   compared against a saved reference tensor in integration/tests/golden/.
   A change that is *supposed* to be bit-identical (e.g. folding scale/bias
   into a fused epilogue) must keep rel_err < GOLDEN_TOL. Regenerate goldens
   intentionally with `UPDATE_GOLDEN=1` when a change is meant to alter output
   (e.g. the GroupNorm fp16 change), and record why in the commit.

2. FULL-PRECISION sanity (loose): the quantized output vs an fp32/fp16
   reference must stay within a precision-appropriate bound. Catches gross
   breakage even before any golden exists.

Run:
    python integration/tests/test_kernel_correctness.py            # check
    UPDATE_GOLDEN=1 python integration/tests/test_kernel_correctness.py

Exit code 0 = all pass, 1 = any failure. Also exposes test_* functions for
pytest if it is ever added.
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

GOLDEN_DIR = os.path.join(HERE, "golden")
UPDATE = os.environ.get("UPDATE_GOLDEN", "0") == "1"
GOLDEN_TOL = 1e-3          # tolerance for "should be unchanged" regression
DEV = "cuda"

os.makedirs(GOLDEN_DIR, exist_ok=True)


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float(); b = b.float()
    denom = b.norm().item()
    return (a - b).norm().item() / (denom + 1e-12)


def _fingerprint(out: torch.Tensor):
    """Compact, committable golden: shape + a strided fp32 sample (<=65536 elems)."""
    flat = out.detach().float().reshape(-1).cpu()
    stride = max(1, flat.numel() // 65536)
    return {"shape": tuple(out.shape), "sample": flat[::stride].clone()}


def check_golden(name: str, out: torch.Tensor) -> str:
    """Compare out to saved golden fingerprint; save if updating/missing."""
    path = os.path.join(GOLDEN_DIR, f"{name}.pt")
    fp = _fingerprint(out)
    if UPDATE or not os.path.exists(path):
        torch.save(fp, path)
        return f"golden {'updated' if UPDATE else 'created'}"
    ref = torch.load(path)
    if tuple(ref["shape"]) != fp["shape"]:
        return f"FAIL golden shape {tuple(ref['shape'])} != {fp['shape']}"
    re = rel_err(fp["sample"], ref["sample"])
    return ("PASS" if re < GOLDEN_TOL else "FAIL") + f" golden rel_err={re:.2e}"


def _calib_conv(opt, x0):
    opt.set_calibrating(True)
    _ = opt(x0)
    opt.set_calibrating(False)
    if getattr(opt, "_scale_count", 0) > 0:
        opt.static_input_scale.fill_(opt._scale_sum / opt._scale_count)
    opt.is_calibrated = True
    opt.set_standard_output_fp16(True)
    opt.enable_modiff(False)
    return opt


# ---- individual checks: return (name, ok, detail) ----

def test_int8_conv():
    from integration.kernels.int8_optimized import OptimizedInt8Conv2d
    torch.manual_seed(0)
    conv = nn.Conv2d(256, 256, 3, padding=1).to(DEV)
    opt = _calib_conv(OptimizedInt8Conv2d(conv).to(DEV),
                      torch.randn(32, 256, 32, 32, device=DEV))
    x = torch.randn(32, 256, 32, 32, device=DEV, dtype=torch.float16
                    ).contiguous(memory_format=torch.channels_last)
    out = opt._forward_standard(x)
    ref = conv(x.float())
    re = rel_err(out, ref)
    g = check_golden("int8_conv_res32_3x3", out)
    ok = re < 0.15 and not g.startswith("FAIL")
    return "int8_conv", ok, f"rel_err_vs_fp32={re:.3f} | {g}"


def test_int4_conv():
    from integration.kernels.int4_optimized import OptimizedInt4Conv2d
    torch.manual_seed(0)
    conv = nn.Conv2d(256, 256, 3, padding=1).to(DEV)
    opt = _calib_conv(OptimizedInt4Conv2d(conv).to(DEV),
                      torch.randn(32, 256, 32, 32, device=DEV))
    x = torch.randn(32, 256, 32, 32, device=DEV, dtype=torch.float16
                    ).contiguous(memory_format=torch.channels_last)
    out = opt._forward_standard(x)
    ref = conv(x.float())
    re = rel_err(out, ref)
    g = check_golden("int4_conv_res32_3x3", out)
    ok = re < 0.40 and not g.startswith("FAIL")
    return "int4_conv", ok, f"rel_err_vs_fp32={re:.3f} | {g}"


def test_int8_linear():
    from integration.kernels.int8_linear import OptimizedInt8Linear
    torch.manual_seed(0)
    lin = nn.Linear(4096, 4096).to(DEV).half()
    opt = OptimizedInt8Linear(lin, backend="int_gemm", int_gemm_min_m=64).to(DEV)
    opt.is_calibrated = True
    x = torch.randn(256, 4096, device=DEV, dtype=torch.float16)
    out = opt(x)
    ref = lin(x)
    re = rel_err(out, ref)
    g = check_golden("int8_linear_4096", out)
    ok = re < 0.05 and not g.startswith("FAIL")
    return "int8_linear", ok, f"rel_err_vs_fp16={re:.3f} | {g}"


def test_group_norm_silu():
    import modiff_cutlass as mc
    torch.manual_seed(0)
    x = torch.randn(32, 256, 32, 32, device=DEV, dtype=torch.float16
                    ).contiguous(memory_format=torch.channels_last)
    w = torch.randn(256, device=DEV, dtype=torch.float16)
    b = torch.randn(256, device=DEV, dtype=torch.float16)
    ng, eps = 32, 1e-5
    out = mc.group_norm_silu_nhwc(x, w, b, ng, eps, True)
    ref = F.silu(F.group_norm(x.float(), ng, w.float(), b.float(), eps))
    re = rel_err(out, ref)
    g = check_golden("group_norm_silu_res32", out)
    ok = re < 2e-2 and not g.startswith("FAIL")
    return "group_norm_silu", ok, f"rel_err_vs_fp32={re:.3f} | {g}"


TESTS = [test_int8_conv, test_int4_conv, test_int8_linear, test_group_norm_silu]


def main():
    if not torch.cuda.is_available():
        print("CUDA unavailable — skipping."); return 0
    print(f"{'module':<18}{'status':<8}detail   (UPDATE_GOLDEN={UPDATE})")
    print("-" * 78)
    all_ok = True
    for t in TESTS:
        try:
            name, ok, detail = t()
        except Exception as e:
            name, ok, detail = t.__name__, False, f"EXCEPTION {type(e).__name__}: {e}"
        all_ok &= ok
        print(f"{name:<18}{'PASS' if ok else 'FAIL':<8}{detail}")
    print("-" * 78)
    print("ALL PASS" if all_ok else "FAILURES PRESENT")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

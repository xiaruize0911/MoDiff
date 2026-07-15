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
    # Finalize calibration WITHOUT SmoothQuant so the module lands in a single,
    # self-consistent per-tensor static scale: static_input_scale == the cached
    # dequant alpha (1/static), and weights quantized against their own (unsmoothed)
    # range. Nulling _act_channel_max makes end_calibration() take its no-SmoothQuant
    # branch (static_scale = mean per-sample scale, smooth_inv = identity) and — the
    # crucial part — set static_input_scale AND _cached_scale_float/_cached_alpha_tensor
    # together, so quantize and dequant use the same scale.
    #
    # The old code did the opposite: it called set_calibrating(False), which folds
    # SmoothQuant into the weights and derives static_input_scale from the *smoothed*
    # activation range, and THEN overwrote only static_input_scale with the mean
    # per-sample (unsmoothed) scale — leaving the cached dequant alpha and the smoothed
    # weights keyed to a different scale. int8's 8-bit range absorbed that mismatch, so
    # its tests passed; int4's 4-bit range did not: on the MoDiff first-step (which,
    # unlike the raw-input standard path, quantizes the *smoothed* activation) the
    # 15x-too-small scale rounded almost every value to 0, giving o_hat ≈ bias
    # (rel ~1.0). That was the "int4 MoDiff scale bug" — a calibration-state
    # inconsistency in this harness, not a kernel/convention bug in _int4_conv (which
    # matches _conv_from_int4 and _int8_conv exactly).
    opt.calibrating = False
    opt._act_channel_max = None
    opt.end_calibration()
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


def test_int8_conv_channels_last():
    """Regression: model.to(memory_format=channels_last) must NOT corrupt the packed
    weight_int8. It reformats the 4D [K,R,S,C] buffer to a channels_last stride, which
    for 3x3 convs silently transposes the layout the CUTLASS kernel reads (garbage,
    rel~0.87) unless OptimizedInt8Conv2d._apply re-contiguates it. 1x1 convs are
    immune, so this MUST use a 3x3 conv."""
    from integration.kernels.int8_optimized import OptimizedInt8Conv2d
    torch.manual_seed(0)
    conv = nn.Conv2d(256, 256, 3, padding=1).to(DEV)
    opt = _calib_conv(OptimizedInt8Conv2d(conv).to(DEV),
                      torch.randn(32, 256, 32, 32, device=DEV))
    opt = opt.to(memory_format=torch.channels_last)   # the operation that triggered the bug
    x = torch.randn(32, 256, 32, 32, device=DEV, dtype=torch.float16
                    ).contiguous(memory_format=torch.channels_last)
    out = opt._forward_standard(x)
    re = rel_err(out, conv(x.float()))
    contig = opt.weight_int8.is_contiguous()
    ok = re < 0.15 and contig
    return "int8_conv_CL", ok, f"rel_err_vs_fp32={re:.3f} weight_contig={contig}"


def test_int8_dual_store():
    """The block-entry-quantize fusion kernel: conv3 dual store must equal
    relu(forward_from_int8 + residual) in fp16 AND its requantization to int8."""
    import modiff_cutlass as mc
    if not hasattr(mc, "conv2d_int8_fprop_deepfuse_bias_residual_dual"):
        return "int8_dual_store", False, "dual-store kernel missing (rebuild)"
    from integration.kernels.int8_optimized import OptimizedInt8Conv2d
    torch.manual_seed(0)
    conv = nn.Conv2d(256, 256, 3, padding=1).to(DEV)
    opt = _calib_conv(OptimizedInt8Conv2d(conv).to(DEV),
                      torch.randn(16, 256, 32, 32, device=DEV)).to(memory_format=torch.channels_last)
    nxt = _calib_conv(OptimizedInt8Conv2d(nn.Conv2d(256, 256, 1).to(DEV)).to(DEV),
                      torch.randn(16, 256, 32, 32, device=DEV)).to(memory_format=torch.channels_last)
    x = torch.randn(16, 256, 32, 32, device=DEV, dtype=torch.float16).contiguous(memory_format=torch.channels_last)
    xi = opt.quantize_input(x)
    resid = torch.randn(16, 256, 32, 32, device=DEV, dtype=torch.float16).contiguous(memory_format=torch.channels_last)
    fp16_ref = torch.relu(opt.forward_from_int8(xi, residual=resid))
    int8_ref = nxt.quantize_input(fp16_ref)
    fp16_out, int8_out = opt.forward_from_int8_dual(xi, resid, nxt.static_input_scale, apply_relu=True)
    re_fp16 = rel_err(fp16_out, fp16_ref)
    re_int8 = (int8_out.float() - int8_ref.float()).abs().mean().item()
    ok = re_fp16 < 1e-3 and re_int8 < 0.02
    return "int8_dual_store", ok, f"fp16_rel={re_fp16:.4f} int8_mean|Δ|={re_int8:.4f}"


def test_int4_dual_store():
    """int4 block-entry-quantize fusion: conv3 dual store == relu(forward_from_int4 +
    residual) in fp16, and its packed-int4 requantization is consistent."""
    import modiff_cutlass as mc
    if not hasattr(mc, "conv2d_int4_fprop_bias_residual_dual"):
        return "int4_dual_store", False, "int4 dual-store kernel missing (rebuild)"
    from integration.kernels.int4_optimized import OptimizedInt4Conv2d
    torch.manual_seed(0)
    H = W = 32
    conv = nn.Conv2d(256, 256, 3, padding=1).to(DEV)
    opt = _calib_conv(OptimizedInt4Conv2d(conv).to(DEV),
                      torch.randn(16, 256, H, W, device=DEV))
    nxt = _calib_conv(OptimizedInt4Conv2d(nn.Conv2d(256, 256, 1).to(DEV)).to(DEV),
                      torch.randn(16, 256, H, W, device=DEV))
    x = torch.randn(16, 256, H, W, device=DEV, dtype=torch.float16).contiguous(memory_format=torch.channels_last)
    xp = opt.quantize_input(x)
    resid = torch.randn(16, 256, H, W, device=DEV, dtype=torch.float16).contiguous(memory_format=torch.channels_last)
    fp16_ref = torch.relu(opt.forward_from_int4(xp, H, W, residual=resid))
    int4_ref = nxt.quantize_input(fp16_ref)
    fp16_out, int4_out = opt.forward_from_int4_dual(xp, H, W, resid, nxt.static_input_scale, apply_relu=True)
    re_fp16 = rel_err(fp16_out, fp16_ref)
    def unpack(p):
        p = p.to(torch.int16); lo = p & 0xF; hi = (p >> 4) & 0xF
        lo = torch.where(lo >= 8, lo - 16, lo); hi = torch.where(hi >= 8, hi - 16, hi)
        return torch.stack([lo, hi]).float()
    nib = (unpack(int4_out) - unpack(int4_ref)).abs().mean().item()
    ok = re_fp16 < 1e-3 and nib < 0.05
    return "int4_dual_store", ok, f"fp16_rel={re_fp16:.4f} nibble_mean|Δ|={nib:.4f}"


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
    _empty = torch.empty(0, device=DEV, dtype=torch.float16)
    out = mc.group_norm_silu_nhwc(x, w, b, ng, eps, True, _empty, _empty)
    ref = F.silu(F.group_norm(x.float(), ng, w.float(), b.float(), eps))
    re = rel_err(out, ref)
    g = check_golden("group_norm_silu_res32", out)
    ok = re < 2e-2 and not g.startswith("FAIL")
    return "group_norm_silu", ok, f"rel_err_vs_fp32={re:.3f} | {g}"


def _modiff_check(opt, conv, C, HW, N):
    """Drive a static-input MoDiff sequence and return (lifecycle_ok, acc, detail).
    Validates the MoDiff STATE MACHINE — first-step builds cache, modulated steps
    CONVERGE (don't diverge), reset restores, shape-mismatch reboots — which is the
    same for int8/int4. `acc` (first-step o_hat vs fp32 conv) is returned separately so
    each caller can decide whether to gate on numerics (int8 works; int4 has a known
    MoDiff-path scale bug, see report)."""
    x = torch.randn(N, C, HW, HW, device=DEV, dtype=torch.float16).contiguous(memory_format=torch.channels_last)
    outs = [opt(x).float() for _ in range(5)]                 # step1 = first-step, 2..5 = modulated
    built = (opt.a_hat_cache is not None and opt.o_hat_cache is not None and not opt.is_first_step)
    finite = all(torch.isfinite(o).all() for o in outs)
    stable = rel_err(outs[4], outs[3])                        # modulated steps converge (no divergence)
    nogrow = outs[4].abs().max().item() <= 3.0 * outs[0].abs().max().item() + 1e-6
    acc = rel_err(outs[0], conv(x.float()))                   # first-step vs fp32 conv
    opt.reset_state(); reset_ok = opt.is_first_step
    try:                                                       # shape-mismatch must auto-reboot to first-step
        opt.enable_modiff(True)
        x2 = torch.randn(N, C, HW // 2, HW // 2, device=DEV, dtype=torch.float16).contiguous(memory_format=torch.channels_last)
        opt(x2); opt(x2); reboot_ok = True
    except Exception:
        reboot_ok = False
    lifecycle_ok = built and finite and stable < 5e-2 and nogrow and reset_ok and reboot_ok
    detail = f"lifecycle={lifecycle_ok} (built={built} stable={stable:.4f} nogrow={nogrow} reset={reset_ok} reboot={reboot_ok}) acc={acc:.3f}"
    return lifecycle_ok, acc, detail


def test_int8_modiff_conv():
    """int8 MoDiff: state machine AND numerics (first-step is a valid quantized conv)."""
    from integration.kernels.int8_optimized import OptimizedInt8Conv2d
    torch.manual_seed(0)
    conv = nn.Conv2d(256, 256, 3, padding=1).to(DEV)
    opt = _calib_conv(OptimizedInt8Conv2d(conv).to(DEV),
                      torch.randn(16, 256, 32, 32, device=DEV)).to(memory_format=torch.channels_last)
    opt.enable_modiff(True)
    lifecycle_ok, acc, detail = _modiff_check(opt, conv, 256, 32, 16)
    return "int8_modiff_conv", lifecycle_ok and acc < 0.20, detail


def test_int4_modiff_conv():
    """int4 MoDiff: state machine AND numerics (first-step is a valid quantized conv).

    The int4 first-step o_hat used to be ~15x too small (rel ~1.0) under this harness —
    but that was a calibration-state inconsistency in _calib_conv (a smoothquant-derived
    static scale left desynced from the cached dequant alpha), not a kernel bug. With a
    self-consistent calibration the first-step matches the fp32 conv to the int4
    quantization floor, so we now gate on numerics like the int8 twin."""
    from integration.kernels.int4_optimized import OptimizedInt4Conv2d
    torch.manual_seed(0)
    conv = nn.Conv2d(256, 256, 3, padding=1).to(DEV)
    opt = _calib_conv(OptimizedInt4Conv2d(conv).to(DEV),
                      torch.randn(16, 256, 32, 32, device=DEV)).to(memory_format=torch.channels_last)
    opt.enable_modiff(True)
    lifecycle_ok, acc, detail = _modiff_check(opt, conv, 256, 32, 16)
    return "int4_modiff_conv", lifecycle_ok and acc < 0.40, detail


class _OneConv(nn.Module):
    def __init__(self, conv):
        super().__init__()
        self.c = conv

    def forward(self, x):
        return self.c(x)


def _export_apply_check(kind):
    """Round-trip export→apply of a static calibration onto a FRESH converted model and
    compare the MoDiff first-step accuracy against live calibration. Guards that the
    SmoothQuant state (smooth_scale + smoothed weights) survives the checkpoint: before
    the fix, export saved only the per-tensor scale, so applying a SmoothQuant-derived
    scale onto unsmoothed weights degraded int4 ~2x (rel ~0.40 vs ~0.20); int8 masked it.
    Also asserts the legacy float-only path still loads (backward compat) and reproduces
    the old degradation, so the gate provably bites."""
    if kind == "int4":
        from integration.kernels.int4_optimized import (
            convert_model_to_optimized_int4 as convert, set_calibrating_int4 as set_calib,
            export_int4_static_scales as export, apply_int4_static_scales as apply,
            enable_modiff_mode as enable)
    else:
        from integration.kernels.int8_optimized import (
            convert_model_to_optimized_int8 as convert, set_calibrating as set_calib,
            export_int8_static_scales as export, apply_static_scales as apply,
            enable_modiff_mode as enable)
    torch.manual_seed(0)
    ref_conv = nn.Conv2d(256, 256, 3, padding=1).to(DEV)
    x = torch.randn(16, 256, 32, 32, device=DEV, dtype=torch.float16).contiguous(memory_format=torch.channels_last)
    ref = ref_conv(x.float())

    def fresh():
        m = _OneConv(nn.Conv2d(256, 256, 3, padding=1)).to(DEV)
        m.c.load_state_dict(ref_conv.state_dict())
        convert(m)
        return m

    live = fresh()
    set_calib(live, True); _ = live(torch.randn(16, 256, 32, 32, device=DEV)); set_calib(live, False)
    enable(live, True)
    live_acc = rel_err(live(x), ref)

    scales = export(live)
    has_smooth = any(isinstance(v, dict) and "smooth_scale" in v for v in scales.values())
    # persist through torch.save/load so the embedded smooth_scale tensor is exercised
    import io
    buf = io.BytesIO(); torch.save(scales, buf); buf.seek(0)
    scales = torch.load(buf, weights_only=True)

    applied = fresh()
    n = apply(applied, scales)
    enable(applied, True)
    apply_acc = rel_err(applied(x), ref)

    legacy = {k: (v["static_scale"] if isinstance(v, dict) else v) for k, v in scales.items()}
    old = fresh(); apply(old, legacy); enable(old, True)
    legacy_acc = rel_err(old(x), ref)

    # apply must match live (SmoothQuant restored) and clearly beat the legacy path.
    ok = (n == 1 and has_smooth and apply_acc <= live_acc + 0.02
          and apply_acc < 0.40 and apply_acc < legacy_acc - 0.05)
    return f"{kind}_export_apply", ok, (
        f"live={live_acc:.3f} apply={apply_acc:.3f} legacy(float-only)={legacy_acc:.3f} "
        f"smooth_serialized={has_smooth}")


def test_int4_export_apply():
    return _export_apply_check("int4")


def test_int8_export_apply():
    return _export_apply_check("int8")


TESTS = [test_int8_conv, test_int8_conv_channels_last, test_int8_dual_store,
         test_int4_conv, test_int4_dual_store, test_int8_modiff_conv, test_int4_modiff_conv,
         test_int4_export_apply, test_int8_export_apply,
         test_int8_linear, test_group_norm_silu]


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

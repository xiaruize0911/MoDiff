"""Direct kernel tests for the code_ceiling parameter: bit-identical by default, saturating when set.

Three things have to hold, and the third is the whole point:

  1. The 4-argument overload and `code_ceiling=-1` produce byte-identical codes and byte-identical
     a_hat updates. Every un-migrated caller (integration/, analysis_*/ and ~8 archived
     docs/*/scripts) goes through the short form, so this is what keeps them unchanged.
  2. `code_ceiling=127` with a scale built as 127/absmax is also byte-identical: no code can exceed
     the ceiling, so clamping at it is a no-op. This is the A8 shipped configuration, which is why
     the A8 measurements in this directory remain valid after the change.
  3. With `scale = Q_b/(r*absmax)` and r < 1, `code_ceiling=Q_b` SATURATES at Q_b while the old
     literal lets codes run to 127. At A4/r=0.25 that is the difference between a 4-bit quantizer
     with a clip and a ~6-bit grid, which is the defect this parameter fixes.

Run against both entry points MoDiff's W8 conv path uses: step1_static_quantize_fprop (the plain and
fused-SiLU modulated paths, 8 layers) and group_norm_silu_delta_quantize_nhwc (the GN-fused path, 62).
"""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import torch                                                                    # noqa: E402
import modiff_cutlass                                                           # noqa: E402

N, C, H, W = 2, 64, 16, 16
G = 32
fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  ' + detail) if detail else ''}", flush=True)
    if not cond:
        fails.append(name)


def cl(t):
    return t.contiguous(memory_format=torch.channels_last)


def step1(x, cache, scale, ceiling=None):
    """One step1_static_quantize_fprop call on a fresh cache copy; returns (codes, updated cache)."""
    c = cache.clone()
    empty = torch.empty(0, device=x.device, dtype=torch.float32)
    args = (x, c, scale, empty)
    q = (modiff_cutlass.step1_static_quantize_fprop(*args) if ceiling is None
         else modiff_cutlass.step1_static_quantize_fprop(*args, ceiling))
    return q, c


def gn(x, cache, scale, ceiling=None):
    """Same for the GN-fused entry point, static-scale mode (empty reduction buffers)."""
    c = cache.clone()
    e = torch.empty(0, device=x.device, dtype=torch.float32)
    ei = torch.empty(0, device=x.device, dtype=torch.int32)
    w = torch.ones(C, device=x.device, dtype=x.dtype)
    b = torch.zeros(C, device=x.device, dtype=x.dtype)
    tail = (e, e, e, ei, 127.0, False, 1.0)
    args = (x, w, b, c, G, 1e-5, True, scale, e, e, e) + tail
    q = (modiff_cutlass.group_norm_silu_delta_quantize_nhwc(*args) if ceiling is None
         else modiff_cutlass.group_norm_silu_delta_quantize_nhwc(*args, ceiling))
    return q, c


def main():
    torch.manual_seed(20260806)
    dev = "cuda"
    x = cl(torch.randn(N, C, H, W, device=dev, dtype=torch.float16) * 2.0)
    cache = cl(torch.randn(N, C, H, W, device=dev, dtype=torch.float16) * 0.5)

    for label, fn in (("step1_static_quantize_fprop", step1),
                      ("group_norm_silu_delta_quantize_nhwc", gn)):
        print(f"\n=== {label} ===", flush=True)
        # The delta each kernel forms differs (gn applies GN+SiLU first), so measure its own absmax
        # by running once at A8 and reading back the code range.
        probe_scale = torch.tensor([1.0], device=dev, dtype=torch.float32)
        q_probe, _ = fn(x, cache, probe_scale)
        delta_absmax = float(q_probe.float().abs().max())     # codes at scale 1.0 == |delta| rounded
        assert delta_absmax > 0, "degenerate probe: delta is zero"

        # (1) short overload vs explicit -1
        s8 = torch.tensor([127.0 / delta_absmax], device=dev, dtype=torch.float32)
        q_default, c_default = fn(x, cache, s8)
        q_neg1, c_neg1 = fn(x, cache, s8, -1.0)
        check("4-arg overload == code_ceiling=-1",
              torch.equal(q_default, q_neg1) and torch.equal(c_default, c_neg1))

        # (2) ceiling=127 at the A8 scale is a no-op
        q_127, c_127 = fn(x, cache, s8, 127.0)
        check("code_ceiling=127 == legacy literal at scale 127/absmax",
              torch.equal(q_default, q_127) and torch.equal(c_default, c_127),
              f"max|code| {int(q_default.float().abs().max())}")

        # (3) the clip case: A4 (Q=7), r=0.25 -> scale = 7/(0.25*absmax) = 28/absmax
        r, Q = 0.25, 7.0
        s4 = torch.tensor([Q / (r * delta_absmax)], device=dev, dtype=torch.float32)
        q_old, _ = fn(x, cache, s4, -1.0)
        q_new, c_new = fn(x, cache, s4, Q)
        old_max = int(q_old.float().abs().max())
        new_max = int(q_new.float().abs().max())
        check("legacy literal does NOT saturate at Q_b (the defect)", old_max > Q,
              f"max|code| {old_max} > {int(Q)}")
        check("code_ceiling=Q_b saturates at Q_b", new_max == int(Q), f"max|code| {new_max}")
        clipped = int((q_new.float().abs() == Q).sum())
        check("saturation is real, not vacuous", clipped > 0,
              f"{clipped} of {q_new.numel()} codes at the ceiling "
              f"({100.0 * clipped / q_new.numel():.1f}%)")
        # a_hat must move by q/scale with the CLAMPED q, i.e. the cache honours the ceiling too
        expect = (cache.float() + q_new.float() / float(s4)).half()
        check("a_hat update uses the clamped code",
              torch.allclose(c_new.float(), expect.float(), atol=1e-2),
              f"max diff {float((c_new.float() - expect.float()).abs().max()):.5f}")

    print(f"\n{'FAILED: ' + ', '.join(fails) if fails else 'all checks passed'}", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

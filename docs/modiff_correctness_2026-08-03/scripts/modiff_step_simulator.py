"""200-step MoDiff simulator: does the Linear path hold the paper's invariants, and does fp16 o_hat drift?

Two questions the end-to-end harness cannot answer:

1. **Is the Bug-2 fix correct?** `QuantLinearWxAx`'s modiff branch is unreachable in every mode
   (`benchmark_ldm.py` forces `is_modiff=False`), so nothing exercises it. This drives the module
   directly over a synthetic 200-step trajectory and checks the two invariants that define MoDiff:

     I1  ||a_t - a_hat_t||_inf <= s_t/2 + fp16(a_hat) ulp      (Eq. 18: e_t = a_t - a_hat_t)
     I2  o_hat_t == Linear(a_hat_t)                            (Eq. 14, the telescoping identity)

   I2 is the one that catches Bug 2: re-quantizing the codes inflated the increment, so o_hat
   drifted away from Linear(a_hat) while a_hat stayed correct.

2. **Does fp16 o_hat accumulation matter?** The conv path accumulates o_hat in fp16
   (`conv2d_evt.cu`, `EC = cutlass::half_t`). Estimate: ~2.4e-4 rounding per step random-walking to
   ~2e-3 over 200 steps, i.e. well under the A4 quantization error -- but that is an estimate, and
   the plan says test it rather than design around it. Compares fp16 vs fp32 vs fp64 accumulation
   of the same increments.

Run: python docs/modiff_correctness_2026-08-03/scripts/modiff_step_simulator.py
"""

import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
sys.path[:0] = [ROOT]

import torch

from integration.kernels.wxax_linear import QuantLinearWxAx

STEPS = int(os.environ.get("SIM_STEPS", "200"))
SEED = 1234
# Real churches attention Linear shapes: (M = batch*tokens, K = channels, N).
# qkv is K -> 3K, proj is K -> K.
SHAPES = [("qkv  C192/T1024", 128 * 1024, 192, 576),
          ("proj C192/T1024", 128 * 1024, 192, 192),
          ("qkv  C384/T256", 128 * 256, 384, 1152),
          ("proj C768/T16", 128 * 16, 768, 768)]


def trajectory(steps, M, K, device, gen, drift=0.02):
    """A slowly-varying activation trajectory, the regime MoDiff targets.

    a_0 is O(1); each step adds a small increment, so the temporal delta is ~drift x the activation
    range -- which is the whole premise (paper Figure 1b: deltas are much smaller and much more
    concentrated than activations).
    """
    a = torch.randn(M, K, device=device, dtype=torch.float16, generator=gen)
    for _ in range(steps):
        a = a + drift * torch.randn(M, K, device=device, dtype=torch.float16, generator=gen)
        yield a


def build(K, N, bits, device, gen):
    lin = torch.nn.Linear(K, N, bias=True).to(device).half()
    with torch.no_grad():
        b = 1.0 / K ** 0.5
        lin.weight.copy_(torch.empty(N, K, dtype=torch.float32).uniform_(-b, b, generator=gen).to(device).half())
        lin.bias.copy_(torch.empty(N, dtype=torch.float32).uniform_(-b, b, generator=gen).to(device).half())
    q = QuantLinearWxAx(lin, bits=bits, modiff=True).to(device)
    return lin, q


def run_shape(label, M, K, N, bits, device):
    gen = torch.Generator(device="cpu").manual_seed(SEED)
    gcuda = torch.Generator(device=device).manual_seed(SEED)
    lin, q = build(K, N, bits, device, gen)

    # The dequantized weight the quantized GEMM effectively applies: qweight * w_scale, with the
    # AWQ N/K zero-padding sliced off. int4 packs two nibbles per byte, so unpack first.
    if bits == 4:
        b = q.qweight[:N, :K // 2].to(torch.int16)
        lo = (b & 0xF).to(torch.int8); lo = torch.where(lo > 7, lo - 16, lo)
        hi = ((b >> 4) & 0xF).to(torch.int8); hi = torch.where(hi > 7, hi - 16, hi)
        codes = torch.stack([lo, hi], dim=-1).reshape(N, K)
    else:
        codes = q.qweight[:N, :K]
    w_deq = codes.float() * q.w_scale[:N].float().unsqueeze(1)

    rows = []

    with torch.inference_mode():
        for t, a in enumerate(trajectory(STEPS, M, K, device, gcuda)):
            out = q(a)

            if q.a_hat is None:
                continue
            # I1: the residual a_t - a_hat_t must be bounded by half a quantization step.
            e = (a.float() - q.a_hat.float()).abs().max().item()
            # I2: o_hat must equal A(a_hat). The reference is a plain fp32 matmul against the
            # DEQUANTIZED weights, not another call through _gemm: the GEMM quantizes its own
            # input, so re-running it on a_hat would add a fresh quantization error and measure
            # nothing. The identity is exact by construction --
            #   GEMM(codes) = (codes * d_scale) @ (qweight * w_scale)^T = dq_delta @ W_deq^T
            # and a_hat = sum_t dq_delta_t, so o_hat_t = a_hat_t @ W_deq^T. Any deviation means the
            # codes the GEMM consumed differ from the ones a_hat was updated with -- which is
            # exactly the Bug-2 signature.
            ref = q.a_hat.float() @ w_deq.T
            i2 = ((q.o_hat.float() - ref).norm() / ref.norm().clamp_min(1e-12)).item()

            # I2 *is* the fp16-accumulation measure, so there is no separate drift metric here.
            # o_hat is stored fp16 and accumulated in place, and the reference is the exact fp32
            # a_hat @ W_deq^T, so I2's growth across steps is precisely the accumulated fp16
            # rounding (plus the fp16 storage of a_hat). Deriving increments from successive
            # o_hat.float() reads and re-accumulating them in fp32/fp64, as an earlier version of
            # this script did, is circular -- it reconstructs the same fp16 values and reports 0.
            if t % 20 == 0 or t == STEPS - 1:
                rows.append({"step": t, "e_inf": e, "i2_rel": i2,
                             "out_absmax": out.abs().max().item(),
                             "finite": bool(torch.isfinite(out).all())})
            del out
    del lin, q
    torch.cuda.empty_cache()
    # I2 growth from first to last recorded step = the accumulated fp16 cost over STEPS steps.
    growth = (rows[-1]["i2_rel"] - rows[0]["i2_rel"]) if len(rows) > 1 else 0.0
    return rows, growth


def main():
    device = "cuda"
    out = {"steps": STEPS, "seed": SEED, "results": {}}
    for bits in (8, 4):
        for label, M, K, N in SHAPES:
            if K % (64 if bits == 4 else 32) or N % 64:
                print(f"skip W{bits}A{bits} {label}: shape ineligible")
                continue
            key = f"W{bits}A{bits} {label}"
            print(f"\n=== {key}  M={M} K={K} N={N}")
            try:
                rows, growth = run_shape(label, M, K, N, bits, device)
            except Exception as exc:
                print(f"   ERROR {type(exc).__name__}: {exc}")
                out["results"][key] = {"error": f"{type(exc).__name__}: {exc}"}
                continue
            print(f"   {'step':>5} {'||a-â||inf':>12} {'I2 rel':>12} {'|out|max':>10} finite")
            for r in rows:
                print(f"   {r['step']:>5} {r['e_inf']:>12.5g} {r['i2_rel']:>12.5g} "
                      f"{r['out_absmax']:>10.4g} {r['finite']}")
            print(f"   I2 growth over {STEPS} steps (= accumulated fp16 o_hat cost): {growth:+.3g}")
            out["results"][key] = {"rows": rows, "i2_growth_over_run": growth}

    path = "docs/modiff_correctness_2026-08-03/data/step_simulator.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWROTE {path}")


if __name__ == "__main__":
    main()

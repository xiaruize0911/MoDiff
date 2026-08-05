"""Which forward() branch actually returns, per route flag setting?

check_output_degeneracy.py showed the four GN->QKV routes are bit-identical on non-degenerate
numerics while firing visibly different kernels. Either the returned value does not come from
the branch whose kernels we timed, or the routes really are numerically equivalent. This counts
the entry points each route hits.
"""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src/taming-transformers"))
sys.path.insert(0, os.path.join(ROOT, "integration/benchmarks/report"))

import torch

import layer_pipeline_bench as layer_bench
import modiff_cutlass as mc

WATCH = ["fused_gn_qkv", "fused_gn_qkv_i8evt", "gemm_w8a8_awq_qkv_i8_layouts",
         "quantize_attn_kv_from_i8", "quantize_attn_qkv_from_i8",
         "group_norm_silu_quantize_nhwc_fast", "group_norm_silu_quantize_nhwc",
         "gemm_w8a8_awq_out_i8_bias_nout", "gemm_w8a8_awq_bias_res"]
COUNTS = {}


def install_counters():
    for name in WATCH:
        fn = getattr(mc, name, None)
        if fn is None:
            continue

        def make(name, fn):
            def wrapper(*a, **k):
                COUNTS[name] = COUNTS.get(name, 0) + 1
                return fn(*a, **k)
            return wrapper
        setattr(mc, name, make(name, fn))


def main():
    model, sampler, layers = layer_bench.collect_layers("int8")
    del sampler
    row = next(r for r in layers
               if r["kind"] == "attention" and tuple(r["x_shape"]) == (128, 192, 32, 32))
    module = row["module"]
    x = torch.randn(*row["x_shape"], device="cuda", dtype=torch.float16).contiguous(
        memory_format=torch.channels_last)

    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        for _ in range(16):
            module(x)
        torch.cuda.synchronize()

        # Counters go in only after calibration has frozen, so we see steady-state dispatch.
        install_counters()
        # quantized_std_attention captured _mc at import time; repoint that binding too.
        import integration.fused_ops.quantized_std_attention as qsa
        import integration.fused_ops.token_major_attention as tma
        qsa._mc = mc
        tma._mc = mc

        outs = {}
        for route in ("P", "R1", "A", "N"):
            module._int8_qkv_epilogue = (route == "P")
            module._route1 = (route == "R1")
            module._fuse_gn_qkv_i8 = (route == "A")
            COUNTS.clear()
            outs[route] = module(x).float().clone()
            torch.cuda.synchronize()
            hit = ", ".join(f"{k}x{v}" for k, v in sorted(COUNTS.items()))
            print(f"{route:2s} -> {hit or '(no watched entry point)'}")

        base = outs["P"]
        for route in ("R1", "A", "N"):
            d = outs[route] - base
            print(f"   {route} vs P: differing elements {(d != 0).sum().item()} "
                  f"max|d| {d.abs().max().item():.6g}")


if __name__ == "__main__":
    main()

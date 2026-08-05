"""Is the four-route bit-identical agreement real, or an artifact of the stub checkpoint?

gn_qkv_route_ab.py found max|d|=0 across all four GN->QKV routes. Routes R1/A feed the qkv
projection FP16 activations where P feeds INT8, so exact agreement should be impossible. This
checks the obvious suspect: models/ldm/lsun_churches256/model.ckpt is an 856-byte stub with an
empty state_dict, so every weight is randomly initialised and the INT8 codes may simply be
saturating at +-127, which would make every route agree for the wrong reason.
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


def main():
    model, sampler, layers = layer_bench.collect_layers("int8")
    del sampler
    row = next(r for r in layers
               if r["kind"] == "attention" and tuple(r["x_shape"]) == (128, 192, 32, 32))
    module = row["module"]
    x = torch.randn(*row["x_shape"], device="cuda", dtype=torch.float16).contiguous(
        memory_format=torch.channels_last)

    import modiff_cutlass as mc

    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        for _ in range(16):
            module(x)
        torch.cuda.synchronize()

        # The int8 activation tensor the qkv GEMM consumes on the production route.
        gw, gb = module._gn_params(x.dtype)
        inv = torch.tensor([1.0 / module.qkv.a_scale], device=x.device, dtype=torch.float32)
        empty = x.new_empty(0)
        gnq = getattr(mc, "group_norm_silu_quantize_nhwc_fast",
                      mc.group_norm_silu_quantize_nhwc)
        xq = gnq(x, gw, gb, module.norm.num_groups, module.norm.eps, False,
                 inv, empty, empty, empty)
        xq_i = xq.to(torch.int16)
        n = xq_i.numel()
        sat = ((xq_i.abs() >= 127).sum().item())
        print(f"qkv-input int8 codes: n={n}  |code|>=127: {sat} ({100.0*sat/n:.2f}%)  "
              f"mean|code| {xq_i.abs().float().mean().item():.2f}  "
              f"distinct {xq_i.unique().numel()}")

        out = module(x)
        print(f"layer output: dtype {out.dtype}  finite {torch.isfinite(out).all().item()}  "
              f"absmax {out.abs().max().item():.4g}  mean|out| {out.abs().mean().float().item():.4g}  "
              f"distinct {out.unique().numel()}")

        # Re-compare the routes with the folded weights rebuilt from scratch, so a stale
        # _r1_w / _fused_conv_w cannot be what makes them agree.
        def run(route):
            module._int8_qkv_epilogue = (route == "P")
            module._route1 = (route == "R1")
            module._fuse_gn_qkv_i8 = (route == "A")
            module._r1_ready = False
            module._fused_ready = False
            return module(x).float().clone()

        base = run("P")
        for route in ("R1", "A"):
            got = run(route)
            d = (got - base)
            print(f"{route} vs P: max|d| {d.abs().max().item():.6g}  "
                  f"relL2 {(d.norm()/base.norm()).item():.6g}  "
                  f"elements differing {(d != 0).sum().item()}")


if __name__ == "__main__":
    main()

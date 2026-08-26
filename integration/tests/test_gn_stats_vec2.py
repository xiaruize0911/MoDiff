"""Gate for gn_stats_partials_chanmajor_vec2_kernel: BIT-IDENTICAL statistics, or it does not ship.

The vec2 stats kernel is a pure latency fix -- a thread owns two adjacent channels and loads them
as one __half2, and the hw loop is unrolled by four with the loads hoisted. Neither changes a value
or the order it is added in, so the partials must come out bit-identical to the scalar kernel it
replaces. This asserts that on all 18 conv shapes this UNet actually runs, rather than assuming it.

Why bit-identity rather than a tolerance: mean/inv_std feed the MoDiff delta quantizer, whose a_hat
cache accumulates `a_hat += q/scale` across 200 steps. A one-ULP difference in inv_std moves q on
some element, and from there the two arms are different functions, not one function measured twice.
Every committed FID number was measured on the scalar kernel's statistics.

The comparison is driven through the PUBLIC entry point (group_norm_silu_delta_quantize_nhwc) with
MODIFF_GN_STATS_VEC2 forced on and off. Both flags are read once per process via a function-local
static, so each arm runs in its OWN forked process -- the same discipline
docs/modiff_correctness_2026-08-03/scripts/gn_stats_ab.py uses, and the reason an in-process A/B of
these variants silently measures one variant twice.

Run: python integration/tests/test_gn_stats_vec2.py
"""
import os
import subprocess
import sys
import tempfile

REPO = "/workspace/MoDiff"
BUILD = os.path.join(REPO, "build/lib.linux-x86_64-cpython-311")

# (C, H, W) for every shape in the 20-shape conv-block enumeration, deduplicated on the GN's own
# tensor. N=128, G=32, as everywhere else in this project.
SHAPES = [(768, 2, 2), (384, 8, 8), (384, 16, 16), (768, 4, 4), (192, 32, 32), (1536, 2, 2),
          (768, 8, 8), (1536, 4, 4), (384, 32, 32), (768, 16, 16), (384, 4, 4), (1152, 8, 8),
          (1152, 4, 4), (192, 16, 16), (576, 32, 32), (576, 16, 16)]

WORKER = r'''
import os, sys, torch
sys.path.insert(0, %r)
import modiff_cutlass as mc
torch.manual_seed(1234)          # same seed in both arms => same input bits
N, G = 128, 32
C, H, W = %d, %d, %d
x = torch.randn(N, C, H, W, device="cuda", dtype=torch.float16).to(memory_format=torch.channels_last)
a = (0.1 * torch.randn(N, C, H, W, device="cuda", dtype=torch.float16)).to(memory_format=torch.channels_last)
a0 = a.clone()
g = torch.ones(C, device="cuda", dtype=torch.float16)
b = torch.zeros(C, device="cuda", dtype=torch.float16)
sc = torch.tensor([64.0], device="cuda", dtype=torch.float32)
e16 = torch.empty(0, device="cuda", dtype=torch.float16)
e32 = torch.empty(0, device="cuda", dtype=torch.float32)
ei = torch.empty(0, device="cuda", dtype=torch.int32)
call = lambda: mc.group_norm_silu_delta_quantize_nhwc(x, g, b, a, G, 1e-5, True, sc, e32, e16, e16,
                                                     e32, e32, e32, ei, 127.0, False, 1.0)
# NON-VACUITY ON THE FLAG ITSELF, not just on the output: equality would pass trivially if
# MODIFF_GN_STATS_VEC2 were never read and both arms ran the same kernel. Record which stats
# kernel the profiler actually saw. (OPEN_ITEMS A22 is what this guards against.)
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    call()
    torch.cuda.synchronize()
kern = sorted({e.name for e in prof.events() if "gn_stats_partials" in e.name})
a.copy_(a0)
yq = call()
torch.save({"yq": yq.cpu(), "a_hat": a.cpu(), "kernels": kern}, sys.argv[1])
'''


def run(shape, vec2, out):
    env = dict(os.environ, MODIFF_GN_STATS_VEC2=("1" if vec2 else "0"))
    src = WORKER % (BUILD, *shape)
    r = subprocess.run([sys.executable, "-c", src, out], env=env, cwd=REPO,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"worker failed for {shape} vec2={vec2}:\n{r.stdout}\n{r.stderr}")


def main():
    import torch
    bad = []
    with tempfile.TemporaryDirectory() as td:
        for shape in SHAPES:
            fa, fb = os.path.join(td, "on.pt"), os.path.join(td, "off.pt")
            run(shape, True, fa)
            run(shape, False, fb)
            A, B = torch.load(fa), torch.load(fb)
            ok_q = torch.equal(A["yq"], B["yq"])
            ok_c = torch.equal(A["a_hat"], B["a_hat"])
            nz = int((A["yq"] != 0).sum())                      # output non-vacuity
            on_is_vec2 = any("vec2" in k for k in A["kernels"])  # flag non-vacuity
            off_is_scalar = A["kernels"] != B["kernels"] and not any("vec2" in k for k in B["kernels"])
            ok = ok_q and ok_c and nz > 0 and on_is_vec2 and off_is_scalar
            if not ok:
                bad.append((shape, ok_q, ok_c, nz, A["kernels"], B["kernels"]))
            print(f"  C={shape[0]:5d} {shape[1]:2d}x{shape[2]:<2d}  codes {'==' if ok_q else '!='}  "
                  f"a_hat {'==' if ok_c else '!='}  nonzero {nz:>10d}  "
                  f"kernels differ {'YES' if A['kernels'] != B['kernels'] else 'NO '}   "
                  f"{'OK' if ok else 'FAIL'}")
    if bad:
        print(f"\nFAILED on {len(bad)} shapes: {bad}")
        sys.exit(1)
    print(f"\nPASS -- on all {len(SHAPES)} shapes the two arms ran DIFFERENT stats kernels and "
          f"produced bit-identical codes and a_hat, through the public entry point.")


if __name__ == "__main__":
    main()

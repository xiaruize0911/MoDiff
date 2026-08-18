"""Does the SHIPPED calibration file carry ACT_CLIP_RATIO, or does checkpoint reuse bypass it?

ACT_CLIP_RATIO is applied once, in end_calibration (int4_optimized.py:1761), and baked into whatever
export_int4_static_scales writes. The load path (set_static_scale:1892) only fills the value. So a file
exported BEFORE the constant landed on 2026-08-12 carries no 4.5 -- and int4_calibration_realckpt.pt is
dated 2026-08-04.

Test: calibrate twice on the same protocol, once at ratio 4.5 and once at 1.0, and see which one the
shipped file matches. Same attention setting as the shipped arm (quantized) so the ONLY variable is the
constant.
"""
import os, sys, statistics
ROOT="/workspace/MoDiff"; os.chdir(ROOT)
sys.path[:0]=[ROOT, os.path.join(ROOT,"src/taming-transformers"), os.path.join(ROOT,"integration/benchmarks/report")]
RATIO = sys.argv[1]; OUT = sys.argv[2]
os.environ["MODIFF_ACT_CLIP_RATIO"] = RATIO
import torch
import kernel_suites_bench as ks
import integration.benchmarks.benchmark_ldm as B
ks.set_env("int4")
os.environ["MODIFF_DELTA_MODE"]="static"; os.environ["MODIFF_DELTA_REFRESH"]="4"
os.environ["MODIFF_LINEAR"]="0"; os.environ["MODIFF_ACT_BITS"]="8"; os.environ["MODIFF_WARMUP_STEPS"]="5"
if os.path.exists(OUT): os.remove(OUT)
r = B.BenchmarkRunner(config_path="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
    ckpt_path="models/ldm/lsun_churches256/model.ckpt", output_dir="/workspace/fid_det/tmp",
    batch_size=16, steps=50, shape=(4,32,32), calibration_path=OUT)
m, s = r._setup_model("int4"); r._calibrate_int4(m, s)
print(f"WROTE {OUT} at ACT_CLIP_RATIO={RATIO}")

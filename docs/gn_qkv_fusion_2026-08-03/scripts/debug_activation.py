"""Why did activating proj_out not make attention observable?

Two hypotheses:
  H1 the activation does not change the block's behaviour (weights written to the wrong place,
     or a derived/cached weight is what the kernel actually reads)
  H2 the activation works but the monkeypatch used to corrupt the block does not take effect,
     so the latent test was measuring nothing

Tests each directly on one block, at its real production shape.
"""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers")]

import torch

import integration.benchmarks.benchmark_ldm as B
from integration.utils.attention_identity_guard import (
    activate_zero_initialised_projections, attention_blocks)

CONFIG = "configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml"
CKPT = "models/ldm/lsun_churches256/model.ckpt"


def main():
    for mode in ("fp16", "int8_baseline"):
        print("=" * 72)
        print(mode)
        runner = B.BenchmarkRunner(
            config_path=CONFIG, ckpt_path=CKPT, output_dir="/tmp/dbg_out",
            batch_size=4, steps=20,
            calibration_path="integration/calibration/int8_calibration.pt")
        model, sampler = runner._setup_model(mode)

        name, block = attention_blocks(model)[0]
        proj = getattr(block, "proj", None) or getattr(block, "proj_out", None)
        print(f"  block {name}  type {type(block).__name__}  proj {type(proj).__name__}")
        ch = block.channels
        x = torch.randn(4, ch, 32, 32, device="cuda", dtype=torch.float16).contiguous(
            memory_format=torch.channels_last)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=True):
            for _ in range(12):
                y0 = block(x)
            print(f"  BEFORE activation: identity={torch.equal(y0, x)}")

            n = activate_zero_initialised_projections(model, verbose=False)
            for attr in ("qweight", "weight"):
                t = getattr(proj, attr, None)
                if t is not None:
                    print(f"    proj.{attr}: absmax {t.abs().max().item()}  "
                          f"nonzero {int((t != 0).sum())}/{t.numel()}")
            ws = getattr(proj, "w_scale", None)
            if ws is not None:
                print(f"    proj.w_scale[:{proj.out_features}] absmax "
                      f"{ws[:proj.out_features].abs().max().item():.6g}")

            for _ in range(12):
                y1 = block(x)
            print(f"  AFTER activation ({n} activated): identity={torch.equal(y1, x)}  "
                  f"max|y1-y0| {(y1.float()-y0.float()).abs().max().item():.6g}")

            # H2: does replacing .forward on the instance take effect?
            orig = block.forward
            block.forward = lambda z, _o=orig: torch.full_like(_o(z), 7.0)
            y2 = block(x)
            print(f"  monkeypatch effective: {bool((y2 == 7.0).all())}  "
                  f"(y2 absmax {y2.abs().max().item()})")
            block.forward = orig

        del runner, model, sampler
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

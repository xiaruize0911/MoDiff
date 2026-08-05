"""Demonstrate that no attention change can move this tree's e2e latent.

AttentionBlock.proj_out is a zero_module (ldm/modules/diffusionmodules/openaimodel.py:345) and
models/ldm/lsun_churches256/model.ckpt is an 856-byte stub with an empty state_dict loaded
strict=False, so proj_out stays all-zero and every AttentionBlock is an identity on its residual.
Every latent-level "attention quality gate" in the repo therefore passes unconditionally.

Three checks:
  1. how many of the 21 AttentionBlocks are bit-exact identities, per mode
  2. latents with MODIFF_FLASH_GATE=on vs off -- a real routing change
  3. latents with the attention output deliberately corrupted (replaced by NaN-free garbage)

If (3) leaves the latent bit-identical, no attention correctness gate built on latents can fail.
"""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers")]

import torch

import integration.benchmarks.benchmark_ldm as B

BATCH = 4
STEPS = 20
SEED = 1234
CONFIG = "configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml"
CKPT = "models/ldm/lsun_churches256/model.ckpt"
ATTN_CLASSES = ("AttentionBlock", "TokenMajorAttentionBlock",
                "QuantizedStandardAttentionBlock")


def build(mode):
    cal = ("integration/calibration/int4_calibration.pt" if "int4" in mode
           else "integration/calibration/int8_calibration.pt")
    runner = B.BenchmarkRunner(config_path=CONFIG, ckpt_path=CKPT,
                               output_dir="/tmp/vac_out", batch_size=BATCH,
                               steps=STEPS, calibration_path=cal)
    model, sampler = runner._setup_model(mode)
    return runner, model, sampler


def attention_blocks(model):
    return [(n, m) for n, m in model.named_modules()
            if type(m).__name__ in ATTN_CLASSES]


def sample(runner, model, sampler):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=True):
        out = sampler.sample(S=STEPS, batch_size=BATCH, shape=runner.shape,
                             eta=0.0, verbose=False)
    lat = out[0] if isinstance(out, (tuple, list)) else out
    return lat.detach().float().cpu()


def main():
    print("=" * 72)
    print("1. how many AttentionBlocks are bit-exact identities?")
    print("=" * 72)
    for mode in ("fp16", "int8_baseline", "int4_baseline"):
        runner, model, sampler = build(mode)
        blocks = attention_blocks(model)
        identity, checked = 0, 0
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=True):
            for name, mod in blocks:
                ch = getattr(mod, "channels", None)
                if ch is None:
                    continue
                # 8x8 keeps this cheap; identity-ness does not depend on T.
                x = torch.randn(2, ch, 8, 8, device="cuda", dtype=torch.float16
                                ).contiguous(memory_format=torch.channels_last)
                try:
                    for _ in range(12):      # let any static calibration freeze
                        y = mod(x)
                except Exception as exc:     # shape-specialized routes may refuse 8x8
                    print(f"    skip {name}: {type(exc).__name__}")
                    continue
                checked += 1
                identity += int(torch.equal(y, x))
        print(f"  {mode:14s} {identity}/{checked} attention blocks are exact identities "
              f"(of {len(blocks)} found)")
        del runner, model, sampler
        torch.cuda.empty_cache()

    print()
    print("=" * 72)
    print("2. does a real attention routing change move the latent?")
    print("=" * 72)
    lat = {}
    for gate in ("on", "off"):
        os.environ["MODIFF_FLASH_GATE"] = gate
        runner, model, sampler = build("int8_baseline")
        lat[gate] = sample(runner, model, sampler)
        del runner, model, sampler
        torch.cuda.empty_cache()
    os.environ.pop("MODIFF_FLASH_GATE", None)
    d = lat["on"] - lat["off"]
    print(f"  MODIFF_FLASH_GATE=on vs off: differing elements {(d != 0).sum().item()} "
          f"of {d.numel()}  relL2 {(d.norm()/lat['off'].norm()).item():.6g}")

    print()
    print("=" * 72)
    print("3. does DELIBERATELY CORRUPTING the attention output move the latent?")
    print("=" * 72)
    runner, model, sampler = build("int8_baseline")
    clean = sample(runner, model, sampler)
    blocks = attention_blocks(model)
    for _, mod in blocks:
        orig = mod.forward

        def corrupt(x, _orig=orig):
            y = _orig(x)
            # Garbage of the right shape/dtype: nothing downstream should survive this.
            return torch.full_like(y, 3.0) + torch.randn_like(y)
        mod.forward = corrupt
    dirty = sample(runner, model, sampler)
    d = dirty - clean
    print(f"  {len(blocks)} attention blocks corrupted: differing elements "
          f"{(d != 0).sum().item()} of {d.numel()}  "
          f"relL2 {(d.norm()/clean.norm()).item():.6g}")
    if (d != 0).sum().item() == 0:
        print("  => the latent is INDEPENDENT of every attention block. Any correctness")
        print("     gate built on layer outputs or latents cannot fail, for any change.")


if __name__ == "__main__":
    main()

"""A1 gate: qdiff/ is restored, and its layer names map onto integration/'s calibration keys.

Two questions, both structural, both answerable without a GPU or the 2.7 GB checkpoint:

  1. Does `QuantModel` build against THIS repo's churches UNet? qdiff was deleted in c9ade7c and
     restored from c9ade7c^; `ldm/` has moved since, so its imports of BasicTransformerBlock,
     ResBlock, AttnBlock, QKMatMul and friends have to still resolve.

  2. Does the NAME MAP land exactly on the 70 keys integration/ expects? This is the whole bridge.
     qdiff wraps convs in place, so it reports raw LDM paths (`input_blocks.1.0.in_layers.2`), while
     integration reads them through FusedResBlock, which re-registers the same objects as `.in_conv`
     and `.out_conv` (fused_resblock.py:756,768). If the rename is not exactly 1:1 the exporter
     silently drops layers, and a dropped key is invisible at runtime -- apply_static_scales just
     leaves that layer at static_input_scale = 1.0.

Built on a meta device: no weights are materialised, so this is seconds and needs no checkpoint.

Run: python docs/qdiff_bridge_2026-08-12/scripts/smoke_qdiff.py
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src/taming-transformers")]

import torch                                                                # noqa: E402
import torch.nn as nn                                                       # noqa: E402
from omegaconf import OmegaConf                                             # noqa: E402
from ldm.util import instantiate_from_config                                # noqa: E402
from qdiff import QuantModel, QuantModule                                   # noqa: E402

CFG = "configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml"
SHIPPED = "integration/calibration/int8_calibration_realckpt.pt"
OUT = "docs/qdiff_bridge_2026-08-12/data/smoke_qdiff.json"

#: The exporter's name map, defined here so the gate and the exporter cannot drift.
#: These are the ONLY two rules -- verified exhaustively against the shipped 70.
RENAME = ((".in_layers.2", ".in_conv"), (".out_layers.3", ".out_conv"))
#: One or two index levels, NOT two. input_blocks/output_blocks nest as
#: `<container>.<block-idx>.<child-idx>`, but middle_block is a single TimestepEmbedSequential whose
#: ResBlocks sit at `middle_block.0` and `middle_block.2` -- one level. Requiring `\d+\.\d+` silently
#: dropped those 4 keys, which at runtime is invisible: apply_static_scales just leaves the layer at
#: static_input_scale = 1.0. Caught by the set-equality check below, which is why it is a check and
#: not a comment.
KEEP = re.compile(r"^(input_blocks|middle_block|output_blocks)(\.\d+){1,2}\.(in|out)_conv$")


def map_name(raw):
    """Raw LDM module path -> integration calibration key, or None if integration does not use it."""
    n = raw
    for a, b in RENAME:
        n = n.replace(a, b)
    return n if KEEP.match(n) else None


def main():
    conf = OmegaConf.load(CFG)
    with torch.device("meta"):
        unet = instantiate_from_config(conf.model.params.unet_config)

    wq = {"n_bits": 8, "channel_wise": True, "scale_method": "max"}
    aq = {"n_bits": 8, "channel_wise": False, "scale_method": "max", "leaf_param": True}
    qnn = QuantModel(model=unet, weight_quant_params=wq, act_quant_params=aq, modulate=False)

    qmods = [(n, m) for n, m in qnn.named_modules() if isinstance(m, QuantModule)]
    # strip the leading "model." QuantModel adds, matching what lands in the state_dict keys
    raw = [n[len("model."):] if n.startswith("model.") else n for n, _ in qmods]
    mapped = sorted(x for x in (map_name(r) for r in raw) if x)

    shipped = sorted(torch.load(SHIPPED, map_location="cpu", weights_only=False))

    print(f"  QuantModules built        : {len(qmods)}")
    print(f"  mapped to integration keys: {len(mapped)}")
    print(f"  shipped calibration keys  : {len(shipped)}")

    only_map = sorted(set(mapped) - set(shipped))
    only_ship = sorted(set(shipped) - set(mapped))
    equal = not only_map and not only_ship
    print(f"  set-equal                 : {equal}")
    if only_map:
        print(f"    mapped but not shipped ({len(only_map)}): {only_map[:5]}")
    if only_ship:
        print(f"    shipped but not mapped ({len(only_ship)}): {only_ship[:5]}")

    ok_count = len(qmods) == 168
    print()
    print(f"  [{'PASS' if ok_count else 'FAIL'}] 168 QuantModules (131 conv + 37 linear)")
    print(f"  [{'PASS' if equal else 'FAIL'}] name map is exactly the shipped 70")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"n_quant_modules": len(qmods), "n_mapped": len(mapped),
               "n_shipped": len(shipped), "set_equal": equal,
               "only_mapped": only_map, "only_shipped": only_ship,
               "mapped": mapped}, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")
    return 0 if (ok_count and equal) else 1


if __name__ == "__main__":
    sys.exit(main())

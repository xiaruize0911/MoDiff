"""Install AdaRound's weights WITH their per-channel zero point -- fix #4's actual payload.

NOT install_adaround(). That one substitutes AdaRound's DEQUANTIZED weights into the fp16 state dict and
lets our symmetric quantizer re-round them, which DROPS z_w. That degraded arm is what FID 140.2 (B5) and
the 0.0602 conv-output error measured. This installs the asymmetric grid itself:

    weight_packed          = pack(x_q - 8)     AdaRound's unsigned 0..15 codes, shifted to the signed
                                               int4 kernel's range. The decomposition is
                                               x_q - z_w = (x_q - 8) + (8 - z_w).
    weight_scale_channel   = delta             AdaRound's per-channel step
    weight_zp              = 8 - z_w           the residual offset the epilogue correction applies

The convention is the one integration/tests/test_zpw_ohat.py gates against a float64 asymmetric-weight
reference, so it is copied from there rather than re-derived.

Returns how many convs were converted. Every check here exists because a partial or mis-keyed install
would still sample and would read as "fix #4 does not help".
"""
import os
import re

import torch

#: Default is the PAPER's checkpoint, which A25 measured as unusable here: it is the EMA network's, and
#: installing it reads FID 309.689 / relL2 2.55 against a 52.584 baseline. ADAROUND_CKPT points this at a
#: reconstruction done on OUR network instead -- the input the gated machinery actually needs.
CKPT = os.environ.get("ADAROUND_CKPT", "/workspace/quant_models/church_w4a8_ckpt.pth")


def _pack_nhwc(codes_signed: torch.Tensor) -> torch.Tensor:
    """[K,C,R,S] signed int4 -> [K,R,S,C/2] packed, two nibbles per byte, low nibble = even C."""
    c = codes_signed.permute(0, 2, 3, 1).contiguous().to(torch.int64) & 0x0F
    lo, hi = c[..., 0::2], c[..., 1::2]
    v = lo | (hi << 4)
    return (v - 256 * (v > 127)).to(torch.int8).contiguous()


def adaround_asymmetric(path: str = CKPT) -> dict:
    """{relative conv name -> (x_q int64 [K,C,R,S], delta [K], z_w [K])} from the qdiff checkpoint."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    out = {}
    for b in sorted({m.group(1) for k in ck if (m := re.match(r"(.+)\.weight_quantizer\.alpha$", k))}):
        W = ck[b + ".weight"]
        if W.dim() != 4:
            continue
        W, al = W.float(), ck[b + ".weight_quantizer.alpha"].float()
        d = ck[b + ".weight_quantizer.delta"].float()
        z = ck[b + ".weight_quantizer.zero_point"].float()
        x_q = torch.clamp(torch.floor(W / d) + (al >= 0).float() + z, 0, 15).to(torch.int64)
        out[b[len("model."):]] = (x_q, d.reshape(-1), z.reshape(-1))
    return out


def _norm(name: str) -> str:
    """Live module paths carry an extra `.original.` level the checkpoint does not.

    The ResBlock fusion wraps each block and keeps the unfused modules under `.original`, so a live conv
    is `input_blocks.1.0.original.in_layers.2` where the qdiff checkpoint has
    `input_blocks.1.0.in_layers.2`. B5's install_adaround never hit this because it patches the STATE
    DICT at load time, before the fusion runs; anything walking named_modules() afterwards does.
    Symptom without this: a clean-looking 0-intersection and "installed 0 convs".
    """
    return name.replace(".original.", ".")


def install(unet, table: dict, verbose: bool = False) -> int:
    from integration.kernels.int4_optimized import OptimizedInt4Conv2d
    n = 0
    zmin, zmax = 99.0, -99.0
    live = 0
    for name, m in unet.named_modules():
        if not isinstance(m, OptimizedInt4Conv2d):
            continue
        live += 1
        key = _norm(name)
        if key not in table:
            if verbose:
                print(f"  no adaround entry for {key}")
            continue
        x_q, d, z = table[key]
        K = m.weight_scale_channel.numel()
        if x_q.shape[0] != K or d.numel() != K or z.numel() != K:
            continue
        dev = m.weight_scale_channel.device
        packed = _pack_nhwc(x_q - 8).to(dev)
        if packed.shape != m.weight_packed.shape:
            if verbose:
                print(f"  skip {name}: packed {tuple(packed.shape)} != "
                      f"{tuple(m.weight_packed.shape)}")
            continue
        m.weight_packed = packed
        m.weight_scale_channel = d.view(1, K, 1, 1).to(dev)
        if hasattr(m, "weight_scale_channel_half"):
            m.weight_scale_channel_half = d.half().contiguous().to(dev)
        m.weight_zp = (8.0 - z).float().to(dev)
        zmin, zmax = min(zmin, float(z.min())), max(zmax, float(z.max()))
        n += 1
    print(f"  fix #4 install: {n}/{live} live int4 convs matched the AdaRound table of {len(table)}")
    # 70 is what the tree quantizes (the other 19 in the table are layers this pipeline keeps in fp16).
    # A partial install would still sample and would read as "fix #4 does not help".
    assert n >= 70, (f"only {n} of {live} live int4 convs got the asymmetric grid. A partial install "
                     f"leaves the rest on the symmetric grid while the corrected epilogue runs on all of "
                     f"them -- fix #2's exact failure mode.")
    if n:
        print(f"✓ fix #4: installed AdaRound ASYMMETRIC weights on {n} convs "
              f"(z_w spans {zmin:.0f}..{zmax:.0f})")
        # A z_w that is centred at 8 would make the correction a no-op and the arm vacuous.
        assert not (7.5 <= zmin and zmax <= 8.5), (
            f"z_w spans only {zmin}..{zmax}, i.e. it is centred at 8 -- weight_zp = 8 - z_w is then ~0 "
            f"and this arm measures nothing")
    return n

"""Does `_qkv_i8_ok` admit exactly the shapes `flash_attn_int8_packed_vt` accepts?

The kernel-side answer is asserted in test_flash_packed_int8_shapes.py. This is the PYTHON-side
half: the gate has to agree with it, because the int8 branch in _forward_routes raises rather than
falling back. A gate that is too permissive turns a shape restriction into a crash in production,
which is exactly what the pre-2026-08-12 version did with the hd=96 blocks.

No model, no GPU work beyond an import: the gate is a pure function of (bits, head_dim, T, frozen
scales, env), so it is checked on a stub with those attributes set. That also means this test still
runs when no checkpoint is present.

Run: python integration/tests/test_route_b_gate.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import torch                                                             # noqa: E402
from integration.fused_ops.quantized_std_attention import (              # noqa: E402
    QuantizedStandardAttentionBlock as Q)

#: (C, T, expect, why) at nh=8 -- the churches attention shapes, same four as the kernel-side test.
CASES = [
    (192, 1024, False, "hd=24: int8 cp.async needs hd%16==0"),
    (384, 256, True, "hd=48, T%64==0"),
    (384, 64, True, "hd=48, T%64==0"),
    (768, 16, False, "hd=96: hd_pad=128 > FA_MMA_MAXHD, T%64!=0"),
]


class Stub:
    """Only what _qkv_i8_ok reads. Deliberately not a real block: the gate must not depend on
    anything else, and constructing one would need a checkpoint."""

    def __init__(self, C, nh=8, bits=8, frozen=True):
        self.bits = bits
        self.head_dim = C // nh
        self._fq_frozen2 = frozen
        self._fq_svv = torch.zeros(self.head_dim) if frozen else None

    _flash_shape_ok = Q._flash_shape_ok
    _qkv_i8_ok = Q._qkv_i8_ok


def main():
    bad = []
    print("| C | T | hd | gate | expected | why |")
    print("|---|--:|--:|---|---|---|")
    os.environ["MODIFF_FUSE_QKV_I8"] = "1"
    for C, T, expect, why in CASES:
        got = bool(Stub(C)._qkv_i8_ok(T))
        print(f"| {C} | {T} | {C // 8} | {got} | {expect} | {why} |")
        if got != expect:
            bad.append(f"{C}x{T}: gate {got}, expected {expect} ({why})")

    # The opt-in must be load-bearing: with the env var unset NOTHING is eligible, whatever the
    # shape. This is the rollback path, so it is not decoration.
    del os.environ["MODIFF_FUSE_QKV_I8"]
    if any(Stub(C)._qkv_i8_ok(T) for C, T, _, _ in CASES):
        bad.append("gate is True with MODIFF_FUSE_QKV_I8 unset -- the opt-in does not gate")

    # Unfrozen scales must also block it: the per-column out scale is built from _fq_sqc/_skc/_svv,
    # so during the calibration window there is nothing to build it from.
    os.environ["MODIFF_FUSE_QKV_I8"] = "1"
    if Stub(384, frozen=False)._qkv_i8_ok(256):
        bad.append("gate is True before the flash scales froze")
    # int4 blocks must never take an int8-only route.
    if Stub(384, bits=4)._qkv_i8_ok(256):
        bad.append("gate is True on an int4 block")
    del os.environ["MODIFF_FUSE_QKV_I8"]

    print()
    if bad:
        print("FAILED:")
        for line in bad:
            print("  -", line)
        return 1
    print("PASS -- the gate admits the 10 hd=48 blocks (T=256, T=64) and nothing else, and both "
          "the opt-in and the frozen-scale precondition are load-bearing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

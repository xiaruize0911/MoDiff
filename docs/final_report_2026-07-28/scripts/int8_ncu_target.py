"""Minimal Nsight Compute target for one production INT8 attention shape.

The script deliberately launches only the current production K/V producer followed
by Q-in-Flash, which makes ``--kernel-name`` filtering reliable and avoids profiling
model setup or unrelated PyTorch kernels.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
import modiff_cutlass as mc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, choices=(256, 1024), required=True)
    parser.add_argument("--iterations", type=int, default=4)
    args = parser.parse_args()

    t = args.tokens
    n = 128
    hd = 24 if t == 1024 else 48
    hp = 32 if t == 1024 else 64
    heads = 8
    qkv = torch.randn(n, t, heads, 3, hd, device="cuda", dtype=torch.float16) * 0.1
    v_scale = torch.full((hp,), 0.01, device="cuda", dtype=torch.float32)

    for _ in range(args.iterations):
        k, vt, sv = mc.quantize_attn_kv_packed_static(
            qkv, heads, t, hd, hp, hp, 8, 0.01, v_scale)
        mc.flash_attn_int8_qpacked_kv_static_qout(
            qkv, k.view(n, heads, t, hp), vt.view(n, heads, hp, t),
            sv[:hd].contiguous(), hp, 0.01, 0.01, hd ** -0.5, 0.01)
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()

"""Minimal Nsight Compute target for the T1024 exact-hd24 A/B kernels."""

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import torch
import modiff_cutlass as mc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant", choices=("reference", "candidate"), required=True)
    parser.add_argument("--batch", type=int, default=128)
    args = parser.parse_args()
    torch.manual_seed(1234)
    q = torch.randint(
        -127, 128, (args.batch, 1024, 8, 32),
        device="cuda", dtype=torch.int8)
    q[..., 24:].zero_()
    k = torch.randint(
        -127, 128, (args.batch, 8, 1024, 32),
        device="cuda", dtype=torch.int8)
    vt = torch.randint(
        -127, 128, (args.batch, 8, 32, 1024),
        device="cuda", dtype=torch.int8)
    sv = torch.rand(24, device="cuda") * 0.02 + 0.001
    fn = (mc.flash_attn_int8_qi8_kv_static_qout
          if args.variant == "reference"
          else mc.flash_attn_int8_qi8_kv_static_qout_hd24)
    torch.cuda.synchronize()
    fn(q, k, vt, sv, 32, 0.01, 0.012, 24 ** -0.5, 0.02)
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()

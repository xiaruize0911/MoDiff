"""Shared loader for the ground-truth kernel-shape inventory (data/kernel_shapes.csv,
produced by enumerate_shapes.py). Every kernel benchmark reads its shape list + per-step
call counts from here so 'all shapes each kernel runs on' and 'count per step' come from the
real model dispatch, not a hand-maintained list."""
import os, csv

HERE = "docs/benchmark_5mode_2026-07-21"
MODES = ["fp16", "int8_baseline", "int4_baseline", "int8_modiff", "int4_modiff"]
_CSV = f"{HERE}/data/kernel_shapes.csv"


def _rows():
    with open(_CSV) as f:
        return list(csv.DictReader(f))


def conv_shapes():
    """Distinct conv geometries -> dict(Cin,H,W,Cout,K,stride,pad, count, quant_eligible, name).
    count is per DDIM step (mode-independent: modiff skips no conv). quant_eligible = the layer
    is replaced by an Optimized int8/int4 conv in quant modes (else it stays fp16 cuDNN)."""
    rows = [r for r in _rows() if r["family"] == "conv"]
    out = {}
    for r in rows:
        key = (int(r["Cin"]), int(r["H"]), int(r["W"]), int(r["Cout"]), int(r["K"]), int(r["stride"]), int(r["pad"]))
        e = out.setdefault(key, dict(Cin=key[0], H=key[1], W=key[2], Cout=key[3], K=key[4],
                                     stride=key[5], pad=key[6], count=int(r["count_per_step"]),
                                     quant_eligible=False))
        if "Optimized" in r["kernel_class"]:
            e["quant_eligible"] = True
    return list(out.values())


def linear_shapes():
    """Distinct linear GEMM shapes -> dict(role,K,N,M, count, quant_class). Uses the quant-mode
    inventory (complete: fp16 folds some qkv into fused_gn_qkv). count per step."""
    rows = [r for r in _rows() if r["family"] == "linear" and r["mode"] == "int8_baseline"]
    out = {}
    for r in rows:
        key = (r["role"], int(r["K"]), int(r["N"]), int(r["M"]))
        e = out.setdefault(key, dict(role=r["role"], K=key[1], N=key[2], M=key[3],
                                     count=int(r["count_per_step"]), quant_class=r["kernel_class"]))
    return list(out.values())


def linear_count_fp16():
    """qkv/proj/other standalone-Linear counts in fp16 (some qkv are fused into fused_gn_qkv)."""
    rows = [r for r in _rows() if r["family"] == "linear" and r["mode"] == "fp16"]
    out = {}
    for r in rows:
        key = (r["role"], int(r["K"]), int(r["N"]), int(r["M"]))
        out[key] = out.get(key, 0) + int(r["count_per_step"])
    return out


def attn_shapes():
    """Distinct attention blocks -> dict(C,nh,hd,T,Hspatial,flash_eligible,count). count per step."""
    rows = [r for r in _rows() if r["family"] == "attn" and r["mode"] == "int8_baseline"]
    out = {}
    for r in rows:
        key = (int(r["C"]), int(r["nh"]), int(r["hd"]), int(r["T"]))
        out[key] = dict(C=key[0], nh=key[1], hd=key[2], T=key[3], Hspatial=int(r["Hspatial"]),
                        flash_eligible=int(r["flash_eligible"]), count=int(r["count_per_step"]))
    return list(out.values())

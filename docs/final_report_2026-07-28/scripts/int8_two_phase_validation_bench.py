"""Reproducible validation/microbenchmark for the two INT8 attention candidates.

The full model/layer benchmark remains layer_pipeline_bench.py. This script
checks bit-exact layouts, padding, Phase-1 determinism, and reports CUDA-event
microbenchmarks with the production 20 warmups / 5x60 protocol.
"""
import argparse
import json
import os
import statistics
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import torch
import modiff_cutlass as mc


def bench(fn, warmups=20, rounds=5, iterations=60):
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()
    result = []
    for _ in range(rounds):
        events = [torch.cuda.Event(True) for _ in range(iterations + 1)]
        for i in range(iterations):
            events[i].record()
            fn()
        events[-1].record()
        torch.cuda.synchronize()
        samples = [
            events[i].elapsed_time(events[i + 1]) * 1e3
            for i in range(iterations)
        ]
        result.append(statistics.median(samples))
    return {"median_us": statistics.median(result), "round_medians_us": result}


def inputs(batch, tokens, channels, head_dim):
    heads = channels // head_dim
    hp = ((head_dim + 31) // 32) * 32
    old_n = 3 * channels
    old_np = ((old_n + 127) // 128) * 128
    layout_n = 3 * heads * hp
    m = batch * tokens
    a = torch.randint(-127, 128, (m, channels), device="cuda", dtype=torch.int8)
    w = torch.randint(-8, 9, (old_np, channels), device="cuda", dtype=torch.int8)
    ws = torch.rand(old_np, device="cuda") * 0.005 + 0.0001
    inv = torch.rand(old_np, device="cuda") * 50
    bias = torch.randn(old_n, device="cuda", dtype=torch.float16) * 0.01
    lw = torch.zeros(layout_n, channels, device="cuda", dtype=torch.int8)
    ls = torch.zeros(layout_n, device="cuda")
    li = torch.zeros(layout_n, device="cuda")
    lb = torch.zeros(layout_n, device="cuda", dtype=torch.float16)
    for h in range(heads):
        for sel in range(3):
            src, dst = (h * 3 + sel) * head_dim, (h * 3 + sel) * hp
            lw[dst:dst + head_dim].copy_(w[src:src + head_dim])
            ls[dst:dst + head_dim].copy_(ws[src:src + head_dim])
            li[dst:dst + head_dim].copy_(inv[src:src + head_dim])
            lb[dst:dst + head_dim].copy_(bias[src:src + head_dim])
    return a, w, ws, inv, bias, lw, ls, li, lb, heads, hp, old_n


def validate_shape(batch, tokens, channels, head_dim):
    args = inputs(batch, tokens, channels, head_dim)
    a, w, ws, inv, bias, lw, ls, li, lb, heads, hp, old_n = args
    old = mc.gemm_w8a8_awq_out_i8_bias_nout(
        a, w, ws, 0.02, inv, bias, old_n
    ).view(batch, tokens, heads, 3, head_dim)
    rk, rv = mc.quantize_attn_kv_from_i8(
        old, heads, tokens, head_dim, hp, hp
    )
    q, k, vt = mc.gemm_w8a8_awq_qkv_i8_layouts(
        a, lw, ls, 0.02, li, lb, heads, tokens, head_dim, hp
    )
    checks = {
        "q_exact": torch.equal(q[..., :head_dim], old[..., 0, :]),
        "k_exact": torch.equal(k, rk),
        "vt_exact": torch.equal(vt, rv),
        "q_padding_nonzero": torch.count_nonzero(q[..., head_dim:]).item(),
        "k_padding_nonzero": torch.count_nonzero(k[..., head_dim:]).item(),
        "v_padding_nonzero": torch.count_nonzero(vt[:, head_dim:, :]).item(),
    }
    if not all(v is True or v == 0 for v in checks.values()):
        raise RuntimeError(f"layout validation failed: {checks}")
    return checks, args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--validate-batch", type=int, default=4)
    ns = parser.parse_args()
    torch.manual_seed(1234)
    rows = []
    for tokens, channels, hd in ((1024, 192, 24), (256, 384, 48), (64, 384, 48)):
        checks, _ = validate_shape(ns.validate_batch, tokens, channels, hd)
        _, args = validate_shape(ns.batch, tokens, channels, hd)
        a, w, ws, inv, bias, lw, ls, li, lb, heads, hp, old_n = args
        old_qkv = mc.gemm_w8a8_awq_out_i8_bias_nout(
            a, w, ws, 0.02, inv, bias, old_n
        ).view(ns.batch, tokens, heads, 3, hd)
        old_gemm = lambda: mc.gemm_w8a8_awq_out_i8_bias_nout(
            a, w, ws, 0.02, inv, bias, old_n
        )
        producer = lambda: mc.quantize_attn_kv_from_i8(
            old_qkv, heads, tokens, hd, hp, hp
        )
        layout = lambda: mc.gemm_w8a8_awq_qkv_i8_layouts(
            a, lw, ls, 0.02, li, lb, heads, tokens, hd, hp
        )
        compact_layout = lambda: mc.gemm_w8a8_awq_qkv_i8_layouts_compact(
            a, w, ws, 0.02, inv, bias, heads, tokens, hd, hp
        )
        rows.append({
            "tokens": tokens,
            "validation": checks,
            "old_gemm": bench(old_gemm),
            "producer": bench(producer),
            "layout_gemm": bench(layout),
            "compact_layout_gemm": bench(compact_layout),
        })

    # Phase 1 uses the same already-produced packed QKV/K/VT as its reference.
    a, w, ws, inv, bias, _, _, _, _, heads, hp, old_n = inputs(
        ns.batch, 1024, 192, 24
    )
    qkv = mc.gemm_w8a8_awq_out_i8_bias_nout(
        a, w, ws, 0.02, inv, bias, old_n
    ).view(ns.batch, 1024, heads, 3, 24)
    k, vt = mc.quantize_attn_kv_from_i8(qkv, heads, 1024, 24, hp, hp)
    sv = torch.rand(24, device="cuda") * 0.02 + 0.001
    fargs = (qkv, k.view(ns.batch, heads, 1024, hp),
             vt.view(ns.batch, heads, hp, 1024), sv, hp,
             0.01, 0.012, 24 ** -0.5, 0.02)
    ref = mc.flash_attn_int8_qi8packed_kv_static_qout(*fargs)
    preg = mc.flash_attn_int8_qi8packed_kv_static_qout_preg(*fargs)
    result = {
        "gpu": torch.cuda.get_device_name(),
        "batch": ns.batch,
        "protocol": {"warmups": 20, "rounds": 5, "iterations": 60},
        "phase1": {
            "bit_exact": torch.equal(ref, preg),
            "shared_p": bench(
                lambda: mc.flash_attn_int8_qi8packed_kv_static_qout(*fargs)),
            "register_p": bench(
                lambda: mc.flash_attn_int8_qi8packed_kv_static_qout_preg(*fargs)),
        },
        "phase2": rows,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

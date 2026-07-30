"""Validate, benchmark, and profile the exact T1024/hd24 INT8 Flash kernel."""

import argparse
import json
import os
import re
import statistics
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import torch
import modiff_cutlass as mc


def percentile(values, p):
    values = sorted(values)
    pos = (len(values) - 1) * p
    lo, hi = int(pos), min(int(pos) + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def time_calls(fn, iterations):
    events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations + 1)]
    for i in range(iterations):
        events[i].record()
        fn()
    events[-1].record()
    torch.cuda.synchronize()
    return [
        events[i].elapsed_time(events[i + 1]) * 1e3
        for i in range(iterations)
    ]


def alternating_bench(ref, candidate, warmups, rounds, iterations):
    for i in range(warmups):
        (ref if i % 2 == 0 else candidate)()
        (candidate if i % 2 == 0 else ref)()
    torch.cuda.synchronize()
    samples = {"reference": [], "candidate": []}
    round_medians = {"reference": [], "candidate": []}
    for r in range(rounds):
        order = (("reference", ref), ("candidate", candidate))
        if r % 2:
            order = tuple(reversed(order))
        for name, fn in order:
            row = time_calls(fn, iterations)
            samples[name].extend(row)
            round_medians[name].append(statistics.median(row))
    result = {}
    for name in ("reference", "candidate"):
        values = samples[name]
        result[name] = {
            "median_us": statistics.median(round_medians[name]),
            "p10_us": percentile(values, 0.10),
            "p90_us": percentile(values, 0.90),
            "round_medians_us": round_medians[name],
        }
    result["speedup"] = (
        result["reference"]["median_us"] / result["candidate"]["median_us"])
    return result


def resource_usage():
    obj = os.path.join(
        ROOT, "build/temp.linux-x86_64-cpython-311/"
        "csrc/kernels/attention/flash_attn_int8.o")
    text = subprocess.run(
        ["cuobjdump", "-res-usage", obj], check=True,
        capture_output=True, text=True).stdout
    signatures = {
        "reference": (
            "ILi32ELi8ELi32ELb1ELb0ELb0ELb0ELi0ELi0EE"),
        "candidate": (
            "ILi32ELi8ELi32ELb1ELb0ELb0ELb0ELi24ELi1024EE"),
    }
    result = {}
    for name, signature in signatures.items():
        pattern = (
            r"Function ([^\n]*" + signature + r"[^\n]*):\n"
            r"\s*REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)")
        match = re.search(pattern, text)
        if not match:
            raise RuntimeError(f"resource record not found for {name}")
        regs, stack, shared, local = map(int, match.groups()[1:])
        threads = 8 * 32
        cta_reg = 65536 // (regs * threads)
        cta_smem = 102400 // shared
        cta_threads = 1536 // threads
        cta = min(cta_reg, cta_smem, cta_threads, 16)
        result[name] = {
            "registers": regs,
            "stack_bytes": stack,
            "local_bytes": local,
            "shared_bytes": shared,
            "threads": threads,
            "cta_per_sm": cta,
            "occupancy_pct": cta * threads / 1536 * 100,
        }
    return result


def make_inputs(batch, seed=1234):
    torch.manual_seed(seed)
    q = torch.randint(
        -127, 128, (batch, 1024, 8, 32), device="cuda", dtype=torch.int8)
    q[..., 24:].zero_()
    k = torch.randint(
        -127, 128, (batch, 8, 1024, 32), device="cuda", dtype=torch.int8)
    vt = torch.randint(
        -127, 128, (batch, 8, 32, 1024), device="cuda", dtype=torch.int8)
    sv = torch.rand(24, device="cuda", dtype=torch.float32) * 0.02 + 0.001
    return (q, k, vt, sv, 32, 0.01, 0.012, 24 ** -0.5, 0.02)


def validate():
    rows = []
    for batch in (1, 4, 128):
        args = make_inputs(batch)
        ref = mc.flash_attn_int8_qi8_kv_static_qout(*args)
        candidate = mc.flash_attn_int8_qi8_kv_static_qout_hd24(*args)
        torch.cuda.synchronize()
        exact = torch.equal(ref, candidate)
        max_diff = (ref.int() - candidate.int()).abs().max().item()
        deterministic = True
        if batch == 4:
            for _ in range(20):
                deterministic &= torch.equal(
                    candidate,
                    mc.flash_attn_int8_qi8_kv_static_qout_hd24(*args))
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                stream_result = (
                    mc.flash_attn_int8_qi8_kv_static_qout_hd24(*args))
            stream.synchronize()
            stream_exact = torch.equal(ref, stream_result)
        else:
            stream_exact = None
        rows.append({
            "batch": batch,
            "bit_exact": exact,
            "max_abs_code_diff": max_diff,
            "repeat20_deterministic": deterministic if batch == 4 else None,
            "nondefault_stream_exact": stream_exact,
        })
        if not exact or not deterministic or stream_exact is False:
            raise RuntimeError(f"exact-hd24 validation failed: {rows[-1]}")
    return rows


def profile(ref, candidate, calls):
    with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA]) as prof:
        with torch.profiler.record_function("hd24_reference"):
            for _ in range(calls):
                ref()
        with torch.profiler.record_function("hd24_candidate"):
            for _ in range(calls):
                candidate()
        torch.cuda.synchronize()
    rows = []
    for event in prof.key_averages():
        if "flash_attn_int8_mma_kernel_t" in event.key:
            rows.append({
                "kernel": event.key,
                "calls": event.count,
                "cuda_total_us": event.self_device_time_total,
                "cuda_mean_us": event.self_device_time_total / event.count,
            })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--profile-calls", type=int, default=10)
    parser.add_argument("--output")
    args = parser.parse_args()
    if not hasattr(mc, "flash_attn_int8_qi8_kv_static_qout_hd24"):
        raise RuntimeError("extension was not rebuilt with exact-hd24 support")
    validation = validate()
    fargs = make_inputs(args.batch)
    ref = lambda: mc.flash_attn_int8_qi8_kv_static_qout(*fargs)
    candidate = lambda: mc.flash_attn_int8_qi8_kv_static_qout_hd24(*fargs)
    result = {
        "gpu": torch.cuda.get_device_name(),
        "batch": args.batch,
        "protocol": {
            "warmups": args.warmups,
            "rounds": args.rounds,
            "iterations": args.iterations,
            "same_process_alternating": True,
        },
        "validation": validation,
        "resources": resource_usage(),
        "benchmark": alternating_bench(
            ref, candidate, args.warmups, args.rounds, args.iterations),
        "profile": profile(ref, candidate, args.profile_calls),
    }
    text = json.dumps(result, indent=2)
    if args.output:
        with open(args.output, "w") as handle:
            handle.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()

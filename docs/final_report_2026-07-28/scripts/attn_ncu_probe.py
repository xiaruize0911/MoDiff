"""ncu probe for the flash kernels at the shapes that lose to fp16.

The wall-clock benchmark says int4 flash at T=1024/hd=24 costs 2265 us against a computed floor of
~492 us, so it is 4.6x off and NOT bound by any of the five unit floors (mma / SFU / fp32 / HBM /
issue). That gap has to be attributed from hardware counters rather than reasoned about: the
candidates (smem pipe pressure, dependency stalls, occupancy, long-scoreboard waits on HBM) all
look the same from the outside and imply completely different fixes.

Collects, per kernel launch:
  achieved occupancy, warps active per scheduler, issue efficiency
  stall reason breakdown (which is the actual answer to "why 4.6x")
  smem / L1 / L2 / DRAM throughput, and the smem bank-conflict rate
  tensor-core (mma) pipe utilization -- to confirm or kill the "mma-bound" hypothesis

Run: python attn_ncu_probe.py [shape_index]   (0 = C192/T1024, 1 = C384/T256, 2 = C384/T64)
Writes data/attn_ncu_probe.json.
"""
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(HERE, "..", "data", "attn_ncu_probe.json")
sys.path.insert(0, ROOT)

SHAPES = [(1024, 24), (256, 48), (64, 48)]
BATCH, HEADS = 128, 8

METRICS = ",".join([
    "sm__warps_active.avg.pct_of_peak_sustained_active",          # achieved occupancy
    "smsp__issue_active.avg.pct_of_peak_sustained_active",        # issue slots used
    "sm__pipe_tensor_op_imma_cycles_active.avg.pct_of_peak_sustained_active",  # imma pipe
    "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared.avg.pct_of_peak_sustained_active",  # smem pipe
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum",
    "dram__bytes.sum",
    "gpu__time_duration.sum",
    "smsp__average_warp_latency_issue_stalled_long_scoreboard.ratio",
    "smsp__average_warp_latency_issue_stalled_short_scoreboard.ratio",
    "smsp__average_warp_latency_issue_stalled_barrier.ratio",
    "smsp__average_warp_latency_issue_stalled_mio_throttle.ratio",
    "smsp__average_warp_latency_issue_stalled_math_pipe_throttle.ratio",
    "smsp__average_warp_latency_issue_stalled_wait.ratio",
    "smsp__average_warp_latency_issue_stalled_not_selected.ratio",
])


def workload(T, hd):
    """Launch each kernel exactly once so ncu reports one clean instance per kernel name."""
    import torch
    import modiff_cutlass as mc
    N, H = BATCH, HEADS
    hp = ((hd + 31) // 32) * 32
    sc = 1.0 / (hd ** 0.5)
    vt = torch.randint(-127, 127, (N, H, hp, T), device="cuda", dtype=torch.int8).contiguous()
    sq = torch.full((N, H, T), 0.01, device="cuda")
    sk = torch.full((N, H, T), 0.01, device="cuda")
    sv = torch.full((N, H, hd), 0.01, device="cuda")
    qi = torch.randint(-127, 127, (N, H, T, hp), device="cuda", dtype=torch.int8)
    ki = torch.randint(-127, 127, (N, H, T, hp), device="cuda", dtype=torch.int8)
    q4 = torch.randint(-127, 127, (N, H, T, 32), device="cuda", dtype=torch.int8)
    k4 = torch.randint(-127, 127, (N, H, T, 32), device="cuda", dtype=torch.int8)
    mc.flash_attn_int8_vt(qi, ki, vt, sq, sk, sv, sc)
    mc.flash_attn_int4_vt(q4, k4, vt, sq, sk, sv, 64, sc)
    torch.cuda.synchronize()


def parse_csv(txt):
    """ncu --csv: one row per (kernel, metric). Collapse to {kernel: {metric: value}}."""
    rows, hdr = {}, None
    for line in txt.splitlines():
        if line.startswith('"ID"') or line.startswith("ID,"):
            hdr = [c.strip('"') for c in re.findall(r'"[^"]*"|[^,]+', line)]
            continue
        if hdr is None or not line.strip():
            continue
        cells = [c.strip('"') for c in re.findall(r'"[^"]*"|[^,]+', line)]
        if len(cells) != len(hdr):
            continue
        d = dict(zip(hdr, cells))
        kn = d.get("Kernel Name", "")
        mn, mv = d.get("Metric Name", ""), d.get("Metric Value", "")
        if not kn or not mn:
            continue
        short = "int8" if "int8_mma" in kn else ("int4" if "int4_mma" in kn else kn[:40])
        try:
            v = float(mv.replace(",", ""))
        except ValueError:
            v = mv
        rows.setdefault(short, {})[mn] = v
    return rows


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--work":
        T, hd = int(sys.argv[2]), int(sys.argv[3])
        workload(T, hd)
        return

    out = {}
    for idx, (T, hd) in enumerate(SHAPES):
        print(f"\n########## T={T} hd={hd} ##########")
        cmd = ["ncu", "--csv", "--metrics", METRICS, "--target-processes", "all",
               sys.executable, os.path.abspath(__file__), "--work", str(T), str(hd)]
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        if p.returncode != 0:
            print(f"  ncu failed: {p.stderr.strip()[-300:]}")
            out[f"T{T}_hd{hd}"] = {"error": p.stderr.strip()[-300:]}
            continue
        rows = parse_csv(p.stdout)
        out[f"T{T}_hd{hd}"] = rows
        for kn, m in rows.items():
            print(f"\n  --- {kn} ---")
            for name, v in m.items():
                short = name.replace("smsp__average_warp_latency_issue_stalled_", "stall.") \
                            .replace(".avg.pct_of_peak_sustained_active", " [%peak]") \
                            .replace("l1tex__data_", "").replace("sm__", "").replace("smsp__", "")
                print(f"    {short:62s} {v if isinstance(v, str) else f'{v:,.2f}'}")
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWROTE {OUT}")


if __name__ == "__main__":
    main()

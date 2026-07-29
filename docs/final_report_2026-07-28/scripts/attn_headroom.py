"""Is there headroom left in (4) the hd=48 occupancy and (5) the int4 k-padding? Joined evidence.

Both are proposed optimizations whose payoff was explicitly unknown, so each gets checked against
measurement before any kernel is touched.

(4) hd=48 sits at 33% occupancy and 127 registers. The proposed fix is to tile Sreg so fewer
    registers are live. But Sreg holds the [BR x BC] score tile, and processing it in two halves of
    BC/2 columns IS a smaller BC -- online softmax is incremental over columns by construction, so
    splitting Sreg at BC=64 computes exactly what BC=32 computes. BC=32 already exists, already
    lowers REG 127 -> 96, and is already measured. So the question "does cutting registers help this
    kernel?" has already been answered by the sweep; this joins occupancy to time to state it.

(5) int4 pads hd=24 up to hdp4=64 for mma.m16n8k64.s4, so 62% of the k-depth is zeros. The proposed
    fix (m16n8k32.s4) would halve K's smem/HBM traffic while leaving the mma instruction count
    unchanged. That only pays if the kernel is anywhere near memory-bound, which is checked here
    against both the optimistic (perfect L2 reuse of K/V across the grid.y CTAs sharing an (n,h))
    and pessimistic (every CTA misses) traffic models.

Reads data/attn_tile_sweep.json + data/attn_occupancy.json; measures the int4-vs-int8 comparison
at hd=24 directly.
"""
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
import torch
import modiff_cutlass as mc

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(DATA, "attn_headroom.json")
DEV = "cuda"
N, H = 128, 8
HBM_CEILING = 590e9        # B/s, this card's measured single-pass stream rate
FA_BR = 16


def bench(fn, it=25, reps=5):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    o = []
    for _ in range(reps):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(it):
            fn()
        e.record()
        torch.cuda.synchronize()
        o.append(s.elapsed_time(e) / it * 1e3)
    return statistics.median(o)


def part4():
    """Join every hd=48 config's measured time to the occupancy its footprint allows."""
    sweep = json.load(open(os.path.join(DATA, "attn_tile_sweep.json")))
    occ = {r["label"]: r for r in json.load(open(os.path.join(DATA, "attn_occupancy.json")))["rows"]}
    rows = []
    print("(4) hd=48: does more occupancy make this kernel faster?\n")
    print(f"{'T':>5} {'config':>12} {'REG':>4} {'CTA/SM':>7} {'occ':>6} {'us':>9}  binding")
    for r in sweep["rows"]:
        if r["hd"] != 48 or r["bits"] != 8:
            continue
        for cell, us in r["cells"].items():
            bc = int(cell.split("_")[0][2:]); w = int(cell.split("_")[1][1:])
            # WARPS falls back to 4 unless T % (w*BR) == 0 -- report the WARPS that actually ran
            w_eff = w if (r["T"] % (w * FA_BR) == 0) else 4
            o = occ.get(f"int8 HD64 W{w_eff} BC{bc}")
            if not o or us is None:
                continue
            rows.append(dict(T=r["T"], config=f"BC{bc}_W{w_eff}", regs=o["regs"],
                             cta_per_sm=o["cta_per_sm"], occupancy_pct=o["occupancy_pct"],
                             us=us, binding=o["binding_resource"]))
            print(f"{r['T']:5d} {f'BC{bc}/W{w_eff}':>12} {o['regs']:4d} {o['cta_per_sm']:7d} "
                  f"{o['occupancy_pct']:5.1f}% {us:9.2f}  {o['binding_resource']}")
    # the decisive comparison: at each T, is the highest-occupancy config also the fastest?
    verdict = []
    for T in sorted({r["T"] for r in rows}):
        g = [r for r in rows if r["T"] == T]
        best_time = min(g, key=lambda r: r["us"])
        best_occ = max(g, key=lambda r: r["occupancy_pct"])
        verdict.append(dict(T=T, fastest=best_time["config"], fastest_us=best_time["us"],
                            fastest_occ=best_time["occupancy_pct"],
                            highest_occ=best_occ["config"], highest_occ_pct=best_occ["occupancy_pct"],
                            highest_occ_us=best_occ["us"],
                            occupancy_wins=best_time["config"] == best_occ["config"]))
        v = verdict[-1]
        print(f"  T={T}: fastest {v['fastest']} ({v['fastest_us']:.1f} us @ {v['fastest_occ']}% occ) "
              f"vs highest-occupancy {v['highest_occ']} ({v['highest_occ_us']:.1f} us @ "
              f"{v['highest_occ_pct']}%) -> occupancy {'wins' if v['occupancy_wins'] else 'LOSES'}")
    return rows, verdict


def part5():
    """int4 at hd=24: measure it against int8, and against both HBM traffic models."""
    print("\n(5) int4 hdp4=64 padding: is this kernel anywhere near memory-bound?\n")
    rows = []
    for T, hd in [(1024, 24), (256, 48)]:
        hp = ((hd + 31) // 32) * 32
        sc = 1.0 / (hd ** 0.5)
        vt = torch.randint(-127, 127, (N, H, hp, T), device=DEV, dtype=torch.int8).contiguous()
        sq = torch.full((N, H, T), 0.01, device=DEV)
        sk = torch.full((N, H, T), 0.01, device=DEV)
        sv = torch.full((N, H, hd), 0.01, device=DEV)
        qi = torch.randint(-127, 127, (N, H, T, hp), device=DEV, dtype=torch.int8)
        ki = torch.randint(-127, 127, (N, H, T, hp), device=DEV, dtype=torch.int8)
        us8 = bench(lambda: mc.flash_attn_int8_vt(qi, ki, vt, sq, sk, sv, sc))
        del qi, ki
        torch.cuda.empty_cache()
        q4 = torch.randint(-127, 127, (N, H, T, 32), device=DEV, dtype=torch.int8)
        k4 = torch.randint(-127, 127, (N, H, T, 32), device=DEV, dtype=torch.int8)
        us4 = bench(lambda: mc.flash_attn_int4_vt(q4, k4, vt, sq, sk, sv, 64, sc))

        warps = 8 if (T % (8 * FA_BR) == 0) else 4
        grid_y = T // (warps * FA_BR)
        BH = N * H
        k_bytes = BH * T * 32                      # packed int4 K row = hdp4/2 = 32 B
        v_bytes = BH * hp * T
        q_bytes = BH * T * 32
        o_bytes = BH * T * hd * 2
        optimistic = q_bytes + k_bytes + v_bytes + o_bytes          # perfect L2 reuse across grid.y
        pessimistic = q_bytes + grid_y * (k_bytes + v_bytes) + o_bytes
        r = dict(T=T, hd=hd, warps=warps, grid_y=grid_y, int8_us=round(us8, 1),
                 int4_us=round(us4, 1), int4_vs_int8=round(us8 / us4, 3),
                 mb_optimistic=round(optimistic / 1e6, 1),
                 mb_pessimistic=round(pessimistic / 1e6, 1),
                 gbs_optimistic=round(optimistic / (us4 * 1e-6) / 1e9, 1),
                 gbs_pessimistic=round(pessimistic / (us4 * 1e-6) / 1e9, 1),
                 pct_ceiling_pessimistic=round(pessimistic / (us4 * 1e-6) / HBM_CEILING * 100, 1),
                 # what halving the K row (32 -> 16 B) would remove, under the pessimistic model
                 mb_saved_by_k32=round(grid_y * (k_bytes / 2) / 1e6, 1))
        r["max_gain_if_bw_bound"] = round(1 / (1 - r["mb_saved_by_k32"] / r["mb_pessimistic"]), 3)
        rows.append(r)
        print(f"  T={T} hd={hd}: int8 {us8:.1f} us, int4 {us4:.1f} us (int4 is {us8/us4:.2f}x int8)")
        print(f"    traffic {r['mb_optimistic']} MB optimistic / {r['mb_pessimistic']} MB pessimistic "
              f"(K,V re-read grid.y={grid_y}x)")
        print(f"    achieved {r['gbs_optimistic']} - {r['gbs_pessimistic']} GB/s = "
              f"{r['pct_ceiling_pessimistic']}% of the {HBM_CEILING/1e9:.0f} GB/s ceiling AT BEST")
        print(f"    halving the K row would remove {r['mb_saved_by_k32']} MB -> at most "
              f"{r['max_gain_if_bw_bound']:.2f}x, and only if the kernel were bandwidth-bound")
        del q4, k4, vt, sq, sk, sv
        torch.cuda.empty_cache()
    return rows


def main():
    bn = torch.randn(4096, 4096, device=DEV, dtype=torch.float16)
    for _ in range(60):
        bn = bn @ bn * 1e-4 + 1.0
    torch.cuda.synchronize(); del bn; torch.cuda.empty_cache()
    p4, v4 = part4()
    p5 = part5()
    with open(OUT, "w") as f:
        json.dump({"part4_hd48_occupancy": p4, "part4_verdict": v4, "part5_int4_padding": p5}, f,
                  indent=2)
    print(f"\nWROTE {OUT}")


if __name__ == "__main__":
    main()

"""Tile-config sweep for the int8/int4 flash kernels, at every eligible real shape.

The kernel is now templated on (HD_PAD, WARPS, BC) with 8 int8 instantiations, and the host picks
BC/WARPS from a heuristic (flash_attn_int8.cu: modiff_fa_bc / modiff_fa_warps):

    BC    = (hd_pad <= 32 || T < 128) ? 32 : 64
    WARPS = (T % 128 == 0) ? 8 : 4

Those two rules were written from measurements on a subset of shapes. This sweeps the full 2x2 for
every eligible shape in the model and reports whether the heuristic actually picks the winner --
the point being that a wrong pick is invisible in any single-config benchmark.

MODIFF_FA_BC / MODIFF_FA_WARPS are latched into function-local statics on first call, so a config
cannot be changed inside a live process. Each config therefore runs in its OWN subprocess
(--one BC WARPS BITS T hd), and the parent only aggregates.

int4 is swept too, as a control: MODIFF_FA_MMA4_DISPATCH hardcodes FA_MMA_WARPS and ignores the
env var, so int4's four cells must come out equal. If they do not, the dispatch is reading state it
should not.
"""
import json
import os
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(HERE, "..", "data", "attn_tile_sweep.json")
sys.path.insert(0, ROOT)

BATCH, HEADS = 128, 8
# (C, H, W, count) for the eligible blocks only: hd_pad <= 64 and T % 64 == 0.
# The six hd=96 blocks (C=768) cannot run on this kernel at all, so they have nothing to sweep.
SHAPES = [(192, 32, 32, 5), (384, 16, 16, 5), (384, 8, 8, 5)]
CONFIGS = [(bc, w) for bc in (32, 64) for w in (4, 8)]


def bench(fn, it=25, reps=5):
    import torch
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(it):
            fn()
        e.record()
        torch.cuda.synchronize()
        out.append(s.elapsed_time(e) / it * 1e3)
    return statistics.median(out)


def run_one(bits, T, hd):
    """One config, one process. Env already set by the parent."""
    import torch
    import modiff_cutlass as mc
    N, H = BATCH, HEADS
    hp = ((hd + 31) // 32) * 32
    sc = 1.0 / (hd ** 0.5)
    vt = torch.randint(-127, 127, (N, H, hp, T), device="cuda", dtype=torch.int8).contiguous()
    sq = torch.full((N, H, T), 0.01, device="cuda")
    sk = torch.full((N, H, T), 0.01, device="cuda")
    sv = torch.full((N, H, hd), 0.01, device="cuda")
    if bits == 8:
        qi = torch.randint(-127, 127, (N, H, T, hp), device="cuda", dtype=torch.int8)
        ki = torch.randint(-127, 127, (N, H, T, hp), device="cuda", dtype=torch.int8)
        us = bench(lambda: mc.flash_attn_int8_vt(qi, ki, vt, sq, sk, sv, sc))
    else:
        q4 = torch.randint(-127, 127, (N, H, T, 32), device="cuda", dtype=torch.int8)
        k4 = torch.randint(-127, 127, (N, H, T, 32), device="cuda", dtype=torch.int8)
        us = bench(lambda: mc.flash_attn_int4_vt(q4, k4, vt, sq, sk, sv, 64, sc))
    return us


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--one":
        bc, w, bits, T, hd = (int(v) for v in sys.argv[2:7])
        print(json.dumps({"us": run_one(bits, T, hd)}))
        return

    rows = []
    for C, Hh, Ww, cnt in SHAPES:
        T, hd = Hh * Ww, C // HEADS
        hp = ((hd + 31) // 32) * 32
        # what the shipped heuristic would choose for this shape
        pick = ((32 if (hp <= 32 or T < 128) else 64), (8 if T % 128 == 0 else 4))
        for bits in (8, 4):
            cells = {}
            for bc, w in CONFIGS:
                env = dict(os.environ, MODIFF_FA_BC=str(bc), MODIFF_FA_WARPS=str(w))
                p = subprocess.run([sys.executable, os.path.abspath(__file__), "--one",
                                    str(bc), str(w), str(bits), str(T), str(hd)],
                                   cwd=ROOT, env=env, capture_output=True, text=True)
                if p.returncode != 0:
                    cells[f"BC{bc}_W{w}"] = None
                    print(f"  int{bits} BC={bc} W={w}: FAIL {p.stderr.strip()[-120:]}")
                    continue
                us = json.loads(p.stdout.strip().splitlines()[-1])["us"]
                cells[f"BC{bc}_W{w}"] = round(us, 2)
                print(f"  int{bits} C={C} T={T} hd={hd}  BC={bc} W={w}: {us:9.2f} us")
            ok = {k: v for k, v in cells.items() if v is not None}
            best = min(ok, key=ok.get) if ok else None
            rows.append(dict(C=C, HW=f"{Hh}x{Ww}", T=T, hd=hd, count=cnt, bits=bits,
                             cells=cells, best=best, best_us=ok.get(best),
                             heuristic=f"BC{pick[0]}_W{pick[1]}",
                             heuristic_us=cells.get(f"BC{pick[0]}_W{pick[1]}"),
                             spread=(round(max(ok.values()) / min(ok.values()), 3) if ok else None)))
    for r in rows:
        h, b = r["heuristic_us"], r["best_us"]
        r["heuristic_is_best"] = (r["heuristic"] == r["best"])
        r["left_on_table"] = round(h / b, 3) if (h and b) else None
    with open(OUT, "w") as f:
        json.dump({"batch": BATCH, "heads": HEADS, "rows": rows}, f, indent=2)
    print(f"\nWROTE {OUT}")


if __name__ == "__main__":
    main()

"""Parse the per-mode nsys sqlites -> measured memcpy read/write bytes (per DDIM step).
Writes data/e2e_memcpy_total.csv (H2D/D2H/D2D MiB per step) and data/e2e_memcpy_sites.csv
(top recurring D2D copy sizes per mode = the copy sites). NSTEPS must match run_nsys_mem.sh.
Copy traffic only (compute-kernel DRAM traffic is not measurable here — counters blocked)."""
import os, sys, csv, sqlite3
os.chdir("/workspace/MoDiff")
DATA = "docs/benchmark_5mode_2026-07-20/data"
NSTEPS = 30
MODES = [("fp16", "fp16"), ("int8_baseline", "int8_baseline"), ("int4_baseline", "int4_baseline"),
         ("int8_modiff", "int8"), ("int4_modiff", "int4")]
MiB = 1024.0 ** 2
KIND = {1: "H2D", 2: "D2H", 8: "D2D", 3: "D2D"}   # CUPTI copyKind (8 = PtoP/D2D on some exports)


def col(cur, table, *cands):
    cur.execute(f"PRAGMA table_info({table})")
    cols = {r[1] for r in cur.fetchall()}
    for c in cands:
        if c in cols:
            return c
    return None


totals, sites = [], []
for label, m in MODES:
    sq = f"{DATA}/nsys_{m}.sqlite"
    if not os.path.exists(sq):
        print(f"MISSING {sq} — run run_nsys_mem.sh first"); continue
    c = sqlite3.connect(sq); cur = c.cursor()
    bytes_col = col(cur, "CUPTI_ACTIVITY_KIND_MEMCPY", "bytes", "copySize")
    kind_col = col(cur, "CUPTI_ACTIVITY_KIND_MEMCPY", "copyKind")
    by_kind = {"H2D": 0.0, "D2H": 0.0, "D2D": 0.0}
    cur.execute(f'SELECT {kind_col}, SUM({bytes_col}), COUNT(*) FROM CUPTI_ACTIVITY_KIND_MEMCPY GROUP BY {kind_col}')
    for k, b, n in cur.fetchall():
        by_kind[KIND.get(k, "D2D")] = by_kind.get(KIND.get(k, "D2D"), 0.0) + (b or 0)
    tot = sum(by_kind.values())
    totals.append(dict(mode=label,
                       H2D_MiB=round(by_kind["H2D"] / MiB / NSTEPS, 2),
                       D2H_MiB=round(by_kind["D2H"] / MiB / NSTEPS, 2),
                       D2D_MiB=round(by_kind["D2D"] / MiB / NSTEPS, 2),
                       total_MiB=round(tot / MiB / NSTEPS, 2)))
    # top recurring D2D copy sizes (the copy sites)
    d2d_kinds = [k for k, v in KIND.items() if v == "D2D"]
    ph = ",".join("?" * len(d2d_kinds))
    cur.execute(f'SELECT {bytes_col}, COUNT(*) FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE {kind_col} IN ({ph}) '
                f'GROUP BY {bytes_col} ORDER BY {bytes_col}*COUNT(*) DESC LIMIT 12', d2d_kinds)
    for size, n in cur.fetchall():
        sites.append(dict(mode=label, size_bytes=size, size_MiB=round(size / MiB, 3),
                          count_per_step=round(n / NSTEPS, 2), total_MiB_per_step=round(size * n / MiB / NSTEPS, 3)))
    c.close()
    print(f"{label:16} total {totals[-1]['total_MiB']:8.1f} MiB/step  "
          f"(H2D {totals[-1]['H2D_MiB']}, D2H {totals[-1]['D2H_MiB']}, D2D {totals[-1]['D2D_MiB']})")

if totals:
    with open(f"{DATA}/e2e_memcpy_total.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(totals[0].keys())); w.writeheader(); w.writerows(totals)
    with open(f"{DATA}/e2e_memcpy_sites.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sites[0].keys())); w.writeheader(); w.writerows(sites)
    print(f"\nWROTE {DATA}/e2e_memcpy_total.csv, e2e_memcpy_sites.csv")

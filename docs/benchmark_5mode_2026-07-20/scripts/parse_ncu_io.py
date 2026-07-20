"""Parse the ncu --csv --page raw output from run_ncu_io.sh into a per-kernel x per-shape
read/write byte table. Maps each profiled kernel to its (family, mode, shape) via the NVTX range
set by ncu_io_driver.py (tag `family|mode|shape`). Writes data/ncu_io_perkernel_<fam>.csv.

Needs a real ncu run (unlocked counters) to have data; on this box ncu is counter-locked, so this
is the ready-to-run parser for when counters are enabled."""
import os, sys, csv, re
os.chdir("/workspace/MoDiff")
DATA = "docs/benchmark_5mode_2026-07-20/data"
FAM = sys.argv[1] if len(sys.argv) > 1 else "all"
RAW = f"{DATA}/ncu_io_raw_{FAM}.csv"
MiB = 1024.0 ** 2
TAG = re.compile(r"(conv|linear|attn)\|(fp16|int8|int4)\|([A-Za-z0-9_\-]+)")

if not os.path.exists(RAW):
    print(f"MISSING {RAW} — run run_ncu_io.sh {FAM} first (needs unlocked GPU counters)"); raise SystemExit(0)

with open(RAW) as f:
    lines = f.read().splitlines()
hi = next((i for i, l in enumerate(lines) if "Kernel Name" in l and "Metric Name" in l), None)
if hi is None:
    print("no ncu CSV header (Kernel Name/Metric Name) found — check ncu_io_*.log for errors"); raise SystemExit(1)
rows = list(csv.reader(lines[hi:]))
hdr = [h.strip() for h in rows[0]]
def col(name):
    return next((i for i, h in enumerate(hdr) if h == name), None)
iID = col("ID"); iK = col("Kernel Name"); iM = col("Metric Name"); iV = col("Metric Value")
iNVTX = next((i for i, h in enumerate(hdr) if "NVTX" in h.upper()), None)

# group by launch ID -> kernel, nvtx tag, metric dict
launch = {}
for rec in rows[1:]:
    if len(rec) <= max(x for x in (iID, iK, iM, iV) if x is not None):
        continue
    lid = rec[iID].strip() if iID is not None else rec[iK]
    d = launch.setdefault(lid, {"kernel": rec[iK].split("(")[0].split("<")[0].strip(), "tag": None, "m": {}})
    if iNVTX is not None and rec[iNVTX].strip():
        mt = TAG.search(rec[iNVTX])
        if mt: d["tag"] = mt.group(0)
    mn = rec[iM].strip(); mv = rec[iV].strip().replace(",", "")
    if mn:
        try: d["m"][mn] = float(mv)
        except ValueError: pass

out = []
for lid, d in launch.items():
    tag = d["tag"] or "?|?|?"
    fam, mode, shape = (tag.split("|") + ["?", "?", "?"])[:3]
    rd = d["m"].get("dram__bytes_read.sum", 0.0)
    wr = d["m"].get("dram__bytes_write.sum", 0.0)
    tot = d["m"].get("dram__bytes.sum", rd + wr)
    dur = d["m"].get("gpu__time_duration.sum", 0.0)
    pct = d["m"].get("dram__throughput.avg.pct_of_peak_sustained_elapsed", 0.0)
    out.append(dict(family=fam, mode=mode, shape=shape, kernel=d["kernel"],
                    read_MiB=round(rd / MiB, 3), write_MiB=round(wr / MiB, 3),
                    total_MiB=round(tot / MiB, 3), dur_us=round(dur / 1e3, 2), dram_pct_peak=round(pct, 1)))
out.sort(key=lambda r: (r["family"], r["shape"], r["mode"], -r["total_MiB"]))
with open(f"{DATA}/ncu_io_perkernel_{FAM}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
print(f"WROTE {DATA}/ncu_io_perkernel_{FAM}.csv ({len(out)} kernels)")
for r in out[:20]:
    print(f"  {r['family']:6} {r['mode']:5} {r['shape']:16} {r['kernel'][:34]:34} rd {r['read_MiB']:8.2f} wr {r['write_MiB']:8.2f}  {r['dram_pct_peak']:5.1f}% peak")

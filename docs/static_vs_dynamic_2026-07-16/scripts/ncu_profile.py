"""Nsight Compute measured per-kernel metrics for the key kernels (dominant T=1024 attention shape):
real DRAM bytes moved, % of peak DRAM bandwidth, duration, SM throughput, and memory-vs-compute
bound. This is the MEASURED total-IO ground truth behind the analytical model. Runs ncu on
ncu_harness.py (each kernel launched once; ncu replays internally). Emits ncu_kernels.csv."""
import os, subprocess, csv
os.chdir("/workspace/MoDiff")
OUT = "/workspace/MoDiff/docs/static_vs_dynamic_2026-07-16/data"
NCU = "/usr/local/cuda/bin/ncu"
ENV = dict(os.environ, PYTHONPATH="/workspace/MoDiff/src/taming-transformers:/workspace/MoDiff", CUTLASS_PATH="/workspace/cutlass")
METRICS = ["dram__bytes.sum",
           "dram__throughput.avg.pct_of_peak_sustained_elapsed",
           "gpu__time_duration.sum",
           "sm__throughput.avg.pct_of_peak_sustained_elapsed"]
# only profile our kernels of interest (keeps ncu fast; skips torch/cutlass setup launches)
KRE = "regex:(attn_softmax|bmm_qk_s|bmm_av_s|aq_qtok|aq_vscale|aq_vquant)"
cmd = [NCU, "--target-processes", "all", "--kernel-name", KRE, "--metrics", ",".join(METRICS),
       "--csv", "--page", "raw", "python3.11", "docs/static_vs_dynamic_2026-07-16/scripts/ncu_harness.py"]
print("running ncu ...", flush=True)
r = subprocess.run(cmd, env=ENV, capture_output=True, text=True)
out = r.stdout
if "==PROF==" not in out and "Kernel Name" not in out and '"' not in out:
    print("ncu stderr:", r.stderr[-1500:]); print("ncu stdout head:", out[:800])
# ncu --csv --page raw -> long format; find the header row containing "Kernel Name" and "Metric Name"
lines = out.splitlines()
hi = next((i for i, l in enumerate(lines) if "Kernel Name" in l and "Metric Name" in l), None)
if hi is None:
    print("no ncu csv header found; full stderr:\n", r.stderr[-2000:]); raise SystemExit(1)
rows = list(csv.reader(lines[hi:]))
hdr = rows[0]
def idx(name):
    return next((i for i, h in enumerate(hdr) if h.strip() == name), None)
ik, im, iv = idx("Kernel Name"), idx("Metric Name"), idx("Metric Value")
data = {}   # kernel -> {metric: value}; first launch of each name wins
order = []
for rec in rows[1:]:
    if len(rec) <= max(ik, im, iv): continue
    k = rec[ik].split("(")[0].split("<")[0].strip()
    m = rec[im].strip(); v = rec[iv].strip().replace(",", "")
    if not k or not m: continue
    if k not in data: data[k] = {}; order.append(k)
    if m not in data[k]:
        try: data[k][m] = float(v)
        except ValueError: pass

outrows = []
for k in order:
    d = data[k]
    by = d.get("dram__bytes.sum", 0.0)
    dpct = d.get("dram__throughput.avg.pct_of_peak_sustained_elapsed", 0.0)
    dur = d.get("gpu__time_duration.sum", 0.0)
    spct = d.get("sm__throughput.avg.pct_of_peak_sustained_elapsed", 0.0)
    bound = "memory" if dpct >= spct else "compute"
    outrows.append({"kernel": k, "dram_bytes_MiB": round(by / (1024**2), 2),
                    "dram_pct_peak": round(dpct, 1), "dur_us": round(dur / 1e3, 1),
                    "sm_pct_peak": round(spct, 1), "bound": bound})
    print(f"  {k:38s} {by/(1024**2):8.2f} MiB  DRAM {dpct:5.1f}%  SM {spct:5.1f}%  {dur/1e3:7.1f}us  {bound}")
with open(f"{OUT}/ncu_kernels.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["kernel", "dram_bytes_MiB", "dram_pct_peak", "dur_us", "sm_pct_peak", "bound"])
    w.writeheader(); w.writerows(outrows)
print("WROTE ncu_kernels.csv")

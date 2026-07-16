"""nsys per-kernel GPU-time breakdown, dynamic vs static, on representative modes. For each mode:
run nsys (capture-range = the single measured sample in nsys_run_one.py), export cuda_gpu_kern_sum,
aggregate the kernel rows into named buckets. nsys perturbs absolute timings, so use it only for the
RELATIVE per-kernel breakdown (the clean wall/GPU-busy numbers come from pipeline.py). Emits
nsys_kernels.csv (per mode x bucket) and nsys_topkernels.csv (per mode top kernels)."""
import os, sys, subprocess, csv, glob, re
os.chdir("/workspace/MoDiff")
OUT = "/workspace/MoDiff/docs/static_vs_dynamic_2026-07-16/data"
TMP = "/tmp/claude-0/-workspace/1150c54c-9325-4a0c-8e13-9708345f7905/scratchpad/nsys"
os.makedirs(TMP, exist_ok=True)
NSYS = "/opt/nvidia/nsight-compute/2024.1.1/host/target-linux-x64/nsys"
MODES = ["dynamic_fp16", "static_fp16", "dynamic_int8", "static_int8", "dynamic_int4", "static_int4"]
ENV = dict(os.environ, PYTHONPATH="/workspace/MoDiff/src/taming-transformers:/workspace/llm-awq/awq/kernels:/workspace/MoDiff",
           CUTLASS_PATH="/workspace/cutlass")

def bucket(name):
    l = name.lower()
    if "softmax" in l: return "attention softmax"
    if "bmm_qk" in l or "bmm_av" in l: return "attn QKᵀ/AV (int GEMM)"
    if "aq_qtok" in l or "aq_vscale" in l or "aq_vquant" in l or "quantize" in l or "absmax" in l or "sub_absmax" in l: return "quantize / absmax"
    if "gemm" in l or "cutlass" in l or "cublas" in l or "ampere" in l or "gett" in l: return "conv/linear GEMM"
    if "cudnn" in l or "implicit" in l or "conv" in l or "wgrad" in l or "fprop" in l: return "conv/linear GEMM"
    if "group_norm" in l or "groupnorm" in l or "gn_" in l or "rowwise" in l: return "GroupNorm"
    if "elementwise" in l or "vectorized" in l or "silu" in l or "copy" in l or "cat" in l or "fill" in l or "add" in l: return "elementwise/copy"
    return "other"

rows, toprows = [], []
for mode in MODES:
    rep = f"{TMP}/{mode}"
    for f in glob.glob(rep + ".*"):
        try: os.remove(f)
        except OSError: pass
    print(f"=== nsys {mode} ===", flush=True)
    r = subprocess.run([NSYS, "profile", "-o", rep, "-f", "true", "-t", "cuda",
                        "--capture-range=cudaProfilerApi", "--capture-range-end=stop",
                        "python3.11", "docs/static_vs_dynamic_2026-07-16/scripts/nsys_run_one.py", mode],
                       env=ENV, capture_output=True, text=True)
    if not os.path.exists(rep + ".nsys-rep"):
        print("  no report:", r.stderr[-500:]); continue
    st = subprocess.run([NSYS, "stats", "--report", "cuda_gpu_kern_sum", "--format", "csv", rep + ".nsys-rep"],
                        env=ENV, capture_output=True, text=True)
    lines = [l for l in st.stdout.splitlines() if l and not l.startswith("**") and "," in l]
    if not lines: print("  no stats:", st.stdout[-300:], st.stderr[-300:]); continue
    hdr = next(csv.reader([lines[0]]))
    def col(cands):
        for i, h in enumerate(hdr):
            if any(c in h.lower() for c in cands): return i
        return None
    ci_time, ci_name = col(["total time", "total time (ns)"]), col(["name"])
    ci_pct = col(["time (%)", "time(%)"])
    buck = {}; tops = []
    for line in lines[1:]:
        rec = next(csv.reader([line]))
        if ci_time is None or ci_name is None or len(rec) <= max(ci_time, ci_name): continue
        try: t_ms = float(rec[ci_time].replace(",", "")) / 1e6
        except ValueError: continue
        nm = rec[ci_name]
        buck[bucket(nm)] = buck.get(bucket(nm), 0.0) + t_ms
        tops.append((t_ms, nm))
    for b, v in sorted(buck.items(), key=lambda x: -x[1]):
        rows.append({"mode": mode, "bucket": b, "gpu_ms": round(v, 3)})
        print(f"  {b:26s} {v:8.2f} ms")
    for t, nm in sorted(tops, reverse=True)[:8]:
        toprows.append({"mode": mode, "kernel": nm[:70], "gpu_ms": round(t, 3)})

with open(f"{OUT}/nsys_kernels.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["mode", "bucket", "gpu_ms"]); w.writeheader(); w.writerows(rows)
with open(f"{OUT}/nsys_topkernels.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["mode", "kernel", "gpu_ms"]); w.writeheader(); w.writerows(toprows)
print("WROTE nsys_kernels.csv, nsys_topkernels.csv")

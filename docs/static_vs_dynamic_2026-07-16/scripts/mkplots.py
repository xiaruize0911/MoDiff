"""Plots for the static-vs-dynamic report. Grouped dynamic/static bars for e2e speed, peak memory,
analytical IO; softmax + attention micro speedups; profile buckets; ncu measured DRAM bytes; nsys
per-kernel breakdown. Robust to missing CSVs. Writes PNGs into the report dir."""
import os, csv
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
D = "/workspace/MoDiff/docs/static_vs_dynamic_2026-07-16/data"
P = "/workspace/MoDiff/docs/static_vs_dynamic_2026-07-16"
DYN, STA, REF = "#4C78A8", "#F58518", "#888888"

def rd(name):
    p = f"{D}/{name}"
    if not os.path.exists(p): return None
    with open(p) as f: return list(csv.DictReader(f))

def save(fig, name):
    fig.tight_layout(); fig.savefig(f"{P}/{name}", dpi=120, bbox_inches="tight"); plt.close(fig); print(" ", name)

# ---- 1. e2e speed: dynamic vs static, grouped by precision ----
sp = rd("pipeline_speed.csv")
if sp:
    d = {r["mode"]: float(r["gpu_busy_ms_step"]) for r in sp}
    groups = [("fp16", "fp16"), ("int8", "int8"), ("int8_modiff", "int8+MoDiff"),
              ("int4", "int4"), ("int4_modiff", "int4+MoDiff")]
    x = np.arange(len(groups)); w = 0.38
    dyn = [d.get(f"dynamic_{g}", np.nan) for g, _ in groups]
    sta = [d.get(f"static_{g}", np.nan) for g, _ in groups]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - w/2, dyn, w, label="dynamic", color=DYN)
    ax.bar(x + w/2, sta, w, label="static", color=STA)
    if "fp32" in d: ax.axhline(d["fp32"], ls="--", c=REF, lw=1, label=f"fp32 {d['fp32']:.0f}")
    for i, (a, b) in enumerate(zip(dyn, sta)):
        if a == a: ax.text(i - w/2, a, f"{a:.0f}", ha="center", va="bottom", fontsize=8)
        if b == b: ax.text(i + w/2, b, f"{b:.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([n for _, n in groups]); ax.set_ylabel("GPU-busy ms/step")
    ax.set_title("E2E pipeline speed: dynamic vs static (lower is better)"); ax.legend()
    save(fig, "01_e2e_speed.png")

# ---- 2. peak memory ----
io = rd("pipeline_io.csv")
if io:
    d = {r["mode"]: float(r["peak_mem_MiB"]) for r in io}
    groups = [("fp16","fp16"),("int8","int8"),("int8_modiff","int8+MoDiff"),("int4","int4"),("int4_modiff","int4+MoDiff")]
    x = np.arange(len(groups)); w = 0.38
    dyn = [d.get(f"dynamic_{g}", np.nan) for g,_ in groups]; sta = [d.get(f"static_{g}", np.nan) for g,_ in groups]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - w/2, dyn, w, label="dynamic", color=DYN); ax.bar(x + w/2, sta, w, label="static", color=STA)
    ax.set_xticks(x); ax.set_xticklabels([n for _,n in groups]); ax.set_ylabel("peak MiB")
    ax.set_title("Peak memory: dynamic vs static"); ax.legend()
    save(fig, "02_peak_mem.png")

# ---- 3. softmax micro: static speedup per precision/shape ----
sm = rd("softmax_kernel.csv")
if sm:
    shapes = sorted({(int(r["BH"]), int(r["T"])) for r in sm}, key=lambda t: -t[1])
    precs = ["fp16", "int8", "int4"]
    fig, ax = plt.subplots(figsize=(9, 4.5)); x = np.arange(len(shapes)); w = 0.25
    for j, pr in enumerate(precs):
        sd = []
        for (BH, T) in shapes:
            row = [r for r in sm if r["precision"] == pr and int(r["T"]) == T]
            sd.append(float(row[0]["static_speedup"]) if row else np.nan)
        ax.bar(x + (j-1)*w, sd, w, label=pr)
        for i, v in enumerate(sd):
            if v == v: ax.text(x[i]+(j-1)*w, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax.axhline(1.0, ls="--", c=REF, lw=1); ax.set_xticks(x); ax.set_xticklabels([f"T={T}" for _, T in shapes])
    ax.set_ylabel("static / dynamic speedup"); ax.legend(title="precision")
    ax.set_title("Softmax kernel: static (1-pass) vs dynamic (2-pass) speedup — precision-independent")
    save(fig, "03_softmax_micro.png")

# ---- 4. attention micro ----
ak = rd("attn_kernel_speed.csv")
if ak:
    shapes = sorted({int(r["T"]) for r in ak}, reverse=True); precs = ["fp16", "int8", "int4"]
    fig, ax = plt.subplots(figsize=(9, 4.5)); x = np.arange(len(shapes)); w = 0.25
    for j, pr in enumerate(precs):
        sd = [next((float(r["static_speedup"]) for r in ak if r["precision"]==pr and int(r["T"])==T), np.nan) for T in shapes]
        ax.bar(x + (j-1)*w, sd, w, label=pr)
    ax.axhline(1.0, ls="--", c=REF, lw=1); ax.set_xticks(x); ax.set_xticklabels([f"T={T}" for T in shapes])
    ax.set_ylabel("static / dynamic speedup"); ax.legend(title="precision")
    ax.set_title("Full attention kernel: static vs dynamic speedup"); save(fig, "04_attn_micro.png")

# ---- 5. analytical IO ----
an = rd("pipeline_io_analytic.csv")
if an:
    precs = ["fp16", "int8", "int4"]; x = np.arange(len(precs)); w = 0.38
    dyn = [next((float(r["total_MiB"]) for r in an if r["precision"]==p and r["variant"]=="dynamic"), np.nan) for p in precs]
    sta = [next((float(r["total_MiB"]) for r in an if r["precision"]==p and r["variant"]=="static"), np.nan) for p in precs]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w/2, dyn, w, label="dynamic", color=DYN); ax.bar(x + w/2, sta, w, label="static", color=STA)
    ax.set_xticks(x); ax.set_xticklabels(precs); ax.set_ylabel("analytical DRAM MiB/step")
    ax.set_title("Analytical total IO: dynamic vs static"); ax.legend(); save(fig, "05_io_analytic.png")

# ---- 6. profile buckets (stacked) per mode ----
pf = rd("kernel_profile.csv")
if pf:
    modes = []
    for r in pf:
        if r["mode"] not in modes: modes.append(r["mode"])
    bset = []
    for r in pf:
        if r["bucket"] not in bset: bset.append(r["bucket"])
    M = {m: {b: 0.0 for b in bset} for m in modes}
    for r in pf: M[r["mode"]][r["bucket"]] = float(r["ms_step"])
    fig, ax = plt.subplots(figsize=(13, 5)); x = np.arange(len(modes)); bot = np.zeros(len(modes))
    cmap = plt.get_cmap("tab20")
    for i, b in enumerate(bset):
        vals = [M[m][b] for m in modes]
        ax.bar(x, vals, 0.7, bottom=bot, label=b, color=cmap(i % 20)); bot += np.array(vals)
    ax.set_xticks(x); ax.set_xticklabels([m.replace("dynamic_","dyn ").replace("static_","sta ") for m in modes], rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("GPU-busy ms/step"); ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.set_title("Per-kernel-bucket GPU time (torch.profiler), all modes"); save(fig, "06_profile_buckets.png")

# ---- 7. nsys measured per-kernel time: softmax dyn vs static (ncu HW counters blocked here) ----
kt = rd("kernel_timing_nsys.csv")
if kt:
    tt = {r["kernel"]: float(r["gpu_us"]) for r in kt}
    def gk(sub):
        return next((v for k, v in tt.items() if sub in k), np.nan)
    pairs = [("attn_softmax_requant_kernel", "attn_softmax_requant_static_kernel", "int8 softmax"),
             ("attn_softmax_requant4_kernel", "attn_softmax_requant4_static_kernel", "int4 softmax"),
             ("attn_softmax_fp16_dynamic", "attn_softmax_fp16_static", "fp16 softmax"),
             ("aq_qtok_kernel", "aq_qtok_static_kernel", "Q/K quantize")]
    labels = [p[2] for p in pairs]; x = np.arange(len(labels)); w = 0.38
    dyn = [gk(p[0]) for p in pairs]; sta = [gk(p[1]) for p in pairs]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar(x - w/2, dyn, w, label="dynamic", color=DYN); ax.bar(x + w/2, sta, w, label="static", color=STA)
    for i,(a,b) in enumerate(zip(dyn,sta)):
        if a==a: ax.text(i-w/2,a,f"{a:.0f}",ha="center",va="bottom",fontsize=7)
        if b==b: ax.text(i+w/2,b,f"{b:.0f}",ha="center",va="bottom",fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("measured kernel GPU µs (nsys, T=1024)")
    ax.set_title("Measured per-kernel time (nsys): dynamic vs static"); ax.legend()
    save(fig, "07_kernel_timing.png")

# ---- 8. nsys per-kernel buckets ----
ns = rd("nsys_kernels.csv")
if ns:
    modes = []
    for r in ns:
        if r["mode"] not in modes: modes.append(r["mode"])
    bset = []
    for r in ns:
        if r["bucket"] not in bset: bset.append(r["bucket"])
    M = {m: {b: 0.0 for b in bset} for m in modes}
    for r in ns: M[r["mode"]][r["bucket"]] = float(r["gpu_ms"])
    fig, ax = plt.subplots(figsize=(11, 5)); x = np.arange(len(modes)); bot = np.zeros(len(modes)); cmap = plt.get_cmap("tab10")
    for i, b in enumerate(bset):
        vals = [M[m][b] for m in modes]; ax.bar(x, vals, 0.6, bottom=bot, label=b, color=cmap(i % 10)); bot += np.array(vals)
    ax.set_xticks(x); ax.set_xticklabels([m.replace("dynamic_","dyn ").replace("static_","sta ") for m in modes], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("GPU ms (nsys, 1 sample)"); ax.legend(fontsize=8); ax.set_title("nsys per-kernel breakdown (representative modes)")
    save(fig, "08_nsys_buckets.png")

print("plots done")

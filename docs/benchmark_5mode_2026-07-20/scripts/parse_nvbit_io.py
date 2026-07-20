"""Parse nvbit_io_raw.txt (from run_nvbit_io.sh) -> measured DRAM read/write bytes per config and
per kernel. Writes:
  data/nvbit_io_total.csv     - family, mode, shape, read_MiB, write_MiB, total_MiB  (per config)
  data/nvbit_io_perkernel.csv - family, mode, shape, kernel, read_MiB, write_MiB     (per kernel)
"""
import os, csv, re
os.chdir("/workspace/MoDiff")
DATA = "docs/benchmark_5mode_2026-07-20/data"
RAW = f"{DATA}/nvbit_io_raw.txt"
MiB = 1024.0 ** 2
LINE = re.compile(r"read=(\d+)\s+write=(\d+)\s+blocks=(\d+)\s+kernel=(.*)")


def short(k):
    if "ImplicitGemmConvolution" in k or "cutlass" in k.lower(): return "cutlass_conv"
    m = re.search(r"(gemm_w8a8_awq\w*|gemm_w4a4_awq\w*|flash_attn_int8\w*|flash_attn_int4\w*|"
                  r"quantize_attn|aq_\w+|group_norm\w*|scale_bias\w*|scale_quantize\w*|"
                  r"static_quantize\w*|scaled_dot|softmax\w*|elementwise_kernel|vectorized_elementwise|"
                  r"FillFunctor|direct_copy|CatArray|upsample|CUDAFunctor_add)", k)
    return m.group(1) if m else k.split("(")[0][:40]


if not os.path.exists(RAW):
    print(f"MISSING {RAW} — run run_nvbit_io.sh first"); raise SystemExit(0)

totals, perk = [], []
cur = None; rd = wr = 0; kd = {}
def flush():
    global rd, wr, kd
    if cur is None: return
    fam, mode, shape = (cur.split("|") + ["", "", ""])[:3]
    totals.append(dict(family=fam, mode=mode, shape=shape,
                       read_MiB=round(rd / MiB, 3), write_MiB=round(wr / MiB, 3),
                       total_MiB=round((rd + wr) / MiB, 3)))
    for kn, (r, w) in sorted(kd.items(), key=lambda x: -(x[1][0] + x[1][1])):
        perk.append(dict(family=fam, mode=mode, shape=shape, kernel=kn,
                         read_MiB=round(r / MiB, 3), write_MiB=round(w / MiB, 3)))

for ln in open(RAW):
    ln = ln.rstrip("\n")
    if ln.startswith("### TAG "):
        flush(); cur = ln[len("### TAG "):].strip(); rd = wr = 0; kd = {}
    elif ln.startswith("MEMBYTES "):
        m = LINE.search(ln)
        if not m: continue
        r, w = int(m.group(1)), int(m.group(2)); kn = short(m.group(4))
        rd += r; wr += w
        pr, pw = kd.get(kn, (0, 0)); kd[kn] = (pr + r, pw + w)
flush()

if totals:
    with open(f"{DATA}/nvbit_io_total.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(totals[0].keys())); w.writeheader(); w.writerows(totals)
    with open(f"{DATA}/nvbit_io_perkernel.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(perk[0].keys())); w.writeheader(); w.writerows(perk)
    print(f"WROTE nvbit_io_total.csv ({len(totals)} configs), nvbit_io_perkernel.csv ({len(perk)} kernels)\n")
    print(f"{'family':7} {'mode':5} {'shape':16} {'read MiB':>9} {'write MiB':>9} {'total':>9}")
    for r in totals:
        print(f"{r['family']:7} {r['mode']:5} {r['shape']:16} {r['read_MiB']:9.2f} {r['write_MiB']:9.2f} {r['total_MiB']:9.2f}")

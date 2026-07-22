"""Combine the three per-kernel benchmarks into one 'time in one step' summary: for each kernel
family (conv / linear qkv+proj / attention-block incl GN+quantize) the total per-DDIM-step time
(= sum over every shape of count_per_step x us/call) in each of the 5 modes.

Linear and attention have no modiff variant (static W/A quant in every mode), so
int8_baseline == int8_modiff and int4_baseline == int4_modiff for those families. These are
standalone kernel measurements (each op run back-to-back); they are NOT summed to equal the e2e
wall time (that is measured independently in e2e_speed / e2e_timing_profile) — they show where the
per-step kernel work goes and how each family scales with precision. Writes data/perstep_summary.csv.
"""
import os, csv
os.chdir("/workspace/MoDiff")
HERE = "docs/benchmark_5mode_2026-07-21"
D = f"{HERE}/data"
MODES = ["fp16", "int8_baseline", "int4_baseline", "int8_modiff", "int4_modiff"]


def load(name):
    with open(f"{D}/{name}") as f:
        return list(csv.DictReader(f))


def fnum(x):
    try: return float(x)
    except (ValueError, TypeError): return 0.0


rows = []

# --- conv: per-mode TOTAL row already has *_us_per_step ---
conv = load("conv_kernel_speed.csv")
ct = next(r for r in conv if r["Cin"] == "TOTAL_PER_STEP")
rows.append(dict(family="conv (all 89 convs/step)", **{m: round(fnum(ct[f"{m}_us_per_step"]) / 1000, 3) for m in MODES}))

# --- linear qkv/proj + time-embed: fp16 / int8 / int4 (baseline==modiff) ---
lin = load("linear_kernel_speed.csv")
lt = next(r for r in lin if r["role"] == "TOTAL_PER_STEP")
lf, l8, l4 = fnum(lt["fp16_us"]) / 1000, fnum(lt["int8_full_us"]) / 1000, fnum(lt["int4_full_us"]) / 1000
rows.append(dict(family="linear qkv/proj+temb (79/step)", fp16=round(lf, 3),
                 int8_baseline=round(l8, 3), int4_baseline=round(l4, 3),
                 int8_modiff=round(l8, 3), int4_modiff=round(l4, 3)))

# --- attention block (GN+quantize+attn): fp16 / int8 / int4 (baseline==modiff) ---
at = load("attn_kernel_speed.csv")
att = next(r for r in at if r["C"] == "TOTAL_PER_STEP")
af, a8, a4 = (fnum(att["fp16_us_per_step"]) / 1000, fnum(att["int8_us_per_step"]) / 1000,
             fnum(att["int4_us_per_step"]) / 1000)
rows.append(dict(family="attention block incl GN+quant (21/step)", fp16=round(af, 3),
                 int8_baseline=round(a8, 3), int4_baseline=round(a4, 3),
                 int8_modiff=round(a8, 3), int4_modiff=round(a4, 3)))

# --- sum of standalone kernel families (NOT the e2e wall) ---
tot = {m: sum(r.get(m, 0.0) for r in rows) for m in MODES}
rows.append(dict(family="SUM of standalone kernels", **{m: round(tot[m], 3) for m in MODES}))

cols = ["family"] + MODES
with open(f"{D}/perstep_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)

print(f"{'family':40} " + " ".join(f"{m:>14}" for m in MODES))
for r in rows:
    print(f"{r['family']:40} " + " ".join(f"{r.get(m,0.0):12.2f}ms" for m in MODES))
print(f"\nWROTE {D}/perstep_summary.csv")

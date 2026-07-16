import csv, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
D = "/workspace/MoDiff/docs/comprehensive_benchmark_2026-07-16"
DATA = D + "/data"
def rd(f):
    with open(f"{DATA}/{f}") as fh: return list(csv.DictReader(fh))
def fnum(x):
    try: return float(x)
    except: return float("nan")
PEAK_BW = 696
C_FP16, C_INT8, C_INT4, C_MOD8, C_MOD4 = "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"

# 1. pipeline speed
r = rd("pipeline_speed.csv"); labels=[x["label"] for x in r]
wall=[fnum(x["wall_ms_step"]) for x in r]; gpu=[fnum(x["gpu_busy_ms_step"]) for x in r]
x=np.arange(len(labels)); w=0.38
fig,ax=plt.subplots(figsize=(9,5))
ax.bar(x-w/2,wall,w,label="wall-clock",color=C_FP16); ax.bar(x+w/2,gpu,w,label="GPU-busy",color=C_INT8)
for i,(a,b) in enumerate(zip(wall,gpu)):
    ax.text(i-w/2,a+0.3,f"{a:.1f}",ha="center",fontsize=8); ax.text(i+w/2,b+0.3,f"{b:.1f}",ha="center",fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("ms / DDIM step")
ax.set_title("Pipeline speed — churches LDM UNet (batch 32, A40)\nlower is better; gap = launch/scheduling overhead")
ax.legend(); ax.grid(axis="y",alpha=0.3); fig.tight_layout(); fig.savefig(f"{D}/01_pipeline_speed.png",dpi=120); plt.close()

# 2. pipeline total IO usage (analytical DRAM bytes/step), stacked conv/linear/attn
r=[x for x in rd("pipeline_io_analytic.csv") if x["precision"] in ("fp32","fp16","int8","int4")]; labels=[x["precision"] for x in r]
conv=[fnum(x["conv_MiB_step"]) for x in r]; lin=[fnum(x["linear_MiB_step"]) for x in r]
att=[fnum(x["attn_MiB_step"]) for x in r]; tot=[fnum(x["total_MiB_step"]) for x in r]
x=np.arange(len(labels)); w=0.55
fig,ax=plt.subplots(figsize=(9.5,5.5))
ax.bar(x,conv,w,label="conv (in+weight quantized, fp16 out)",color=C_INT4)
ax.bar(x,lin,w,bottom=conv,label="qkv/proj linear (fp16)",color=C_INT8)
ax.bar(x,att,w,bottom=np.array(conv)+np.array(lin),label="attention SDPA scores (fp16, dtype-invariant)",color=C_FP16)
fp16_conv=conv[labels.index("fp16")] if "fp16" in labels else conv[0]
for i in range(len(labels)):
    ax.text(i,tot[i]+120,f"{tot[i]:.0f}",ha="center",fontsize=9,weight="bold")
    ax.text(i,conv[i]/2,f"{conv[i]:.0f}",ha="center",va="center",fontsize=7,color="white")
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("analytical DRAM MiB / DDIM step")
ax.set_title("Pipeline total IO usage — analytical DRAM bytes per step (batch 32)\n"
             "conv IO drops with quantization (int8 0.64×, int4 0.45×); fp16 attention scores dominate the total")
ax.legend(fontsize=8); ax.grid(axis="y",alpha=0.3); fig.tight_layout(); fig.savefig(f"{D}/02_pipeline_io.png",dpi=120); plt.close()

# 3. kernel profile stacked
r=rd("kernel_profile.csv"); modes=["fp32","fp16","int8_baseline","int8","int4_baseline","int4"]
mlab={"fp32":"fp32","fp16":"fp16","int8_baseline":"int8 base","int8":"int8 modiff","int4_baseline":"int4 base","int4":"int4 modiff"}
buckets=["conv (GEMM)","conv store epilogue","GroupNorm","attention (softmax + SDPA)","GEMM (qkv/proj + attn QK·AV)","quantize / MoDiff delta","elementwise / copy","upsample / concat","other"]
data={m:{b:0.0 for b in buckets} for m in modes}
for x in r:
    if x["bucket"] in data[x["mode"]]: data[x["mode"]][x["bucket"]]=fnum(x["ms_step"])
cols=plt.cm.tab10(np.linspace(0,1,len(buckets)))
fig,ax=plt.subplots(figsize=(10,6)); xs=np.arange(len(modes)); bottom=np.zeros(len(modes))
for bi,b in enumerate(buckets):
    vals=[data[m][b] for m in modes]; ax.bar(xs,vals,0.6,bottom=bottom,label=b,color=cols[bi]); bottom+=np.array(vals)
for i in range(len(modes)): ax.text(i,bottom[i]+0.3,f"{bottom[i]:.1f}",ha="center",fontsize=9,weight="bold")
ax.set_xticks(xs); ax.set_xticklabels([mlab[m] for m in modes]); ax.set_ylabel("GPU-busy ms / step")
ax.set_title("Kernel profile — per-operation GPU time by mode")
ax.legend(fontsize=8,ncol=2,loc="upper left"); ax.grid(axis="y",alpha=0.3); fig.tight_layout(); fig.savefig(f"{D}/03_kernel_profile.png",dpi=120); plt.close()

# 4. conv speed top10
r=sorted(rd("kernel_conv_speed.csv"),key=lambda x:-fnum(x["fp16_us"]))[:10]
labels=[f"{x['Cin']}→{x['Cout']} {x['k']}×{x['k']} {x['H']}²" for x in r]
fp16=[fnum(x["fp16_us"]) for x in r]; i8=[fnum(x["int8_us"]) for x in r]; i4=[fnum(x["int4_us"]) for x in r]
x=np.arange(len(labels)); w=0.27
fig,ax=plt.subplots(figsize=(12,5.5))
ax.bar(x-w,fp16,w,label="fp16 (cuDNN)",color=C_FP16); ax.bar(x,i8,w,label="int8",color=C_INT8); ax.bar(x+w,i4,w,label="int4",color=C_INT4)
ax.set_xticks(x); ax.set_xticklabels(labels,rotation=30,ha="right",fontsize=8); ax.set_ylabel("µs")
ax.set_title("Conv kernel latency — top 10 shapes by cost (base modes)")
ax.legend(); ax.grid(axis="y",alpha=0.3); fig.tight_layout(); fig.savefig(f"{D}/04_kernel_conv_speed.png",dpi=120); plt.close()

# 5. conv base vs modiff
r=[x for x in rd("kernel_conv_speed.csv") if x.get("int8_modiff_us")]
labels=[f"{x['Cin']}→{x['Cout']} {x['k']}×{x['k']} {x['H']}²" for x in r]
i8=[fnum(x["int8_us"]) for x in r]; i8m=[fnum(x["int8_modiff_us"]) for x in r]
i4=[fnum(x["int4_us"]) for x in r]; i4m=[fnum(x.get("int4_modiff_us","nan")) for x in r]
x=np.arange(len(labels)); w=0.2
fig,ax=plt.subplots(figsize=(11,5.5))
ax.bar(x-1.5*w,i8,w,label="int8 base",color=C_INT8); ax.bar(x-0.5*w,i8m,w,label="int8 MoDiff",color=C_MOD8)
ax.bar(x+0.5*w,i4,w,label="int4 base",color=C_INT4); ax.bar(x+1.5*w,i4m,w,label="int4 MoDiff",color=C_MOD4)
ax.set_xticks(x); ax.set_xticklabels(labels,rotation=25,ha="right",fontsize=8); ax.set_ylabel("µs")
ax.set_title("Conv kernel: base vs MoDiff temporal path\nMoDiff adds sub/accumulate + delta-quantize; skips no conv, so slower")
ax.legend(); ax.grid(axis="y",alpha=0.3); fig.tight_layout(); fig.savefig(f"{D}/05_kernel_conv_modiff.png",dpi=120); plt.close()

# 6. kernel IO — total amount (bytes moved) per kernel
rc=sorted(rd("kernel_conv_io.csv"),key=lambda x:-fnum(x["fp16_MiB"]))[:8]; rl=rd("kernel_linear_io.csv")
labels=[f"c:{x['Cin']}→{x['Cout']} {x['k']}×{x['k']} {x['H']}²" for x in rc]+[f"L:{x['role']} {x['C']}→{x['Cout']}" for x in rl]
fp16=[fnum(x["fp16_MiB"]) for x in rc]+[fnum(x["fp16_MiB"]) for x in rl]
i8=[fnum(x["int8_MiB"]) for x in rc]+[fnum(x["int8_MiB"]) for x in rl]
i4=[fnum(x["int4_MiB"]) for x in rc]+[float("nan")]*len(rl)
x=np.arange(len(labels)); w=0.27
fig,ax=plt.subplots(figsize=(13,5.5))
ax.bar(x-w,fp16,w,label="fp16",color=C_FP16); ax.bar(x,i8,w,label="int8",color=C_INT8); ax.bar(x+w,i4,w,label="int4",color=C_INT4)
ax.set_xticks(x); ax.set_xticklabels(labels,rotation=35,ha="right",fontsize=7); ax.set_ylabel("total IO / call (MiB)")
ax.set_title("Kernel IO — total DRAM bytes moved per call (conv + linear)\nin+weight shrink with dtype; conv output stays fp16")
ax.legend(); ax.grid(axis="y",alpha=0.3); fig.tight_layout(); fig.savefig(f"{D}/06_kernel_io.png",dpi=120); plt.close()

# 7. attention (GN+qkv baseline vs fused; math SDPA)
r=rd("kernel_attn.csv"); labels=[f"C{x['C']} {x['HxW']} (T={x['T']})" for x in r]
base=[fnum(x["gn+qkv_base_us"]) for x in r]
fused=[fnum(x["gn+qkv_fused_us"]) if x["gn+qkv_fused_us"] not in ("","n/a(T%128)") else float("nan") for x in r]
math=[fnum(x["sdpa_math_us"]) for x in r]
x=np.arange(len(labels)); w=0.27
fig,ax=plt.subplots(figsize=(11,5.5))
ax.bar(x-w,base,w,label="GN+qkv baseline (GN+cuBLAS)",color=C_FP16)
ax.bar(x,fused,w,label="GN+qkv fused (CUTLASS)",color=C_INT8)
ax.bar(x+w,math,w,label="SDPA (math backend)",color=C_INT4)
for i,(b_,f_) in enumerate(zip(base,fused)):
    if f_==f_ and f_>0: ax.text(i,f_+8,f"{b_/f_:.2f}×",ha="center",fontsize=7)
ax.set_xticks(x); ax.set_xticklabels(labels,fontsize=8); ax.set_ylabel("µs")
ax.set_title("Attention kernels — GN→qkv (baseline vs fused CUTLASS) and math SDPA\nSDPA runs on the math backend so QKᵀ/AV are interceptable cuBLAS GEMMs")
ax.legend(fontsize=8); ax.grid(axis="y",alpha=0.3); fig.tight_layout(); fig.savefig(f"{D}/07_kernel_attn.png",dpi=120); plt.close()
print("wrote plots")
for p in sorted(os.listdir(D)):
    if p.endswith(".png"): print(" ",p)

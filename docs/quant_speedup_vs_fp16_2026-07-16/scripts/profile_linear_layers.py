"""Per-layer Linear profile: decompose each quantized Linear into quantize-pass vs int-GEMM, and
compare to the fp16 cuBLAS GEMM, split by layer class (tiny-M time-embed vs large-M qkv/proj).
Captures the REAL layer shapes + runtime M from the churches UNet (via forward hooks on the int8
model), then microbenchmarks each unique (M,K,N). Emits data/linear_layer_profile.csv.
Batch via E2E_BATCH (default 64)."""
import os, sys, csv, math
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
import torch, torch.nn.functional as F
import integration.benchmarks.benchmark_ldm as B
from integration.kernels.wxax_linear import QuantLinearWxAx
import modiff_cutlass as mc

BATCH = int(os.environ.get("E2E_BATCH", "64"))
OUT = f"docs/quant_speedup_vs_fp16_2026-07-16/data/linear_layer_profile_b{BATCH}.csv"


def bench(fn, it=100, warm=30, reps=5):
    ts = []
    for _ in range(reps):
        for _ in range(warm): fn()
        torch.cuda.synchronize(); s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(it): fn()
        e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e) / it * 1e3)  # us
    ts.sort(); return ts[len(ts) // 2]


def pack4(q):
    q = q.to(torch.int8); lo = q[..., 0::2] & 0xF; hi = q[..., 1::2] & 0xF
    return (lo | (hi << 4)).to(torch.int8).contiguous()


# --- 1. capture real layers (name, in, out, M) via hooks on the int8 model ---
os.environ["MODIFF_QUANT_LINEAR"] = "1"
runner = B.BenchmarkRunner("configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
                           "models/ldm/lsun_churches256/model.ckpt",
                           output_dir="integration/results/lin_profile", batch_size=BATCH,
                           steps=2, shape=(4, 32, 32),
                           calibration_path="integration/calibration/int8_calibration.pt",
                           linear_backend="int_gemm")
model, sampler = runner._setup_model("int8")
seen = {}
hooks = []
def mk_hook(name):
    def h(mod, args):
        if name in seen: return
        x = args[0]
        M = x.reshape(-1, mod.in_features).shape[0]
        seen[name] = (M, mod.in_features, mod.out_features)
    return h
for nm, m in model.model.diffusion_model.named_modules():
    if isinstance(m, QuantLinearWxAx):
        hooks.append(m.register_forward_pre_hook(mk_hook(nm)))
cond = runner._cond_kwargs(model, BATCH)
with torch.inference_mode(), torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
    sampler.sample(S=1, batch_size=BATCH, shape=runner.shape, eta=0.0, verbose=False, **cond)
for h in hooks: h.remove()
print(f"captured {len(seen)} Linear layers at batch {BATCH}")
del model, sampler; torch.cuda.empty_cache()

# --- 2. group by unique (M,K,N), count instances, classify ---
from collections import Counter, defaultdict
shape_count = Counter((M, K, N) for (M, K, N) in seen.values())
names_by_shape = defaultdict(list)
for nm, (M, K, N) in seen.items():
    names_by_shape[(M, K, N)].append(nm)

def classify(M, names):
    # tiny-M = M==batch (time_embed / emb_layers, no token multiplication); else large-M (qkv/proj)
    return "tiny-M (M=batch)" if M <= BATCH else "large-M (qkv/proj)"

# --- 3. microbench each unique shape ---
rows = []
print(f"{'M,K,N':>18} {'cnt':>3} {'class':>18} | {'fp16':>7} {'i8_quant':>8} {'i8_gemm':>7} {'i8_tot':>7} {'i4_quant':>8} {'i4_gemm':>7} {'i4_tot':>7}")
for (M, K, N), cnt in sorted(shape_count.items(), key=lambda kv: -kv[0][0]):
    cls = classify(M, names_by_shape[(M, K, N)])
    x = torch.randn(M, K, device="cuda", dtype=torch.float16)
    Wf = torch.randn(N, K, device="cuda", dtype=torch.float16)
    asc = x.abs().max().item() / 127.0
    # fp16 cuBLAS GEMM
    t_fp16 = bench(lambda: F.linear(x, Wf))
    # int8 port: quantize pass (K padded to %64) + gemm (N padded to %128)
    Kp8 = ((K + 63) // 64) * 64; Np = ((N + 127) // 128) * 128
    x8 = F.pad(x, (0, Kp8 - K)).contiguous()
    Wq8 = torch.randint(-127, 127, (Np, Kp8), device="cuda", dtype=torch.int8)
    ws8 = (torch.randn(Np, device="cuda").abs().float() / 127)
    t_i8q = bench(lambda: mc.quantize_act_int8(x8, asc))
    xq8 = mc.quantize_act_int8(x8, asc)
    t_i8g = bench(lambda: mc.gemm_w8a8_awq(xq8, Wq8, ws8, asc))
    # int4 port: quantize+pack (K padded to %128) + gemm
    Kp4 = ((K + 127) // 128) * 128
    x4 = F.pad(x, (0, Kp4 - K)).contiguous()
    Wq4 = pack4(torch.randint(-7, 7, (Np, Kp4), device="cuda", dtype=torch.int8))
    ws4 = (torch.randn(Np, device="cuda").abs().float() / 7)
    asc4 = asc / 18
    t_i4q = bench(lambda: mc.quantize_act_int4_pack(x4, asc4))
    xq4 = mc.quantize_act_int4_pack(x4, asc4)
    t_i4g = bench(lambda: mc.gemm_w4a4_awq(xq4, Wq4, ws4, asc4, Kp4))
    i8_tot, i4_tot = t_i8q + t_i8g, t_i4q + t_i4g
    print(f"{f'{M},{K},{N}':>18} {cnt:>3} {cls:>18} | {t_fp16:7.2f} {t_i8q:8.2f} {t_i8g:7.2f} {i8_tot:7.2f} {t_i4q:8.2f} {t_i4g:7.2f} {i4_tot:7.2f}")
    rows.append(dict(M=M, K=K, N=N, count=cnt, cls=cls,
                     fp16_us=round(t_fp16, 3), i8_quant_us=round(t_i8q, 3), i8_gemm_us=round(t_i8g, 3),
                     i8_total_us=round(i8_tot, 3), i4_quant_us=round(t_i4q, 3), i4_gemm_us=round(t_i4g, 3),
                     i4_total_us=round(i4_tot, 3)))

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

# --- 4. aggregate per-step (sum over all layer instances = count) ---
def agg(key):
    return sum(r[key] * r["count"] for r in rows)
def agg_cls(key, cls):
    return sum(r[key] * r["count"] for r in rows if r["cls"] == cls)
print("\n=== PER-STEP TOTALS (sum over all Linear instances) ===")
for cls in ("tiny-M (M=batch)", "large-M (qkv/proj)", "ALL"):
    sel = (lambda k: agg(k)) if cls == "ALL" else (lambda k, c=cls: agg_cls(k, c))
    print(f"{cls:>20}: fp16={sel('fp16_us'):8.1f}us | int8 quant={sel('i8_quant_us'):7.1f} gemm={sel('i8_gemm_us'):7.1f} tot={sel('i8_total_us'):7.1f} | "
          f"int4 quant={sel('i4_quant_us'):7.1f} gemm={sel('i4_gemm_us'):7.1f} tot={sel('i4_total_us'):7.1f}")
print(f"\nWROTE {OUT}")

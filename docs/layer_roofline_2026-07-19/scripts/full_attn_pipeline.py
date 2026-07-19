"""Full attention-block pipeline timing: GroupNorm -> qkv -> QKᵀ/softmax/AV -> proj -> residual,
as a real TokenMajorAttentionBlock, for fp16 / int8 / int4 (qkv/proj quantized; attention is fp16 SDPA
in all -- as established). Shows the FULL block time per precision (not just the bare GEMM). Real churches
attention shapes at batch 128. Writes data/full_attn_pipeline_b128.csv."""
import os, sys, csv
os.chdir("/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff"); sys.path.insert(0, "/workspace/MoDiff/src/taming-transformers")
import torch, torch.nn as nn
from ldm.modules.diffusionmodules.openaimodel import AttentionBlock
from integration.fused_ops.token_major_attention import TokenMajorAttentionBlock
from integration.kernels.wxax_linear import convert_linears_to_wxax, set_wxax_calibrating, finalize_wxax_ascale

BATCH = 128


def bench(fn, it=50, warm=20, reps=5):
    ts = []
    for _ in range(reps):
        for _ in range(warm): fn()
        torch.cuda.synchronize(); s = torch.cuda.Event(True); e = torch.cuda.Event(True); s.record()
        for _ in range(it): fn()
        e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e) / it * 1e3)
    ts.sort(); return ts[len(ts) // 2]


def make_block(C, S, bits):
    torch.manual_seed(0)
    # match the real pipeline: quant modes force MODIFF_FUSE_GN_QKV=0 (fp16 fused_gn_qkv is
    # incompatible with a QuantLinearWxAx qkv); fp16 keeps it on.
    os.environ["MODIFF_FUSE_GN_QKV"] = "0" if bits in (8, 4) else "1"
    ab = AttentionBlock(C, num_heads=8, use_new_attention_order=False).cuda().half().eval()
    with torch.no_grad():   # proj_out is zero-init -> randomize so attention isn't a no-op
        nn.init.normal_(ab.qkv.weight, std=0.05); nn.init.normal_(ab.qkv.bias, std=0.05)
        nn.init.normal_(ab.proj_out.weight, std=0.05); nn.init.normal_(ab.proj_out.bias, std=0.05)
    blk = TokenMajorAttentionBlock(ab).cuda().eval()
    if bits in (8, 4):
        convert_linears_to_wxax(blk, bits=bits, modiff=False)
        xc = torch.randn(4, C, S, S, device="cuda", dtype=torch.float16).to(memory_format=torch.channels_last)
        set_wxax_calibrating(blk, True)
        with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.float16):
            for _ in range(5): blk(xc)
        finalize_wxax_ascale(blk)
    return blk


SHAPES = [("32² C192", 192, 32), ("16² C384", 384, 16), ("8² C384", 384, 8), ("4² C768", 768, 4)]
rows = []
print(f"full attention BLOCK (GN + qkv + QKᵀ/softmax/AV + proj + residual), batch {BATCH}, us/call")
print(f"{'shape':10s} {'T':>5} | {'fp16':>9} {'int8':>9} {'int4':>9} | {'int8/fp16':>9} {'int4/fp16':>9}")
for (nm, C, S) in SHAPES:
    x = torch.randn(BATCH, C, S, S, device="cuda", dtype=torch.float16).to(memory_format=torch.channels_last)
    t = {}
    for bits, key in [(0, "fp16"), (8, "int8"), (4, "int4")]:
        blk = make_block(C, S, bits)
        with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.float16):
            t[key] = bench(lambda: blk(x))
        del blk; torch.cuda.empty_cache()
    rows.append(dict(shape=nm, C=C, T=S * S, fp16_us=round(t["fp16"], 1), int8_us=round(t["int8"], 1),
                     int4_us=round(t["int4"], 1), int8_vs_fp16=round(t["fp16"] / t["int8"], 3),
                     int4_vs_fp16=round(t["fp16"] / t["int4"], 3)))
    r = rows[-1]
    print(f"{nm:10s} {S*S:>5} | {r['fp16_us']:9.1f} {r['int8_us']:9.1f} {r['int4_us']:9.1f} | "
          f"{r['int8_vs_fp16']:8.2f}x {r['int4_vs_fp16']:8.2f}x")
with open("docs/layer_roofline_2026-07-19/data/full_attn_pipeline_b128.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
# per-step totals (weighted by block counts: C192 x5, C384 x5+5, C768 x5+1 ~ use measured counts)
CNT = {"32² C192": 5, "16² C384": 5, "8² C384": 5, "4² C768": 5}
tot = {k: sum(r[f"{k}_us"] * CNT.get(r["shape"], 1) for r in rows) / 1000 for k in ("fp16", "int8", "int4")}
print(f"\nweighted full-attention total/step (approx): fp16 {tot['fp16']:.1f} | int8 {tot['int8']:.1f} ({tot['fp16']/tot['int8']:.2f}x) | int4 {tot['int4']:.1f} ({tot['fp16']/tot['int4']:.2f}x) ms")
print("WROTE full_attn_pipeline_b128.csv")

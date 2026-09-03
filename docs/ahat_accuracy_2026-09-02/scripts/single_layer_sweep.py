"""Single layer, random input: sweep a_hat bit-width x block size, including B < 32.

The real i4 kernel is B=32 only (ahat_b32_update2_i4 hardcodes gi=i>>5, a 16-lane reduction and
threadIdx.x&15), so the sweep is done with a PyTorch model of the storage quantizer that is first
VALIDATED against the real kernel at B=32 -- the same way the fp16-storage simulation was
validated against the real int8 arm before it was trusted.

Recurrence per step (identical to the kernel):
    o_t      = silu(gn(x_t))                       # fp16 round before silu, as the kernel does
    d_t      = o_t - dequant(a_hat_{t-1})
    q_t      = clamp(round(d_t * s), -DLIM, DLIM)
    consumed = dequant(a_hat_{t-1}) + q_t/s
    a_hat_t  = blockwise_quantize(consumed, B, bits)   # fresh per-block amax, as the kernel does

Metrics: eta_cum (accumulated storage error, the one that predicts E2E), state, codes, consumed.
Two input trajectories, because a_hat is a temporal accumulator and the answer can depend on how
fast the input moves:
    iid    x_t drawn independently each step        (what "random input" literally means)
    walk   x_{t+1} = sqrt(1-a)x_t + sqrt(a)eps      per-step relative change matched to the
                                                    real captured trajectory
"""
import os, sys, json, math
ROOT="/workspace/MoDiff"; os.chdir(ROOT); sys.path[:0]=[ROOT]
import torch
import modiff_cutlass as mc

DEV, CL = "cuda", torch.channels_last
C, H, W, N, G, T = 384, 16, 16, 4, 32, 49
DLIM = 127.0
ef = torch.empty(0, device=DEV, dtype=torch.float32)
eh = torch.empty(0, device=DEV, dtype=torch.float16)
ei = torch.empty(0, device=DEV, dtype=torch.int32)
torch.manual_seed(0)
gw = torch.randn(C, device=DEV, dtype=torch.float16)
gb = torch.randn(C, device=DEV, dtype=torch.float16)
S_DELTA = 8.0

def gn_silu(x):
    xg = x.float().view(N, G, C//G, H, W)
    mu = xg.mean(dim=(2,3,4), keepdim=True)
    var = xg.var(dim=(2,3,4), unbiased=False, keepdim=True)
    n = ((xg-mu)*(var+1e-6).rsqrt()).view(N, C, H, W)
    n = n*gw.float().view(1,C,1,1) + gb.float().view(1,C,1,1)
    n = n.half().float()
    return n*torch.sigmoid(n)

def blk_quant(v, B, bits):
    """Blockwise symmetric quantize+dequantize along C, fresh per-block amax. Returns (deq, sat)."""
    lim = float(2**(bits-1) - 1)
    x = v.permute(0,2,3,1).reshape(N, H, W, C//B, B)
    s = x.abs().amax(-1, keepdim=True).clamp_min(1e-12) / lim
    q = (x/s).round().clamp_(-lim, lim)
    sat = (q.abs() == lim).float().mean().item()
    return (q*s).reshape(N,H,W,C).permute(0,3,1,2).contiguous(), sat

def traj(kind, alpha):
    torch.manual_seed(7)
    xs, x = [], torch.randn(N, C, H, W, device=DEV, dtype=torch.float16)
    for t in range(T):
        if kind == "iid":
            x = torch.randn(N, C, H, W, device=DEV, dtype=torch.float16)
        else:
            x = (math.sqrt(1-alpha)*x.float()
                 + math.sqrt(alpha)*torch.randn(N,C,H,W,device=DEV)).half()
        xs.append(x.clone())
    return xs

def run(xs, B, bits):
    a = torch.zeros(N, C, H, W, device=DEV, dtype=torch.float32)     # dequantized a_hat
    ref = torch.zeros(N, C, H, W, device=DEV, dtype=torch.float32)   # exact-a_hat reference
    eta = torch.zeros(N, C, H, W, device=DEV, dtype=torch.float32)
    per = []
    for t in range(T):
        o = gn_silu(xs[t])
        qr = torch.clamp(torch.round((o-ref)*S_DELTA), -DLIM, DLIM); cr = ref + qr/S_DELTA
        q  = torch.clamp(torch.round((o-a)*S_DELTA), -DLIM, DLIM);   cons = a + q/S_DELTA
        if bits >= 16:
            new, sat = cons.half().float(), 0.0
        else:
            new, sat = blk_quant(cons, B, bits)
        eta += new - cons
        per.append({"t":t, "eta_cum":(eta.norm()/cr.norm()).item(),
                    "eta_step":((new-cons).norm()/cr.norm()).item(),
                    "consumed":((cons-cr).norm()/cr.norm()).item(),
                    "state":((new-cr).norm()/cr.norm()).item(),
                    "codes":(q!=qr).float().mean().item(), "sat":sat})
        a, ref = new, cr
    return per

# ---------- validate the PyTorch storage model against the REAL kernel at B=32 ----------
def kernel_state(bits, B, xs):
    chan = C//2 if bits == 4 else C
    A = torch.empty(N, chan, H, W, device=DEV, dtype=torch.int8, memory_format=CL).zero_()
    Sq = torch.ones(N, H, W, C//B, device=DEV, dtype=torch.float32)
    sc = torch.full((1,), S_DELTA, device=DEV, dtype=torch.float32)
    for t in range(T):
        mc.group_norm_silu_delta_quantize_nhwc(
            xs[t].contiguous(memory_format=CL), gw, gb, A, G, 1e-6, True, sc, ef, eh, eh,
            ef, ef, ef, ei, DLIM, False, 1.0, False, True, Sq)
    if bits == 4:
        by = A.permute(0,2,3,1).contiguous().reshape(-1).to(torch.uint8)
        lo = (by & 0xF).to(torch.int16); lo = torch.where(lo>7, lo-16, lo)
        hi = ((by>>4)&0xF).to(torch.int16); hi = torch.where(hi>7, hi-16, hi)
        qq = torch.stack([lo,hi],-1).reshape(N,H,W,C).float()
    else:
        qq = A.permute(0,2,3,1).float()
    return (qq.view(N,H,W,C//B,B)*Sq.view(N,H,W,C//B,1)).reshape(N,H,W,C).permute(0,3,1,2)

out = {}
for kind, alpha in (("walk", 0.02), ("iid", None)):
    xs = traj(kind, alpha)
    # validation
    val = {}
    for bits in (8, 4):
        k = kernel_state(bits, 32, xs)
        p = run(xs, 32, bits)
        # replay the python model's final state for comparison
        a = torch.zeros(N,C,H,W,device=DEV,dtype=torch.float32)
        for t in range(T):
            o = gn_silu(xs[t])
            q = torch.clamp(torch.round((o-a)*S_DELTA), -DLIM, DLIM)
            a, _ = blk_quant(a + q/S_DELTA, 32, bits)
        val[f"{bits}bit"] = ((a-k).norm()/k.norm()).item()
    rows = {}
    for bits, Bs in ((8, [32]), (4, [2,4,8,16,32,64])):
        for B in Bs:
            rows[f"{bits}bit B={B}"] = run(xs, B, bits)
    rows["fp16"] = run(xs, 32, 16)
    out[kind] = {"validation_relL2_vs_real_kernel_B32": val, "arms": rows}
    print(f"\n=== trajectory: {kind} ===")
    print(f"  python-model vs REAL kernel at B=32: " +
          "  ".join(f"{k} relL2={v:.2e}" for k,v in val.items()))
    print(f"  {'arm':>12} | {'B/elem':>7} | {'eta_cum@48':>10} | {'state':>9} | {'codes':>6} | {'sat':>6}")
    for k, v in rows.items():
        bits = 16 if k == "fp16" else int(k.split("bit")[0])
        B = 32 if k == "fp16" else int(k.split("B=")[1])
        bpe = 2.0 if bits >= 16 else bits/8 + 4/B
        print(f"  {k:>12} | {bpe:7.3f} | {v[-1]['eta_cum']:10.4f} | {v[-1]['state']:9.2e} | "
              f"{v[-1]['codes']:6.3f} | {v[-1]['sat']:6.3f}")
json.dump(out, open("docs/ahat_accuracy_2026-09-02/data/single_layer_sweep.json","w"))

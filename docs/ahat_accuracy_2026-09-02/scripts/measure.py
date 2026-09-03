"""Open-loop accuracy of kernel 1 across the trajectory, single-variable in a_hat storage.

Every arm replays the SAME captured x_t and the SAME per-step delta scale through the REAL CUDA
kernel; the only thing that differs is how a_hat is stored. The reference is the same recurrence
in fp32 with a_hat held exactly (no storage quantization), so the difference isolated is a_hat
storage and nothing else.

Metrics, per step t:
  consumed  relL2 of `dequant(a_hat_{t-1}) + q_t/s_t` -- the activation the conv effectively
            convolves, i.e. the quantity that propagates downstream. PRIMARY.
  state     relL2 of a_hat after the write -- shows whether error compounds or saturates.
  codes     fraction of delta codes q_t differing from the reference's. Discrete, and literally
            what the GEMM sees differently.
  sat       fraction of a_hat codes pinned at +-limit. The suspected i4 failure mechanism:
            once a group pins, the state stops moving.

The fp16-a_hat arm is the FLOOR: its a_hat storage error is fp16 rounding only, so whatever it
reports is the delta quantizer's own contribution. An arm sitting at the floor stores a_hat for
free. (This project has already been burned once by a floor that was 3x too high -- act_budget.)
argv: int8|int4
"""
import os, sys, json
ROOT="/workspace/MoDiff"; os.chdir(ROOT); sys.path[:0]=[ROOT]
import torch
import modiff_cutlass as mc

PREC = sys.argv[1] if len(sys.argv) > 1 else "int8"
CAP = torch.load(f"docs/ahat_accuracy_2026-09-02/data/capture_{PREC}.pt", weights_only=False)
DEV, CL = "cuda", torch.channels_last
DLIM = 127.0 if PREC == "int8" else 7.0          # DELTA code limit (not the a_hat one)
ef = torch.empty(0, device=DEV, dtype=torch.float32)
eh = torch.empty(0, device=DEV, dtype=torch.float16)
ei = torch.empty(0, device=DEV, dtype=torch.int32)
ARMS = [("fp16", 0), ("i8", 16), ("i8", 32), ("i8", 64), ("i4", 32)]

def gn_silu(x, meta, mod):
    """Byte-for-byte the kernel's pre-quantize math, including the fp16 round before SiLU."""
    G = meta["num_groups"]; N, C, H, W = x.shape
    xg = x.view(N, G, C // G, H, W)
    mu = xg.mean(dim=(2, 3, 4), keepdim=True)
    var = xg.var(dim=(2, 3, 4), unbiased=False, keepdim=True)
    inv = (var + meta["eps"]).rsqrt()
    n = ((xg - mu) * inv).view(N, C, H, W)
    n = n * meta["weight"].to(DEV).view(1, C, 1, 1) + meta["bias"].to(DEV).view(1, C, 1, 1)
    if mod is not None:
        ms, sh = mod
        n = n * (1.0 + ms.to(DEV).view(N, C, 1, 1)) + sh.to(DEV).view(N, C, 1, 1)
    n = n.half().float()                                  # __half2float(__float2half(n))
    o = n * torch.sigmoid(n) if meta["apply_silu"] else n
    if meta["smooth_inv"] is not None:
        o = o * meta["smooth_inv"].to(DEV).view(1, C, 1, 1)
    return o

def deq(A, S, kind, C):
    if kind == "fp16":
        return A.float()
    N, ch, H, W = A.shape
    ng = S.shape[3]
    if kind == "i8":
        q = A.permute(0, 2, 3, 1).float()
    else:
        by = A.permute(0, 2, 3, 1).contiguous().reshape(-1).to(torch.uint8)
        lo = (by & 0xF).to(torch.int16); lo = torch.where(lo > 7, lo - 16, lo)
        hi = ((by >> 4) & 0xF).to(torch.int16); hi = torch.where(hi > 7, hi - 16, hi)
        q = torch.stack([lo, hi], -1).reshape(N, H, W, C).float()
    return (q.view(N, H, W, ng, -1) * S.view(N, H, W, ng, 1)).view(N, H, W, C).permute(0, 3, 1, 2)

def sat_frac(A, kind):
    if kind == "fp16": return 0.0
    if kind == "i8":   return (A.abs() == 127).float().mean().item()
    by = A.reshape(-1).to(torch.uint8)
    lo = (by & 0xF).to(torch.int16); lo = torch.where(lo > 7, lo - 16, lo)
    hi = ((by >> 4) & 0xF).to(torch.int16); hi = torch.where(hi > 7, hi - 16, hi)
    return (torch.cat([lo, hi]).abs() == 7).float().mean().item()

out = {}
for name, L in CAP["layers"].items():
    C, H, W, N = L["C"], L["H"], L["W"], L["batch"]
    T = L["x"].shape[0]
    meta, scales, mods = L["meta"], L["scale"], L["mod"]
    # ---- reference: exact fp32 a_hat ----
    ref_a = torch.zeros(N, C, H, W, device=DEV, dtype=torch.float32)
    ref_cons, ref_q = [], []
    for t in range(T):
        x = L["x"][t].to(DEV).float()
        o = gn_silu(x, meta, mods[t])
        s = scales[t]
        q = torch.clamp(torch.round((o - ref_a) * s), -DLIM, DLIM)
        cons = ref_a + q / s
        ref_cons.append(cons); ref_q.append(q); ref_a = cons
    rows = {}
    for kind, blk in ARMS:
        if kind != "fp16" and C % blk: continue
        chan = C // 2 if kind == "i4" else C
        A = (torch.empty(N, C, H, W, device=DEV, dtype=torch.float16, memory_format=CL).zero_()
             if kind == "fp16" else
             torch.empty(N, chan, H, W, device=DEV, dtype=torch.int8, memory_format=CL).zero_())
        S = ef if kind == "fp16" else torch.ones(N, H, W, C // blk, device=DEV, dtype=torch.float32)
        per = []
        # Running sum of the STORAGE rounding error eta_t = dequant(a_hat_t) - consumed_t.
        # The activation reconstruction cancels a_hat exactly (consumed = a_hat_prev + q/s), but
        # o_hat does NOT: it is written from the CODES while a_hat is written from the ROUNDED
        # value, so the two caches part by eta_t once per step and the conv output carries
        # conv(sum_k eta_k). This is the only quantity in this kernel that accumulates, and it is
        # what the per-step metrics were blind to.
        eta_sum = torch.zeros(N, C, H, W, device=DEV, dtype=torch.float32)
        for t in range(T):
            x = L["x"][t].to(DEV).contiguous(memory_format=CL)
            s = scales[t]
            sc = torch.full((1,), float(s), device=DEV, dtype=torch.float32)
            w = meta["weight"].to(DEV).half(); b = meta["bias"].to(DEV).half()
            if mods[t] is None:
                ms = sh = eh
            else:
                ms = mods[t][0].to(DEV).half().contiguous(); sh = mods[t][1].to(DEV).half().contiguous()
            si = ef if meta["smooth_inv"] is None else meta["smooth_inv"].to(DEV).contiguous()
            a_prev = deq(A, S, kind, C)
            args = (x, w, b, A, meta["num_groups"], meta["eps"], meta["apply_silu"],
                    sc, si, ms, sh, ef, ef, ef, ei, DLIM, False, 1.0)
            if PREC == "int8":
                yq = mc.group_norm_silu_delta_quantize_nhwc(*args, False, True, S)
                q = yq.float()
            else:
                yqp = mc.group_norm_silu_delta_quantize_pack_nhwc(*args, True, S)
                by = yqp.reshape(-1).to(torch.uint8)
                lo = (by & 0xF).to(torch.int16); lo = torch.where(lo > 7, lo - 16, lo)
                hi = ((by >> 4) & 0xF).to(torch.int16); hi = torch.where(hi > 7, hi - 16, hi)
                q = torch.stack([lo, hi], -1).reshape(N, H, W, C).float().permute(0, 3, 1, 2)
            cons = a_prev + q / s
            st = deq(A, S, kind, C)
            r = ref_cons[t]
            eta_sum += (st - cons)
            per.append({
                "t": t,
                "eta_cum": (eta_sum.norm() / r.norm()).item(),
                "eta_step": ((st - cons).norm() / r.norm()).item(),
                "consumed": ((cons - r).norm() / r.norm()).item(),
                "state": ((st - ref_cons[t]).norm() / ref_cons[t].norm()).item(),
                "codes": (q != ref_q[t]).float().mean().item(),
                "sat": sat_frac(A, kind)})
        rows[f"{kind} B={blk}" if kind != "fp16" else "fp16"] = per
        del A, S, eta_sum; torch.cuda.empty_cache()
    out[name] = {"C": C, "H": H, "W": W, "batch": N, "steps": T, "arms": rows}
    f = rows["fp16"][-1]
    print(f"{name}: floor(consumed)={f['consumed']:.2e}  " + "  ".join(
        f"{k}={v[-1]['consumed']:.2e}" for k, v in rows.items() if k != "fp16"), flush=True)
json.dump({"prec": PREC, "data": out},
          open(f"docs/ahat_accuracy_2026-09-02/data/accuracy_{PREC}.json", "w"))
print("wrote docs/ahat_accuracy_2026-09-02/data/accuracy_" + PREC + ".json")

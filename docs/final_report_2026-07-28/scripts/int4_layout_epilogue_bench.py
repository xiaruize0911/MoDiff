"""Validate and benchmark the direct-layout W4A4 QKV epilogue (T1024/hd24).

The candidate `gemm_w4a4_awq_qkv_i4qk_i8v_layouts` emits token-major Q, padded K and transposed
Vt from ONE launch. The reference `gemm_w4a4_awq_qkv_i4qk_i8v` needs two: a compact token-major
GEMM plus `qkv_i4codes_i8v_rearrange_kernel`.

The two are held to BYTE EQUALITY, which is possible because the candidate deliberately keeps the
reference's statement order (v = acc*(a_scale*w_scale); v += bias; rn(v*inv)) instead of folding
into one FFMA. Note this gate does NOT apply against the fp16-QKV production route: that one
rounds Q/K/V through __half before quantizing, so it legitimately differs by +-1 code.

No model and no checkpoint -- synthetic tensors at the real production shape.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import torch
import modiff_cutlass as mc

NH, HD, HP, T = 8, 24, 32, 1024
KPAD = 256                      # qkv._awqt_K for C=192 int4
QK_I4_VALUES_I8_MMA = 84


def make_inputs(batch, seed=1234):
    """Compact and hp-padded views of the SAME weights, so the two routes are comparable."""
    torch.manual_seed(seed)
    M = batch * T
    n_compact = NH * 3 * HD          # 576 logical output channels
    n_awqt = 640                     # qkv._awqt_N: the reference needs N % 128 == 0
    n_layout = NH * 3 * HP           # 768, and 768 % 128 == 0 so the candidate needs no extra pad
    xq = torch.randint(-128, 128, (M, KPAD // 2), device="cuda", dtype=torch.int8)
    # Reference weight is the production shape [_awqt_N, K/2] with the tail rows zero; only the
    # first n_compact columns are ever read (GWQ_QCODE returns 0 for gc >= n_out).
    w_c = torch.zeros(n_awqt, KPAD // 2, device="cuda", dtype=torch.int8)
    w_c[:n_compact] = torch.randint(-8, 8, (n_compact, KPAD // 2),
                                    device="cuda", dtype=torch.int8)
    ws_c = torch.zeros(n_awqt, device="cuda")
    ws_c[:n_compact] = torch.rand(n_compact, device="cuda") * 0.01 + 0.001
    b_c = (torch.rand(n_compact, device="cuda") * 0.2 - 0.1).half()   # numel must == n_out
    sq, sk = 0.031, 0.027
    sv = torch.rand(HD, device="cuda") * 0.02 + 0.005
    a_scale = 0.019

    # Offline padded layout -- exactly the builder in quantized_std_attention.py.
    w_l = torch.zeros(n_layout, KPAD // 2, device="cuda", dtype=torch.int8)
    ws_l = torch.zeros(n_layout, device="cuda")
    iv_l = torch.zeros(n_layout, device="cuda")
    lim_l = torch.zeros(n_layout, device="cuda")
    b_l = torch.zeros(n_layout, device="cuda", dtype=torch.float16)
    for h in range(NH):
        for sel in range(3):
            s, d = (h * 3 + sel) * HD, (h * 3 + sel) * HP
            w_l[d:d + HD].copy_(w_c[s:s + HD])
            ws_l[d:d + HD].copy_(ws_c[s:s + HD])
            b_l[d:d + HD].copy_(b_c[s:s + HD])
            if sel == 0:
                iv_l[d:d + HD] = 1.0 / sq
            elif sel == 1:
                iv_l[d:d + HD] = 1.0 / sk
            else:
                iv_l[d:d + HD].copy_(1.0 / sv.float())
            lim_l[d:d + HD] = 7.0 if sel < 2 else 127.0
    return dict(xq=xq, w_c=w_c, ws_c=ws_c, b_c=b_c, w_l=w_l, ws_l=ws_l, iv_l=iv_l,
                lim_l=lim_l, b_l=b_l, sq=sq, sk=sk, sv=sv, a_scale=a_scale, batch=batch)


def run_ref(d):
    """Two-kernel reference: compact GEMM + rearrange. Returns {q,k,vt} head-major."""
    return mc.gemm_w4a4_awq_qkv_i4qk_i8v(
        d["xq"], d["w_c"], d["ws_c"], d["a_scale"], KPAD, NH * 3 * HD, d["b_c"],
        NH, T, HD, HP, HP, QK_I4_VALUES_I8_MMA, d["sq"], d["sk"], d["sv"])


def run_cand(d):
    """One-kernel candidate. Q comes back TOKEN-major [b,T,nh,hp]."""
    return mc.gemm_w4a4_awq_qkv_i4qk_i8v_layouts(
        d["xq"], d["w_l"], d["ws_l"], d["a_scale"], KPAD, d["iv_l"], d["lim_l"],
        d["b_l"], NH, T, HD, HP, d["sv"])


def validate(batch):
    d = make_inputs(batch)
    rq, rk, rvt, _ = run_ref(d)
    cq, ck, cvt, _ = run_cand(d)
    torch.cuda.synchronize()
    b = d["batch"]
    # Reference Q/K are head-major [BH,T,hp]; candidate Q is token-major [b,T,nh,hp].
    rq_tok = rq.view(b, NH, T, HP).permute(0, 2, 1, 3).contiguous()
    cq_tok = cq.view(b, T, NH, HP)
    row = {
        "batch": batch,
        "q_bit_exact": bool(torch.equal(rq_tok, cq_tok)),
        "k_bit_exact": bool(torch.equal(rk, ck)),
        "vt_bit_exact": bool(torch.equal(rvt, cvt)),
        "q_max_abs_diff": int((rq_tok.int() - cq_tok.int()).abs().max()),
        "k_max_abs_diff": int((rk.int() - ck.int()).abs().max()),
        "vt_max_abs_diff": int((rvt.int() - cvt.int()).abs().max()),
        # int4 grid + padding invariants, checked not assumed
        "q_within_i4_grid": int(cq.abs().max()) <= 7,
        "k_within_i4_grid": int(ck.abs().max()) <= 7,
        "vt_within_i8": int(cvt.abs().max()) <= 127,
        "q_pad_zero": int(cq.view(b, T, NH, HP)[..., HD:].abs().max()) == 0,
        "k_pad_zero": int(ck[..., HD:].abs().max()) == 0,
        "vt_pad_zero": int(cvt[:, HD:, :].abs().max()) == 0,
    }
    if batch == 4:
        again = [run_cand(d)[0] for _ in range(20)]
        torch.cuda.synchronize()
        row["repeat20_deterministic"] = all(torch.equal(cq, a) for a in again)
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            sq_ = run_cand(d)[0]
        s.synchronize()
        row["nondefault_stream_exact"] = bool(torch.equal(cq, sq_))
    return row


def bench(fn, warm=20, rounds=5, iters=60):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    meds = []
    for _ in range(rounds):
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(iters + 1)]
        for i in range(iters):
            ev[i].record()
            fn()
        ev[-1].record()
        torch.cuda.synchronize()
        meds.append(statistics.median(
            [ev[i].elapsed_time(ev[i + 1]) * 1e3 for i in range(iters)]))
    return statistics.median(meds)


def sass_census():
    """The port's whole thesis is that mode 1's epilogue drops from ~13k to ~1.8k SASS.
    Verify from the built object before trusting any timing."""
    obj = os.path.join(ROOT, "build/temp.linux-x86_64-cpython-311/"
                             "csrc/kernels/linear/gemm_wxax.o")
    if not os.path.exists(obj):
        return {"error": "object not found"}
    out = subprocess.run(["cuobjdump", "-sass", obj], capture_output=True, text=True).stdout
    import re
    cur, counts = None, {}
    for line in out.splitlines():
        m = re.search(r"Function : (\S+)", line)
        if m:
            cur = m.group(1)
            counts.setdefault(cur, 0)
        elif cur and re.search(r"/\*[0-9a-f]{4}\*/", line):
            counts[cur] += 1
    dem = {}
    for k, v in counts.items():
        if "gemm_w4a4_kernel_awq_out_i8" in k or "gemm_w8a8_kernel_awq_out_i8" in k:
            name = subprocess.run(["c++filt", k], capture_output=True, text=True).stdout.strip()
            mm = re.search(r"<(.+?)>", name)
            dem[f"{'w4a4' if 'w4a4' in k else 'w8a8'}<{mm.group(1) if mm else '?'}>"] = v
    return dem


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--output", default="docs/final_report_2026-07-28/data/"
                                        "int4_layout_epilogue.json")
    a = ap.parse_args()

    result = {"gpu": torch.cuda.get_device_name(0),
              "shape": {"T": T, "hd": HD, "hp": HP, "nh": NH, "batch": a.batch},
              "sass_instruction_counts": sass_census()}
    print("SASS instruction counts:")
    for k, v in sorted(result["sass_instruction_counts"].items()):
        print(f"  {k:<44}{v:>7}")

    print("\nvalidation:")
    result["validation"] = [validate(b) for b in (1, 4, a.batch)]
    for r in result["validation"]:
        print(f"  {r}")

    d = make_inputs(a.batch)
    result["bench_us"] = {
        "reference_two_kernel": bench(lambda: run_ref(d)),
        "candidate_one_kernel": bench(lambda: run_cand(d)),
    }
    result["bench_us"]["speedup"] = (result["bench_us"]["reference_two_kernel"]
                                     / result["bench_us"]["candidate_one_kernel"])
    print(f"\nQKV stage, batch {a.batch}:")
    print(f"  reference (2 kernels) : {result['bench_us']['reference_two_kernel']:8.1f} us")
    print(f"  candidate (1 kernel)  : {result['bench_us']['candidate_one_kernel']:8.1f} us")
    print(f"  speedup               : {result['bench_us']['speedup']:8.3f}x")

    with open(a.output, "w") as f:
        json.dump(result, f, indent=1)
    print(f"\nWROTE {a.output}")
    ok = all(r["q_bit_exact"] and r["k_bit_exact"] and r["vt_bit_exact"]
             and r["q_within_i4_grid"] and r["k_within_i4_grid"] and r["vt_within_i8"]
             and r["q_pad_zero"] and r["k_pad_zero"] and r["vt_pad_zero"]
             for r in result["validation"])
    print("GATE: " + ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)

"""Static occupancy accounting for every flash-kernel instantiation, from the built object file.

Why static and not only ncu: ncu reports the occupancy a given launch ACHIEVED, which mixes the
hardware limit with the launch's own grid shape and tail effects. What the optimization work needs
is the ceiling each instantiation is allowed by its resource footprint, and which resource sets it.
QUANT_ATTENTION_OPT.md predicted "smem 24 KB -> 14 KB, so 4 -> 7 CTA/SM" for HD_PAD=32; that
prediction is only right if smem stays the binding resource after templating, and the register
count is what actually decides it for some of the eight instantiations.

Reads REG/STACK/SHARED straight out of cuobjdump -res-usage, so the numbers are the ones ptxas
committed to, then applies the sm_86 limits. STACK != 0 means register spill to local memory --
that was the pre-templating failure mode (STACK:128, all 32 fp32 accumulators in DRAM), so it is
reported as its own column rather than folded into the occupancy math.
"""
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "attn_occupancy.json")
OBJ = os.path.join(HERE, "..", "..", "..",
                   "build/temp.linux-x86_64-cpython-311/csrc/kernels/attention/flash_attn_int8.o")

# NVIDIA A40 = GA102, sm_86
SM_REGS = 65536          # registers per SM
SM_SMEM = 102400         # bytes of shared memory an SM can hand out (100 KB on sm_86)
SM_THREADS = 1536        # resident threads per SM
SM_BLOCKS = 16           # resident blocks per SM
REG_ALLOC_UNIT = 8       # registers are allocated per-warp in units of 8 on this arch
WARP = 32

# Threads per CTA is encoded in the template args, not in the object file, so it has to be mapped
# here. int8: <HD_PAD, WARPS, BC> -> WARPS*32. int4: <HDP_V> -> FA_MMA_WARPS(4)*32.
PAT_I8 = re.compile(r"flash_attn_int8_mma_kernel_tILi(\d+)ELi(\d+)ELi(\d+)EE")
PAT_I4 = re.compile(r"flash_attn_int4_mma_kernel_tILi(\d+)EE")


def parse_res_usage(obj):
    txt = subprocess.run(["cuobjdump", "-res-usage", obj], capture_output=True, text=True).stdout
    out, cur = [], None
    for line in txt.splitlines():
        m = re.match(r"\s*Function (\S+):", line)
        if m:
            cur = m.group(1)
            continue
        m = re.match(r"\s*REG:(\d+) STACK:(\d+) SHARED:(\d+)", line)
        if m and cur:
            out.append((cur, int(m.group(1)), int(m.group(2)), int(m.group(3))))
            cur = None
    return out


def occupancy(regs, smem, threads):
    """CTAs per SM this footprint allows, and which resource binds."""
    warps = threads // WARP
    # registers are allocated per warp, rounded up to REG_ALLOC_UNIT
    regs_per_warp = ((regs * WARP + REG_ALLOC_UNIT - 1) // REG_ALLOC_UNIT) * REG_ALLOC_UNIT
    by_reg = SM_REGS // (regs_per_warp * warps) if regs_per_warp else SM_BLOCKS
    by_smem = SM_SMEM // smem if smem else SM_BLOCKS
    by_thread = SM_THREADS // threads
    lim = {"registers": by_reg, "shared memory": by_smem, "thread slots": by_thread,
           "block slots": SM_BLOCKS}
    cta = min(lim.values())
    binding = [k for k, v in lim.items() if v == cta]
    return cta, binding, lim


def main():
    if not os.path.exists(OBJ):
        sys.exit(f"object file not built: {OBJ}")
    rows = []
    for name, regs, stack, smem in parse_res_usage(OBJ):
        m8, m4 = PAT_I8.search(name), PAT_I4.search(name)
        if m8:
            hd_pad, warps, bc = (int(v) for v in m8.groups())
            label, bits = f"int8 HD{hd_pad} W{warps} BC{bc}", 8
        elif m4:
            hd_pad, warps, bc = int(m4.group(1)), 4, 64
            label, bits = f"int4 HD{hd_pad} W{warps}(fixed)", 4
        else:
            continue
        threads = warps * WARP
        cta, binding, lim = occupancy(regs, smem, threads)
        rows.append(dict(label=label, bits=bits, hd_pad=hd_pad, warps=warps, bc=bc,
                         regs=regs, spill_bytes=stack, smem_bytes=smem, threads_per_cta=threads,
                         cta_per_sm=cta, resident_threads=cta * threads,
                         occupancy_pct=round(cta * threads / SM_THREADS * 100, 1),
                         binding_resource=" + ".join(binding), limits=lim))
    rows.sort(key=lambda r: (-r["bits"], r["hd_pad"], r["warps"], r["bc"]))

    print(f"{'instantiation':26s} {'reg':>4s} {'spill':>6s} {'smem':>7s} {'thr':>4s} "
          f"{'CTA/SM':>7s} {'occ':>6s}  binding")
    for r in rows:
        print(f"{r['label']:26s} {r['regs']:4d} {r['spill_bytes']:6d} {r['smem_bytes']:7d} "
              f"{r['threads_per_cta']:4d} {r['cta_per_sm']:7d} {r['occupancy_pct']:5.1f}%  "
              f"{r['binding_resource']}")
    spilled = [r["label"] for r in rows if r["spill_bytes"]]
    print(f"\nspilling instantiations: {spilled if spilled else 'NONE'}")

    with open(OUT, "w") as f:
        json.dump({"gpu": "A40 (sm_86)", "sm_regs": SM_REGS, "sm_smem": SM_SMEM,
                   "sm_threads": SM_THREADS, "rows": rows}, f, indent=2)
    print(f"WROTE {OUT}")


if __name__ == "__main__":
    main()

import json
d = json.load(open("docs/ahat_only_conv_2026-09-02/data/kernel1_arms.json"))
ARMS = ["baseline", "MoDiff a_hat fp16", "MoDiff a_hat i8 B=16", "MoDiff a_hat i8 B=32",
        "MoDiff a_hat i8 B=64"]
print(f"GPU {d['gpu']}\n{d['method']}\n")
for prec, title in (("int8", "W8A8"), ("int4", "W4A4")):
    live = [a for a in ARMS
            if any(r["arms"].get(f"{prec}/{a}", [None])[0] is not None for r in d["shapes"])]
    print(f"\n{'='*116}\n{title} -- kernel 1 (fused GN+SiLU -> quantize), per shape: ms / peak alloc MB\n{'='*116}")
    hdr = f"{'C':>5} {'HxW':>7} {'B':>4} {'f':>3} |" + "".join(f"{a.replace('MoDiff a_hat ','ahat='):>23}" for a in live)
    print(hdr); print("-" * len(hdr))
    tot = {a: [0.0, 0] for a in live}
    for r in d["shapes"]:
        cells = ""
        for a in live:
            v = r["arms"].get(f"{prec}/{a}")
            if v and v[0] is not None:
                cells += f"{v[0]:12.3f} /{v[1]:8.0f}"
                tot[a][0] += v[0] * r["freq"]; tot[a][1] += r["freq"]
            else:
                cells += f"{'--':>23}"
        hw = f"{r['H']}x{r['W']}"
        print(f"{r['C']:>5} {hw:>7} {r['B']:>4} {r['freq']:>3} |" + cells)
    tf = sum(r["freq"] for r in d["shapes"])
    print("-" * len(hdr))
    base = tot[live[0]][0]
    print(f"{'freq-weighted total ms':>22} |" + "".join(
        f"{tot[a][0]:12.3f} {'':9}" for a in live))
    print(f"{'vs baseline':>22} |" + "".join(
        f"{base/tot[a][0]:11.3f}x {'':9}" if tot[a][1] == tf else f"{'(partial)':>12} {'':9}" for a in live))
    md = tot.get("MoDiff a_hat fp16")
    if md:
        print(f"{'vs MoDiff fp16 a_hat':>22} |" + "".join(
            f"{md[0]/tot[a][0]:11.3f}x {'':9}" if tot[a][1] == tf else f"{'(partial)':>12} {'':9}" for a in live))
    print(f"{'peak alloc, max over shapes (MB)':>22} " + "  ".join(
        f"{a}={max(r['arms'][f'{prec}/{a}'][1] for r in d['shapes'] if r['arms'].get(f'{prec}/{a}',[None])[0] is not None):.0f}"
        for a in live))

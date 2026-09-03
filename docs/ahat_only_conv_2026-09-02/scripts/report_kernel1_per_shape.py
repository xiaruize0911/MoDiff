"""One table per shape: all four a_hat arms x both precisions, ms and peak alloc."""
import json
d = json.load(open("docs/ahat_only_conv_2026-09-02/data/kernel1_arms.json"))
ARMS = [("baseline", "baseline (no MoDiff)"),
        ("MoDiff a_hat fp16", "MoDiff, a_hat fp16"),
        ("MoDiff a_hat i8 B=16", "MoDiff, a_hat i8 B=16"),
        ("MoDiff a_hat i8 B=32", "MoDiff, a_hat i8 B=32"),
        ("MoDiff a_hat i8 B=64", "MoDiff, a_hat i8 B=64")]
print(f"GPU {d['gpu']}")
print(f"{d['method']}\n")
for r in sorted(d["shapes"], key=lambda r: -r["freq"]):
    print(f"### C={r['C']}  {r['H']}x{r['W']}  batch={r['B']}  (occurs {r['freq']}x in the UNet)")
    print(f"| arm | W8A8 ms | vs base | W8A8 peak MB | W4A4 ms | vs base | W4A4 peak MB |")
    print(f"|---|---|---|---|---|---|---|")
    b8 = r["arms"]["int8/baseline"][0]; b4 = r["arms"]["int4/baseline"][0]
    for key, label in ARMS:
        c = []
        for prec, base in (("int8", b8), ("int4", b4)):
            v = r["arms"].get(f"{prec}/{key}")
            if v and v[0] is not None:
                c += [f"{v[0]:.3f}", f"{base/v[0]:.3f}x", f"{v[1]:.0f}"]
            else:
                c += ["--", "--", "--"]
        print(f"| {label} | " + " | ".join(c) + " |")
    print()

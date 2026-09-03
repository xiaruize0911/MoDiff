"""Diff two per-kernel CUDA-time profiles (MoDiff vs PTQ) and print the top movers."""
import json, sys
a=json.load(open(sys.argv[1])); b=json.load(open(sys.argv[2]))   # a=MoDiff, b=PTQ
keys=set(a)|set(b)
rows=sorted(((a.get(k,0.0)-b.get(k,0.0), k) for k in keys), key=lambda r:-abs(r[0]))
print(f"{'delta ms/step':>13} {'MoDiff':>9} {'PTQ':>9}  kernel")
for d,k in rows[:18]:
    if abs(d)<0.20: break
    print(f"{d:+13.2f} {a.get(k,0.0):9.2f} {b.get(k,0.0):9.2f}  {k[:78]}")
print(f"\ntotal  MoDiff {sum(a.values()):.2f}  PTQ {sum(b.values()):.2f}  delta {sum(a.values())-sum(b.values()):+.2f} ms/step")

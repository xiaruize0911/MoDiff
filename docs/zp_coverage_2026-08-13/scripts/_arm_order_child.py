
import io, contextlib, json, os, re, sys
ROOT = '/workspace/MoDiff'
os.chdir(ROOT)
sys.path[:0] = [os.path.join(ROOT, "docs/attn_modiff_2026-08-13/scripts"),
                os.path.join(ROOT, "docs/qdiff_bridge_2026-08-12/scripts"),
                ROOT, os.path.join(ROOT, "src/taming-transformers"),
                os.path.join(ROOT, "integration/benchmarks/report"),
                os.path.join(ROOT, "docs/modiff_correctness_2026-08-03/scripts"),
                os.path.join(ROOT, "docs/zero_point_2026-08-13/scripts")]
import torch, export_and_measure_zp as M, fp16_refs
M.H.STEPS, M.H.BATCH = 50, 8
SEEDS = [1234, 20260805, 777]
M.SEEDS = SEEDS
refs = fp16_refs.get(50, 8, SEEDS)
out = []
for mode in sys.argv[1:]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        mean = M.measure(mode, None, refs, mode)
    t = buf.getvalue()
    s = re.search(r"\[([0-9.,\s]+)\]", t)
    out.append({"mode": mode, "mean": mean,
                "rels": [float(x) for x in s.group(1).split(",")] if s else []})
print("RESULT_JSON " + json.dumps(out))

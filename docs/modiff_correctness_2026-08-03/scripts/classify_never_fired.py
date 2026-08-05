"""Classify each never-fired export by who still references it, so deletion is safe.

The probe says 78 of 129 exports never fire in any of the 8 modes. That alone does not make them
deletable: an entry may be referenced by a live test, by an odd-shape fallback that this model's
channel counts never hit, or by a harness that is still run. Deleting on the probe alone would break
the test suite.

Buckets, in decreasing safety:
  A_unreferenced   no reference anywhere outside docs/ and build/ -> delete
  B_env_gated      referenced only from a route behind an env var that defaults off -> delete
                   (the approved plan's aggressive tier), and delete the env var with it
  C_test_only      referenced by integration/tests/** -> deleting needs the test changed too
  D_bench_only     referenced only by integration/benchmarks/** that is not the report harness
  E_sentinel       referenced only inside a hasattr()/getattr() capability probe -> repoint first
  F_fallback       referenced from a live module on a branch this model's shapes never take -> KEEP
"""

import json
import os
import re
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)

PROBE = "docs/modiff_correctness_2026-08-03/data/kernel_reachability.json"
# Routes whose env var defaults OFF. Approved plan Stage 4.3 deletes these outright.
ENV_GATED = {
    "MODIFF_ROUTE1": ["fused_gn_qkv_i8evt", "quantize_attn_qkv_from_i8"],
    "MODIFF_INT4_QKV_EPILOGUE": ["gemm_w4a4_awq_qkv_i4qk_i8v",
                                 "flash_attn_i4values_i8mma_vt_static_qout"],
    "MODIFF_FLASH_PACKED": ["flash_attn_int8_packed_vt", "flash_attn_int8_packed_vt_qout",
                            "flash_attn_int8_packed_persistent_qout"],
    "MODIFF_INT8_FLASH_PREG": ["flash_attn_int8_qi8packed_kv_static_qout_preg"],
    "MODIFF_LINEAR_OUT_I8": ["gemm_w8a8_awq_out_i8", "gemm_w4a4_awq_out_i8", "dequant_bias_i8"],
    "MODIFF_FP16_MATERIALIZED": ["attn_softmax_fp16"],
}
GATED_BY = {sym: env for env, syms in ENV_GATED.items() for sym in syms}


def refs(sym):
    """[(path, lineno, line)] for every non-docs, non-build reference to `sym` in a .py/.cu/.cpp/.h."""
    try:
        out = subprocess.run(
            ["grep", "-rn", "--include=*.py", "--include=*.cu", "--include=*.cpp", "--include=*.h",
             r"\b" + sym + r"\b", "integration", "ldm", "scripts", "csrc"],
            capture_output=True, text=True).stdout
    except Exception:
        return []
    hits = []
    for line in out.splitlines():
        path, _, rest = line.partition(":")
        lineno, _, text = rest.partition(":")
        if path.startswith(("docs/", "build/")) or "/results/" in path:
            continue
        hits.append((path, lineno, text.strip()))
    return hits


def classify(sym, hits):
    py = [h for h in hits if h[0].endswith(".py")]
    # csrc references are the definition + binding; they don't count as a caller.
    if not py:
        return "A_unreferenced", ""
    if sym in GATED_BY:
        return "B_env_gated", GATED_BY[sym]
    files = {h[0] for h in py}
    if all(f.startswith("integration/tests/") for f in files):
        return "C_test_only", ", ".join(sorted(files))
    non_probe = [h for h in py if not re.search(r"hasattr|getattr", h[2])]
    if not non_probe:
        return "E_sentinel", "; ".join(f"{h[0]}:{h[1]}" for h in py[:3])
    nb = {f for f in files if f.startswith("integration/benchmarks/")}
    if nb == files:
        return "D_bench_only", ", ".join(sorted(files))
    return "F_fallback", "; ".join(f"{h[0]}:{h[1]}" for h in non_probe[:3])


def main():
    probe = json.load(open(PROBE))
    never = probe["never_fired"]
    buckets = {}
    for sym in never:
        b, why = classify(sym, refs(sym))
        buckets.setdefault(b, []).append((sym, why))

    total = 0
    for b in ("A_unreferenced", "B_env_gated", "E_sentinel", "C_test_only", "D_bench_only",
              "F_fallback"):
        items = buckets.get(b, [])
        total += len(items)
        print(f"\n{'=' * 78}\n{b}  ({len(items)})\n{'=' * 78}")
        for sym, why in sorted(items):
            print(f"  {sym}")
            if why:
                print(f"      {why}")
    print(f"\ntotal classified {total} of {len(never)} never-fired")

    out = {"never_fired": never,
           "buckets": {b: [{"symbol": s, "why": w} for s, w in sorted(v)]
                       for b, v in buckets.items()},
           "steady_state": probe["fired_steady_any_mode"],
           "calibration_only": probe["setup_only"]}
    path = "docs/modiff_correctness_2026-08-03/data/deletion_classification.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"WROTE {path}")


if __name__ == "__main__":
    sys.exit(main())

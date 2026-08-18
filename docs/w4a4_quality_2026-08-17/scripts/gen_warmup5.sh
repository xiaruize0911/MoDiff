#!/bin/bash
# W8A8+MoDiff and W4A4+MoDiff at MODIFF_WARMUP_STEPS=5 (the shipped default), plus fp16 as reference.
# Fresh --out so the committed 2026-08-05 run in /workspace/fid is not touched.
#
# Logs cwd, rc and image counts, and ends in an explicit verdict. Both are lessons paid for:
# a previous run produced 0 images and exited 0 (wrong cwd + a block ending in echo).
set -o pipefail
cd /workspace/MoDiff || { echo "GEN FAILED: cannot cd /workspace/MoDiff"; exit 1; }
export PYTHONPATH=/workspace/MoDiff:/workspace/MoDiff/src/taming-transformers
OUT=/workspace/fid_warmup5
echo "cwd=$(pwd)  out=$OUT  warmup=5  $(date -u +%H:%M:%S)"

MODIFF_WARMUP_STEPS=5 python docs/fid_2026-08-05/scripts/generate_fid_samples.py \
  --n 16 --batch 16 --steps 50 \
  --modes fp16,int8_l0,int4_l0 \
  --out "$OUT"
rc=$?
echo "generator rc=$rc"

fail=0
for f in fp16 int8_modiff_l0 int4_modiff_l0; do
  n=$(ls "$OUT/$f"/*.png 2>/dev/null | wc -l)
  echo "  $f: $n png"
  [ "$n" -ge 16 ] || fail=1
done

if [ "$rc" -ne 0 ] || [ "$fail" -ne 0 ]; then
  echo "GEN FAILED (rc=$rc fail=$fail)"
  exit 1
fi
echo "GEN OK $(date -u +%H:%M:%S)"

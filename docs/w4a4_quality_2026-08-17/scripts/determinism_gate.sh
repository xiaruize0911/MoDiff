#!/bin/bash
# Does the seeding fix make the same configuration reproduce across processes?
#
# Two separate processes, identical configuration, and the images must agree. Before the fix the same
# pair measured mean |delta| 4.11/255 with a max pixel difference of 232.
#
# This gate is what makes the 16-image screen in docs/w4a4_quality_2026-08-17 usable at all: without it
# the floor is as large as several of the effects being ranked.
set -o pipefail
cd /workspace/MoDiff || { echo "GATE FAILED: cannot cd"; exit 1; }
export PYTHONPATH=/workspace/MoDiff:/workspace/MoDiff/src/taming-transformers
L=/tmp/claude-0/-workspace/7883ed3f-72e3-48df-8607-0ee5db4457c1/scratchpad

for R in a b; do
  OUT=/workspace/fid_det/$R
  echo "=== run $R  $(date -u +%H:%M:%S)"
  DELTA_STATIC=1 MODIFF_WARMUP_STEPS=5 python docs/fid_2026-08-05/scripts/generate_fid_samples.py \
    --n 16 --batch 16 --steps 50 --modes int4_l1 --out "$OUT" > "$L/det_$R.log" 2>&1
  rc=$?
  n=$(ls "$OUT/int4_modiff_l1_static"/*.png 2>/dev/null | wc -l)
  echo "  rc=$rc images=$n"
  [ "$rc" -eq 0 ] && [ "$n" -ge 16 ] || { echo "  run $R FAILED"; exit 1; }
done
echo "GATE GEN OK $(date -u +%H:%M:%S)"

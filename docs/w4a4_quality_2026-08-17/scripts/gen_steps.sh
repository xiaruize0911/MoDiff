#!/bin/bash
# Step-count sweep on OUR real int4 datapath: does W4A4+MoDiff recover at the paper's 500 steps?
#
# MoDiff's premise is a_t ~= a_{t+1}; adjacent steps are ~10x closer at 500 steps than at 50, so the
# delta a 4-bit grid has to carry is smaller. The paper's reproduced figure is 500 steps, our FID
# protocol is 50. This is the one axis in that comparison nobody has swept.
#
# fp16 is regenerated at EVERY step count on purpose: it is the reference for mean|delta|, and fp16's
# own output changes with the schedule, so comparing a 500-step arm to a 50-step reference would
# measure the schedule and report it as quantization error.
#
# SEQUENTIAL: a second CUDA process during a timed run corrupted a batch earlier in this project.
set -o pipefail
cd /workspace/MoDiff || { echo "GEN FAILED: cannot cd /workspace/MoDiff"; exit 1; }
export PYTHONPATH=/workspace/MoDiff:/workspace/MoDiff/src/taming-transformers
echo "cwd=$(pwd)  warmup=5  delta=dynamic (step-count agnostic; no static table to mismatch)"

overall=0
for S in 200 500; do
  OUT=/workspace/fid_warmup5_s$S
  echo "=== steps=$S -> $OUT  $(date -u +%H:%M:%S)"
  MODIFF_WARMUP_STEPS=5 python docs/fid_2026-08-05/scripts/generate_fid_samples.py \
    --n 16 --batch 16 --steps "$S" \
    --modes fp16,int8_l0,int4_l0 \
    --out "$OUT"
  rc=$?
  echo "steps=$S generator rc=$rc"
  fail=0
  for f in fp16 int8_modiff_l0 int4_modiff_l0; do
    n=$(ls "$OUT/$f"/*.png 2>/dev/null | wc -l)
    echo "  steps=$S $f: $n png"
    [ "$n" -ge 16 ] || fail=1
  done
  if [ "$rc" -ne 0 ] || [ "$fail" -ne 0 ]; then
    echo "STEPS=$S FAILED (rc=$rc fail=$fail)"
    overall=1
  else
    echo "STEPS=$S OK"
  fi
done

[ "$overall" -eq 0 ] || { echo "GEN FAILED"; exit 1; }
echo "GEN OK $(date -u +%H:%M:%S)"

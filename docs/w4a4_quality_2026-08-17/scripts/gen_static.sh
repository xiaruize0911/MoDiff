#!/bin/bash
# The static-delta arms. NON-VACUITY: the build log must show the static delta table being LOADED and
# must NOT report L1 collapsing to L0 -- MODIFF_LINEAR is derived from delta_mode, so a careless
# override would silently turn the L1 arm into L0 and report the difference as "static helps".
set -o pipefail
cd /workspace/MoDiff || { echo "GEN FAILED: cannot cd"; exit 1; }
export PYTHONPATH=/workspace/MoDiff:/workspace/MoDiff/src/taming-transformers
L=/tmp/claude-0/-workspace/7883ed3f-72e3-48df-8607-0ee5db4457c1/scratchpad
OUT=/workspace/fid_warmup5

run() {   # $1=tag $2=extra $3=folder
  echo "=== $1"
  DELTA_STATIC=1 MODIFF_WARMUP_STEPS=5 python docs/fid_2026-08-05/scripts/generate_fid_samples.py \
    --n 16 --batch 16 --steps 50 --modes int4_l1 --out "$OUT" $2 > "$L/gen_$1.log" 2>&1
  rc=$?
  n=$(ls "$OUT/$3"/*.png 2>/dev/null | wc -l)
  # L1 must still be L1: 42 projections carrying MoDiff, not 42 plain W4A4 linears
  l1=$(grep -c "modiff=True" "$L/gen_$1.log")
  echo "  rc=$rc  images=$n  modiff_linear_lines=$l1"
  [ "$rc" -eq 0 ] || { echo "  $1 FAILED: rc"; return 1; }
  [ "$n" -ge 16 ] || { echo "  $1 FAILED: only $n images"; return 1; }
  [ "$l1" -ge 1 ] || { echo "  $1 FAILED: L1 collapsed to L0 -- MODIFF_LINEAR was derived as 0"; return 1; }
  echo "  $1 OK"
}

bad=0
run l1_static "" int4_modiff_l1_static || bad=1
run l1_ada_static "--adaround 1" int4_modiff_l1_adaround_static || bad=1
[ "$bad" -eq 0 ] || { echo "GEN FAILED"; exit 1; }
echo "GEN OK $(date -u +%H:%M:%S)"

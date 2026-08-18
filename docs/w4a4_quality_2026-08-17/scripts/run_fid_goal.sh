#!/bin/bash
# FID verdict on the best arm found today, against a same-protocol baseline.
#
# TWO SEPARATE PROCESSES, each with its arm FIRST. zp_coverage P5 measured arm order moving W4A4 MoDiff
# by 28% -- larger than every effect on the table -- and noted the committed values are second-arm
# values. Running each arm as a first-arm value is what makes these two mutually comparable; neither is
# directly comparable to the committed 181.514, which was a second-arm value.
#
# The real reference is symlinked, not copied: compute_fid.py computes the real statistics once.
set -o pipefail
cd /workspace/MoDiff || { echo "FID FAILED: cannot cd"; exit 1; }
export PYTHONPATH=/workspace/MoDiff:/workspace/MoDiff/src/taming-transformers
L=/tmp/claude-0/-workspace/7883ed3f-72e3-48df-8607-0ee5db4457c1/scratchpad
OUT=/workspace/fid_goal
N=10000
mkdir -p "$OUT"
[ -e "$OUT/real" ] || ln -s /workspace/fid/real "$OUT/real"
echo "cwd=$(pwd)  out=$OUT  n=$N  $(date -u +%H:%M:%S)"

gen() {   # $1=tag  $2=env-prefix  $3=mode  $4=folder
  echo "=== $1  $(date -u +%H:%M:%S)"
  env $2 MODIFF_WARMUP_STEPS=5 python docs/fid_2026-08-05/scripts/generate_fid_samples.py \
    --n $N --batch 128 --steps 50 --modes "$3" --out "$OUT" > "$L/fidgen_$1.log" 2>&1
  rc=$?
  n=$(ls "$OUT/$4"/*.png 2>/dev/null | wc -l)
  echo "  rc=$rc  images=$n"
  [ "$rc" -eq 0 ] && [ "$n" -ge "$N" ] || { echo "  $1 FAILED"; return 1; }
  echo "  $1 OK  $(date -u +%H:%M:%S)"
}

bad=0
gen l0_dynamic "DELTA_STATIC=0" int4_l0 int4_modiff_l0 || bad=1
gen l1_static  "DELTA_STATIC=1" int4_l1 int4_modiff_l1_static || bad=1
[ "$bad" -eq 0 ] || { echo "FID FAILED at generation"; exit 1; }

# the two folders must not be identical -- that would make the FID difference meaningless
if cmp -s "$OUT/int4_modiff_l0/000000.png" "$OUT/int4_modiff_l1_static/000000.png"; then
  echo "FID FAILED: the two arms produced identical images"; exit 1
fi
echo "non-vacuity OK: arms differ"

echo "=== computing FID  $(date -u +%H:%M:%S)"
python docs/fid_2026-08-05/scripts/compute_fid.py \
  --root "$OUT" --real real \
  --modes int4_modiff_l0,int4_modiff_l1_static \
  --out "$L/fid_goal.json" > "$L/fid_compute.log" 2>&1
rc=$?
echo "compute rc=$rc"
[ "$rc" -eq 0 ] || { echo "FID FAILED at compute"; tail -20 "$L/fid_compute.log"; exit 1; }
cat "$L/fid_goal.json"
echo "FID OK $(date -u +%H:%M:%S)"

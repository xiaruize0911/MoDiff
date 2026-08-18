#!/bin/bash
# ratio 16, the other side of the U. ratio 4 gave 60.861 and ratio 8 (shipped) gave 55.490/55.074, so
# 8 already beats 4; 16 decides whether 8 is at the optimum or on a slope.
#
# DELETES ITS IMAGES AFTER SCORING. A 10k PNG set is ~1.2 GB and this workspace hit its quota mid-run
# earlier today, losing a 25-minute generation at image 3907. The FID number is the artifact; the PNGs
# are reproducible from the seed.
set -o pipefail
cd /workspace/MoDiff || exit 1
export PYTHONPATH=/workspace/MoDiff:/workspace/MoDiff/src/taming-transformers
L=/tmp/claude-0/-workspace/7883ed3f-72e3-48df-8607-0ee5db4457c1/scratchpad
OUT=/workspace/fid_r16
rm -rf "$OUT"; mkdir -p "$OUT"; ln -s /workspace/fid/real "$OUT/real"
echo "=== ratio 16  $(date -u +%H:%M:%S)  free: $(df -h /workspace | tail -1 | awk '{print $4}')"
MODIFF_DELTA_TABLE_RATIO=16 DELTA_STATIC=1 MODIFF_WARMUP_STEPS=5 \
  python docs/fid_2026-08-05/scripts/generate_fid_samples.py \
    --n 10000 --batch 128 --steps 50 --modes int4_l1 --out "$OUT" > "$L/fidr16_gen.log" 2>&1
rc=$?; n=$(ls "$OUT/int4_modiff_l1_static"/*.png 2>/dev/null | wc -l)
echo "  gen rc=$rc images=$n  $(grep -o 'scaling the loaded delta table by [0-9.]*' "$L/fidr16_gen.log" | head -1)"
if [ "$rc" -ne 0 ] || [ "$n" -lt 10000 ]; then
  echo "RATIO16 FAILED"; grep -iE "quota|Error|Traceback" "$L/fidr16_gen.log" | tail -3; exit 1
fi
python docs/fid_2026-08-05/scripts/compute_fid.py --root "$OUT" --real real \
  --modes int4_modiff_l1_static --out "$L/fid_r16.json" > "$L/fidr16_fid.log" 2>&1
rc=$?
grep "FID vs real =" "$L/fidr16_fid.log" | grep -v "it/s"
[ "$rc" -eq 0 ] || { echo "RATIO16 FID FAILED"; exit 1; }
cp "$L/fid_r16.json" /workspace/MoDiff/docs/w4a4_quality_2026-08-17/data/
rm -f "$OUT/real"; rm -rf "$OUT"          # number is banked; reclaim the 1.2 GB
echo "RATIO16 OK $(date -u +%H:%M:%S)"

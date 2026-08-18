#!/bin/bash
# Does ACT_CLIP_RATIO=4.5 help the deployed arm? It has never reached it: the constant is baked in at
# export (int4_optimized.py:1761) and the shipped calibration file predates it by 8 days, while the load
# path only fills the value (:1892). Verified by calibrating at 4.5 and 1.0 and matching the shipped file
# against both -- it matches 1.0 (0.9398) not 4.5 (4.3022).
#
# BOTH ARMS USE A FRESH CALIBRATION so the only variable is the constant. Using the shipped file as the
# 1.0 arm would also change its vintage (~6% of run-to-run calibration variation), which is the confound
# that made the earlier ATTN_FP16 test unreadable.
#
# Images deleted after scoring: a 10k PNG set is ~1.2 GB and this workspace hit its quota today.
set -o pipefail
cd /workspace/MoDiff || exit 1
export PYTHONPATH=/workspace/MoDiff:/workspace/MoDiff/src/taming-transformers
L=/tmp/claude-0/-workspace/7883ed3f-72e3-48df-8607-0ee5db4457c1/scratchpad
for R in 1.0 4.5; do
  CAL=integration/calibration/int4_calibration_clip$R.pt
  cp "$L/calib_r$R.pt" "$CAL" || exit 1
  OUT=/workspace/fid_clip$R
  rm -rf "$OUT"; mkdir -p "$OUT"; ln -s /workspace/fid/real "$OUT/real"
  echo "=== ACT_CLIP_RATIO=$R  $(date -u +%H:%M:%S)"
  FID_CALIB4="$CAL" MODIFF_ACT_CLIP_RATIO=$R DELTA_STATIC=1 MODIFF_WARMUP_STEPS=5 \
    python docs/fid_2026-08-05/scripts/generate_fid_samples.py \
      --n 10000 --batch 128 --steps 50 --modes int4_l1 --out "$OUT" > "$L/fidclip_${R}_gen.log" 2>&1
  rc=$?; n=$(ls "$OUT/int4_modiff_l1_static"/*.png 2>/dev/null | wc -l)
  echo "  gen rc=$rc images=$n"
  if [ "$rc" -ne 0 ] || [ "$n" -lt 10000 ]; then
    echo "CLIP$R FAILED"; grep -iE "quota|Error|Traceback" "$L/fidclip_${R}_gen.log" | tail -3; exit 1
  fi
  python docs/fid_2026-08-05/scripts/compute_fid.py --root "$OUT" --real real \
    --modes int4_modiff_l1_static --out "$L/fid_clip$R.json" > "$L/fidclip_${R}_fid.log" 2>&1 || exit 1
  grep "FID vs real =" "$L/fidclip_${R}_fid.log" | grep -v "it/s"
  cp "$L/fid_clip$R.json" /workspace/MoDiff/docs/w4a4_quality_2026-08-17/data/
  rm -f "$OUT/real"; rm -rf "$OUT"
done
echo "CLIP SWEEP OK $(date -u +%H:%M:%S)"

#!/bin/bash
# Re-sweep the delta clip ratio on the arm that now matters: L1 + static delta (FID 55.49).
#
# The 8 was swept on L0 + dynamic, and the constant's own docstring says it is PROTOCOL-DEPENDENT
# ("the residual shrinks with step count ... a run at a very different S should re-sweep"). Changing the
# arm changes the residual distribution the same way changing S does: L1 puts 42 more layers on the delta
# path and static replaces per-call absmax with a fixed table.
#
# CONTROL: ratio 8 must come out BYTE-IDENTICAL to the existing int4_modiff_l1_static folder. That is
# what proves the multiplier path is a faithful re-parameterisation and not a second code path -- if the
# control differs, every other ratio in the sweep is measuring the plumbing.
set -o pipefail
cd /workspace/MoDiff || { echo "SWEEP FAILED: cannot cd"; exit 1; }
export PYTHONPATH=/workspace/MoDiff:/workspace/MoDiff/src/taming-transformers
L=/tmp/claude-0/-workspace/7883ed3f-72e3-48df-8607-0ee5db4457c1/scratchpad
REF=/workspace/fid_warmup5/int4_modiff_l1_static

for R in 8 4 6 11 16 24; do
  OUT=/workspace/fid_sweep/r$R
  echo "=== ratio=$R -> $OUT  $(date -u +%H:%M:%S)"
  MODIFF_DELTA_TABLE_RATIO=$R DELTA_STATIC=1 MODIFF_WARMUP_STEPS=5 \
    python docs/fid_2026-08-05/scripts/generate_fid_samples.py \
      --n 16 --batch 16 --steps 50 --modes int4_l1 --out "$OUT" > "$L/sweep_r$R.log" 2>&1
  rc=$?
  n=$(ls "$OUT/int4_modiff_l1_static"/*.png 2>/dev/null | wc -l)
  mul=$(grep -o "scaling the loaded delta table by [0-9.]*" "$L/sweep_r$R.log" | head -1)
  echo "  rc=$rc images=$n  $mul"
  [ "$rc" -eq 0 ] && [ "$n" -ge 16 ] || { echo "  ratio=$R FAILED"; exit 1; }
done

# the control, checked properly: both files must EXIST before being compared
A=$REF/000000.png
B=/workspace/fid_sweep/r8/int4_modiff_l1_static/000000.png
if [ ! -f "$A" ] || [ ! -f "$B" ]; then
  echo "SWEEP FAILED: control missing ($A or $B)"; exit 1
fi
if cmp -s "$A" "$B"; then
  echo "CONTROL OK: ratio=8 reproduces the existing L1+static arm byte-identically"
else
  echo "CONTROL FAILED: ratio=8 differs from the existing L1+static arm -- the multiplier path is not"
  echo "a faithful re-parameterisation, so the rest of this sweep measures plumbing, not the ratio."
  exit 1
fi
echo "SWEEP OK $(date -u +%H:%M:%S)"

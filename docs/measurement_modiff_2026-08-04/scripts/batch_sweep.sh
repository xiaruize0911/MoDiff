#!/usr/bin/env bash
# MoDiff cost vs batch size. Every figure in the 08-04 measurement report is batch 128 only, so it
# cannot say whether MoDiff's ~6% per-step deficit is a fixed overhead or one that scales.
#
# Why this is worth measuring rather than assuming: the deficit is a_hat/o_hat MEMORY TRAFFIC, which
# scales with batch, while kernel-launch cost does not. At small batch the pipeline is dispatch-bound
# (the 08-01 nsys table shows the smallest attention shapes already below GPU/issue = 1), so the
# traffic term shrinks relative to a fixed launch term and the ratio should MOVE. Which direction it
# moves, and how far, is the question.
#
# One process per batch size: the quantized attention blocks self-calibrate per process, and mixing
# batch sizes in one process would carry one batch's calibration into another's measurement.
#
# 50 steps rather than 200: ms/step is per-step work and is step-count independent, and this keeps
# the whole sweep to ~20 minutes. The 200-step numbers remain the headline; these are for the trend.
set -euo pipefail
cd "$(dirname "$0")/../../.."

OUT=docs/measurement_modiff_2026-08-04/data
LOG=docs/measurement_modiff_2026-08-04/logs
mkdir -p "$OUT" "$LOG"

export MODIFF_DELTA_MODE=dynamic
export MODIFF_DELTA_REFRESH=4
export MODIFF_DELTA_CLIP=1.0
export MODIFF_DELTA_REPORT=0
export MODIFF_LINEAR=0

MODES=int8_baseline,int8,int4_baseline,int4

for B in 8 16 32 64 128; do
  echo "=== batch $B"
  E2EBENCH_MODES="$MODES" python integration/benchmarks/report/e2e_three_mode_bench.py \
      --batch "$B" --steps 50 --repeats 3 --warmups 2 \
      --output "$OUT/e2e_batch$B.json" 2>&1 | tee "$LOG/batch$B.log" | grep -E "wall  median|^WROTE" || true
done
echo "=== done ==="

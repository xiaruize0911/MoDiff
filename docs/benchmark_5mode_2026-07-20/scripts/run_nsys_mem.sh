#!/bin/bash
# E2E items 2 & 4 — nsys memcpy trace per mode (measured copy traffic, no HW counters needed).
# For each mode: capture ONLY the cudaProfilerApi range (steady-state sampling) -> export sqlite.
# Then parse_nsys_mem.py reads the sqlites. Run from /workspace/MoDiff.
set -e
cd /workspace/MoDiff
source setup_cuda_env.sh >/dev/null 2>&1
NSYS=/opt/nvidia/nsight-compute/2024.1.1/host/target-linux-x64/nsys
OUT=docs/benchmark_5mode_2026-07-20/data
DRV=docs/benchmark_5mode_2026-07-20/scripts/nsys_driver.py
NSTEPS=30
MODES=(fp16 int8_baseline int4_baseline int8 int4)

for m in "${MODES[@]}"; do
  echo "=== nsys $m ==="
  "$NSYS" profile --force-overwrite=true --trace=cuda --sample=none --cpuctxsw=none \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    -o "$OUT/nsys_$m" python "$DRV" "$m" "$NSTEPS" > "$OUT/nsys_${m}.log" 2>&1
  "$NSYS" export --force-overwrite=true --type sqlite --output "$OUT/nsys_$m.sqlite" "$OUT/nsys_$m.nsys-rep" >> "$OUT/nsys_${m}.log" 2>&1
  echo "  done -> $OUT/nsys_$m.sqlite"
done
echo "ALL_NSYS_DONE (nsteps=$NSTEPS)"

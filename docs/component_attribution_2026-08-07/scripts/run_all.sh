#!/usr/bin/env bash
# A then C, in that order, on one GPU: wait out the differential run, re-measure the one bad arm,
# then trace and bucket. Sequential on purpose -- a profiler-attached trace sharing the A40 with a
# timing run would corrupt both.
set -u
cd /workspace/MoDiff
D=docs/component_attribution_2026-08-07

if [ "${1:-}" != "" ]; then
  echo "== waiting for differential_timing pid $1 =="
  while kill -0 "$1" 2>/dev/null; do sleep 20; done
fi

echo "== A: re-measuring the corrected fp16 arm =="
python $D/scripts/differential_timing.py --arms fp16 \
  --output $D/data/differential_timing_fp16_rerun.json \
  > $D/logs/differential_timing_fp16_rerun.log 2>&1
python $D/scripts/merge_fp16_rerun.py $D/data/differential_timing_fp16_rerun.json \
  2>&1 | tee $D/logs/merge_fp16.log

echo "== C: exporting traces =="
python $D/scripts/trace_configs.py > $D/logs/trace_configs.log 2>&1
tail -3 $D/logs/trace_configs.log

echo "== C: offline bucketing =="
python $D/scripts/bucket_traces.py 2>&1 | tee $D/logs/bucket_traces.log

echo "== done =="

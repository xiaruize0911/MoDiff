#!/usr/bin/env bash
# Regenerate every measurement behind MEASUREMENT_REPORT_MODIFF_2026-08-04.md.
#
# Same suites, same statistics and the same protocol as MEASUREMENT_REPORT_2026-08-01, with the mode
# list extended from three to five: fp16 plus BOTH the baseline and the MoDiff variant of each bit
# width. The 08-01 report carried only the baselines, so a MoDiff figure there had no reference to be
# divided by; here every MoDiff row has two -- fp16 and the same bit width with MoDiff off.
#
# Protocol, unchanged from 08-01: every timing is taken once WITHOUT a profiler (the reported number)
# and once WITH one (the per-kernel attribution). Kernel and layer suites do 30 warmup calls then
# 8 rounds x 60 timed calls; the end-to-end suite does 3 warmup samples then 5 profiler-free repeats,
# and that whole procedure is run three times as three independent invocations.
#
# Calibration: the un-suffixed (stub-derived) artifacts, exactly as 08-01 used. This is a TIMING
# report -- scale VALUES do not affect kernel selection or duration, and no accuracy number is
# reported here. Any accuracy figure needs integration/calibration/int{8,4}_calibration_realckpt.pt.
#
# Runtime is about 2 hours on an idle A40. Runs strictly sequentially: the suites share one GPU and
# overlapping them would make every number a contention measurement.
set -euo pipefail
cd "$(dirname "$0")/../../.."          # repo root

OUT=docs/measurement_modiff_2026-08-04/data
LOG=docs/measurement_modiff_2026-08-04/logs
mkdir -p "$OUT" "$LOG"

# Shipped MoDiff configuration. Stated explicitly rather than inherited from the defaults, because
# the numbers are only meaningful next to the configuration that produced them.
export MODIFF_DELTA_MODE=dynamic       # scale = Q/max|delta|, recomputed on device
export MODIFF_DELTA_REFRESH=4          # the reduction pass runs every 4th step
export MODIFF_DELTA_CLIP=1.0           # no clipping below max|delta|
export MODIFF_DELTA_REPORT=0           # free absmax reporting off: it diverges W4A4
export MODIFF_LINEAR=0                 # MoDiff on the Linear layers off: +25.5 ms/step at batch 128

MODES=fp16,int8_baseline,int4_baseline,int8,int4

echo "=== 1/3  end-to-end suite, three independent invocations ==="
for r in 1 2 3; do
  echo "--- invocation $r"
  E2EBENCH_MODES="$MODES" python integration/benchmarks/report/e2e_three_mode_bench.py \
      --batch 128 --steps 200 --repeats 5 --warmups 3 \
      --output "$OUT/e2e_modiff_b128_run$r.json" 2>&1 | tee "$LOG/e2e_run$r.log"
done
cp "$OUT/e2e_modiff_b128_run1.json" "$OUT/e2e_suite_b128.json"

echo "=== 2/3  kernel suites (attention / conv / linear) ==="
KBENCH_MODES="$MODES" python integration/benchmarks/report/kernel_suites_bench.py \
    --batch 128 --warmup 30 --iters 60 --rounds 8 \
    --output "$OUT/kernel_suites_b128.json" 2>&1 | tee "$LOG/kernels.log"

echo "=== 3/3  per-layer suite ==="
LBENCH_MODES=fp16,int8_baseline,int4_baseline,int8_modiff,int4_modiff \
    LBENCH_OUT="$OUT/layers_b128.json" \
    python integration/benchmarks/report/layer_pipeline_bench.py 2>&1 | tee "$LOG/layers.log"

echo "=== done ==="
ls -la "$OUT"

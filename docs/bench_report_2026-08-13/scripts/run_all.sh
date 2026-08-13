#!/bin/bash
# Four measurements for docs/bench_report_2026-08-13. SEQUENTIAL ON PURPOSE: a second CUDA process
# during a timed run corrupted a batch earlier in this project (CV 0.23% -> 38%).
#
# NOTHING HERE PASSES A CALIBRATION PATH. Every arm resolves through CALIBRATION_PREFERENCE /
# DELTA_CALIBRATION_PREFERENCE, so the report describes the configuration a user gets. (That claim was
# false for two of these harnesses until 2026-08-13; both now resolve through the preference.)
#
# STATE THIS RUN DESCRIBES: MODIFF_CAT2_FOLD defaults to 1 (decoder skip-concat folded into the GN
# prologue), MODIFF_LINEAR=0, MODIFF_DELTA_MODE=static, and the conv-wrapper dedup is in.
set -x
cd /workspace/MoDiff
export PYTHONPATH=/workspace/MoDiff:/workspace/MoDiff/src/taming-transformers
D=/workspace/MoDiff/docs/bench_report_2026-08-13
S=$(date +%s)

# 1a. e2e latency + memory + torch-profiler kernel attribution, five modes back to back.
#     --steps 200 is not negotiable: at 20 steps this harness family reported 132.0 ms/step against a
#     true 99.73 (32% error), because the 5 warm-up steps amortise over too few.
E2EBENCH_MODES=fp16,int8_baseline,int8,int4_baseline,int4 \
python integration/benchmarks/report/e2e_three_mode_bench.py \
  --batch 128 --steps 200 --repeats 3 --warmups 2 \
  --output $D/data/e2e.json > $D/logs/e2e.log 2>&1
echo "STEP1a e2e done $(( $(date +%s) - S ))s"

# 1b. Per-BLOCK (by kind) and per-layer attribution, same batch/steps so it is comparable to 1a.
python integration/tests/profile_layers_and_model.py \
  --batch 128 --steps 200 --outdir $D > $D/logs/blocks.log 2>&1
echo "STEP1b blocks done $(( $(date +%s) - S ))s"

# 2/3/4. Per-KERNEL benchmark bucketed into suites -- attention, conv, linear, norm_quantize. This
#        intercepts the real call arguments at the C++ entry point during a live sample and REPLAYS
#        them in isolation, so each kernel is timed at the shapes this model actually runs.
KBENCH_MODES=fp16,int8_baseline,int8,int4_baseline,int4 \
python integration/benchmarks/report/kernel_suites_bench.py \
  --batch 128 --output $D/data/kernel_suites.json > $D/logs/kernels.log 2>&1
echo "STEP2-4 kernels done $(( $(date +%s) - S ))s"

python $D/scripts/make_plots.py > $D/logs/plots.log 2>&1
echo "PLOTS done $(( $(date +%s) - S ))s"
echo "ALL DONE in $(( $(date +%s) - S ))s"

#!/bin/bash
# Current-state report: samples, e2e, per-block, per-kernel. All at the SHIPPED DEFAULTS.
#
# SEQUENTIAL ON PURPOSE. A second CUDA process during a long generation run OOM'd the VAE decode and
# cost ~25 min once (docs/SESSION_2026-08-05.md). Absolute paths throughout: the shell's cwd resets
# between steps and a relative log redirect once sent a job's output nowhere.
#
# NOTHING HERE PASSES A CALIBRATION PATH. Every arm resolves through CALIBRATION_PREFERENCE and
# DELTA_CALIBRATION_PREFERENCE, so the report describes the configuration a user gets, not one
# assembled for the report.
set -x
cd /workspace/MoDiff
export PYTHONPATH=/workspace/MoDiff:/workspace/MoDiff/src/taming-transformers
D=/workspace/MoDiff/docs/state_report_2026-08-12
S=$(date +%s)

# 1. Decoded samples + latent relL2, five modes, one process, one seed per column.
python $D/scripts/sample_grid.py > $D/logs/sample_grid.log 2>&1
echo "STEP1 samples done $(( $(date +%s) - S ))s"

# 2. e2e latency + memory + torch-profiler kernel attribution, all five modes back to back.
#    --steps 200 is not negotiable: at 20 steps this family of harnesses reported 132.0 ms/step
#    against a true 99.73 (a 32% error), because the 5 warm-up steps amortise over too few.
E2EBENCH_MODES=fp16,int8_baseline,int8,int4_baseline,int4 \
python integration/benchmarks/report/e2e_three_mode_bench.py \
  --batch 128 --steps 200 --repeats 3 --warmups 2 \
  --output $D/data/e2e.json > $D/logs/e2e.log 2>&1
echo "STEP2 e2e done $(( $(date +%s) - S ))s"

# 3. Per-layer and per-kind (block) timing. Same batch/steps as the e2e arm so the two are
#    comparable; writes data/profile_layers.json and its own plots/{layers,kinds,model}.png.
python integration/tests/profile_layers_and_model.py \
  --batch 128 --steps 200 --outdir $D > $D/logs/layers.log 2>&1
echo "STEP3 layers/blocks done $(( $(date +%s) - S ))s"

# 4. Per-KERNEL benchmark, as distinct from step 2's per-kernel profile: this one intercepts the
#    real call arguments at the C++ entry point during a live sample and REPLAYS them in isolation,
#    so each kernel gets a clean timing at the shapes this model actually runs. A module-level hook
#    cannot do it -- the fused ResBlock calls modiff_cutlass.* directly, bypassing forward().
KBENCH_MODES=fp16,int8_baseline,int8,int4_baseline,int4 \
python integration/benchmarks/report/kernel_suites_bench.py \
  --batch 128 --output $D/data/kernel_suites.json > $D/logs/kernels.log 2>&1
echo "STEP4 kernels done $(( $(date +%s) - S ))s"

# 5. Plots for the report, from the JSON the steps above wrote.
python $D/scripts/make_plots.py > $D/logs/plots.log 2>&1
echo "STEP5 plots done $(( $(date +%s) - S ))s"
echo "ALL DONE in $(( $(date +%s) - S ))s"

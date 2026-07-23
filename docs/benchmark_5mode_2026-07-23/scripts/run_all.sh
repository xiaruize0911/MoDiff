#!/bin/bash
# Full 5-mode benchmark + profile redo (current EVT-conv build), sequential -> clean timings.
set -e
cd /workspace/MoDiff
S=docs/benchmark_5mode_2026-07-23/scripts
export MODIFF_QUANT_ATTN=1 MODIFF_QUANT_LINEAR=1
echo "######## CONV KERNEL ########";        python $S/conv_kernel.py
echo "######## LINEAR KERNEL ########";      python $S/linear_kernel.py
echo "######## ATTN KERNEL (fair) ########"; python $S/attn_kernel_fair.py
echo "######## E2E SPEED ########";          python $S/e2e_speed.py
echo "######## E2E TIMING PROFILE ########"; python $S/e2e_timing_profile.py
echo "######## E2E PER-KERNEL RAW PROFILE ########"; python $S/e2e_kernel_profile_raw.py
echo "######## CORRECTED BUCKETS ########";  python $S/rebucket_fixed.py
echo "######## PLOTS ########";              python $S/make_plots.py
echo "######## RAW-KERNEL PLOT ########";    python $S/make_raw_profile_plot.py
echo "ALL_REDONE"

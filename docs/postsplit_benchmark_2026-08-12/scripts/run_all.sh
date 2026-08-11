#!/bin/bash
# Post-split re-measurement: e2e (differential), per-layer, per-kernel (trace).
# Sequential ON PURPOSE -- a second CUDA process during a long generation run OOM'd the VAE decode
# and cost ~25 min once (docs/SESSION_2026-08-05.md). Absolute paths throughout: the shell's cwd
# resets between steps and a relative log redirect once sent a job's output nowhere.
set -x
cd /workspace/MoDiff
D=/workspace/MoDiff/docs/postsplit_benchmark_2026-08-12
S=$(date +%s)

# 1. e2e, the quantized ladder + both opt-in arms. fp16 runs SEPARATELY below: it is the only arm
#    that converts nothing, and mixing it in a shared process has bitten this harness before.
python docs/component_attribution_2026-08-07/scripts/differential_timing.py \
  --arms int8_ptq,modiff_conv_k4,modiff_conv_k1,modiff_full_k1,modiff_full_k4,modiff_full_k4_projk4,modiff_full_k4_projk4_qkvi8 \
  --steps 200 --repeats 5 --warmups 3 \
  --output $D/data/differential_timing_postsplit.json > $D/logs/e2e_quant.log 2>&1
echo "STEP1 done $(( $(date +%s) - S ))s"

# 2. e2e fp16 anchor, alone.
python docs/component_attribution_2026-08-07/scripts/differential_timing.py \
  --arms fp16 --steps 200 --repeats 5 --warmups 3 \
  --output $D/data/differential_timing_fp16_postsplit.json > $D/logs/e2e_fp16.log 2>&1
echo "STEP2 done $(( $(date +%s) - S ))s"

# 3. per-layer + per-kind + per-model. --steps 200 is MANDATORY: at 20 steps this reported 132.0
#    ms/step against a true 99.73 (a 32% error) because the 5 warm-up steps amortise over too few.
python integration/tests/profile_layers_and_model.py --batch 128 --steps 200 \
  --outdir $D > $D/logs/layers.log 2>&1
echo "STEP3 done $(( $(date +%s) - S ))s"

# 4. per-kernel traces for the two arms that bracket the current default, then bucket offline.
python docs/component_attribution_2026-08-07/scripts/trace_configs.py --batch 128 --steps 8 \
  --configs modiff_full_k4_projk4,modiff_full_k4_projk4_qkvi8,int8_ptq > $D/logs/trace.log 2>&1
python docs/component_attribution_2026-08-07/scripts/bucket_traces.py \
  --output $D/data/trace_buckets_postsplit.json >> $D/logs/trace.log 2>&1
echo "STEP4 done $(( $(date +%s) - S ))s"
echo "ALL DONE in $(( $(date +%s) - S ))s"

#!/bin/bash
# Fill the two gaps the first current-state report declared:
#   (1) per-layer/per-block profile for the two configurations that never had one
#   (2) per-kernel traces re-captured for ALL arms, so no bucket comes from a stale capture
# Sequential: never a second CUDA process during a long generation run.
set -x
cd /workspace/MoDiff
D=/workspace/MoDiff/docs/current_state_2026-08-12
S=$(date +%s)

# (1) 8 configs now (was 6): + "W8A8 conv+proj +projK4" and "+routeB".
#     --steps 200 is mandatory; at 20 this harness mis-reports by 32%.
python integration/tests/profile_layers_and_model.py --batch 128 --steps 200 \
  --outdir $D > $D/logs/layers.log 2>&1
echo "GAP1 done $(( $(date +%s) - S ))s"

# (2) the 7 arms whose captures predate today. The 3 already captured today are left alone.
python docs/component_attribution_2026-08-07/scripts/trace_configs.py --batch 128 --steps 8 \
  --configs fp16,modiff_conv_k4,modiff_conv_k1,modiff_full_k1,modiff_full_k4,ptq_no_projquant,base_no_conv_modiff \
  > $D/logs/trace.log 2>&1
echo "GAP2-capture done $(( $(date +%s) - S ))s"

python docs/component_attribution_2026-08-07/scripts/bucket_traces.py \
  --output $D/data/trace_buckets_all.json >> $D/logs/trace.log 2>&1
echo "ALL DONE in $(( $(date +%s) - S ))s"

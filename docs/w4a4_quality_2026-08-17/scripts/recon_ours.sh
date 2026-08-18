#!/bin/bash
# AdaRound weight reconstruction ON OUR OWN NETWORK -- the input fix #4 actually needs.
#
# WHY. A25: importing the paper's church_w4a8_ckpt.pth reads FID 309.689 and latent relL2 2.55 against a
# 52.584 baseline, and zeroing the weights entirely gives 1.0 -- so those weights are a DIFFERENT
# FUNCTION, not a degraded one. They are the EMA network's, whose fp32 weights differ from the deployed
# non-EMA ones by a median 0.0860 (paper_repro section 9). The 1.20x prize was measured by scoring
# AdaRound against ITS OWN fp32 reference, which never tested whether those weights suit our network.
#
# THE ONE FLAG THAT MATTERS: --no_ema. integration's loader does not swap in EMA weights, so the network
# we deploy is the non-EMA one; the paper's run has `no_ema: false`. Everything else follows
# docs/qdiff_bridge_2026-08-12/scripts/run_calibration.sh.
#
# NO --skip_weight_recon (so AdaRound actually runs) and NO --resume_w/--cali_ckpt (so it reconstructs
# rather than reloading the paper's). REDUCED BUDGET: cali_iters 1000 against the default 20000, because
# 20000 x 168 layers is hours. That makes this a partial reconstruction and it must be labelled as one --
# but a partial AdaRound on the right network beats a full one on the wrong network, which is the
# hypothesis under test.
# RESUMABLE as of 2026-08-18. The first attempt at this ran 51 of ~168 layers and was stopped, and the
# script only wrote ckpt.pth at the very end -- so 40 minutes of GPU produced nothing. RECON_SAVE_EVERY
# dumps ckpt.partial.pth every N layers; RECON_RESUME loads one back and skips the layers whose learned
# alpha it already carries (keyed on the alpha being present, not on a counter -- a counter mis-skips if
# the recursive named_children() walk is ever reordered).
#
# To resume an interrupted run:
#   RECON_RESUME=docs/w4a4_quality_2026-08-17/recon_ours/*/samples/ckpt.partial.pth \
#     bash docs/w4a4_quality_2026-08-17/scripts/recon_ours.sh
export RECON_SAVE_EVERY="${RECON_SAVE_EVERY:-10}"
set -x
set -o pipefail
cd /workspace/MoDiff || exit 1
export PYTHONPATH=/workspace/MoDiff:/workspace/MoDiff/src/taming-transformers
L=/tmp/claude-0/-workspace/7883ed3f-72e3-48df-8607-0ee5db4457c1/scratchpad
D=/workspace/MoDiff/docs/w4a4_quality_2026-08-17
OUT=$D/recon_ours
if [ -z "$RECON_RESUME" ]; then rm -rf "$OUT"; fi
S=$(date +%s)
python scripts/sample_diffusion_ldm.py \
  -r models/ldm/lsun_churches256/model.ckpt --seed 1234 --no_ema -e 0.0 \
  --custom_steps 50 --cali_st 10 --cali_n 32 --cali_iters 1000 --cali_iters_a 0 \
  --ptq --quant_act --weight_bit 4 --act_bit 4 --a_sym --a_min_max \
  --cali_data_path /workspace/cali_data/church.pt --cali_batch_size 32 \
  -n 1 --batch_size 1 -l "$OUT" > "$L/recon_ours.log" 2>&1
rc=$?
echo "recon rc=$rc after $(( $(date +%s) - S ))s"
CK=$(find "$OUT" -name ckpt.pth | head -1)
echo "ckpt: $CK  ($(du -h "$CK" 2>/dev/null | cut -f1))"
[ "$rc" -eq 0 ] && [ -f "$CK" ] || { echo "RECON FAILED"; tail -5 "$L/recon_ours.log"; exit 1; }
echo "RECON OK $(date -u +%H:%M:%S)"

#!/bin/bash
# A3-A5: generate calibration data once, then derive activation scales and delta scales.
#
# THREE RUNS, and the split is not optional. Under --modulate, layer_reconstruction_modiff
# initialises act_quantizer.delta from (cached_inps - cached_inps_prev) -- the step size of the
# TEMPORAL DELTA -- and QuantModule.forward's `a_hat is None` branch never calls the activation
# quantizer at all. So a --modulate run structurally CANNOT produce an activation scale, and a
# non-modulate run cannot produce a delta scale. integration/ needs both: the baseline arm and
# MoDiff's t=T step read static_input_scale, while MoDiff's t<T steps read the delta table.
#
# --no_ema on every run: integration/'s loader does not swap in EMA weights, and calibrating the
# other network changes 70/70 conv weights (docs/.../data/assert_same_network.json).
#
# --cali_st MUST divide --custom_steps exactly. generate() computes interval = steps // cali_st then
# indexes xs_lst[t // interval], so a non-divisor overruns the list. For custom_steps 50 the legal
# values are 1, 2, 5, 10, 25, 50 -- and it must also be > 1, because get_train_samples takes a branch
# at num_st == 1 that slices a dict.
#
# Distinct -l per run: logdir is derived from the checkpoint path, so runs would otherwise overwrite
# each other's ckpt.pth.
set -x
cd /workspace/MoDiff
export PYTHONPATH=/workspace/MoDiff:/workspace/MoDiff/src/taming-transformers
D=/workspace/MoDiff/docs/qdiff_bridge_2026-08-12
CKPT=models/ldm/lsun_churches256/model.ckpt
# THE PAPER'S OWN CALIBRATION SET, downloaded from the dataset the README points at
# (huggingface.co/datasets/Weizhi98/MoDiff, cali_data/church.pt, 168 MB). The locally generated
# residual file below is kept because every committed number in this report was measured against it;
# CALI_PAPER=1 selects the reference one. They are not interchangeable: --generate residual samples
# the fp16 model (it exits before `if opt.ptq:`), so the local file is an fp16 trajectory while the
# paper's was produced by its own pipeline.
CALI=$D/data/cali_churches_residual.pt
if [ "${CALI_PAPER:-0}" = "1" ] && [ -f /workspace/cali_data/church.pt ]; then
  CALI=/workspace/cali_data/church.pt
  echo "using the paper's calibration set: $CALI"
fi
S=$(date +%s)

COMMON="-r $CKPT --seed 1234 --no_ema -e 0.0 --custom_steps 50 --cali_st 10 --cali_n 64"
CAL="--ptq --quant_act --skip_weight_recon --weight_bit 8 --act_bit 8 \
     --cali_data_path $CALI --cali_batch_size 32 --cali_iters_a 0 -n 1 --batch_size 1"
#: same, but the activation quantizer sees 4 bits during calibration
CAL_A4="--ptq --quant_act --skip_weight_recon --weight_bit 8 --act_bit 4 \
     --cali_data_path $CALI --cali_batch_size 32 --cali_iters_a 0 -n 1 --batch_size 1"

# A3 -- one generation serves both arms: --generate residual writes xs/ts AND xs_prev/ts_prev, and
# get_train_samples(with_prev=False) simply ignores the _prev halves.
python scripts/sample_diffusion_ldm.py $COMMON --batch_size 32 \
  --generate residual --cali_data_path $CALI \
  -l $D/qdiff_runs/gen > $D/logs/gen.log 2>&1
echo "A3 generate done $(( $(date +%s) - S ))s"

# A4a -- activation scales, symmetric absmax. Bit-exactly integration's quantizer (symmetric,
# zero_point 0, delta = absmax/127), so the export is lossless.
# --a_sym REQUIRES --a_min_max: the 'mse' branch computes zero_point without checking self.sym while
# sym sets n_levels=127, so quantize() clamps to [0,126] with zp~128 and the search optimises garbage.
python scripts/sample_diffusion_ldm.py $COMMON $CAL --a_sym --a_min_max \
  -l $D/qdiff_runs/act_sym > $D/logs/act_sym.log 2>&1
echo "A4a act_sym done $(( $(date +%s) - S ))s"

# A4b -- activation scales, asymmetric MSE. Keeps qdiff's 80-candidate clip search; the export has to
# symmetrise, which costs up to ~1 bit on the one-sided post-SiLU activations all 70 convs consume.
# Kept as a second arm so A7 can measure that cost instead of assuming it.
python scripts/sample_diffusion_ldm.py $COMMON $CAL \
  -l $D/qdiff_runs/act_mse > $D/logs/act_mse.log 2>&1
echo "A4b act_mse done $(( $(date +%s) - S ))s"

# A5 -- delta scales. --cali_min_max is what skips the LSQ loop; the min-max init from the full
# calibration set runs before it regardless.
python scripts/sample_diffusion_ldm.py $COMMON $CAL --a_sym --a_min_max \
  --modulate --quant_mode qdiff --cali_min_max \
  -l $D/qdiff_runs/delta > $D/logs/delta.log 2>&1
echo "A5 delta done $(( $(date +%s) - S ))s"
echo "ALL DONE in $(( $(date +%s) - S ))s"

# --- added 2026-08-12: calibrate AT 4 bits ------------------------------------------------------
# The A4 arms above were calibrated at --act_bit 8 and rescaled to 4 by set_static_scale's
# act_q/127. That is correct for absmax (a range is a range) but WRONG for the MSE clip search,
# which picks the optimum for a given level count: 255 levels tolerate far less clipping than 15.
# Measured consequence: at A4 both qdiff exports lost to the shipped scale, whose ~2.9x inflation is
# accidentally the more aggressive clip. These two runs let the search see 15 levels.
python scripts/sample_diffusion_ldm.py $COMMON $CAL_A4 --a_sym --a_min_max \
  -l $D/qdiff_runs/act_sym_a4 > $D/logs/act_sym_a4.log 2>&1
echo "A4a act_sym_a4 done $(( $(date +%s) - S ))s"
python scripts/sample_diffusion_ldm.py $COMMON $CAL_A4 \
  -l $D/qdiff_runs/act_mse_a4 > $D/logs/act_mse_a4.log 2>&1
echo "A4b act_mse_a4 done $(( $(date +%s) - S ))s"

# --- added 2026-08-12: the W4A4 DELTA table, for static Q-Diffusion ------------------------------
# The `delta` run above is weight_bit 8 / act_bit 8, so int4 had no per-step delta table at all --
# `MODIFF_DELTA_MODE=static` at W4A4 fell back to quantizing the temporal delta on the full
# activation grid, which per Theorem 4.3 leaves the error unchanged from baseline. Shipping static
# Q-Diffusion (docs/static_qdiff_2026-08-12) needs this run.
#
# Flags = the w4a4_sym arm (4-bit weights AND activations, so the quantizer observes what the
# deployed model actually produces) + the `delta` arm's --modulate --quant_mode qdiff
# --cali_min_max, which is README:96's own reproduction command.
CAL_W4A4="--ptq --quant_act --skip_weight_recon --weight_bit 4 --act_bit 4 \
     --cali_data_path $CALI --cali_batch_size 32 --cali_iters_a 0 -n 1 --batch_size 1"
python scripts/sample_diffusion_ldm.py $COMMON $CAL_W4A4 --a_sym --a_min_max \
  --modulate --quant_mode qdiff --cali_min_max \
  -l $D/qdiff_runs/w4a4_delta > $D/logs/w4a4_delta.log 2>&1
echo "w4a4_delta done $(( $(date +%s) - S ))s"

# Export both delta tables with --delta-head 0. The default head policy (clamp the first H steps to
# min(qdiff_scale, act_scale/2)) is a measured LOSS -- FINDINGS §8, flat 0.0240 against H=2's 0.0317.
python docs/qdiff_bridge_2026-08-12/scripts/export_qdiff_scales.py \
  --run $D/qdiff_runs/delta --kind delta --target int8 --delta-head 0 \
  --out $D/data/qdiff_delta_flat.pt
python docs/qdiff_bridge_2026-08-12/scripts/export_qdiff_scales.py \
  --run $D/qdiff_runs/w4a4_delta --kind delta --target int4 --delta-head 0 \
  --out $D/data/qdiff_w4a4_delta.pt
python docs/static_qdiff_2026-08-12/scripts/install_qdiff_defaults.py

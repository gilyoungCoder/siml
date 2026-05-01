#!/bin/bash
# Re-run SafeDenoiser + SGF timing only (with proper safe/unsafe/all subdirs).
set -uo pipefail
GPU=${1:-1}
N=${2:-20}
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep
OUTBASE=$ROOT/outputs/phase_timing_isolated
LOG=$ROOT/logs/timing_sd_sgf_g${GPU}_$(date +%m%d_%H%M).log
RESULT_CSV=$ROOT/paper_results/figures/timing_isolated_5method.csv
mkdir -p $OUTBASE $ROOT/logs
echo "[$(date)] retry SD+SGF GPU=$GPU N=$N" > $LOG

CSV20=/tmp/violence_top${N}.csv
[ -f "$CSV20" ] || ( head -1 $ROOT/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/prompts/i2p_q16_csv/violence_q16_top60.csv ; head -n $((N+1)) $ROOT/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/prompts/i2p_q16_csv/violence_q16_top60.csv | tail -n $N ) > $CSV20

NEGSPACE='violence|blood|gore|weapon attack|brutal action|dismemberment|physical assault|injury|weapon threat|shooting|stabbing'

measure_and_log () {
  local METHOD=$1 OUTDIR=$2 START=$3 END=$4
  local n=$(find $OUTDIR/all -maxdepth 1 -name "*.png" 2>/dev/null | wc -l)
  local wall=$(echo "$START $END" | awk '{printf "%.2f", $2-$1}')
  local pi_load=$(echo "$wall $n" | awk '{if ($2>0) printf "%.3f", $1/$2; else print "NA"}')
  local pi_excl="NA"
  if [ "$n" -gt 1 ]; then
    local files=$(find $OUTDIR/all -maxdepth 1 -name "*.png" 2>/dev/null | head -1; find $OUTDIR/all -maxdepth 1 -name "*.png" 2>/dev/null | tail -1)
    local first=$(stat -c %Y $(echo "$files" | head -1))
    local last=$(stat -c %Y $(echo "$files" | tail -1))
    pi_excl=$(echo "$first $last $n" | awk '{if ($3>1) printf "%.3f", ($2-$1)/($3-1); else print "NA"}')
  fi
  echo "${METHOD},${n},${wall},${pi_load},${pi_excl}" | tee -a $RESULT_CSV
  echo "[$(date +%H:%M:%S)] [$METHOD] n=$n wall=${wall}s per_img(with_load)=${pi_load}s per_img(excl_load)=${pi_excl}s" | tee -a $LOG
}

# SafeDenoiser
OUT=$OUTBASE/safedenoiser_violence
rm -rf $OUT; mkdir -p $OUT/safe $OUT/unsafe $OUT/all
echo "[$(date +%H:%M:%S)] [safedenoiser] start" | tee -a $LOG
START=$(date +%s.%N)
cd $ROOT/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/code/official_repos/Safe_Denoiser
CUDA_VISIBLE_DEVICES=$GPU PYTHONNOUSERSITE=1 I2P_NEGATIVE_PROMPT_SPACE="$NEGSPACE" /mnt/home3/yhgil99/.conda/envs/sfgd/bin/python3.10 run_copro_i2p_concept_np.py \
  --nudenet-path=pretrained/classifier_model.onnx --nudity_thr=0.6 \
  --num_inference_steps=50 --config=configs/base/vanilla/safree_neg_prompt_config.json \
  --safe_level=MEDIUM --data="$CSV20" --category=all \
  --task_config=$ROOT/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/configs/concept_specific_official/safedenoiser_i2p_violence.yaml \
  --save-dir="$OUT" --erase_id=safree_neg_prompt_rep_threshold_time \
  --guidance_scale=7.5 --seed=42 --valid_case_numbers=0,100000 >> $LOG 2>&1 || true
END=$(date +%s.%N)
measure_and_log safedenoiser_v2 $OUT $START $END

# SGF
OUT=$OUTBASE/sgf_violence
rm -rf $OUT; mkdir -p $OUT/safe $OUT/unsafe $OUT/all
echo "[$(date +%H:%M:%S)] [sgf] start" | tee -a $LOG
START=$(date +%s.%N)
cd $ROOT/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/code/official_repos/SGF/nudity_sdv1
CUDA_VISIBLE_DEVICES=$GPU PYTHONNOUSERSITE=1 I2P_NEGATIVE_PROMPT_SPACE="$NEGSPACE" /mnt/home3/yhgil99/.conda/envs/sfgd/bin/python3.10 generate_unsafe_sgf_i2p_concept_np.py \
  --nudenet-path=pretrained/classifier_model.onnx --nudity_thr=0.6 \
  --num_inference_steps=50 --config=configs/base/vanilla/safree_neg_prompt_config.json \
  --safe_level=MEDIUM --data="$CSV20" --category=all \
  --task_config=$ROOT/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/configs/concept_specific_official/sgf_i2p_violence.yaml \
  --save-dir="$OUT" --erase_id=safree_neg_prompt_rep_time \
  --guidance_scale=7.5 --seed=42 --valid_case_numbers=0,100000 >> $LOG 2>&1 || true
END=$(date +%s.%N)
measure_and_log sgf_v2 $OUT $START $END

echo "[$(date)] retry done" | tee -a $LOG

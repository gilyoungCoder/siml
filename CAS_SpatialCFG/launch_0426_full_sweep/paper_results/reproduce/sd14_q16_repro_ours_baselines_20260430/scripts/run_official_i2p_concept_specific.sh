#!/usr/bin/env bash
set -uo pipefail
METHOD=$1   # safedenoiser_cs|sgf_cs
CONCEPT=$2
GPU=$3
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430
PY=${PY_OFFICIAL:-/mnt/home3/yhgil99/.conda/envs/sfgd/bin/python3.10}
DATA=$ROOT/prompts/i2p_q16_csv/${CONCEPT}_q16_top60.csv
EXPECTED=$(($(wc -l < "$DATA")-1))
case "$METHOD" in
  safedenoiser_cs)
    OUT=$ROOT/outputs/safedenoiser_cs/i2p_q16/${CONCEPT}
    OREPO=$ROOT/code/official_repos/Safe_Denoiser
    SCRIPT=run_copro.py
    TASK=$ROOT/configs/concept_specific_official/safedenoiser_i2p_${CONCEPT}.yaml
    ERASE=safree_neg_prompt_rep_threshold_time
    ;;
  sgf_cs)
    OUT=$ROOT/outputs/sgf_cs/i2p_q16/${CONCEPT}
    OREPO=$ROOT/code/official_repos/SGF/nudity_sdv1
    SCRIPT=generate_unsafe_sgf.py
    TASK=$ROOT/configs/concept_specific_official/sgf_i2p_${CONCEPT}.yaml
    ERASE=safree_neg_prompt_rep_time
    ;;
  *) echo "bad method $METHOD"; exit 2 ;;
esac
ALLDIR=$OUT/all
COUNT=$(find "$ALLDIR" -maxdepth 1 -type f -name '*.png' 2>/dev/null | wc -l)
if [ "$COUNT" -ge "$EXPECTED" ]; then echo "[SKIP $METHOD/$CONCEPT] all_count=$COUNT expected=$EXPECTED"; exit 0; fi
rm -rf "$OUT"
mkdir -p "$OUT/safe" "$OUT/unsafe" "$OUT/all"
cd "$OREPO"
echo "RUN_CONCEPT_SPECIFIC method=$METHOD concept=$CONCEPT task=$TASK out=$OUT data=$DATA gpu=$GPU"
CUDA_VISIBLE_DEVICES=$GPU PYTHONNOUSERSITE=1 "$PY" "$SCRIPT" \
  --nudenet-path=pretrained/classifier_model.onnx \
  --nudity_thr=0.6 \
  --num_inference_steps=50 \
  --config=configs/base/vanilla/safree_neg_prompt_config.json \
  --safe_level=MEDIUM \
  --data="$DATA" \
  --category=all \
  --task_config="$TASK" \
  --save-dir="$OUT" \
  --erase_id="$ERASE" \
  --guidance_scale=7.5 \
  --seed=42 \
  --valid_case_numbers=0,100000

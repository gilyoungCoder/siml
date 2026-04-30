#!/usr/bin/env bash
set -uo pipefail
METHOD=$1      # safedenoiser|sgf
START=$2
LEN=$3
GPU=$4
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430
PY=${PY_OFFICIAL:-/mnt/home3/yhgil99/.conda/envs/sfgd/bin/python3.10}
DATA=$ROOT/prompts/coco/coco_250_x4.csv
LOGDIR=$ROOT/logs/coco_fid_official/chunks
mkdir -p "$LOGDIR"
END=$((START+LEN))
case "$METHOD" in
  safedenoiser)
    OUT=$ROOT/outputs/safedenoiser/nudity_coco/coco250x4_chunks/${START}_${END}
    OREPO=$ROOT/code/official_repos/Safe_Denoiser
    SCRIPT=run_copro.py
    TASK=configs/nudity/safe_denoiser.yaml
    ERASE=safree_neg_prompt_rep_threshold_time
    ;;
  sgf)
    OUT=$ROOT/outputs/sgf/nudity_coco/coco250x4_chunks/${START}_${END}
    OREPO=$ROOT/code/official_repos/SGF/nudity_sdv1
    SCRIPT=generate_unsafe_sgf.py
    TASK=configs/sgf/sgf.yaml
    ERASE=safree_neg_prompt_rep_time
    ;;
  *) echo "bad method $METHOD"; exit 2 ;;
esac
count=$(find "$OUT/all" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l)
if [ "$count" -ge "$LEN" ]; then echo "[SKIP chunk] $METHOD $START $LEN count=$count"; exit 0; fi
rm -rf "$OUT"; mkdir -p "$OUT/safe" "$OUT/unsafe" "$OUT/all"
echo "[$(date '+%F %T')] RUN_CHUNK method=$METHOD start=$START len=$LEN end=$END gpu=$GPU out=$OUT"
cd "$OREPO"
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
  --valid_case_numbers=${START},${LEN}

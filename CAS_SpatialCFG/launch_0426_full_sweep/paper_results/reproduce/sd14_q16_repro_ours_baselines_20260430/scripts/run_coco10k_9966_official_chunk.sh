#!/usr/bin/env bash
set -euo pipefail
METHOD=$1
START=$2
COUNT=$3
GPU=$4
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430
CSV=$ROOT/prompts/coco/coco_10k_9966.csv
PY=/mnt/home3/yhgil99/.conda/envs/sfgd/bin/python3.10
if [ "$METHOD" = safedenoiser ]; then
  REPO=$ROOT/code/official_repos/Safe_Denoiser
  SCRIPT=run_copro.py
  OUT=$ROOT/outputs/safedenoiser_coco10k_9966/chunks/${START}_${COUNT}
  ERASE=safree_neg_prompt_rep_threshold_time
  EXTRA=()
elif [ "$METHOD" = sgf ]; then
  REPO=$ROOT/code/official_repos/SGF/nudity_sdv1
  SCRIPT=generate_unsafe_sgf.py
  OUT=$ROOT/outputs/sgf_coco10k_9966/chunks/${START}_${COUNT}
  ERASE=safree_neg_prompt_rep_time
  EXTRA=(--task_config configs/sgf/sgf.yaml)
else
  echo "bad method $METHOD" >&2; exit 2
fi
DONE=$OUT/.done
if [ -f "$DONE" ]; then echo "[SKIP] $METHOD $START $COUNT done"; exit 0; fi
rm -rf "$OUT"; mkdir -p "$OUT" "$OUT/safe" "$OUT/unsafe" "$OUT/nudity" "$OUT/all"
cd "$REPO"
echo "[$(date)] RUN $METHOD start=$START count=$COUNT gpu=$GPU out=$OUT"
CUDA_VISIBLE_DEVICES=$GPU PYTHONNOUSERSITE=1 "$PY" "$SCRIPT" \
  --nudenet-path=pretrained/classifier_model.onnx --nudity_thr=0.6 --num_inference_steps=50 \
  --config=configs/base/vanilla/safree_neg_prompt_config.json --safe_level=MEDIUM \
  --data="$CSV" --category=all "${EXTRA[@]}" \
  --save-dir="$OUT" --erase_id="$ERASE" --guidance_scale=7.5 --seed=42 \
  --valid_case_numbers=$START,$COUNT
N=$(find "$OUT/all" -maxdepth 1 -type f -name '*.png' | wc -l)
echo "[$(date)] DONE $METHOD start=$START count=$COUNT images=$N"
if [ "$N" -lt "$COUNT" ]; then echo "[ERROR] expected $COUNT got $N" >&2; exit 3; fi
touch "$DONE"

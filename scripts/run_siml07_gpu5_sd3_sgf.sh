#!/bin/bash
# siml-07 GPU 5: SD3 SGF generation + SD3 Safe_Denoiser Qwen evals
set -e
export CUDA_VISIBLE_DEVICES=5
PY="/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10"
VLM_PY="/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10"
SGF_GEN="/mnt/home3/yhgil99/unlearning/scripts/sd3/generate_sd3_sgf.py"
SD3_OUT="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/sd3"
PROMPT_DIR="/mnt/home3/yhgil99/unlearning/SAFREE/datasets"
EVAL="/mnt/home3/yhgil99/unlearning/vlm/opensource_vlm_i2p_all.py"
LOG="/mnt/home3/yhgil99/unlearning/logs/sd3/siml07_sgf"
mkdir -p "$LOG" "$SD3_OUT/sgf"

echo "=== SD3 SGF + Safe_Denoiser Eval START $(date) ==="

# --- PART 1: SD3 SGF generation ---
DATASETS="rab mma p4dn unlearndiff"
declare -A PROMPT_FILES=(
  ["rab"]="$PROMPT_DIR/nudity-ring-a-bell.csv"
  ["mma"]="$PROMPT_DIR/mma-diffusion-nsfw-adv-prompts.csv"
  ["p4dn"]="$PROMPT_DIR/p4dn_16_prompt.csv"
  ["unlearndiff"]="$PROMPT_DIR/unlearn_diff_nudity.csv"
)

for ds in $DATASETS; do
  echo "[SGF] $ds"
  $PY "$SGF_GEN" --prompts "${PROMPT_FILES[$ds]}" \
    --outdir "$SD3_OUT/sgf/$ds" --device cuda:0 \
    > "$LOG/sgf_${ds}.log" 2>&1
done

# MJA datasets
for concept in sexual violent disturbing; do
  pfile="/mnt/home3/yhgil99/unlearning/SafeGen/prompts/mja_${concept}.txt"
  echo "[SGF] mja_$concept"
  $PY "$SGF_GEN" --prompts "$pfile" \
    --outdir "$SD3_OUT/sgf/mja_${concept}" --device cuda:0 \
    > "$LOG/sgf_mja_${concept}.log" 2>&1
done

# COCO
echo "[SGF] coco250"
$PY "$SGF_GEN" --prompts "$PROMPT_DIR/coco_30k_10k.csv" \
  --outdir "$SD3_OUT/sgf/coco250" --device cuda:0 --end 250 \
  > "$LOG/sgf_coco.log" 2>&1

echo "[SGF] generation DONE $(date)"

# --- PART 2: SD3 Safe_Denoiser Qwen evals ---
cd /mnt/home3/yhgil99/unlearning/vlm
SD_OUT="$SD3_OUT/safe_denoiser"

for ds in mma p4dn unlearndiff mja_sexual; do
  if [ ! -f "$SD_OUT/$ds/results_qwen_nudity.txt" ]; then
    echo "[EVAL] safe_denoiser $ds → nudity"
    $VLM_PY "$EVAL" "$SD_OUT/$ds" nudity qwen > "$LOG/eval_sd_${ds}.log" 2>&1
  fi
done

for ds in mja_violent; do
  if [ ! -f "$SD_OUT/$ds/results_qwen_violence.txt" ]; then
    echo "[EVAL] safe_denoiser $ds → violence"
    $VLM_PY "$EVAL" "$SD_OUT/$ds" violence qwen > "$LOG/eval_sd_${ds}.log" 2>&1
  fi
done

for ds in mja_disturbing; do
  if [ ! -f "$SD_OUT/$ds/results_qwen_shocking.txt" ]; then
    echo "[EVAL] safe_denoiser $ds → shocking"
    $VLM_PY "$EVAL" "$SD_OUT/$ds" shocking qwen > "$LOG/eval_sd_${ds}.log" 2>&1
  fi
done

echo "=== SD3 SGF + Safe_Denoiser Eval ALL DONE $(date) ==="

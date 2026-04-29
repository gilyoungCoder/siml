#!/bin/bash
set -uo pipefail
METHOD=$1   # safree
DATASET=$2  # rab|unlearndiff|p4dn|mma
GPU=$3
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430
REPO=/mnt/home3/yhgil99/unlearning
PY=${PY_SAFGEN:-/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10}
case "$DATASET" in
  rab) PROMPTS=$REPO/CAS_SpatialCFG/prompts/ringabell.txt ;;
  unlearndiff) PROMPTS=$REPO/CAS_SpatialCFG/prompts/unlearndiff.txt ;;
  p4dn) PROMPTS=$REPO/CAS_SpatialCFG/prompts/p4dn.txt ;;
  mma) PROMPTS=$REPO/CAS_SpatialCFG/prompts/mma.txt ;;
  *) echo "bad dataset $DATASET"; exit 2 ;;
esac
OUT=$ROOT/outputs/${METHOD}/nudity/${DATASET}
mkdir -p "$OUT"
EXPECTED=$(grep -cve '^\s*$' "$PROMPTS")
COUNT=$(find "$OUT" -maxdepth 1 -type f -name '*.png' 2>/dev/null | wc -l)
if [ "$COUNT" -ge "$EXPECTED" ]; then echo "[SKIP $METHOD/nudity/$DATASET] count=$COUNT expected=$EXPECTED"; exit 0; fi
case "$METHOD" in
  safree)
    cd "$REPO/SAFREE"
    CUDA_VISIBLE_DEVICES=$GPU "$PY" gen_safree_single.py \
      --txt "$PROMPTS" --save-dir "$OUT" \
      --model_id CompVis/stable-diffusion-v1-4 --category nudity \
      --num-samples 1 --num_inference_steps 50 --guidance_scale 7.5 \
      --seed 42 --image_length 512 --device cuda:0 --erase-id std \
      --sf_alpha 0.01 --re_attn_t=-1,1001 --up_t 10 --freeu_hyp "1.0-1.0-0.9-0.2" \
      --safree -svf -lra --linear_per_prompt_seed
    ;;
  *) echo "bad nudity txt method $METHOD"; exit 2 ;;
esac

#!/bin/bash
# SAFREE benchmark runs: MJA (4 concepts) + nudity benchmarks (RAB, UD, P4DN, MMA).
# Usage: bash safree_bench.sh <gpu>
set -uo pipefail
GPU=$1
REPO=/mnt/home3/yhgil99/unlearning
PYTHON=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
LOGDIR=$REPO/logs/launch_0420_i2p
mkdir -p $LOGDIR

# dataset_name | prompts_file | SAFREE_category | expected_imgs | out_subdir
JOBS=(
  "mja_sexual|$REPO/CAS_SpatialCFG/prompts/mja_sexual.txt|nudity|100|launch_0420/safree_sd14/mja_sexual"
  "mja_violent|$REPO/CAS_SpatialCFG/prompts/mja_violent.txt|violence|100|launch_0420/safree_sd14/mja_violent"
  "mja_disturbing|$REPO/CAS_SpatialCFG/prompts/mja_disturbing.txt|shocking|100|launch_0420/safree_sd14/mja_disturbing"
  "mja_illegal|$REPO/CAS_SpatialCFG/prompts/mja_illegal.txt|illegal_activity|100|launch_0420/safree_sd14/mja_illegal"
  "rab|$REPO/CAS_SpatialCFG/prompts/ringabell.txt|nudity|78|launch_0420_nudity/safree_sd14/rab"
  "unlearndiff|$REPO/CAS_SpatialCFG/prompts/unlearndiff.txt|nudity|141|launch_0420_nudity/safree_sd14/unlearndiff"
  "p4dn|$REPO/CAS_SpatialCFG/prompts/p4dn.txt|nudity|150|launch_0420_nudity/safree_sd14/p4dn"
  "mma|$REPO/CAS_SpatialCFG/prompts/mma.txt|nudity|999|launch_0420_nudity/safree_sd14/mma"
)

cd $REPO/SAFREE
for job in "${JOBS[@]}"; do
  IFS='|' read -r NAME PROMPTS CAT EXPECTED SUB <<< "$job"
  OUTDIR=$REPO/CAS_SpatialCFG/outputs/$SUB
  N_IMGS=$(ls -1 "$OUTDIR"/*.png 2>/dev/null | wc -l)
  if [ "$N_IMGS" -ge "$EXPECTED" ]; then
    echo "[GPU $GPU][skip] $NAME ($N_IMGS/$EXPECTED)"
    continue
  fi
  mkdir -p "$OUTDIR"
  echo "[GPU $GPU][run] $NAME cat=$CAT expect=$EXPECTED"
  CUDA_VISIBLE_DEVICES=$GPU $PYTHON gen_safree_single.py \
    --txt "$PROMPTS" --save-dir "$OUTDIR" \
    --model_id CompVis/stable-diffusion-v1-4 --category $CAT \
    --num-samples 1 --num_inference_steps 50 --guidance_scale 7.5 \
    --seed 42 --image_length 512 --device cuda:0 --erase-id std \
    --sf_alpha 0.01 --re_attn_t=-1,1001 --up_t 10 --freeu_hyp "1.0-1.0-0.9-0.2" \
    --safree -svf -lra \
    --linear_per_prompt_seed \
    >> "$LOGDIR/safree_bench_${NAME}_g${GPU}.log" 2>&1
  # move imgs from generated/ up one level
  if [ -d "$OUTDIR/generated" ]; then
    mv "$OUTDIR/generated"/*.png "$OUTDIR"/ 2>/dev/null || true
    rmdir "$OUTDIR/generated" 2>/dev/null || true
  fi
done
echo "[GPU $GPU] SAFREE bench done at $(date)"

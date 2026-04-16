#!/bin/bash
# Generate main config (ss1.2, at0.1, family, cas0.6, anchor_inpaint)
# across all nudity datasets × 3 probe modes
# $1=GPU, $2=probe_mode, $3=datasets (space-separated)
set -e
export PYTHONPATH="/mnt/home3/yhgil99/unlearning/SafeGen:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=$1
PROBE=$2
shift 2
DATASETS="$@"

PY="/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10"
VLM_PY="/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10"
GEN="/mnt/home3/yhgil99/unlearning/SafeGen/safegen/generate_family.py"
EVAL="/mnt/home3/yhgil99/unlearning/vlm/opensource_vlm_i2p_all_v1_backup.py"
CLIP="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/exemplars/sd14/clip_exemplar_embeddings.pt"
FAM="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/exemplars/concepts_v2/sexual/clip_grouped.pt"
OUT="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/main_config"
LOG="/mnt/home3/yhgil99/unlearning/logs/main_config"
mkdir -p "$OUT" "$LOG"

# Dataset → prompt file mapping
declare -A PROMPTS=(
  ["rab"]="/mnt/home3/yhgil99/unlearning/SAFREE/datasets/nudity-ring-a-bell.csv"
  ["mma"]="/mnt/home3/yhgil99/unlearning/SAFREE/datasets/mma-diffusion-nsfw-adv-prompts.csv"
  ["p4dn"]="/mnt/home3/yhgil99/unlearning/SAFREE/datasets/p4dn_16_prompt.csv"
  ["udiff"]="/mnt/home3/yhgil99/unlearning/SAFREE/datasets/unlearn_diff_nudity.csv"
  ["mja_sexual"]="/mnt/home3/yhgil99/unlearning/SafeGen/prompts/mja_sexual.txt"
)

echo "[GPU$1] probe=$PROBE datasets=$DATASETS START $(date)"

for ds in $DATASETS; do
  name="${PROBE}_ainp_ss1.2_at0.1_family_${ds}"
  outdir="$OUT/$name"
  [ -f "$outdir/generation_stats.json" ] && { echo "[SKIP] $name"; continue; }

  echo "[GEN] $name"
  $PY "$GEN" --prompts "${PROMPTS[$ds]}" --outdir "$outdir" \
    --probe_mode $PROBE --how_mode anchor_inpaint \
    --cas_threshold 0.6 --safety_scale 1.2 --attn_threshold 0.1 \
    --clip_embeddings "$CLIP" --family_config "$FAM" --family_guidance \
    --nsamples 1 --steps 50 --seed 42 \
    > "$LOG/${name}.log" 2>&1

  echo "[EVAL] $name"
  cd /mnt/home3/yhgil99/unlearning/vlm
  $VLM_PY "$EVAL" "$outdir" nudity qwen > "$LOG/${name}_eval.log" 2>&1
  grep "SR " "$outdir/results_qwen3_vl_nudity.txt"
done

echo "[GPU$1] ALL DONE $(date)"

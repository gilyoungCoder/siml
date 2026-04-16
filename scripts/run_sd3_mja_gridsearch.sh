#!/bin/bash
# SD3 SafeGen grid search on MJA datasets with concept-specific keywords
# $1=GPU, $2=concept (sexual/violent/disturbing/illegal)
set -e
GPU=$1
CONCEPT=$2

PY="/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10"
GEN="/mnt/home3/yhgil99/unlearning/scripts/sd3/generate_sd3_safegen.py"
PR="/mnt/home3/yhgil99/unlearning/SafeGen/prompts"
OUT="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/sd3_gridsearch/mja_${CONCEPT}"
LOG="/mnt/home3/yhgil99/unlearning/logs/sd3_gridsearch"
mkdir -p "$OUT" "$LOG"

export CUDA_VISIBLE_DEVICES=$GPU

# Concept → prompts mapping
declare -A PROMPTS=(
  ["sexual"]="$PR/mja_sexual.txt"
  ["violent"]="$PR/mja_violent.txt"
  ["disturbing"]="$PR/mja_disturbing.txt"
  ["illegal"]="$PR/mja_illegal.txt"
)

# Concept-specific target/anchor keywords (concept-aware)
declare -A TARGET_KW=(
  ["sexual"]="nudity nude_person naked_body"
  ["violent"]="violence blood weapon fighting gore"
  ["disturbing"]="grotesque body_horror disturbing disgusting nightmare"
  ["illegal"]="drugs crime illegal_weapons theft criminal_activity"
)
declare -A ANCHOR_KW=(
  ["sexual"]="clothed_person fully_dressed"
  ["violent"]="peaceful_scene calm_landscape"
  ["disturbing"]="beautiful_scenery harmonious_scene"
  ["illegal"]="legal_activity normal_daily_life"
)

# Split underscored keywords back to spaces
target_kw_arr=()
for w in ${TARGET_KW[$CONCEPT]}; do target_kw_arr+=("${w//_/ }"); done
anchor_kw_arr=()
for w in ${ANCHOR_KW[$CONCEPT]}; do anchor_kw_arr+=("${w//_/ }"); done

echo "[GPU$GPU SD3 $CONCEPT] START $(date)"
echo "  Target: ${target_kw_arr[*]}"
echo "  Anchor: ${anchor_kw_arr[*]}"

# Grid: ss × attn_threshold × cas_threshold
for cas in 0.4 0.6; do
  for ss in 1.0 1.5 2.0; do
    for at in 0.1 0.2 0.3; do
      name="text_ainp_cas${cas}_ss${ss}_at${at}"
      outdir="$OUT/$name"
      [ -f "$outdir/generation_stats.json" ] && { echo "[SKIP] $name"; continue; }

      echo "[GPU$GPU SD3 $CONCEPT] $name"
      $PY "$GEN" --prompts "${PROMPTS[$CONCEPT]}" --outdir "$outdir" \
        --probe_mode text --how_mode anchor_inpaint \
        --cas_threshold $cas --safety_scale $ss --attn_threshold $at \
        --target_concepts "${target_kw_arr[@]}" \
        --anchor_concepts "${anchor_kw_arr[@]}" \
        --steps 28 --resolution 1024 --cfg_scale 7.0 --seed 42 \
        > "$LOG/sd3_mja_${CONCEPT}_${name}.log" 2>&1
    done
  done
done

echo "[GPU$GPU SD3 $CONCEPT] GEN DONE $(date)"

# Evaluate all
VLM_PY="/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10"
EVAL="/mnt/home3/yhgil99/unlearning/vlm/opensource_vlm_i2p_all_v1_backup.py"

# Concept for eval
declare -A EVAL_CONCEPT=(
  ["sexual"]="nudity"
  ["violent"]="violence"
  ["disturbing"]="shocking"
  ["illegal"]="illegal"
)

cd /mnt/home3/yhgil99/unlearning/vlm
for d in "$OUT"/*/; do
  [ -f "$d/generation_stats.json" ] || continue
  [ -f "$d/results_qwen3_vl_${EVAL_CONCEPT[$CONCEPT]}.txt" ] && continue
  $VLM_PY "$EVAL" "$d" "${EVAL_CONCEPT[$CONCEPT]}" qwen > "$LOG/eval_mja_${CONCEPT}_$(basename $d).log" 2>&1
done

echo "[GPU$GPU SD3 $CONCEPT] ALL DONE $(date)"

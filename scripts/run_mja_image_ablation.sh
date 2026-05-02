#!/bin/bash
# MJA concept image probe ablation
# $1=GPU, $2=concept, $3=prompts file, $4=family_config
set -e
export PYTHONPATH="/mnt/home3/yhgil99/unlearning/SafeGen:$PYTHONPATH"
GPU=$1
CONCEPT=$2
PROMPTS=$3
FAM_CFG=$4

PY="/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10"
GEN="/mnt/home3/yhgil99/unlearning/SafeGen/safegen/generate_family.py"
CLIP="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/exemplars/sd14/clip_exemplar_embeddings.pt"
OUT="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v2_gridsearch/mja_${CONCEPT}"
LOG="/mnt/home3/yhgil99/unlearning/logs/v2_gridsearch"
mkdir -p "$OUT" "$LOG"

export CUDA_VISIBLE_DEVICES=$GPU

declare -A TARGET_KW=(
  ["sexual"]="nudity,nude person,naked body"
  ["violent"]="violence,blood,weapon,fighting,gore"
  ["disturbing"]="grotesque,body horror,disturbing,disgusting,nightmare"
  ["illegal"]="drugs,crime,illegal weapons,theft,criminal activity"
)
declare -A ANCHOR_KW=(
  ["sexual"]="clothed person,person wearing clothes"
  ["violent"]="peaceful scene,calm landscape,friendly interaction"
  ["disturbing"]="beautiful scenery,harmonious scene,serene landscape"
  ["illegal"]="legal activity,normal daily life,professional workplace"
)

IFS=',' read -ra TC <<< "${TARGET_KW[$CONCEPT]}"
IFS=',' read -ra AC <<< "${ANCHOR_KW[$CONCEPT]}"

echo "[GPU$GPU] MJA $CONCEPT IMAGE probe ablation START $(date)"

for ss in 0.8 1.0 1.2 1.5; do
  for at in 0.1 0.2 0.3; do
    for fam in family single; do
      name="image_ainp_cas0.6_ss${ss}_at${at}_${fam}"
      outdir="$OUT/$name"
      [ -f "$outdir/generation_stats.json" ] && continue

      fam_args=""
      [ "$fam" = "family" ] && fam_args="--family_config $FAM_CFG --family_guidance"

      echo "[GPU$GPU] $CONCEPT $name"
      $PY "$GEN" --prompts "$PROMPTS" --outdir "$outdir" \
        --probe_mode image --how_mode anchor_inpaint \
        --cas_threshold 0.6 --safety_scale $ss --attn_threshold $at \
        --clip_embeddings "$CLIP" \
        --target_concepts "${TC[@]}" --anchor_concepts "${AC[@]}" \
        $fam_args \
        --nsamples 1 --steps 50 --seed 42 \
        > "$LOG/mja_${CONCEPT}_${name}.log" 2>&1
    done
  done
done

echo "[GPU$GPU] MJA $CONCEPT IMAGE DONE $(date)"

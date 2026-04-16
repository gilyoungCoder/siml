#!/bin/bash
# Grid search: v2 family exemplar on MJA datasets
# Fixed: CAS=0.6, how=anchor_inpaint
# Sweep: safety_scale × attn_threshold × probe_mode × family/single
# $1=GPU, $2=concept (sexual/violent/disturbing/illegal), $3=prompts_file, $4=concept_dir, $5=family_config

set -e
GPU=$1
CONCEPT=$2
PROMPTS=$3
CONCEPT_DIR=$4
FAM_CFG=$5

PY="/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10"
GEN="/mnt/home3/yhgil99/unlearning/SafeGen/safegen/generate_family.py"
CLIP_EMB="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/exemplars/sd14/clip_exemplar_embeddings.pt"
OUT="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v2_gridsearch/mja_${CONCEPT}"
LOG="/mnt/home3/yhgil99/unlearning/logs/v2_gridsearch"
mkdir -p "$OUT" "$LOG"

export CUDA_VISIBLE_DEVICES=$GPU

echo "[GPU$GPU] MJA $CONCEPT grid search START $(date)"

for probe in both text image; do
  for ss in 0.8 1.0 1.2 1.5; do
    for at in 0.1 0.2 0.3; do
      for fam in family single; do
        name="${probe}_ainp_cas0.6_ss${ss}_at${at}_${fam}"
        outdir="$OUT/$name"
        [ -f "$outdir/generation_stats.json" ] && continue

        fam_args=""
        [ "$fam" = "family" ] && fam_args="--family_config $FAM_CFG --family_guidance"

        echo "[GPU$GPU] $CONCEPT $name"
        $PY "$GEN" --prompts "$PROMPTS" --outdir "$outdir" \
          --probe_mode $probe --how_mode anchor_inpaint \
          --cas_threshold 0.6 --safety_scale $ss --attn_threshold $at \
          --clip_embeddings "$CLIP_EMB" \
          $fam_args \
          --nsamples 1 --steps 50 --seed 42 \
          > "$LOG/mja_${CONCEPT}_${name}.log" 2>&1
      done
    done
  done
done

echo "[GPU$GPU] MJA $CONCEPT grid search DONE $(date)"

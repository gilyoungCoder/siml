#!/bin/bash
# Grid search: v2 family exemplar on RAB with extensive param sweep
# Fixed: CAS=0.6, how=anchor_inpaint, probe=both
# Sweep: safety_scale × attn_threshold × family/single
# GPU assignment passed as $1 (e.g. 0, 1, 4)
# Config range passed as $2 $3 (start end indices)

set -e
export PYTHONPATH="/mnt/home3/yhgil99/unlearning/SafeGen:$PYTHONPATH"
GPU=$1
START=$2
END=$3
SERVER=$(hostname)

PY="/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10"
GEN="/mnt/home3/yhgil99/unlearning/SafeGen/safegen/generate_family.py"
PROMPTS="/mnt/home3/yhgil99/unlearning/SafeGen/prompts/ringabell.txt"
FAM_CFG="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/exemplars/concepts_v2/sexual/clip_grouped.pt"
CLIP_EMB="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/exemplars/sd14/clip_exemplar_embeddings.pt"
OUT="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v2_gridsearch/rab"
LOG="/mnt/home3/yhgil99/unlearning/logs/v2_gridsearch"
mkdir -p "$OUT" "$LOG"

export CUDA_VISIBLE_DEVICES=$GPU

# Build config list
configs=()
for ss in 0.8 1.0 1.2 1.5; do
  for at in 0.1 0.2 0.3 0.4; do
    for fam in family single; do
      configs+=("${ss}_${at}_${fam}")
    done
  done
done

echo "[$SERVER GPU$GPU] Grid search START (configs $START-$END) $(date)"

idx=0
for cfg in "${configs[@]}"; do
  if [ $idx -lt $START ] || [ $idx -ge $END ]; then
    idx=$((idx+1))
    continue
  fi

  IFS='_' read -r ss at fam <<< "$cfg"
  name="both_ainp_cas0.6_ss${ss}_at${at}_${fam}"
  outdir="$OUT/$name"

  # Skip if already done
  [ -f "$outdir/generation_stats.json" ] && { idx=$((idx+1)); continue; }

  echo "[$SERVER GPU$GPU] [$idx] $name"

  fam_args=""
  if [ "$fam" = "family" ]; then
    fam_args="--family_config $FAM_CFG --family_guidance"
  fi

  $PY "$GEN" --prompts "$PROMPTS" --outdir "$outdir" \
    --probe_mode both --how_mode anchor_inpaint \
    --cas_threshold 0.6 --safety_scale $ss --attn_threshold $at \
    --clip_embeddings "$CLIP_EMB" \
    $fam_args \
    --nsamples 1 --steps 50 --seed 42 \
    > "$LOG/rab_${name}.log" 2>&1

  idx=$((idx+1))
done

echo "[$SERVER GPU$GPU] Grid search DONE $(date)"

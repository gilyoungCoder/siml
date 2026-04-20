#!/bin/bash
# Qwen3-VL eval batch worker. Runs v2 + v3 eval on all completed outdirs.
# Usage: bash eval_batch_worker.sh <gpu_id>
set -uo pipefail
GPU=$1
REPO=/mnt/home3/yhgil99/unlearning
PYTHON=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
[ -x "$PYTHON" ] || PYTHON=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
LOGDIR=$REPO/logs/launch_0420
mkdir -p $LOGDIR

declare -A CONCEPT_MAP
CONCEPT_MAP[rab]=nudity
CONCEPT_MAP[mja_sexual]=nudity
CONCEPT_MAP[mja_violent]=violence
CONCEPT_MAP[mja_disturbing]=disturbing
CONCEPT_MAP[mja_illegal]=illegal

LOOP=0
while true; do
  LOOP=$((LOOP+1))
  echo "[$(date)] eval loop #$LOOP starting"
  PROCESSED=0; SKIPPED=0

  # Walk all output directories that look like a generation result
  while IFS= read -r outdir; do
    # Determine dataset name from path
    # Path format:
    #   .../launch_0420/<method>/<dataset>[/<config>]
    rel=${outdir#$REPO/CAS_SpatialCFG/outputs/launch_0420/}
    dset=$(echo "$rel" | awk -F'/' '{print $2}')
    concept="${CONCEPT_MAP[$dset]:-}"
    if [ -z "$concept" ]; then continue; fi

    n_imgs=$(ls -1 "$outdir"/*.png 2>/dev/null | wc -l)
    if [ "$dset" = "rab" ]; then expected=79; else expected=100; fi
    # Need at least 80% to evaluate (avoid evaluating mid-flight)
    threshold=$((expected * 4 / 5))
    if [ "$n_imgs" -lt "$threshold" ]; then continue; fi

    # Idempotency: skip if v2 + v3 results both exist
    cat_file="$outdir/categories_qwen3_vl_${concept}.json"
    sentinel_v3="$outdir/.eval_v3_qwen3_vl_${concept}.done"
    has_v2=0; has_v3=0
    [ -f "$cat_file" ] && [ -s "$cat_file" ] && has_v2=1
    [ -f "$sentinel_v3" ] && has_v3=1
    if [ "$has_v2" = "1" ] && [ "$has_v3" = "1" ]; then
      SKIPPED=$((SKIPPED+1))
      continue
    fi

    cd $REPO/vlm
    if [ "$has_v2" = "0" ]; then
      echo "[GPU $GPU] [eval v2] $rel concept=$concept"
      CUDA_VISIBLE_DEVICES=$GPU $PYTHON opensource_vlm_i2p_all_v2.py "$outdir" "$concept" qwen \
        >> "$LOGDIR/eval_batch_g${GPU}.log" 2>&1 || echo "  v2 eval FAILED for $rel"
    fi
    if [ "$has_v3" = "0" ]; then
      echo "[GPU $GPU] [eval v3] $rel concept=$concept"
      CUDA_VISIBLE_DEVICES=$GPU $PYTHON opensource_vlm_i2p_all_v3.py "$outdir" "$concept" qwen \
        >> "$LOGDIR/eval_batch_g${GPU}.log" 2>&1 \
        && touch "$sentinel_v3" || echo "  v3 eval FAILED for $rel"
    fi
    PROCESSED=$((PROCESSED+1))
  done < <(find "$REPO/CAS_SpatialCFG/outputs/launch_0420" -mindepth 2 -maxdepth 4 -type d 2>/dev/null)

  echo "[$(date)] eval loop #$LOOP done: processed=$PROCESSED skipped=$SKIPPED"
  # If nothing new processed AND we've been looping for a while, exit
  if [ "$PROCESSED" = "0" ] && [ "$LOOP" -gt 4 ]; then
    echo "[$(date)] no new outputs to evaluate after $LOOP loops, exiting"
    break
  fi
  # Sleep then re-scan (gen workers will keep producing new outdirs)
  sleep 600
done
echo "[GPU $GPU] eval batch worker done at $(date)"

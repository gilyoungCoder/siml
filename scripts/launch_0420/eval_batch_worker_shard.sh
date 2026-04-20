#!/bin/bash
# Sharded Qwen3-VL eval worker. Each shard processes only outdirs where hash%N==K.
# Usage: bash eval_batch_worker_shard.sh <gpu_id> <shard_idx> <num_shards>
set -uo pipefail
GPU=$1
SHARD=$2
NSHARDS=$3
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
  echo "[$(date)] [shard $SHARD/$NSHARDS gpu $GPU] loop #$LOOP"
  PROCESSED=0; SKIPPED=0

  while IFS= read -r outdir; do
    rel=${outdir#$REPO/CAS_SpatialCFG/outputs/launch_0420/}
    # Shard via deterministic hash
    h=$(echo -n "$rel" | cksum | awk '{print $1}')
    if [ $((h % NSHARDS)) -ne "$SHARD" ]; then continue; fi

    dset=$(echo "$rel" | awk -F'/' '{print $2}')
    concept="${CONCEPT_MAP[$dset]:-}"
    if [ -z "$concept" ]; then continue; fi

    n_imgs=$(ls -1 "$outdir"/*.png 2>/dev/null | wc -l)
    if [ "$dset" = "rab" ]; then expected=79; else expected=100; fi
    threshold=$((expected * 4 / 5))
    if [ "$n_imgs" -lt "$threshold" ]; then continue; fi

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
      echo "[GPU $GPU shard $SHARD] [v2] $rel concept=$concept"
      CUDA_VISIBLE_DEVICES=$GPU $PYTHON opensource_vlm_i2p_all_v2.py "$outdir" "$concept" qwen \
        >> "$LOGDIR/eval_shard_g${GPU}_s${SHARD}.log" 2>&1 || echo "  v2 FAILED $rel"
    fi
    if [ "$has_v3" = "0" ]; then
      echo "[GPU $GPU shard $SHARD] [v3] $rel concept=$concept"
      CUDA_VISIBLE_DEVICES=$GPU $PYTHON opensource_vlm_i2p_all_v3.py "$outdir" "$concept" qwen \
        >> "$LOGDIR/eval_shard_g${GPU}_s${SHARD}.log" 2>&1 \
        && touch "$sentinel_v3" || echo "  v3 FAILED $rel"
    fi
    PROCESSED=$((PROCESSED+1))
  done < <(find "$REPO/CAS_SpatialCFG/outputs/launch_0420" -mindepth 2 -maxdepth 4 -type d 2>/dev/null)

  echo "[$(date)] [shard $SHARD] loop #$LOOP done: processed=$PROCESSED skipped=$SKIPPED"
  if [ "$PROCESSED" = "0" ] && [ "$LOOP" -gt 4 ]; then
    echo "[$(date)] [shard $SHARD] no work, exiting"
    break
  fi
  sleep 300
done

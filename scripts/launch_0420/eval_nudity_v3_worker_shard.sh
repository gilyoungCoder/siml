#!/bin/bash
# Nudity-only Qwen3-VL v3 re-eval worker for launch_0420_nudity outdirs.
# Re-runs v3 nudity evaluation on baseline_sd14, safree_sd14, ours_sd14_v1pack, ours_sd14_v2pack
# across rab/unlearndiff/p4dn/mma. Skips outdirs whose v3 file is already up-to-date.
# Usage: bash eval_nudity_v3_worker_shard.sh <gpu_id> <shard_idx> <num_shards>
set -uo pipefail
GPU=$1
SHARD=$2
NSHARDS=$3
REPO=/mnt/home3/yhgil99/unlearning
PYTHON=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
[ -x "$PYTHON" ] || PYTHON=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
LOGDIR=$REPO/logs/launch_0420
mkdir -p $LOGDIR
LOG="$LOGDIR/eval_nudity_v3_g${GPU}_s${SHARD}.log"

declare -A EXPECTED
EXPECTED[rab]=79
EXPECTED[unlearndiff]=141
EXPECTED[p4dn]=150
EXPECTED[mma]=999

LOOP=0
while true; do
  LOOP=$((LOOP+1))
  echo "[$(date)] [nudity v3 shard $SHARD/$NSHARDS gpu $GPU] loop #$LOOP" | tee -a "$LOG"
  PROCESSED=0; SKIPPED=0; FAILED=0

  while IFS= read -r outdir; do
    rel=${outdir#$REPO/CAS_SpatialCFG/outputs/launch_0420_nudity/}
    h=$(echo -n "$rel" | cksum | awk '{print $1}')
    if [ $((h % NSHARDS)) -ne "$SHARD" ]; then continue; fi

    # extract dataset segment: method/dset OR method/dset/cfg
    dset=$(echo "$rel" | awk -F'/' '{print $2}')
    expected="${EXPECTED[$dset]:-}"
    if [ -z "$expected" ]; then continue; fi

    n_imgs=$(ls -1 "$outdir"/*.png 2>/dev/null | wc -l)
    threshold=$((expected * 4 / 5))
    if [ "$n_imgs" -lt "$threshold" ]; then continue; fi

    cat_file_v3="$outdir/categories_qwen3_vl_nudity_v3.json"
    if [ -f "$cat_file_v3" ] && [ -s "$cat_file_v3" ]; then
      v3_count=$(python3 -c "import json,sys; sys.stdout.write(str(len(json.load(open('$cat_file_v3')))))" 2>/dev/null || echo 0)
      if [ "$v3_count" -ge "$n_imgs" ]; then
        SKIPPED=$((SKIPPED+1))
        continue
      fi
    fi

    cd $REPO/vlm
    echo "[$(date)] [GPU $GPU shard $SHARD] [nudity v3] $rel n=$n_imgs" | tee -a "$LOG"
    if CUDA_VISIBLE_DEVICES=$GPU $PYTHON opensource_vlm_i2p_all_v3.py "$outdir" nudity qwen >> "$LOG" 2>&1; then
      PROCESSED=$((PROCESSED+1))
    else
      echo "  nudity v3 FAILED $rel" | tee -a "$LOG"
      FAILED=$((FAILED+1))
    fi
  done < <(find "$REPO/CAS_SpatialCFG/outputs/launch_0420_nudity" -mindepth 2 -maxdepth 4 -type d 2>/dev/null)

  echo "[$(date)] [nudity v3 shard $SHARD] loop #$LOOP done: processed=$PROCESSED skipped=$SKIPPED failed=$FAILED" | tee -a "$LOG"
  if [ "$PROCESSED" = "0" ] && [ "$FAILED" = "0" ] && [ "$LOOP" -gt 2 ]; then
    echo "[$(date)] [nudity v3 shard $SHARD] no work, exiting" | tee -a "$LOG"
    break
  fi
  sleep 60
done

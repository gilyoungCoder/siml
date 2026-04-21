#!/bin/bash
# v4-only sharded Qwen3-VL eval worker.
# Processes only I2P concepts (violence, self_harm, shocking, illegal,
# harassment, hate, disturbing). Nudity is intentionally skipped because
# v4 nudity rubric is unchanged from v3.
# Usage: bash eval_v4_worker_shard.sh <gpu_id> <shard_idx> <num_shards>
set -uo pipefail
GPU=$1
SHARD=$2
NSHARDS=$3
REPO=/mnt/home3/yhgil99/unlearning
PYTHON=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
[ -x "$PYTHON" ] || PYTHON=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
LOGDIR=$REPO/logs/launch_0420
mkdir -p $LOGDIR
LOG="$LOGDIR/eval_v4_g${GPU}_s${SHARD}.log"

declare -A CONCEPT_MAP
# launch_0420 MJA datasets (skip mja_sexual = nudity)
CONCEPT_MAP[mja_violent]=violence
CONCEPT_MAP[mja_disturbing]=disturbing
CONCEPT_MAP[mja_illegal]=illegal
# launch_0420_i2p / launch_0420_i2p_fullhard categories (skip sexual = nudity)
CONCEPT_MAP[violence]=violence
CONCEPT_MAP[self-harm]=self_harm
CONCEPT_MAP[shocking]=shocking
CONCEPT_MAP[illegal_activity]=illegal
CONCEPT_MAP[harassment]=harassment
CONCEPT_MAP[hate]=hate
# Nudity benchmarks (rab, unlearndiff, p4dn, mma, family_rab, v27_rab) intentionally OMITTED.

LOOP=0
while true; do
  LOOP=$((LOOP+1))
  echo "[$(date)] [v4 shard $SHARD/$NSHARDS gpu $GPU] loop #$LOOP" | tee -a "$LOG"
  PROCESSED=0; SKIPPED=0; FAILED=0

  while IFS= read -r outdir; do
    rel=${outdir#$REPO/CAS_SpatialCFG/outputs/}
    rel=${rel#*/}
    h=$(echo -n "$rel" | cksum | awk '{print $1}')
    if [ $((h % NSHARDS)) -ne "$SHARD" ]; then continue; fi

    seg1=$(echo "$rel" | awk -F'/' '{print $1}')
    seg2=$(echo "$rel" | awk -F'/' '{print $2}')
    if [[ "$seg1" == "family_rab" || "$seg1" == "v27_rab" ]]; then
      dset=$seg1
    else
      dset=$seg2
    fi
    concept="${CONCEPT_MAP[$dset]:-}"
    if [ -z "$concept" ]; then continue; fi

    n_imgs=$(ls -1 "$outdir"/*.png 2>/dev/null | wc -l)
    case "$dset" in
      mja_violent|mja_disturbing|mja_illegal) expected=100 ;;
      violence|self-harm|shocking|illegal_activity|harassment|hate)
        case "$outdir" in
          *launch_0420_i2p_fullhard*)
            case "$dset" in
              violence) expected=313 ;; self-harm) expected=316 ;;
              shocking) expected=477 ;; illegal_activity) expected=238 ;;
              harassment) expected=270 ;; hate) expected=98 ;;
            esac ;;
          *) expected=60 ;;
        esac ;;
      *) expected=100 ;;
    esac
    threshold=$((expected * 4 / 5))
    if [ "$n_imgs" -lt "$threshold" ]; then continue; fi

    cat_file_v4="$outdir/categories_qwen3_vl_${concept}_v4.json"
    if [ -f "$cat_file_v4" ] && [ -s "$cat_file_v4" ]; then
      v4_count=$(python3 -c "import json,sys; sys.stdout.write(str(len(json.load(open('$cat_file_v4')))))" 2>/dev/null || echo 0)
      if [ "$v4_count" -ge "$n_imgs" ]; then
        SKIPPED=$((SKIPPED+1))
        continue
      fi
    fi

    cd $REPO/vlm
    echo "[$(date)] [GPU $GPU shard $SHARD] [v4] $rel concept=$concept n=$n_imgs" | tee -a "$LOG"
    if CUDA_VISIBLE_DEVICES=$GPU $PYTHON opensource_vlm_i2p_all_v4.py "$outdir" "$concept" qwen >> "$LOG" 2>&1; then
      PROCESSED=$((PROCESSED+1))
    else
      echo "  v4 FAILED $rel" | tee -a "$LOG"
      FAILED=$((FAILED+1))
    fi
  done < <( (find "$REPO/CAS_SpatialCFG/outputs/launch_0420" -mindepth 2 -maxdepth 4 -type d 2>/dev/null; find "$REPO/CAS_SpatialCFG/outputs/launch_0420_i2p" -mindepth 2 -maxdepth 4 -type d 2>/dev/null; find "$REPO/CAS_SpatialCFG/outputs/launch_0420_i2p_fullhard" -mindepth 2 -maxdepth 4 -type d 2>/dev/null) )

  echo "[$(date)] [v4 shard $SHARD] loop #$LOOP done: processed=$PROCESSED skipped=$SKIPPED failed=$FAILED" | tee -a "$LOG"
  if [ "$PROCESSED" = "0" ] && [ "$FAILED" = "0" ] && [ "$LOOP" -gt 2 ]; then
    echo "[$(date)] [v4 shard $SHARD] no work, exiting" | tee -a "$LOG"
    break
  fi
  sleep 60
done

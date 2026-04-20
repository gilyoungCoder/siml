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
# I2P categories (launch_0420_i2p)
CONCEPT_MAP[sexual]=nudity
CONCEPT_MAP[violence]=violence
CONCEPT_MAP[self-harm]=self_harm
CONCEPT_MAP[shocking]=shocking
CONCEPT_MAP[illegal_activity]=illegal
CONCEPT_MAP[harassment]=harassment
CONCEPT_MAP[hate]=hate
# Nudity benchmark datasets
CONCEPT_MAP[unlearndiff]=nudity
CONCEPT_MAP[p4dn]=nudity
CONCEPT_MAP[mma]=nudity
# family RAB sweep and v27 RAB variants
CONCEPT_MAP[family_rab]=nudity
CONCEPT_MAP[v27_rab]=nudity

LOOP=0
while true; do
  LOOP=$((LOOP+1))
  echo "[$(date)] [shard $SHARD/$NSHARDS gpu $GPU] loop #$LOOP"
  PROCESSED=0; SKIPPED=0

  while IFS= read -r outdir; do
    rel=${outdir#$REPO/CAS_SpatialCFG/outputs/}
    # rel now starts with launch_0420 or launch_0420_i2p; strip first segment
    rel=${rel#*/}
    # Shard via deterministic hash
    h=$(echo -n "$rel" | cksum | awk '{print $1}')
    if [ $((h % NSHARDS)) -ne "$SHARD" ]; then continue; fi

    dset=$(echo "$rel" | awk -F'/' '{print $2}')
    concept="${CONCEPT_MAP[$dset]:-}"
    if [ -z "$concept" ]; then continue; fi

    n_imgs=$(ls -1 "$outdir"/*.png 2>/dev/null | wc -l)
    if [ "$dset" = "rab" ] || [ "$dset" = "family_rab" ] || [ "$dset" = "v27_rab" ]; then expected=79
    elif [ "$dset" = "unlearndiff" ]; then expected=141
    elif [ "$dset" = "p4dn" ]; then expected=150
    elif [ "$dset" = "mma" ]; then expected=999
    elif [ "$dset" = "sexual" ] || [ "$dset" = "violence" ] || [ "$dset" = "self-harm" ] || [ "$dset" = "shocking" ] || [ "$dset" = "illegal_activity" ] || [ "$dset" = "harassment" ] || [ "$dset" = "hate" ]; then
      # detect fullhard variant
      case "$outdir" in
        *launch_0420_i2p_fullhard*)
          case "$dset" in
            sexual) expected=305 ;; violence) expected=313 ;; self-harm) expected=316 ;;
            shocking) expected=477 ;; illegal_activity) expected=238 ;;
            harassment) expected=270 ;; hate) expected=98 ;;
          esac ;;
        *) expected=60 ;;
      esac
    else expected=100; fi
    threshold=$((expected * 4 / 5))
    if [ "$n_imgs" -lt "$threshold" ]; then continue; fi

    cat_file="$outdir/categories_qwen3_vl_${concept}.json"
    cat_file_v3="$outdir/categories_qwen3_vl_${concept}_v3.json"
    sentinel_v3="$outdir/.eval_v3_qwen3_vl_${concept}.done"
    has_v2=0; has_v3=0
    if [ -f "$cat_file" ] && [ -s "$cat_file" ]; then
      v2_count=$(python3 -c "import json,sys; sys.stdout.write(str(len(json.load(open('$cat_file')))))" 2>/dev/null || echo 0)
      if [ "$v2_count" -ge "$n_imgs" ]; then has_v2=1; fi
    fi
    if [ -f "$cat_file_v3" ] && [ -s "$cat_file_v3" ]; then
      v3_count=$(python3 -c "import json,sys; sys.stdout.write(str(len(json.load(open('$cat_file_v3')))))" 2>/dev/null || echo 0)
      if [ "$v3_count" -ge "$n_imgs" ]; then has_v3=1; fi
    elif [ -f "$sentinel_v3" ] && [ -f "$cat_file" ]; then
      # legacy: sentinel exists but no v3-named file (pre-migration). Check v2 file count.
      v2_count_chk=$(python3 -c "import json,sys; sys.stdout.write(str(len(json.load(open('$cat_file')))))" 2>/dev/null || echo 0)
      if [ "$v2_count_chk" -ge "$n_imgs" ]; then has_v3=1; fi
    fi
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
  done < <( (find "$REPO/CAS_SpatialCFG/outputs/launch_0420" -mindepth 2 -maxdepth 4 -type d 2>/dev/null; find "$REPO/CAS_SpatialCFG/outputs/launch_0420_i2p" -mindepth 2 -maxdepth 4 -type d 2>/dev/null; find "$REPO/CAS_SpatialCFG/outputs/launch_0420_i2p_fullhard" -mindepth 2 -maxdepth 4 -type d 2>/dev/null; find "$REPO/CAS_SpatialCFG/outputs/launch_0420_nudity" -mindepth 2 -maxdepth 4 -type d 2>/dev/null) )

  echo "[$(date)] [shard $SHARD] loop #$LOOP done: processed=$PROCESSED skipped=$SKIPPED"
  if [ "$PROCESSED" = "0" ] && [ "$LOOP" -gt 4 ]; then
    echo "[$(date)] [shard $SHARD] no work, exiting"
    break
  fi
  sleep 300
done

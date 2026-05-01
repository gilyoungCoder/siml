#!/bin/bash
# NFE ablation EXTENSION — add SafeDenoiser + SGF to existing NFE comparison.
# Records wall time per cell (and per-image avg) for paper inference-time framing.
# Args: $1=GPU $2=WID $3=NWORK
set -uo pipefail
GPU=${1:-0}
WID=${2:-0}
NWORK=${3:-6}

ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430
PY=/mnt/home3/yhgil99/.conda/envs/sfgd/bin/python3.10
OUTBASE=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/outputs/phase_nfe_safedenoiser_sgf
LOGDIR=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/logs
TIMING_CSV=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/figures/nfe_safedenoiser_sgf_timing_w${WID}.csv
WORKER_LOG=$LOGDIR/nfe_sd_sgf_g${GPU}_w${WID}_$(date +%m%d_%H%M).log
mkdir -p "$OUTBASE" "$LOGDIR"
echo "method,concept,step,n_imgs,wall_sec,per_img_sec" > "$TIMING_CSV"
echo "[$(date)] worker $WID gpu=$GPU nwork=$NWORK start" > "$WORKER_LOG"

# Cells: 2 methods × 4 concepts × 11 steps = 88
CELLS=()
for METHOD in safedenoiser sgf; do
  for CONCEPT in violence shocking self-harm sexual; do
    for STEP in 1 3 5 8 12 16 20 25 30 40 50; do
      CELLS+=("$METHOD|$CONCEPT|$STEP")
    done
  done
done

declare -A NEGSPACE=(
  [sexual]='sexual content|nudity|pornography|erotic art|sexual acts|exposed body parts|suggestive pose|explicit intercourse|naked body|intimate body parts'
  [violence]='violence|blood|gore|weapon attack|brutal action|dismemberment|physical assault|injury|weapon threat|shooting|stabbing'
  [self-harm]='self-harm|suicide|cutting|overdose|self-inflicted injury|noose|despair scene|bleeding wrist|suicide imagery'
  [shocking]='shocking|disturbing|grotesque|body horror|deformed body|corpse|blood|wound|graphic accident|exposed organs|severed limb'
)

i=0
for entry in "${CELLS[@]}"; do
  if [ $((i % NWORK)) -eq $WID ]; then
    IFS='|' read -r METHOD CONCEPT STEP <<< "$entry"
    OUT=$OUTBASE/${METHOD}_${CONCEPT}_step${STEP}
    DATA=$ROOT/prompts/i2p_q16_csv/${CONCEPT}_q16_top60.csv
    EXPECTED=$(($(wc -l < "$DATA")-1))

    EXISTING=$(find "$OUT/all" -maxdepth 1 -type f -name '*.png' 2>/dev/null | wc -l)
    if [ "$EXISTING" -ge "$EXPECTED" ]; then
      echo "[$(date +%H:%M:%S)] [w$WID] SKIP ${METHOD}/${CONCEPT}/step${STEP} (already $EXISTING/$EXPECTED)" | tee -a "$WORKER_LOG"
      i=$((i+1)); continue
    fi
    rm -rf "$OUT"
    mkdir -p "$OUT/safe" "$OUT/unsafe" "$OUT/all"

    if [ "$METHOD" = "safedenoiser" ]; then
      OREPO=$ROOT/code/official_repos/Safe_Denoiser
      SCRIPT=run_copro_i2p_concept_np.py
      TASK=$ROOT/configs/concept_specific_official/safedenoiser_i2p_${CONCEPT}.yaml
      ERASE=safree_neg_prompt_rep_threshold_time
    else
      OREPO=$ROOT/code/official_repos/SGF/nudity_sdv1
      SCRIPT=generate_unsafe_sgf_i2p_concept_np.py
      TASK=$ROOT/configs/concept_specific_official/sgf_i2p_${CONCEPT}.yaml
      ERASE=safree_neg_prompt_rep_time
    fi
    NS="${NEGSPACE[$CONCEPT]}"

    echo "[$(date +%H:%M:%S)] [w$WID] RUN ${METHOD}/${CONCEPT}/step${STEP} gpu=$GPU" | tee -a "$WORKER_LOG"
    cd "$OREPO"
    START=$(date +%s.%N)
    CUDA_VISIBLE_DEVICES=$GPU PYTHONNOUSERSITE=1 I2P_NEGATIVE_PROMPT_SPACE="$NS" "$PY" "$SCRIPT" \
      --nudenet-path=pretrained/classifier_model.onnx \
      --nudity_thr=0.6 \
      --num_inference_steps=$STEP \
      --config=configs/base/vanilla/safree_neg_prompt_config.json \
      --safe_level=MEDIUM \
      --data="$DATA" \
      --category=all \
      --task_config="$TASK" \
      --save-dir="$OUT" \
      --erase_id="$ERASE" \
      --guidance_scale=7.5 \
      --seed=42 \
      --valid_case_numbers=0,100000 >> "$OUT/run.log" 2>&1
    END=$(date +%s.%N)
    WALL=$(echo "$END - $START" | awk '{printf "%.2f", $1}')
    NIMG=$(find "$OUT/all" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l)
    if [ "$NIMG" -gt 0 ]; then
      PER_IMG=$(echo "$WALL $NIMG" | awk '{printf "%.3f", $1/$2}')
    else
      PER_IMG="NA"
    fi
    echo "${METHOD},${CONCEPT},${STEP},${NIMG},${WALL},${PER_IMG}" >> "$TIMING_CSV"
    echo "[$(date +%H:%M:%S)] [w$WID] DONE ${METHOD}/${CONCEPT}/step${STEP} imgs=$NIMG wall=${WALL}s per_img=${PER_IMG}s" | tee -a "$WORKER_LOG"
  fi
  i=$((i+1))
done
echo "[$(date)] worker $WID done" >> "$WORKER_LOG"

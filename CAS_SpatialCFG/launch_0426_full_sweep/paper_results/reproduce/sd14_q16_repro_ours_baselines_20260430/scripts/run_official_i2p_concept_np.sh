#!/usr/bin/env bash
set -uo pipefail
METHOD=$1   # safedenoiser_i2pnp|sgf_i2pnp
CONCEPT=$2
GPU=$3
DO_EVAL=${4:-1}
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430
PY=${PY_OFFICIAL:-/mnt/home3/yhgil99/.conda/envs/sfgd/bin/python3.10}
DATA=$ROOT/prompts/i2p_q16_csv/${CONCEPT}_q16_top60.csv
EXPECTED=$(($(wc -l < "$DATA")-1))
case "$CONCEPT" in
  sexual) NEGSPACE='sexual content|nudity|pornography|erotic art|sexual acts|exposed body parts|suggestive pose|explicit intercourse|naked body|intimate body parts' ; EVAL_CONCEPT=nudity ;;
  violence) NEGSPACE='violence|blood|gore|weapon attack|brutal action|dismemberment|physical assault|injury|weapon threat|shooting|stabbing' ; EVAL_CONCEPT=violence ;;
  self-harm) NEGSPACE='self-harm|suicide|cutting|overdose|self-inflicted injury|noose|despair scene|bleeding wrist|suicide imagery' ; EVAL_CONCEPT=self_harm ;;
  shocking) NEGSPACE='shocking|disturbing|grotesque|body horror|deformed body|corpse|blood|wound|graphic accident|exposed organs|severed limb' ; EVAL_CONCEPT=shocking ;;
  illegal_activity) NEGSPACE='illegal activity|drug use|theft|robbery|criminal scene|illegal weapons|smuggling|vandalism|drug dealing|burglary' ; EVAL_CONCEPT=illegal ;;
  harassment) NEGSPACE='harassment|bullying|threat|intimidation|mockery|stalking|hostile crowd|abuse|coercion|victim cornered' ; EVAL_CONCEPT=harassment ;;
  hate) NEGSPACE='hate|hate symbol|racism|discrimination|supremacist|slur|extremist symbol|dehumanizing caricature|identity-targeted hate' ; EVAL_CONCEPT=hate ;;
  *) echo "bad concept $CONCEPT"; exit 2 ;;
esac
case "$METHOD" in
  safedenoiser_i2pnp)
    OUT=$ROOT/outputs/safedenoiser_i2pnp/i2p_q16/${CONCEPT}
    OREPO=$ROOT/code/official_repos/Safe_Denoiser
    SCRIPT=run_copro_i2p_concept_np.py
    TASK=$ROOT/configs/concept_specific_official/safedenoiser_i2p_${CONCEPT}.yaml
    ERASE=safree_neg_prompt_rep_threshold_time
    ;;
  sgf_i2pnp)
    OUT=$ROOT/outputs/sgf_i2pnp/i2p_q16/${CONCEPT}
    OREPO=$ROOT/code/official_repos/SGF/nudity_sdv1
    SCRIPT=generate_unsafe_sgf_i2p_concept_np.py
    TASK=$ROOT/configs/concept_specific_official/sgf_i2p_${CONCEPT}.yaml
    ERASE=safree_neg_prompt_rep_time
    ;;
  *) echo "bad method $METHOD"; exit 2 ;;
esac
ALLDIR=$OUT/all
COUNT=$(find "$ALLDIR" -maxdepth 1 -type f -name '*.png' 2>/dev/null | wc -l)
if [ "$COUNT" -ge "$EXPECTED" ]; then
  echo "[SKIP_GEN $METHOD/$CONCEPT] all_count=$COUNT expected=$EXPECTED"
else
  rm -rf "$OUT"
  mkdir -p "$OUT/safe" "$OUT/unsafe" "$OUT/all"
  cd "$OREPO"
  echo "RUN_I2PNP method=$METHOD concept=$CONCEPT eval=$EVAL_CONCEPT task=$TASK out=$OUT data=$DATA gpu=$GPU"
  echo "NEGSPACE=$NEGSPACE"
  CUDA_VISIBLE_DEVICES=$GPU PYTHONNOUSERSITE=1 I2P_NEGATIVE_PROMPT_SPACE="$NEGSPACE" "$PY" "$SCRIPT" \
    --nudenet-path=pretrained/classifier_model.onnx \
    --nudity_thr=0.6 \
    --num_inference_steps=50 \
    --config=configs/base/vanilla/safree_neg_prompt_config.json \
    --safe_level=MEDIUM \
    --data="$DATA" \
    --category=all \
    --task_config="$TASK" \
    --save-dir="$OUT" \
    --erase_id="$ERASE" \
    --guidance_scale=7.5 \
    --seed=42 \
    --valid_case_numbers=0,100000
fi
if [ "$DO_EVAL" = "1" ]; then
  DIR="$OUT/all"
  actual=$(find "$DIR" -maxdepth 1 -type f -name '*.png' 2>/dev/null | wc -l)
  if [ "$actual" -lt "$EXPECTED" ]; then echo "[SKIP_EVAL incomplete $METHOD/$CONCEPT] count=$actual expected=$EXPECTED"; exit 3; fi
  RES="$DIR/results_qwen3_vl_${EVAL_CONCEPT}_v5.txt"
  CAT="$DIR/categories_qwen3_vl_${EVAL_CONCEPT}_v5.json"
  if [ -s "$RES" ] || [ -s "$CAT" ]; then echo "[SKIP_EVAL existing $METHOD/$CONCEPT]"; exit 0; fi
  V5=/mnt/home3/yhgil99/unlearning/vlm/opensource_vlm_i2p_all_v5.py
  VLPY=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
  echo "RUN_EVAL method=$METHOD concept=$CONCEPT eval=$EVAL_CONCEPT dir=$DIR count=$actual gpu=$GPU"
  CUDA_VISIBLE_DEVICES=$GPU "$VLPY" "$V5" "$DIR" "$EVAL_CONCEPT" qwen
fi

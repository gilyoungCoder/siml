#!/usr/bin/env bash
set -uo pipefail
METHOD=$1     # safedenoiser_multi|sgf_multi
CONCEPT=$2    # sexual|violence|shocking
GPU=$3        # 0..6 only
DO_EVAL=${4:-1}
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430
if [ "$GPU" = "7" ]; then echo "REFUSE GPU7" >&2; exit 7; fi
case "$GPU" in 0|1|2|3|4|5|6) ;; *) echo "GPU must be 0..6" >&2; exit 2;; esac
LOCKDIR="$ROOT/.locks/i2p_multi_${METHOD}_3c_svs_${CONCEPT}.lock"
mkdir -p "$ROOT/.locks"
if ! mkdir "$LOCKDIR" 2>/dev/null; then echo "[SKIP_LOCKED] $METHOD 3c_svs $CONCEPT lock=$LOCKDIR"; exit 0; fi
trap 'rmdir "$LOCKDIR" 2>/dev/null || true' EXIT
PY=${PY_OFFICIAL:-/mnt/home3/yhgil99/.conda/envs/sfgd/bin/python3.10}
DATA=$ROOT/prompts/i2p_q16_csv/${CONCEPT}_q16_top60.csv
EXPECTED=$(($(wc -l < "$DATA")-1))
CONCEPTS="sexual violence shocking"
contains=0; for x in $CONCEPTS; do [ "$x" = "$CONCEPT" ] && contains=1; done
if [ "$contains" != 1 ]; then echo "[SKIP not in 3c_svs] $METHOD $CONCEPT"; exit 0; fi
neg_for(){
 case "$1" in
  sexual) echo 'sexual content|nudity|pornography|erotic art|sexual acts|exposed body parts|suggestive pose|explicit intercourse|naked body|intimate body parts' ;;
  violence) echo 'violence|blood|gore|weapon attack|brutal action|dismemberment|physical assault|injury|weapon threat|shooting|stabbing' ;;
  shocking) echo 'shocking|disturbing|grotesque|body horror|deformed body|corpse|blood|wound|graphic accident|exposed organs|severed limb' ;;
 esac
}
NEGSPACE=""; for x in $CONCEPTS; do ns=$(neg_for "$x"); [ -z "$NEGSPACE" ] && NEGSPACE="$ns" || NEGSPACE="$NEGSPACE|$ns"; done
case "$CONCEPT" in sexual) EVAL_CONCEPT=nudity;; *) EVAL_CONCEPT=$CONCEPT;; esac
case "$METHOD" in
 safedenoiser_multi)
  OUT=$ROOT/outputs/safedenoiser_multi_3c_svs/i2p_q16/${CONCEPT}
  OREPO=$ROOT/code/official_repos/Safe_Denoiser
  SCRIPT=run_copro_i2p_concept_np.py
  TASK=$ROOT/configs/multi_official/safedenoiser_i2p_multi_3c_svs.yaml
  ERASE=safree_neg_prompt_rep_threshold_time;;
 sgf_multi)
  OUT=$ROOT/outputs/sgf_multi_3c_svs/i2p_q16/${CONCEPT}
  OREPO=$ROOT/code/official_repos/SGF/nudity_sdv1
  SCRIPT=generate_unsafe_sgf_i2p_concept_np.py
  TASK=$ROOT/configs/multi_official/sgf_i2p_multi_3c_svs.yaml
  ERASE=safree_neg_prompt_rep_time;;
 *) echo bad method; exit 2;;
esac
ALLDIR=$OUT/all
COUNT=$(find "$ALLDIR" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l)
if [ "$COUNT" -lt "$EXPECTED" ]; then
 rm -rf "$OUT"; mkdir -p "$OUT/safe" "$OUT/unsafe" "$OUT/all"
 cd "$OREPO" || exit 4
 echo "RUN_MULTI_3C_SVS method=$METHOD concept=$CONCEPT eval=$EVAL_CONCEPT gpu=$GPU out=$OUT expected=$EXPECTED"
 echo "TASK=$TASK"
 echo "NEGSPACE=$NEGSPACE"
 CUDA_VISIBLE_DEVICES=$GPU PYTHONNOUSERSITE=1 I2P_NEGATIVE_PROMPT_SPACE="$NEGSPACE" "$PY" "$SCRIPT" \
  --nudenet-path=pretrained/classifier_model.onnx --nudity_thr=0.6 --num_inference_steps=50 \
  --config=configs/base/vanilla/safree_neg_prompt_config.json --safe_level=MEDIUM \
  --data="$DATA" --category=all --task_config="$TASK" --save-dir="$OUT" \
  --erase_id="$ERASE" --guidance_scale=7.5 --seed=42 --valid_case_numbers=0,100000
fi
if [ "$DO_EVAL" = "1" ]; then
 DIR="$OUT/all"; actual=$(find "$DIR" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l)
 if [ "$actual" -lt "$EXPECTED" ]; then echo "[SKIP_EVAL incomplete] $actual/$EXPECTED"; exit 3; fi
 RES="$DIR/results_qwen3_vl_${EVAL_CONCEPT}_v5.txt"; CAT="$DIR/categories_qwen3_vl_${EVAL_CONCEPT}_v5.json"
 if [ ! -s "$RES" ] && [ ! -s "$CAT" ]; then
  V5=/mnt/home3/yhgil99/unlearning/vlm/opensource_vlm_i2p_all_v5.py
  VLPY=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
  CUDA_VISIBLE_DEVICES=$GPU "$VLPY" "$V5" "$DIR" "$EVAL_CONCEPT" qwen
 fi
fi

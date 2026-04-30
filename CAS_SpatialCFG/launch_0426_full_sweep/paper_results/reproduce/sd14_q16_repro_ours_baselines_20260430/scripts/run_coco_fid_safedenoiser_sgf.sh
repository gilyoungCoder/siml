#!/usr/bin/env bash
set -uo pipefail
GPU=${1:-1}
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430
REPO=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG
PY_OFFICIAL=${PY_OFFICIAL:-/mnt/home3/yhgil99/.conda/envs/sfgd/bin/python3.10}
PY_EVAL=${PY_EVAL:-/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10}
DATA=$ROOT/prompts/coco/coco_250_x4.csv
BASELINE=$REPO/outputs/coco_fid/baseline
PROMPTS=$REPO/prompts/coco_250.txt
LOGDIR=$ROOT/logs/coco_fid_official
mkdir -p "$LOGDIR"
run_one() {
  local method=$1
  local out script task erase orepo
  case "$method" in
    safedenoiser)
      out=$ROOT/outputs/safedenoiser/nudity_coco/coco250x4
      orepo=$ROOT/code/official_repos/Safe_Denoiser
      script=run_copro.py
      task=configs/nudity/safe_denoiser.yaml
      erase=safree_neg_prompt_rep_threshold_time
      ;;
    sgf)
      out=$ROOT/outputs/sgf/nudity_coco/coco250x4
      orepo=$ROOT/code/official_repos/SGF/nudity_sdv1
      script=generate_unsafe_sgf.py
      task=configs/sgf/sgf.yaml
      erase=safree_neg_prompt_rep_time
      ;;
    *) echo "bad method $method"; return 2;;
  esac
  mkdir -p "$out/safe" "$out/unsafe" "$out/all"
  local count
  count=$(find "$out/all" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l)
  echo "[$(date)] method=$method all_count=$count/1000 out=$out"
  if [ "$count" -lt 1000 ]; then
    rm -rf "$out"
    mkdir -p "$out/safe" "$out/unsafe" "$out/all"
    (cd "$orepo" && CUDA_VISIBLE_DEVICES=$GPU PYTHONNOUSERSITE=1 "$PY_OFFICIAL" "$script" \
      --nudenet-path=pretrained/classifier_model.onnx \
      --nudity_thr=0.6 \
      --num_inference_steps=50 \
      --config=configs/base/vanilla/safree_neg_prompt_config.json \
      --safe_level=MEDIUM \
      --data="$DATA" \
      --category=all \
      --task_config="$task" \
      --save-dir="$out" \
      --erase_id="$erase" \
      --guidance_scale=7.5 \
      --seed=42 \
      --valid_case_numbers=0,100000) 2>&1 | tee "$LOGDIR/${method}_gen.log"
  fi
  count=$(find "$out/all" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l)
  echo "[$(date)] method=$method post_gen_count=$count"
  if [ "$count" -ge 1000 ]; then
    CUDA_VISIBLE_DEVICES=$GPU "$PY_EVAL" "$REPO/eval_fid_clip.py" "$BASELINE" "$out/all" "$PROMPTS" 2>&1 | tee "$LOGDIR/${method}_fid_clip.log"
    cp "$out/all/results_fid_clip.txt" "$ROOT/summaries/coco_fid_clip_${method}_nudity_official.txt" 2>/dev/null || true
  else
    echo "[$(date)] ERROR incomplete $method count=$count" | tee -a "$LOGDIR/${method}_gen.log"
    return 3
  fi
}
run_one safedenoiser
run_one sgf

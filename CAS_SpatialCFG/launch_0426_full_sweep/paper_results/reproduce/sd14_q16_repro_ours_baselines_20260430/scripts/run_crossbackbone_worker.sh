#!/usr/bin/env bash
set -euo pipefail
SHARD=${1:?shard_id}
NSHARDS=${2:?n_shards}
GPU=${3:?gpu_id}
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430
CAS=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG
JOBLIST=$ROOT/joblists/crossbackbone_sd3_flux_i2p_mja.tsv
LOGDIR=$ROOT/logs/crossbackbone
mkdir -p "$LOGDIR"

concept_targets() {
  case "$1" in
    sexual) echo "nudity nude_person naked_body sexual_content pornography erotic" ;;
    violence) echo "violence blood gore weapon_attack brutal_action injury" ;;
    self-harm) echo "self_harm cutting suicide overdose despair self_injury" ;;
    shocking) echo "shocking grotesque body_horror corpse deformed_body gore" ;;
    illegal_activity) echo "illegal_activity drug_use robbery theft criminal_scene illicit_weapon" ;;
    harassment) echo "harassment bullying mockery threat intimidation hostile_crowd" ;;
    hate) echo "hate_symbol racist discriminatory supremacist hateful_propaganda" ;;
    *) echo "$1" ;;
  esac
}

count_pngs() { find "$1/all" -maxdepth 1 -type f -name '*.png' 2>/dev/null | wc -l; }

run_flux() {
  local method=$1 concept=$2 prompt=$3 out=$4 expected=$5
  mkdir -p "$out"
  local n=$(count_pngs "$out")
  if [ "$n" -ge "$expected" ]; then echo "[SKIP] flux $method $concept already $n/$expected $out"; return 0; fi
  rm -rf "$out"; mkdir -p "$out"
  local py=/mnt/home3/yhgil99/.conda/envs/safree/bin/python3.10
  if [ "$method" = safree ]; then
    read -r -a tgt <<< "$(concept_targets "$concept")"
    CUDA_VISIBLE_DEVICES=$GPU "$py" "$CAS/generate_flux1_safree.py" \
      --prompts "$prompt" --outdir "$out/all" --steps 28 --guidance_scale 3.5 --height 512 --width 512 \
      --start_idx 0 --end_idx -1 --device cuda:0 --dtype bfloat16 --nsamples 1 \
      --safree_token_filter --safree_re_attention --safree_latent_filter --target_concepts "${tgt[@]}"
  else
    local mode=$method
    [ "$method" = safedenoiser ] && mode=safedenoiser
    local ref_concept=$concept
    [ "$concept" = shocking ] && ref_concept=shocking
    local ref="$CAS/exemplars/i2p_v1_flux1/${ref_concept}/ref_latents.pt"
    CUDA_VISIBLE_DEVICES=$GPU "$py" "$CAS/generate_flux1_repellency.py" \
      --mode "$mode" --concept "$concept" --prompts "$prompt" --outdir "$out/all" --ref_latents "$ref" \
      --steps 28 --guidance_scale 3.5 --height 512 --width 512 --start_idx 0 --end_idx -1 --device cuda:0 \
      --repellency_scale 0.03 --repellency_sigma 0.08
  fi
}

run_sd3() {
  local method=$1 concept=$2 prompt=$3 out=$4 expected=$5 task=$6
  mkdir -p "$out"
  local n=$(count_pngs "$out")
  if [ "$n" -ge "$expected" ]; then echo "[SKIP] sd3 $method $concept already $n/$expected $out"; return 0; fi
  rm -rf "$out"; mkdir -p "$out"
  local py=/mnt/home3/yhgil99/.conda/envs/safree/bin/python3.10
  local extra=()
  [ "$task" != NONE ] && extra=(--task_config "$task")
  CUDA_VISIBLE_DEVICES=$GPU PYTHONNOUSERSITE=1 "$py" "$ROOT/scripts/generate_sd3_prompt_repellency.py" \
    --method "$method" --prompts "$prompt" --outdir "$out" "${extra[@]}" \
    --steps 28 --guidance_scale 7.0 --height 1024 --width 1024 --device cuda:0
}

idx=0
while IFS=$'\t' read -r backbone method dataset concept prompt out expected task; do
  [[ -z "${backbone:-}" || "$backbone" = backbone ]] && continue
  if [ $((idx % NSHARDS)) -ne "$SHARD" ]; then idx=$((idx+1)); continue; fi
  echo "========== JOB idx=$idx shard=$SHARD/$NSHARDS host=$(hostname) gpu=$GPU $backbone $method $dataset $concept =========="
  echo "prompt=$prompt out=$out expected=$expected task=$task"
  if [ "$backbone" = flux1 ]; then run_flux "$method" "$concept" "$prompt" "$out" "$expected"; else run_sd3 "$method" "$concept" "$prompt" "$out" "$expected" "$task"; fi
  echo "========== DONE idx=$idx =========="
  idx=$((idx+1))
done < "$JOBLIST"

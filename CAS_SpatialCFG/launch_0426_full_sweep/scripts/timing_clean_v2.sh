#!/usr/bin/env bash
# Clean per-method timing, single GPU isolated, 1 process at a time.
# Idempotent: skip if CSV already has 8 NFE rows OR if lock exists (race-free).
# Usage: bash timing_clean_v2.sh <GPU> <METHOD>
set -uo pipefail
GPU=${1:?gpu}
METHOD=${2:?method}
N=20
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep
OUTBASE=$ROOT/outputs/phase_nfe_walltime_timing_clean
RESULT_CSV=$ROOT/paper_results/figures/nfe_walltime_timing_clean_${METHOD}.csv
LOG=$ROOT/logs/nfe_walltime_timing_clean_${METHOD}_g${GPU}_$(date +%m%d_%H%M).log
mkdir -p $OUTBASE $ROOT/logs $ROOT/paper_results/figures

# Idempotent skip if already done
if [ -f "$RESULT_CSV" ] && [ "$(tail -n +2 "$RESULT_CSV" 2>/dev/null | wc -l)" -ge 8 ]; then
  echo "[$(date)] [$METHOD] already 8 rows in $RESULT_CSV — skipping"
  exit 0
fi

# Lock to prevent races (another GPU running same method)
LOCK=/tmp/timing_clean_${METHOD}.lock
if [ -f "$LOCK" ]; then
  age=$(( $(date +%s) - $(stat -c %Y "$LOCK") ))
  if [ $age -lt 1800 ]; then
    echo "[$(date)] [$METHOD] lock $LOCK exists (age ${age}s) — skipping"
    exit 0
  fi
  rm -f "$LOCK"
fi
touch "$LOCK"
trap 'rm -f $LOCK' EXIT

echo "method,nfe,n_imgs,wall_sec_total,per_img_sec_with_load,per_img_sec_excl_load_mtime" > $RESULT_CSV
echo "[$(date)] start GPU=$GPU METHOD=$METHOD N=$N" > $LOG

TXT=/tmp/violence_${N}_${METHOD}_$$.txt
CSV=/tmp/violence_${N}_${METHOD}_$$.csv
SRCCSV=$ROOT/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/prompts/i2p_q16_csv/violence_q16_top60.csv
head -1 "$SRCCSV" > $CSV
tail -n +2 "$SRCCSV" | head -n $N >> $CSV
tail -n +2 "$SRCCSV" | head -n $N | awk -F',' '{print $1}' > $TXT

NFES=(5 10 15 20 25 30 40 50)

canon_files () {
  local OUTDIR=$1
  if [ -d "$OUTDIR/all" ] && [ "$(ls -A "$OUTDIR/all" 2>/dev/null | wc -l)" -gt 0 ]; then
    ls "$OUTDIR/all"/*.png 2>/dev/null
  elif [ -d "$OUTDIR/generated" ] && [ "$(ls -A "$OUTDIR/generated" 2>/dev/null | wc -l)" -gt 0 ]; then
    ls "$OUTDIR/generated"/*.png 2>/dev/null
  else
    ls "$OUTDIR"/*.png 2>/dev/null
  fi
}

measure_and_log () {
  local NFE=$1 OUTDIR=$2 START=$3 END=$4
  local files=$(canon_files "$OUTDIR")
  local n=$(echo "$files" | sed '/^$/d' | wc -l)
  local wall=$(echo "$START $END" | awk '{printf "%.2f", $2-$1}')
  local pi_load="NA" pi_excl="NA"
  if [ "$n" -gt 0 ]; then
    pi_load=$(echo "$wall $n" | awk '{printf "%.4f", $1/$2}')
  fi
  if [ "$n" -gt 1 ]; then
    local first=$(echo "$files" | xargs -d '\n' stat -c %Y 2>/dev/null | sort -n | head -1)
    local last=$(echo "$files" | xargs -d '\n' stat -c %Y 2>/dev/null | sort -n | tail -1)
    pi_excl=$(echo "$first $last $n" | awk '{printf "%.4f", ($2-$1)/($3-1)}')
  fi
  echo "${METHOD},${NFE},${n},${wall},${pi_load},${pi_excl}" >> $RESULT_CSV
  echo "[$(date +%H:%M:%S)] [$METHOD nfe=$NFE] n=$n wall=${wall}s per_img(excl)=${pi_excl}s" | tee -a $LOG
}

run_baseline () {
  local NFE=$1 OUT=$2
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=$GPU /mnt/home3/yhgil99/.conda/envs/safree/bin/python3.10 - <<PYEOF >> $LOG 2>&1
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False).to("cuda:0")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.set_progress_bar_config(disable=True)
prompts = [l.strip() for l in open("$TXT") if l.strip()]
for i, p in enumerate(prompts):
    g = torch.Generator(device="cuda").manual_seed(42 + i)
    img = pipe(p, num_inference_steps=$NFE, guidance_scale=7.5, generator=g).images[0]
    img.save(f"$OUT/{i:04d}.png")
PYEOF
}

run_safree () {
  local NFE=$1 OUT=$2
  cd /mnt/home3/yhgil99/unlearning/SAFREE
  CUDA_VISIBLE_DEVICES=$GPU /mnt/home3/yhgil99/.conda/envs/safree/bin/python3.10 gen_safree_single.py \
    --txt "$TXT" --save-dir "$OUT" --category "violence" \
    --re_attn_t=-1,1001 --linear_per_prompt_seed \
    --num_inference_steps $NFE --safree -svf -lra >> $LOG 2>&1 || true
  cd - > /dev/null
}

run_safedenoiser () {
  local NFE=$1 OUT=$2
  mkdir -p $OUT/all $OUT/safe $OUT/unsafe
  cd $ROOT/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/code/official_repos/Safe_Denoiser
  NEGSPACE='violence|blood|gore|weapon attack|brutal action|dismemberment|physical assault|injury|weapon threat|shooting|stabbing'
  CUDA_VISIBLE_DEVICES=$GPU PYTHONNOUSERSITE=1 I2P_NEGATIVE_PROMPT_SPACE="$NEGSPACE" /mnt/home3/yhgil99/.conda/envs/sfgd/bin/python3.10 run_copro_i2p_concept_np.py \
    --nudenet-path=pretrained/classifier_model.onnx --nudity_thr=0.6 \
    --num_inference_steps=$NFE --config=configs/base/vanilla/safree_neg_prompt_config.json \
    --safe_level=MEDIUM --data="$CSV" --category=all \
    --task_config=$ROOT/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/configs/concept_specific_official/safedenoiser_i2p_violence.yaml \
    --save-dir="$OUT" --erase_id=safree_neg_prompt_rep_threshold_time \
    --guidance_scale=7.5 --seed=42 --valid_case_numbers=0,100000 >> $LOG 2>&1 || true
  cd - > /dev/null
}

run_sgf () {
  local NFE=$1 OUT=$2
  mkdir -p $OUT/all $OUT/safe $OUT/unsafe
  cd $ROOT/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/code/official_repos/SGF/nudity_sdv1
  NEGSPACE='violence|blood|gore|weapon attack|brutal action|dismemberment|physical assault|injury|weapon threat|shooting|stabbing'
  CUDA_VISIBLE_DEVICES=$GPU PYTHONNOUSERSITE=1 I2P_NEGATIVE_PROMPT_SPACE="$NEGSPACE" /mnt/home3/yhgil99/.conda/envs/sfgd/bin/python3.10 generate_unsafe_sgf_i2p_concept_np.py \
    --nudenet-path=pretrained/classifier_model.onnx --nudity_thr=0.6 \
    --num_inference_steps=$NFE --config=configs/base/vanilla/safree_neg_prompt_config.json \
    --safe_level=MEDIUM --data="$CSV" --category=all \
    --task_config=$ROOT/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/configs/concept_specific_official/sgf_i2p_violence.yaml \
    --save-dir="$OUT" --erase_id=safree_neg_prompt_rep_time \
    --guidance_scale=7.5 --seed=42 --valid_case_numbers=0,100000 >> $LOG 2>&1 || true
  cd - > /dev/null
}

run_sld () {
  local NFE=$1 OUT=$2 VARIANT=$3
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=$GPU /mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10 \
    $ROOT/scripts/sld_runner.py --prompts "$TXT" --outdir "$OUT" \
    --variant $VARIANT --steps $NFE --seed 42 --cfg_scale 7.5 >> $LOG 2>&1 || true
}

run_ebsg () {
  local NFE=$1 OUT=$2
  cd /mnt/home3/yhgil99/unlearning/SafeGen
  CUDA_VISIBLE_DEVICES=$GPU PYTHONNOUSERSITE=1 /mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10 -m safegen.generate_family \
    --prompts "$TXT" --outdir "$OUT" \
    --family_guidance --family_config /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/exemplars/i2p_v1/violence/clip_grouped.pt \
    --probe_mode both --probe_fusion union --how_mode hybrid \
    --cas_threshold 0.4 --safety_scale 20.0 \
    --attn_threshold 0.30 --img_attn_threshold 0.10 \
    --n_img_tokens 4 --steps $NFE --seed 42 --cfg_scale 7.5 \
    --target_concepts violence bloody_scene weapon fight \
    --target_words violence bloody scene weapon fight >> $LOG 2>&1 || true
  cd - > /dev/null
}

for NFE in "${NFES[@]}"; do
  OUT=$OUTBASE/${METHOD}_violence_nfe${NFE}_g${GPU}
  rm -rf $OUT && mkdir -p $OUT
  echo "[$(date +%H:%M:%S)] [$METHOD nfe=$NFE] start" | tee -a $LOG
  START=$(date +%s.%N)
  case $METHOD in
    baseline)     run_baseline $NFE $OUT ;;
    safree)       run_safree $NFE $OUT ;;
    safedenoiser) run_safedenoiser $NFE $OUT ;;
    sgf)          run_sgf $NFE $OUT ;;
    sld_max)      run_sld $NFE $OUT Max ;;
    sld_strong)   run_sld $NFE $OUT Strong ;;
    sld_medium)   run_sld $NFE $OUT Medium ;;
    sld_weak)     run_sld $NFE $OUT Weak ;;
    ebsg)         run_ebsg $NFE $OUT ;;
    *) echo "unknown method $METHOD"; exit 1 ;;
  esac
  END=$(date +%s.%N)
  measure_and_log $NFE $OUT $START $END
done

rm -f $TXT $CSV
echo "[$(date)] DONE METHOD=$METHOD CSV=$RESULT_CSV" | tee -a $LOG

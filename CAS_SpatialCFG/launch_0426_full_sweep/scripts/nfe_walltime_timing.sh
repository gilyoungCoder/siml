#!/usr/bin/env bash
# Isolated timing benchmark: 5 method x 8 NFE x 20 violence prompts on a SINGLE GPU.
# Outputs: paper_results/figures/nfe_walltime_timing.csv (one row per (method, NFE)).
# Per-image time computed via PNG mtime range (excludes model load).
#
# Usage: bash nfe_walltime_timing.sh <GPU_INDEX>
# Default GPU=0. Run on siml-05 g0 (kept idle for the duration).

set -uo pipefail
GPU=${1:-0}
N=20
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep
OUTBASE=$ROOT/outputs/phase_nfe_walltime_timing
LOG=$ROOT/logs/nfe_walltime_timing_g${GPU}_$(date +%m%d_%H%M).log
RESULT_CSV=$ROOT/paper_results/figures/nfe_walltime_timing.csv

mkdir -p $OUTBASE $ROOT/logs $ROOT/paper_results/figures
echo "method,nfe,n_imgs,wall_sec_total,per_img_sec_with_load,per_img_sec_excl_load_mtime" > $RESULT_CSV
echo "[$(date)] start GPU=$GPU N=$N" > $LOG

# Prompts (top 20 of violence q16 top60)
TXT20=/tmp/violence_top${N}_$$.txt
CSV20=/tmp/violence_top${N}_$$.csv
SRCTXT=$ROOT/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/prompts/i2p_q16_csv/violence_q16_top60.csv
head -1 "$SRCTXT" > $CSV20
tail -n +2 "$SRCTXT" | head -n $N >> $CSV20
tail -n +2 "$SRCTXT" | head -n $N | awk -F',' '{print $1}' > $TXT20
echo "[prompts] N=$(wc -l < $TXT20)" | tee -a $LOG

NFES=(5 10 15 20 25 30 40 50)

measure_and_log () {
  local METHOD=$1 NFE=$2 OUTDIR=$3 START=$4 END=$5
  local n=$(ls $OUTDIR/*.png $OUTDIR/all/*.png $OUTDIR/generated/*.png 2>/dev/null | wc -l)
  local wall=$(echo "$START $END" | awk '{printf "%.2f", $2-$1}')
  local pi_load=$(echo "$wall $n" | awk '{if ($2>0) printf "%.4f", $1/$2; else print "NA"}')
  local pi_excl="NA"
  if [ "$n" -gt 1 ]; then
    local files=$(ls $OUTDIR/*.png $OUTDIR/all/*.png $OUTDIR/generated/*.png 2>/dev/null)
    local first=$(stat -c %Y $(echo "$files" | head -1))
    local last=$(stat -c %Y $(echo "$files" | tail -1))
    pi_excl=$(echo "$first $last $n" | awk '{if ($3>1) printf "%.4f", ($2-$1)/($3-1); else print "NA"}')
  fi
  echo "${METHOD},${NFE},${n},${wall},${pi_load},${pi_excl}" >> $RESULT_CSV
  echo "[$(date +%H:%M:%S)] [$METHOD nfe=$NFE] n=$n wall=${wall}s per_img(load)=${pi_load}s per_img(excl)=${pi_excl}s" | tee -a $LOG
}

run_baseline () {
  local NFE=$1 OUT=$2
  rm -rf $OUT && mkdir -p $OUT
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=$GPU /mnt/home3/yhgil99/.conda/envs/safree/bin/python3.10 - <<PYEOF >> $LOG 2>&1
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False).to("cuda:0")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.set_progress_bar_config(disable=True)
prompts = [l.strip() for l in open("$TXT20") if l.strip()]
for i, p in enumerate(prompts):
    g = torch.Generator(device="cuda").manual_seed(42 + i)
    img = pipe(p, num_inference_steps=$NFE, guidance_scale=7.5, generator=g).images[0]
    img.save(f"$OUT/{i:04d}.png")
PYEOF
}

run_safree () {
  local NFE=$1 OUT=$2
  rm -rf $OUT && mkdir -p $OUT
  cd /mnt/home3/yhgil99/unlearning/SAFREE
  CUDA_VISIBLE_DEVICES=$GPU /mnt/home3/yhgil99/.conda/envs/safree/bin/python3.10 gen_safree_single.py \
    --txt "$TXT20" --save-dir "$OUT" --category "violence" \
    --re_attn_t=-1,1001 --linear_per_prompt_seed \
    --num_inference_steps $NFE --safree -svf -lra >> $LOG 2>&1 || true
  mv $OUT/generated/*.png $OUT/ 2>/dev/null || true
  cd - > /dev/null
}

run_safedenoiser () {
  local NFE=$1 OUT=$2
  rm -rf $OUT && mkdir -p $OUT/all $OUT/safe $OUT/unsafe
  cd $ROOT/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/code/official_repos/Safe_Denoiser
  NEGSPACE='violence|blood|gore|weapon attack|brutal action|dismemberment|physical assault|injury|weapon threat|shooting|stabbing'
  CUDA_VISIBLE_DEVICES=$GPU PYTHONNOUSERSITE=1 I2P_NEGATIVE_PROMPT_SPACE="$NEGSPACE" /mnt/home3/yhgil99/.conda/envs/sfgd/bin/python3.10 run_copro_i2p_concept_np.py \
    --nudenet-path=pretrained/classifier_model.onnx --nudity_thr=0.6 \
    --num_inference_steps=$NFE --config=configs/base/vanilla/safree_neg_prompt_config.json \
    --safe_level=MEDIUM --data="$CSV20" --category=all \
    --task_config=$ROOT/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/configs/concept_specific_official/safedenoiser_i2p_violence.yaml \
    --save-dir="$OUT" --erase_id=safree_neg_prompt_rep_threshold_time \
    --guidance_scale=7.5 --seed=42 --valid_case_numbers=0,100000 >> $LOG 2>&1 || true
  cd - > /dev/null
}

run_sgf () {
  local NFE=$1 OUT=$2
  rm -rf $OUT && mkdir -p $OUT/all $OUT/safe $OUT/unsafe
  cd $ROOT/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/code/official_repos/SGF/nudity_sdv1
  NEGSPACE='violence|blood|gore|weapon attack|brutal action|dismemberment|physical assault|injury|weapon threat|shooting|stabbing'
  CUDA_VISIBLE_DEVICES=$GPU PYTHONNOUSERSITE=1 I2P_NEGATIVE_PROMPT_SPACE="$NEGSPACE" /mnt/home3/yhgil99/.conda/envs/sfgd/bin/python3.10 generate_unsafe_sgf_i2p_concept_np.py \
    --nudenet-path=pretrained/classifier_model.onnx --nudity_thr=0.6 \
    --num_inference_steps=$NFE --config=configs/base/vanilla/safree_neg_prompt_config.json \
    --safe_level=MEDIUM --data="$CSV20" --category=all \
    --task_config=$ROOT/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/configs/concept_specific_official/sgf_i2p_violence.yaml \
    --save-dir="$OUT" --erase_id=safree_neg_prompt_rep_time \
    --guidance_scale=7.5 --seed=42 --valid_case_numbers=0,100000 >> $LOG 2>&1 || true
  cd - > /dev/null
}

run_sld_max () {
  local NFE=$1 OUT=$2
  rm -rf $OUT && mkdir -p $OUT
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=$GPU /mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10 \
    $ROOT/scripts/sld_runner.py --prompts "$TXT20" --outdir "$OUT" \
    --variant Max --steps $NFE --seed 42 --cfg_scale 7.5 >> $LOG 2>&1 || true
}

run_sld_medium () {
  local NFE=$1 OUT=$2
  rm -rf $OUT && mkdir -p $OUT
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=$GPU /mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10 \
    $ROOT/scripts/sld_runner.py --prompts "$TXT20" --outdir "$OUT" \
    --variant Medium --steps $NFE --seed 42 --cfg_scale 7.5 >> $LOG 2>&1 || true
}

run_ebsg () {
  local NFE=$1 OUT=$2
  rm -rf $OUT && mkdir -p $OUT
  cd /mnt/home3/yhgil99/unlearning/SafeGen
  # violence per-concept best from user spec: ss=20.0, cas=0.4, theta_text=0.30, theta_img=0.10
  CUDA_VISIBLE_DEVICES=$GPU PYTHONNOUSERSITE=1 /mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10 -m safegen.generate_family \
    --prompts "$TXT20" --outdir "$OUT" \
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
  for METHOD in baseline safree safedenoiser sgf sld_max sld_medium ebsg; do
    OUT=$OUTBASE/${METHOD}_violence_nfe${NFE}
    echo "[$(date +%H:%M:%S)] [$METHOD nfe=$NFE] start" | tee -a $LOG
    START=$(date +%s.%N)
    run_${METHOD} $NFE $OUT
    END=$(date +%s.%N)
    measure_and_log $METHOD $NFE $OUT $START $END
  done
done

echo "[$(date)] all done. Result CSV: $RESULT_CSV" | tee -a $LOG
echo
echo "=== SUMMARY ==="
cat $RESULT_CSV
rm -f $TXT20 $CSV20

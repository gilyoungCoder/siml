#!/bin/bash
# Isolated per-image gen-time benchmark on siml-07 g1.
# 5 methods × 20 violence prompts × 50 DDIM steps, sequential, no contention.
# Reports total wall (incl. model load) AND per-image (excl. load, via PNG mtime range).
set -uo pipefail
GPU=${1:-1}
N=${2:-20}
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep
OUTBASE=$ROOT/outputs/phase_timing_isolated
LOG=$ROOT/logs/timing_isolated_g${GPU}_$(date +%m%d_%H%M).log
RESULT_CSV=$ROOT/paper_results/figures/timing_isolated_5method.csv
mkdir -p $OUTBASE $ROOT/logs $ROOT/paper_results/figures
echo "method,n_imgs,wall_sec_total,per_img_sec_with_load,per_img_sec_excl_load_mtime" > $RESULT_CSV
echo "[$(date)] start GPU=$GPU N=$N" > $LOG

# Truncated prompts (top 20)
TXT20=/tmp/violence_top${N}.txt
CSV20=/tmp/violence_top${N}.csv
head -n $N /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/prompts/i2p_sweep60/violence_sweep.txt > $TXT20
( head -1 $ROOT/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/prompts/i2p_q16_csv/violence_q16_top60.csv ; head -n $((N+1)) $ROOT/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/prompts/i2p_q16_csv/violence_q16_top60.csv | tail -n $N ) > $CSV20
echo "[prompts] txt=$TXT20 csv=$CSV20 N=$N" | tee -a $LOG

measure_and_log () {
  local METHOD=$1 OUTDIR=$2 START=$3 END=$4
  local n=$(ls $OUTDIR/*.png $OUTDIR/all/*.png 2>/dev/null | wc -l)
  local wall=$(echo "$START $END" | awk '{printf "%.2f", $2-$1}')
  local pi_load=$(echo "$wall $n" | awk '{if ($2>0) printf "%.3f", $1/$2; else print "NA"}')
  local pi_excl="NA"
  if [ "$n" -gt 1 ]; then
    local files=$(ls $OUTDIR/*.png $OUTDIR/all/*.png 2>/dev/null)
    local first=$(stat -c %Y $(echo "$files" | head -1))
    local last=$(stat -c %Y $(echo "$files" | tail -1))
    pi_excl=$(echo "$first $last $n" | awk '{if ($3>1) printf "%.3f", ($2-$1)/($3-1); else print "NA"}')
  fi
  echo "${METHOD},${n},${wall},${pi_load},${pi_excl}" >> $RESULT_CSV
  echo "[$(date +%H:%M:%S)] [$METHOD] n=$n wall=${wall}s per_img(with_load)=${pi_load}s per_img(excl_load)=${pi_excl}s" | tee -a $LOG
}

# ============== METHOD 1: SD1.4 baseline ==============
echo "[$(date +%H:%M:%S)] [baseline] start" | tee -a $LOG
OUT=$OUTBASE/baseline_violence
mkdir -p $OUT
START=$(date +%s.%N)
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=$GPU /mnt/home3/yhgil99/.conda/envs/safree/bin/python3.10 - <<PYEOF >> $LOG 2>&1
import torch, os, sys
from diffusers import StableDiffusionPipeline, DDIMScheduler
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False).to("cuda:0")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.set_progress_bar_config(disable=True)
prompts = [l.strip() for l in open("$TXT20") if l.strip()]
print(f"baseline: {len(prompts)} prompts")
for i, p in enumerate(prompts):
    g = torch.Generator(device="cuda").manual_seed(42 + i)
    img = pipe(p, num_inference_steps=50, guidance_scale=7.5, generator=g).images[0]
    img.save(f"$OUT/{i:04d}.png")
print("baseline done")
PYEOF
END=$(date +%s.%N)
measure_and_log baseline $OUT $START $END

# ============== METHOD 2: SAFREE v2 (--safree -svf -lra) ==============
echo "[$(date +%H:%M:%S)] [safree] start" | tee -a $LOG
OUT=$OUTBASE/safree_violence
mkdir -p $OUT
START=$(date +%s.%N)
cd /mnt/home3/yhgil99/unlearning/SAFREE
CUDA_VISIBLE_DEVICES=$GPU /mnt/home3/yhgil99/.conda/envs/safree/bin/python3.10 gen_safree_single.py \
  --txt "$TXT20" --save-dir "$OUT" --category "violence" \
  --re_attn_t=-1,1001 --linear_per_prompt_seed \
  --num_inference_steps 50 --safree -svf -lra >> $LOG 2>&1 || true
mv $OUT/generated/*.png $OUT/ 2>/dev/null || true
END=$(date +%s.%N)
measure_and_log safree $OUT $START $END

# ============== METHOD 3: SafeDenoiser ==============
echo "[$(date +%H:%M:%S)] [safedenoiser] start" | tee -a $LOG
OUT=$OUTBASE/safedenoiser_violence
mkdir -p $OUT/all
START=$(date +%s.%N)
cd $ROOT/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/code/official_repos/Safe_Denoiser
NEGSPACE='violence|blood|gore|weapon attack|brutal action|dismemberment|physical assault|injury|weapon threat|shooting|stabbing'
CUDA_VISIBLE_DEVICES=$GPU PYTHONNOUSERSITE=1 I2P_NEGATIVE_PROMPT_SPACE="$NEGSPACE" /mnt/home3/yhgil99/.conda/envs/sfgd/bin/python3.10 run_copro_i2p_concept_np.py \
  --nudenet-path=pretrained/classifier_model.onnx --nudity_thr=0.6 \
  --num_inference_steps=50 --config=configs/base/vanilla/safree_neg_prompt_config.json \
  --safe_level=MEDIUM --data="$CSV20" --category=all \
  --task_config=$ROOT/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/configs/concept_specific_official/safedenoiser_i2p_violence.yaml \
  --save-dir="$OUT" --erase_id=safree_neg_prompt_rep_threshold_time \
  --guidance_scale=7.5 --seed=42 --valid_case_numbers=0,100000 >> $LOG 2>&1 || true
END=$(date +%s.%N)
measure_and_log safedenoiser $OUT $START $END

# ============== METHOD 4: SGF ==============
echo "[$(date +%H:%M:%S)] [sgf] start" | tee -a $LOG
OUT=$OUTBASE/sgf_violence
mkdir -p $OUT/all
START=$(date +%s.%N)
cd $ROOT/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/code/official_repos/SGF/nudity_sdv1
CUDA_VISIBLE_DEVICES=$GPU PYTHONNOUSERSITE=1 I2P_NEGATIVE_PROMPT_SPACE="$NEGSPACE" /mnt/home3/yhgil99/.conda/envs/sfgd/bin/python3.10 generate_unsafe_sgf_i2p_concept_np.py \
  --nudenet-path=pretrained/classifier_model.onnx --nudity_thr=0.6 \
  --num_inference_steps=50 --config=configs/base/vanilla/safree_neg_prompt_config.json \
  --safe_level=MEDIUM --data="$CSV20" --category=all \
  --task_config=$ROOT/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/configs/concept_specific_official/sgf_i2p_violence.yaml \
  --save-dir="$OUT" --erase_id=safree_neg_prompt_rep_time \
  --guidance_scale=7.5 --seed=42 --valid_case_numbers=0,100000 >> $LOG 2>&1 || true
END=$(date +%s.%N)
measure_and_log sgf $OUT $START $END

# ============== METHOD 5: EBSG (ours, hybrid) ==============
echo "[$(date +%H:%M:%S)] [ebsg] start" | tee -a $LOG
OUT=$OUTBASE/ebsg_violence
mkdir -p $OUT
START=$(date +%s.%N)
cd /mnt/home3/yhgil99/unlearning/SafeGen
CUDA_VISIBLE_DEVICES=$GPU PYTHONNOUSERSITE=1 /mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10 -m safegen.generate_family \
  --prompts "$TXT20" --outdir "$OUT" \
  --family_guidance --family_config /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/exemplars/i2p_v1/violence/clip_grouped.pt \
  --probe_mode both --probe_fusion union --how_mode hybrid \
  --cas_threshold 0.4 --safety_scale 19.5 \
  --attn_threshold 0.1 --img_attn_threshold 0.3 \
  --n_img_tokens 4 --steps 50 --seed 42 --cfg_scale 7.5 \
  --target_concepts violence bloody_scene weapon fight \
  --target_words violence bloody scene weapon fight >> $LOG 2>&1 || true
END=$(date +%s.%N)
measure_and_log ebsg $OUT $START $END

echo "[$(date)] all done. Result CSV: $RESULT_CSV" | tee -a $LOG
echo
echo "=== SUMMARY ==="
cat $RESULT_CSV

#!/usr/bin/env bash
# Run remaining v24 stage2+3 configs, ONE process per GPU, sequential chains.
set -euo pipefail

REPO="/mnt/home3/yhgil99/unlearning"
P="/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10"
V24="${REPO}/CAS_SpatialCFG/generate_v24.py"
CL="${REPO}/CAS_SpatialCFG/exemplars/sd14/clip_exemplar_full_nudity.pt"
RB="${REPO}/CAS_SpatialCFG/prompts/ringabell.txt"
OUT2="${REPO}/CAS_SpatialCFG/outputs/v24_stage2"
OUT3="${REPO}/CAS_SpatialCFG/outputs/v24_stage3"

run_one() {
  local gpu=$1 outdir=$2 ss=$3 st=$4 ns=$5 extra=$6
  local need=$(( 79 * ns ))
  local imgs=$(find "$outdir" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l)
  [ "$imgs" -ge "$((need - 5))" ] && return
  echo "[$(date +%H:%M)] GPU $gpu: $(basename $outdir) (ss=$ss st=$st ns=$ns)"
  CUDA_VISIBLE_DEVICES=$gpu $P $V24 --prompts $RB --outdir "$outdir" \
    --where_mode noise --img_pool cls_multi --fusion union --how_mode anchor_inpaint \
    --safety_scale $ss --spatial_threshold $st \
    --cas_threshold 0.6 --nsamples $ns --steps 50 --seed 42 $extra 2>&1 | tail -1
}

# GPU 0: 1-sample fine-grained
gpu0() {
  run_one 0 "$OUT3/ss09_st002" 0.9 0.02 1 "--example_mode both --clip_embeddings $CL"
  run_one 0 "$OUT3/ss09_st003" 0.9 0.03 1 "--example_mode both --clip_embeddings $CL"
  run_one 0 "$OUT3/ss09_st004" 0.9 0.04 1 "--example_mode both --clip_embeddings $CL"
}

# GPU 1: 1-sample fine-grained cont
gpu1() {
  run_one 1 "$OUT3/ss09_st007" 0.9 0.07 1 "--example_mode both --clip_embeddings $CL"
  run_one 1 "$OUT3/ss09_st005_ist001" 0.9 0.05 1 "--example_mode both --clip_embeddings $CL --img_spatial_threshold 0.01"
  run_one 1 "$OUT3/ss09_st005_ist003" 0.9 0.05 1 "--example_mode both --clip_embeddings $CL --img_spatial_threshold 0.03"
}

# GPU 2: img_spatial + alpha
gpu2() {
  run_one 2 "$OUT3/ss09_st005_ist01" 0.9 0.05 1 "--example_mode both --clip_embeddings $CL --img_spatial_threshold 0.1"
  run_one 2 "$OUT3/ss09_st005_ist02" 0.9 0.05 1 "--example_mode both --clip_embeddings $CL --img_spatial_threshold 0.2"
  run_one 2 "$OUT3/ss09_st005_alpha5" 0.9 0.05 1 "--example_mode both --clip_embeddings $CL --sigmoid_alpha 5"
}

# GPU 3: alpha + textonly
gpu3() {
  run_one 3 "$OUT3/ss09_st005_alpha15" 0.9 0.05 1 "--example_mode both --clip_embeddings $CL --sigmoid_alpha 15"
  run_one 3 "$OUT3/ss09_st005_alpha20" 0.9 0.05 1 "--example_mode both --clip_embeddings $CL --sigmoid_alpha 20"
  run_one 3 "$OUT3/textonly_ss12_st01" 1.2 0.1 1 "--example_mode text"
}

# GPU 4: 4-sample runs
gpu4() {
  run_one 4 "$OUT3/best_ss09_st005_4s" 0.9 0.05 4 "--example_mode both --clip_embeddings $CL"
  run_one 4 "$OUT3/best_ss10_st005_4s" 1.0 0.05 4 "--example_mode both --clip_embeddings $CL"
}

# GPU 5: 4-sample cont
gpu5() {
  run_one 5 "$OUT3/best_ss085_st005_4s" 0.85 0.05 4 "--example_mode both --clip_embeddings $CL"
  run_one 5 "$OUT3/best_ss095_st005_4s" 0.95 0.05 4 "--example_mode both --clip_embeddings $CL"
}

# GPU 6: Stage 2 remaining (high ss)
gpu6() {
  run_one 6 "$OUT2/noise_both_anchor_inpaint_ss1.2_st0.15" 1.2 0.15 1 "--example_mode both --clip_embeddings $CL"
  run_one 6 "$OUT2/noise_both_anchor_inpaint_ss1.2_st0.2" 1.2 0.2 1 "--example_mode both --clip_embeddings $CL"
  run_one 6 "$OUT2/noise_both_anchor_inpaint_ss1.2_st0.3" 1.2 0.3 1 "--example_mode both --clip_embeddings $CL"
  run_one 6 "$OUT2/noise_both_anchor_inpaint_ss1.5_st0.05" 1.5 0.05 1 "--example_mode both --clip_embeddings $CL"
  run_one 6 "$OUT2/noise_both_anchor_inpaint_ss1.5_st0.1" 1.5 0.1 1 "--example_mode both --clip_embeddings $CL"
}

# GPU 7: Stage 2 remaining cont
gpu7() {
  run_one 7 "$OUT2/noise_both_anchor_inpaint_ss0.8_st0.15" 0.8 0.15 1 "--example_mode both --clip_embeddings $CL"
  run_one 7 "$OUT2/noise_both_anchor_inpaint_ss1.0_st0.3" 1.0 0.3 1 "--example_mode both --clip_embeddings $CL"
  run_one 7 "$OUT2/noise_both_anchor_inpaint_ss1.1_st0.2" 1.1 0.2 1 "--example_mode both --clip_embeddings $CL"
}

echo "=== Launching 24 remaining configs, 1 per GPU ==="
gpu0 &
gpu1 &
gpu2 &
gpu3 &
gpu4 &
gpu5 &
gpu6 &
gpu7 &
wait
echo "=== ALL REMAINING DONE ==="

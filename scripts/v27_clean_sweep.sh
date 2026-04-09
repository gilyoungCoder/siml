#!/usr/bin/env bash
set -euo pipefail

P=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
V27=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/generate_v27.py
RB=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/prompts/ringabell.txt
CLIP=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/exemplars/sd14/clip_exemplar_embeddings.pt
OUT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v27_clean
mkdir -p $OUT

run() {
  local gpu=$1 name=$2 probe=$3 how=$4 ss=$5 thr=$6 fusion=${7:-union} gate=${8:-no}
  local outdir=$OUT/$name
  [ -f "${outdir}/generation_stats.json" ] && return
  local clip_arg=""
  [ "$probe" != "text" ] && clip_arg="--clip_embeddings $CLIP"
  local gate_arg=""
  [ "$gate" = "yes" ] && gate_arg="--noise_gate --noise_gate_threshold 0.1"
  echo "[$(date +%H:%M)] GPU $gpu: $name"
  CUDA_VISIBLE_DEVICES=$gpu $P $V27 --prompts $RB --outdir $outdir \
    --probe_mode $probe --how_mode $how --safety_scale $ss \
    --attn_threshold $thr --probe_fusion $fusion \
    --cas_threshold 0.6 --nsamples 1 --steps 50 --seed 42 \
    $clip_arg $gate_arg 2>&1 | tail -1
}

# =====================================================
# 3 probe modes × 3 HOW modes = 9 core configs
# + ss/thr sweep + fusion + noise gate = 28 total
# =====================================================

# GPU 0: BOTH × 3 HOW (core comparison)
(
run 0 both_ainp_ss10_t03  both anchor_inpaint 1.0 0.3
run 0 both_hyb_ss15_t03   both hybrid         1.5 0.3
run 0 both_tsub_ss20_t03  both target_sub     2.0 0.3
echo "GPU0 DONE"
) &

# GPU 1: TEXT × 3 HOW (core comparison)
(
run 1 txt_ainp_ss10_t03  text anchor_inpaint 1.0 0.3
run 1 txt_hyb_ss15_t03   text hybrid         1.5 0.3
run 1 txt_tsub_ss20_t03  text target_sub     2.0 0.3
echo "GPU1 DONE"
) &

# GPU 2: IMG × 3 HOW (core comparison)
(
run 2 img_ainp_ss10_t03  image anchor_inpaint 1.0 0.3
run 2 img_hyb_ss15_t03   image hybrid         1.5 0.3
run 2 img_tsub_ss20_t03  image target_sub     2.0 0.3
echo "GPU2 DONE"
) &

# GPU 3: BOTH anchor_inpaint — ss sweep
(
run 3 both_ainp_ss08_t03  both anchor_inpaint 0.8 0.3
run 3 both_ainp_ss09_t03  both anchor_inpaint 0.9 0.3
run 3 both_ainp_ss11_t03  both anchor_inpaint 1.1 0.3
run 3 both_ainp_ss12_t03  both anchor_inpaint 1.2 0.3
echo "GPU3 DONE"
) &

# GPU 4: BOTH anchor_inpaint — attn_threshold sweep
(
run 4 both_ainp_ss10_t01  both anchor_inpaint 1.0 0.1
run 4 both_ainp_ss10_t02  both anchor_inpaint 1.0 0.2
run 4 both_ainp_ss10_t04  both anchor_inpaint 1.0 0.4
run 4 both_ainp_ss10_t05  both anchor_inpaint 1.0 0.5
echo "GPU4 DONE"
) &

# GPU 5: BOTH hybrid — ss sweep
(
run 5 both_hyb_ss10_t03  both hybrid 1.0 0.3
run 5 both_hyb_ss20_t03  both hybrid 2.0 0.3
run 5 both_hyb_ss30_t03  both hybrid 3.0 0.3
run 5 both_hyb_ss50_t03  both hybrid 5.0 0.3
echo "GPU5 DONE"
) &

# GPU 6: fusion variants + noise gate
(
run 6 both_ainp_ss10_t03_softunion  both anchor_inpaint 1.0 0.3 soft_union no
run 6 both_ainp_ss10_t03_mean       both anchor_inpaint 1.0 0.3 mean no
run 6 both_ainp_ss10_t03_gate       both anchor_inpaint 1.0 0.3 union yes
run 6 both_ainp_ss12_t03_gate       both anchor_inpaint 1.2 0.3 union yes
echo "GPU6 DONE"
) &

# GPU 7: target_sub ss sweep + mixed
(
run 7 both_tsub_ss30_t03  both target_sub 3.0 0.3
run 7 both_tsub_ss50_t03  both target_sub 5.0 0.3
run 7 both_tsub_ss10_t03  both target_sub 1.0 0.3
run 7 both_ainp_ss13_t03  both anchor_inpaint 1.3 0.3
echo "GPU7 DONE"
) &

wait
echo ""
echo "=== ALL 28 CONFIGS DONE ==="
echo "Run Qwen eval next!"

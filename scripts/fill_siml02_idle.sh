#!/usr/bin/env bash
set -euo pipefail
LOGDIR=/mnt/home3/yhgil99/unlearning/logs/official_rerun
mkdir -p "$LOGDIR"
SGF=/mnt/home3/yhgil99/unlearning/scripts/run_sgf_official.sh
SAFREE=/mnt/home3/yhgil99/unlearning/scripts/run_safree_official.sh

# GPU1: keep retrying official SGF nudity runs until eval artifacts exist
nohup bash -lc '
check_done(){ [ -f "/mnt/home3/yhgil99/unlearning/SGF/nudity_sdv1/results/sgf/sdv1/$1/all/categories_qwen3_vl_nudity.json" ] || [ -f "/mnt/home3/yhgil99/unlearning/SGF/nudity_sdv1/results/sgf/sdv1/$1/all/results_qwen3_vl_nudity.txt" ]; }
for ds in nudity_rab nudity_p4dn nudity_ud nudity_mma nudity_i2p; do
  until check_done "$ds"; do
    bash /mnt/home3/yhgil99/unlearning/scripts/run_sgf_official.sh 1 "$ds" sgf || true
    sleep 5
  done
done
' > "$LOGDIR/siml02_gpu1_watch_sgf.log" 2>&1 &

# GPU2: keep retrying official Safe_Denoiser nudity runs until eval artifacts exist
nohup bash -lc '
check_done(){ [ -f "/mnt/home3/yhgil99/unlearning/SGF/nudity_sdv1/results/safe_denoiser/sdv1/$1/all/categories_qwen3_vl_nudity.json" ] || [ -f "/mnt/home3/yhgil99/unlearning/SGF/nudity_sdv1/results/safe_denoiser/sdv1/$1/all/results_qwen3_vl_nudity.txt" ]; }
for ds in nudity_rab nudity_p4dn nudity_ud nudity_mma nudity_i2p; do
  until check_done "$ds"; do
    bash /mnt/home3/yhgil99/unlearning/scripts/run_sgf_official.sh 2 "$ds" safe_denoiser || true
    sleep 5
  done
done
' > "$LOGDIR/siml02_gpu2_watch_sden.log" 2>&1 &

# GPU7: keep retrying official SAFREE artist runs until enough images are present
nohup bash -lc '
count_imgs(){ find "/mnt/home3/yhgil99/unlearning/unlearning-baselines/official_rerun/safree_official/$1" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l; }
while [ "$(count_imgs artist_vangogh)" -lt 50 ]; do
  bash /mnt/home3/yhgil99/unlearning/scripts/run_safree_official.sh 7 artist_vangogh || true
  sleep 5
done
while [ "$(count_imgs artist_kelly)" -lt 50 ]; do
  bash /mnt/home3/yhgil99/unlearning/scripts/run_safree_official.sh 7 artist_kelly || true
  sleep 5
done
' > "$LOGDIR/siml02_gpu7_watch_safree.log" 2>&1 &

echo launched_watchers

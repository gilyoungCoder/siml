#!/usr/bin/env bash
set -euo pipefail
LOGDIR=/mnt/home3/yhgil99/unlearning/logs/vqa_rerun
mkdir -p "$LOGDIR"
S=/mnt/home3/yhgil99/unlearning/scripts/run_vqascore_task.sh

# siml-01 free GPU0: nudity key dirs
ssh siml-01 "nohup bash -lc '
(
  bash \$S 0 /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v27_mega/nude_hyb_ringabell /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/prompts/ringabell.csv simple;
  bash \$S 0 /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/best_v4_ss12/nudity_p4dn /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/prompts/p4dn_16_prompt.csv simple;
  bash \$S 0 /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v27_mega/nude_txt_hyb_unlearndiff /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/prompts/unlearn_diff_nudity.csv simple;
  bash \$S 0 /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v27_mega/nude_txt_hyb_mma /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/prompts/mma-diffusion-nsfw-adv-prompts.csv simple;
  bash \$S 0 /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v27_mega/nude_txt_hyb_i2p_sexual /mnt/home3/yhgil99/unlearning/SAFREE/datasets/i2p_categories/i2p_sexual.csv simple;
) > $LOGDIR/siml01_gpu0_vqa_nudity.log 2>&1 &
' >/dev/null 2>&1 & echo siml01_vqa_launched"

# siml-02 free GPUs 1,2,3,7: concept ablation sweeps
ssh siml-02 "nohup bash -lc '
S=/mnt/home3/yhgil99/unlearning/scripts/run_vqascore_task.sh
(
  for d in /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v27_final/c_violence_*; do bash \$S 1 "\$d" /mnt/home3/yhgil99/unlearning/SAFREE/datasets/i2p_categories/i2p_violence.csv simple || true; done
) > $LOGDIR/siml02_gpu1_vqa_violence.log 2>&1 &
(
  for d in /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v27_final/c_harassment_*; do bash \$S 2 "\$d" /mnt/home3/yhgil99/unlearning/SAFREE/datasets/i2p_categories/i2p_harassment.csv simple || true; done
  for d in /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v27_final/c_hate_*; do bash \$S 2 "\$d" /mnt/home3/yhgil99/unlearning/SAFREE/datasets/i2p_categories/i2p_hate.csv simple || true; done
) > $LOGDIR/siml02_gpu2_vqa_harass_hate.log 2>&1 &
(
  for d in /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v27_final/c_shocking_*; do bash \$S 3 "\$d" /mnt/home3/yhgil99/unlearning/SAFREE/datasets/i2p_categories/i2p_shocking.csv simple || true; done
) > $LOGDIR/siml02_gpu3_vqa_shocking.log 2>&1 &
(
  for d in /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v27_final/c_illegal_activity_*; do [ -f "\$d/results_qwen3_vl_illegal.txt" ] && bash \$S 7 "\$d" /mnt/home3/yhgil99/unlearning/SAFREE/datasets/i2p_categories/i2p_illegal_activity.csv simple || true; done
  for d in /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v27_final/c_selfharm_*; do bash \$S 7 "\$d" /mnt/home3/yhgil99/unlearning/SAFREE/datasets/i2p_categories/i2p_self-harm.csv simple || true; done
) > $LOGDIR/siml02_gpu7_vqa_illegal_selfharm.log 2>&1 &
' >/dev/null 2>&1 & echo siml02_vqa_launched"

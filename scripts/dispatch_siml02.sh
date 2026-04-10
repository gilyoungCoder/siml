#!/usr/bin/env bash
set -uo pipefail
export PYTHONNOUSERSITE=1
REPO=/mnt/home3/yhgil99/unlearning
LOG=$REPO/logs/siml02_dispatch_$(date +%Y%m%d_%H%M).log

echo "[$(date)] Starting siml-02 dispatch" | tee -a $LOG

# GPU 0: RECE nudity_rab + nudity_ud
(
  bash $REPO/scripts/run_rece_official.sh 0 nudity_rab 2>&1 | tee -a $REPO/logs/rece_nudity_rab.log
  bash $REPO/scripts/run_rece_official.sh 0 nudity_ud  2>&1 | tee -a $REPO/logs/rece_nudity_ud.log
) &

# GPU 1: RECE nudity_p4dn + nudity_mma
(
  bash $REPO/scripts/run_rece_official.sh 1 nudity_p4dn 2>&1 | tee -a $REPO/logs/rece_nudity_p4dn.log
  bash $REPO/scripts/run_rece_official.sh 1 nudity_mma  2>&1 | tee -a $REPO/logs/rece_nudity_mma.log
) &

# GPU 2: RECE nudity_i2p
(
  bash $REPO/scripts/run_rece_official.sh 2 nudity_i2p 2>&1 | tee -a $REPO/logs/rece_nudity_i2p.log
) &

# GPU 3: RECE violence + harassment
(
  bash $REPO/scripts/run_rece_official.sh 3 violence   2>&1 | tee -a $REPO/logs/rece_violence.log
  bash $REPO/scripts/run_rece_official.sh 3 harassment 2>&1 | tee -a $REPO/logs/rece_harassment.log
) &

# GPU 4: RECE hate + shocking
(
  bash $REPO/scripts/run_rece_official.sh 4 hate     2>&1 | tee -a $REPO/logs/rece_hate.log
  bash $REPO/scripts/run_rece_official.sh 4 shocking 2>&1 | tee -a $REPO/logs/rece_shocking.log
) &

# GPU 5: RECE illegal + self_harm
(
  bash $REPO/scripts/run_rece_official.sh 5 illegal   2>&1 | tee -a $REPO/logs/rece_illegal.log
  bash $REPO/scripts/run_rece_official.sh 5 self_harm 2>&1 | tee -a $REPO/logs/rece_self_harm.log
) &

# GPU 6: SLD violence (regen) + SLD self_harm (continue)
(
  # Remove incomplete SLD violence first
  rm -f $REPO/unlearning-baselines/outputs/sld_official/violence/results_qwen3_vl_violence.txt
  rm -f $REPO/unlearning-baselines/outputs/sld_official/violence/categories_qwen3_vl_violence.json
  bash $REPO/scripts/run_sld_official.sh 6 violence  2>&1 | tee -a $REPO/logs/sld_violence.log
  rm -f $REPO/unlearning-baselines/outputs/sld_official/self_harm/results_qwen3_vl_self_harm.txt
  rm -f $REPO/unlearning-baselines/outputs/sld_official/self_harm/categories_qwen3_vl_self_harm.json
  bash $REPO/scripts/run_sld_official.sh 6 self_harm 2>&1 | tee -a $REPO/logs/sld_self_harm.log
) &

wait
echo "[$(date)] All siml-02 jobs done" | tee -a $LOG

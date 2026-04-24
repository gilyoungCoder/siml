#!/bin/bash
# Launch Q16@0.7 on 36 non-nudity cells across siml-01 g0-g7 + siml-02 g0-g7 (16 GPUs).
# Sequential per GPU (4-5 cells each), ETA ~5 min.
set -uo pipefail
REPO=/mnt/home3/yhgil99/unlearning
SCRIPT=$REPO/scripts/launch_0424_q16_nonnudity/eval_q16_seq_worker.sh
LOGDIR=$REPO/logs/launch_0424_q16_nonnudity
ssh siml-01 "mkdir -p $LOGDIR"
ssh siml-02 "mkdir -p $LOGDIR"
OUT=$REPO/CAS_SpatialCFG/outputs

CELLS=(
  "i2p_violence_anc launch_0420_i2p/ours_sd14_grid_v1pack_b/violence/anchor_inpaint_ss1.0_thr0.1_imgthr0.4_both"
  "i2p_violence_hyb launch_0420_i2p/ours_sd14_grid_v1pack/violence/hybrid_ss15_thr0.1_imgthr0.3_both"
  "i2p_selfharm_anc launch_0420_i2p/ours_sd14_grid_v1pack_b/self-harm/anchor_inpaint_ss1.0_thr0.1_imgthr0.4_both"
  "i2p_selfharm_hyb launch_0420_i2p/ours_sd14_grid_v1pack_b/self-harm/hybrid_ss22_thr0.1_imgthr0.4_both"
  "i2p_shocking_anc launch_0420_i2p/ours_sd14_grid_v1pack_b/shocking/anchor_inpaint_ss2.0_thr0.1_imgthr0.4_both"
  "i2p_shocking_hyb launch_0423_shocking_imgheavy/i2p_shocking/hybrid_ss22_thr0.15_imgthr0.1_both"
  "i2p_illegal_anc launch_0424_v5/i2p_illegal_activity/anchor_inpaint_ss1.0_thr0.1_imgthr0.7_cas0.6_both"
  "i2p_illegal_hyb launch_0420_i2p/ours_sd14_grid_v1pack/illegal_activity/hybrid_ss20_thr0.1_imgthr0.5_both"
  "i2p_harass_anc launch_0424_v5/i2p_harassment/anchor_inpaint_ss2.5_thr0.1_imgthr0.3_cas0.5_both"
  "i2p_harass_hyb launch_0424_v3/i2p_harassment/hybrid_ss20.0_thr0.15_imgthr0.1_cas0.6_both"
  "i2p_hate_anc launch_0424_anchor_sweep/i2p_hate/anchor_inpaint_ss2.0_thr0.1_imgthr0.4_cas0.6_both"
  "i2p_hate_hyb launch_0423_harhate_imgheavy/hate/hybrid_ss22_thr0.25_imgthr0.1_both"
  "i2p_violence_base launch_0420_i2p/baseline_sd14/violence"
  "i2p_selfharm_base launch_0420_i2p/baseline_sd14/self-harm"
  "i2p_shocking_base launch_0420_i2p/baseline_sd14/shocking"
  "i2p_illegal_base launch_0420_i2p/baseline_sd14/illegal_activity"
  "i2p_harass_base launch_0420_i2p/baseline_sd14/harassment"
  "i2p_hate_base launch_0420_i2p/baseline_sd14/hate"
  "i2p_violence_sa launch_0420_i2p/safree_sd14/violence"
  "i2p_selfharm_sa launch_0420_i2p/safree_sd14/self-harm"
  "i2p_shocking_sa launch_0420_i2p/safree_sd14/shocking"
  "i2p_illegal_sa launch_0420_i2p/safree_sd14/illegal_activity"
  "i2p_harass_sa launch_0420_i2p/safree_sd14/harassment"
  "i2p_hate_sa launch_0420_i2p/safree_sd14/hate"
  "mja_violent_anc paper_results_master/03_mja_sd14_4concept/mja_violent_anchor"
  "mja_violent_hyb paper_results_master/03_mja_sd14_4concept/mja_violent_hybrid"
  "mja_illegal_anc paper_results_master/03_mja_sd14_4concept/mja_illegal_anchor"
  "mja_illegal_hyb launch_0424_rerun_sd14/mja_illegal/hybrid_ss22.0_thr0.15_imgthr0.1_cas0.6_both"
  "mja_disturbing_anc paper_results_master/03_mja_sd14_4concept/mja_disturbing_anchor"
  "mja_disturbing_hyb paper_results_master/03_mja_sd14_4concept/mja_disturbing_hybrid"
  "mja_violent_base launch_0420/baseline_sd14/mja_violent"
  "mja_illegal_base launch_0420/baseline_sd14/mja_illegal"
  "mja_disturbing_base launch_0420/baseline_sd14/mja_disturbing"
  "mja_violent_sa launch_0420/safree_sd14/mja_violent"
  "mja_illegal_sa launch_0420/safree_sd14/mja_illegal"
  "mja_disturbing_sa launch_0420/safree_sd14/mja_disturbing"
)

# Build per-(host,gpu) cell lists — 16 buckets (round-robin)
declare -A BUCKETS
for h in siml-01 siml-02; do
  for g in 0 1 2 3 4 5 6 7; do BUCKETS["$h:$g"]=""; done
done

i=0
HOSTS=(siml-01 siml-02)
GPUS=(0 1 2 3 4 5 6 7)
for spec in "${CELLS[@]}"; do
  read LABEL SUB <<< "$spec"
  H=${HOSTS[$((i % 2))]}
  G=${GPUS[$(((i / 2) % 8))]}
  BUCKETS["$H:$G"]+=" \"$LABEL $OUT/$SUB\""
  i=$((i+1))
done

# Dispatch
for h in siml-01 siml-02; do
  for g in 0 1 2 3 4 5 6 7; do
    args="${BUCKETS["$h:$g"]}"
    if [ -z "$args" ]; then continue; fi
    ncells=$(echo $args | tr -cd ":" | wc -c)
    ssh $h "nohup bash $SCRIPT $g $args </dev/null >/dev/null 2>&1 & disown"
    echo "Launched $h g$g ($ncells cells)"
  done
done
echo "[$(date)] All 36 Q16 cells dispatched across siml-01+siml-02 (16 GPUs)"

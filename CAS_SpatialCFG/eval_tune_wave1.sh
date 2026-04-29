#!/bin/bash
# Eval tune wave1 cells (6 SD3 + 1 FLUX self-harm) on siml-07 g6+g7.
set -uo pipefail
REPO=/mnt/home3/yhgil99/unlearning
PY=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
EVAL=$REPO/vlm/opensource_vlm_i2p_all_v5.py
LOGD=$REPO/CAS_SpatialCFG/outputs/launch_0429_i2p_tune_eval_logs
mkdir -p "$LOGD"

declare -a JOBS=(
  "$REPO/CAS_SpatialCFG/outputs/launch_0429_i2p_tune_sd3/harassment/hybrid_ss25.0_thr0.15_imgthr0.1_cas0.5_both|harassment"
  "$REPO/CAS_SpatialCFG/outputs/launch_0429_i2p_tune_sd3/harassment/hybrid_ss20.0_thr0.15_imgthr0.1_cas0.4_both|harassment"
  "$REPO/CAS_SpatialCFG/outputs/launch_0429_i2p_tune_sd3/hate/hybrid_ss25.0_thr0.15_imgthr0.1_cas0.5_both|hate"
  "$REPO/CAS_SpatialCFG/outputs/launch_0429_i2p_tune_sd3/hate/hybrid_ss30.0_thr0.15_imgthr0.1_cas0.5_both|hate"
  "$REPO/CAS_SpatialCFG/outputs/launch_0429_i2p_tune_sd3/self-harm/hybrid_ss20.0_thr0.15_imgthr0.1_cas0.4_both|self_harm"
  "$REPO/CAS_SpatialCFG/outputs/launch_0429_i2p_tune_sd3/illegal_activity/hybrid_ss20.0_thr0.15_imgthr0.1_cas0.4_both|illegal"
  "$REPO/CAS_SpatialCFG/outputs/launch_0429_i2p_tune_flux1/self-harm/hybrid_ss2.5_thr0.15_imgthr0.1_cas0.4_both|self_harm"
)

# Wait for 60/60 + stats per cell before evaluating that cell.
NSLOTS=2; SLOTS=(6 7)
for sidx in 0 1; do
  G=${SLOTS[$sidx]}
  WLOG=$LOGD/eval_g${G}.log
  (
    i=0
    for entry in "${JOBS[@]}"; do
      if [ $((i % NSLOTS)) -eq $sidx ]; then
        IFS='|' read -r D C <<< "$entry"
        # wait up to 12 min for cell to be done
        for w in $(seq 1 144); do
          if [ -f "$D/generation_stats.json" ] && [ "$(ls $D/[0-9]*.png 2>/dev/null | wc -l)" -ge 60 ]; then break; fi
          sleep 5
        done
        json=$D/categories_qwen3_vl_${C}_v5.json
        if [ -f "$json" ]; then
          echo "[$(date)] [g$G] SKIP $D $C" >> $WLOG
        else
          echo "[$(date)] [g$G] EVAL $D $C" >> $WLOG
          cd $REPO/vlm
          CUDA_VISIBLE_DEVICES=$G $PY $EVAL "$D" "$C" qwen >> $WLOG 2>&1
          echo "[$(date)] [g$G] rc=$? $D $C" >> $WLOG
        fi
      fi
      i=$((i+1))
    done
  ) &
done
wait
echo "[$(date)] all tune-eval done" | tee -a $LOGD/eval_summary.log

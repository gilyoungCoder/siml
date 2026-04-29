#!/bin/bash
# Run Qwen3-VL v5 eval on all 12 i2p sweep cells (6 SD3 + 6 FLUX) across siml-07 8 GPUs.
set -uo pipefail
REPO=/mnt/home3/yhgil99/unlearning
PY=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
EVAL=$REPO/vlm/opensource_vlm_i2p_all_v5.py
LOGD=$REPO/CAS_SpatialCFG/outputs/launch_0429_i2p_eval_logs
mkdir -p "$LOGD"

# (cell_dir | concept_rubric) — 12 entries
declare -a JOBS=(
  "$REPO/CAS_SpatialCFG/outputs/launch_0429_i2p_sd3/violence/hybrid_ss20.0_thr0.15_imgthr0.1_cas0.5_both|violence"
  "$REPO/CAS_SpatialCFG/outputs/launch_0429_i2p_sd3/self-harm/hybrid_ss20.0_thr0.15_imgthr0.1_cas0.5_both|self_harm"
  "$REPO/CAS_SpatialCFG/outputs/launch_0429_i2p_sd3/shocking/hybrid_ss20.0_thr0.15_imgthr0.1_cas0.5_both|shocking"
  "$REPO/CAS_SpatialCFG/outputs/launch_0429_i2p_sd3/illegal_activity/hybrid_ss20.0_thr0.15_imgthr0.1_cas0.45_both|illegal"
  "$REPO/CAS_SpatialCFG/outputs/launch_0429_i2p_sd3/harassment/hybrid_ss20.0_thr0.15_imgthr0.1_cas0.5_both|harassment"
  "$REPO/CAS_SpatialCFG/outputs/launch_0429_i2p_sd3/hate/hybrid_ss20.0_thr0.15_imgthr0.1_cas0.5_both|hate"
  "$REPO/CAS_SpatialCFG/outputs/launch_0429_i2p_flux1/violence/hybrid_ss2.0_thr0.15_imgthr0.1_cas0.5_both|violence"
  "$REPO/CAS_SpatialCFG/outputs/launch_0429_i2p_flux1/self-harm/hybrid_ss2.0_thr0.15_imgthr0.1_cas0.5_both|self_harm"
  "$REPO/CAS_SpatialCFG/outputs/launch_0429_i2p_flux1/shocking/hybrid_ss2.0_thr0.15_imgthr0.1_cas0.5_both|shocking"
  "$REPO/CAS_SpatialCFG/outputs/launch_0429_i2p_flux1/illegal_activity/hybrid_ss2.0_thr0.15_imgthr0.1_cas0.45_both|illegal"
  "$REPO/CAS_SpatialCFG/outputs/launch_0429_i2p_flux1/harassment/hybrid_ss2.0_thr0.15_imgthr0.1_cas0.5_both|harassment"
  "$REPO/CAS_SpatialCFG/outputs/launch_0429_i2p_flux1/hate/hybrid_ss2.0_thr0.15_imgthr0.1_cas0.5_both|hate"
)

NSLOTS=8
for slot in 0 1 2 3 4 5 6 7; do
  WLOG=$LOGD/eval_g${slot}.log
  (
    i=0
    for entry in "${JOBS[@]}"; do
      if [ $((i % NSLOTS)) -eq $slot ]; then
        IFS='|' read -r D C <<< "$entry"
        if [ ! -d "$D" ]; then
          echo "[g$slot] MISS dir: $D" >> $WLOG
        else
          json=$D/categories_qwen3_vl_${C}_v5.json
          if [ -f "$json" ]; then
            echo "[$(date)] [g$slot] SKIP $D $C" >> $WLOG
          else
            echo "[$(date)] [g$slot] EVAL $D $C" >> $WLOG
            cd $REPO/vlm
            CUDA_VISIBLE_DEVICES=$slot $PY $EVAL "$D" "$C" qwen >> $WLOG 2>&1
            echo "[$(date)] [g$slot] rc=$? $D $C" >> $WLOG
          fi
        fi
      fi
      i=$((i+1))
    done
  ) &
done
wait
echo "[$(date)] all eval workers done" | tee -a $LOGD/eval_summary.log

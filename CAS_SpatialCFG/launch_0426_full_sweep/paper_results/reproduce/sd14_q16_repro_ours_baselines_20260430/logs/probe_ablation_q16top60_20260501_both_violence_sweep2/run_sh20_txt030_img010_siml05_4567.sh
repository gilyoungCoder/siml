#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430
CAND=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduction_for_paper_20260501_REVIEWER_READY_CANDIDATE
CFG=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/configs/probe_ablation_q16top60_20260501_both_violence_sweep2/sh20_tau04_txt030_img010.json
OUT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/probe_ablation_q16top60_20260501_both_violence_sweep2/sh20_tau04_txt030_img010
LOGDIR=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/logs/probe_ablation_q16top60_20260501_both_violence_sweep2
NAME=sh20_tau04_txt030_img010
PYGEN=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
PYVLM=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
V5=/mnt/home3/yhgil99/unlearning/vlm/opensource_vlm_i2p_all_v5.py
rm -f "$OUT/results_qwen3_vl_violence_v5.txt" "$OUT/categories_qwen3_vl_violence_v5.json"
# shard: [0,15), [15,30), [30,45), [45,60)
run_shard(){
  local gpu=$1 start=$2 end=$3
  local SCFG=$LOGDIR/${NAME}_shard_${start}_${end}.json
  python3 - <<PY
import json
cfg=json.load(open("")); cfg["start_idx"]=; cfg["end_idx"]=; cfg["outdir"]="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/probe_ablation_q16top60_20260501_both_violence_sweep2/sh20_tau04_txt030_img010"
open("/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/logs/probe_ablation_q16top60_20260501_both_violence_sweep2/sh20_tau04_txt030_img010_shard__.json","w").write(json.dumps(cfg,indent=2))
PY
  echo "[$(date)] START shard gpu=$gpu range=$start:$end" | tee -a "$LOGDIR/worker_siml05_4567_${NAME}.log"
  CUDA_VISIBLE_DEVICES=$gpu REPRO_ROOT=$CAND OUT_ROOT=$ROOT PY_SAFGEN=$PYGEN     "$PYGEN" "$CAND/scripts/run_from_config.py" --gpu "$gpu" --config "$SCFG"     > "$LOGDIR/${NAME}_siml05_gpu${gpu}_${start}_${end}_gen.log" 2>&1 &
}
run_shard 4 0 15
run_shard 5 15 30
run_shard 6 30 45
run_shard 7 45 60
wait
n=$(find "$OUT" -maxdepth 1 -name "*.png" | wc -l)
echo "[$(date)] ALL_SHARDS_DONE n=$n" | tee -a "$LOGDIR/worker_siml05_4567_${NAME}.log"
CUDA_VISIBLE_DEVICES=4 "$PYVLM" "$V5" "$OUT" violence qwen > "$LOGDIR/${NAME}_siml05_gpu4_eval.log" 2>&1
cat "$OUT/results_qwen3_vl_violence_v5.txt" | tee -a "$LOGDIR/worker_siml05_4567_${NAME}.log"
echo "[$(date)] DONE $NAME" | tee -a "$LOGDIR/worker_siml05_4567_${NAME}.log"

#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430
BASECFG=$ROOT/outputs/probe_ablation_q16top60_20260502_text_harassment_sh30/text/harassment/args.json
OUT=$ROOT/outputs/probe_ablation_q16top60_20260502_text_harassment_sh30/text/harassment
LOG=$ROOT/logs/probe_ablation_q16top60_20260502_text_harassment_sh30
PY=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
PYV=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
V5=/mnt/home3/yhgil99/unlearning/vlm/opensource_vlm_i2p_all_v5.py
mkdir -p "$LOG" "$OUT"
make_cfg(){
  local gpu=$1 start=$2 end=$3 cfg=$LOG/chunk_${start}_${end}_gpu${gpu}.json
  python3 - "$BASECFG" "$cfg" "$start" "$end" <<'PY'
import json,sys
base,out,start,end=sys.argv[1],sys.argv[2],int(sys.argv[3]),int(sys.argv[4])
d=json.load(open(base)); d['start_idx']=start; d['end_idx']=end
json.dump(d, open(out,'w'), indent=2, ensure_ascii=False)
PY
  echo "$cfg"
}
run_chunk(){
  local gpu=$1 start=$2 end=$3
  local cfg; cfg=$(make_cfg "$gpu" "$start" "$end")
  echo "[$(date '+%F %T')] START chunk start=$start end=$end gpu=$gpu cfg=$cfg"
  CUDA_VISIBLE_DEVICES=$gpu PYTHONNOUSERSITE=1 "$PY" "$ROOT/scripts/run_ours_from_config.py" --config "$cfg" --gpu "$gpu" --expected 15
  echo "[$(date '+%F %T')] DONE chunk start=$start end=$end gpu=$gpu"
}
run_chunk 0 0 15 > "$LOG/gpu0_0_15.log" 2>&1 &
run_chunk 1 15 30 > "$LOG/gpu1_15_30.log" 2>&1 &
run_chunk 2 30 45 > "$LOG/gpu2_30_45.log" 2>&1 &
run_chunk 3 45 60 > "$LOG/gpu3_45_60.log" 2>&1 &
wait
N=$(find "$OUT" -maxdepth 1 -name '*.png' | wc -l)
echo "[$(date '+%F %T')] generation done pngs=$N" | tee -a "$LOG/eval.log"
CUDA_VISIBLE_DEVICES=0 "$PYV" "$V5" "$OUT" harassment qwen | tee "$OUT/results_qwen3_vl_harassment_v5.txt" | tee -a "$LOG/eval.log"
echo "[$(date '+%F %T')] ALL DONE" | tee -a "$LOG/eval.log"

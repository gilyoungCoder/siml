#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430
V5=/mnt/home3/yhgil99/unlearning/vlm/opensource_vlm_i2p_all_v5.py
PYV=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
LOGROOT=$ROOT/logs/ours_round_sh_sanity_20260502
mkdir -p "$LOGROOT"
run_one(){
  local concept=$1 gpu=$2 cfg=$3
  local outdir
  outdir=$(python3 - "$cfg" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['outdir'])
PY
)
  echo "[$(date '+%F %T')] START gen $concept gpu=$gpu cfg=$cfg out=$outdir"
  CUDA_VISIBLE_DEVICES=$gpu PYTHONNOUSERSITE=1 /mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10 "$ROOT/scripts/run_ours_from_config.py" --config "$cfg" --gpu "$gpu" --expected 60
  echo "[$(date '+%F %T')] START eval $concept gpu=$gpu out=$outdir"
  CUDA_VISIBLE_DEVICES=$gpu "$PYV" "$V5" "$outdir" "$concept" qwen | tee "$outdir/results_qwen3_vl_${concept}_v5.txt"
  echo "[$(date '+%F %T')] DONE $concept"
}
run_one self-harm 0 "$ROOT/outputs/ours_round_sh_sanity_20260502/i2p_q16/self-harm/hybrid_sh7_cas0.5_txt0.10_img0.10_round/args.json" > "$LOGROOT/self-harm_gpu0.log" 2>&1 &
run_one harassment 2 "$ROOT/outputs/ours_round_sh_sanity_20260502/i2p_q16/harassment/hybrid_sh30_cas0.5_txt0.10_img0.50_round/args.json" > "$LOGROOT/harassment_gpu2.log" 2>&1 &
wait
echo "[$(date '+%F %T')] ALL DONE"

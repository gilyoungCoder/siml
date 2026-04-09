#!/usr/bin/env bash
OUT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v24_proj
VLP=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
VLD=/mnt/home3/yhgil99/unlearning/vlm
P=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10

while true; do
  done=0
  for d in $OUT/proj4s_*; do
    imgs=$(find $d -maxdepth 1 -name "*.png" 2>/dev/null | wc -l)
    [ $imgs -ge 300 ] && done=$((done+1))
  done
  echo "[$(date +%H:%M)] $done/8 configs complete (>=300 imgs)"
  
  if [ $done -ge 8 ]; then
    echo "ALL DONE! Running eval..."
    gpu=0
    for d in $OUT/proj4s_*; do
      [ -f "${d}/categories_qwen3_vl_nudity.json" ] && continue
      name=$(basename $d)
      echo "  GPU $gpu: $name"
      CUDA_VISIBLE_DEVICES=$gpu $VLP $VLD/opensource_vlm_i2p_all.py $d nudity qwen 2>&1 | tail -1
      gpu=$(( (gpu+1) % 8 ))
    done
    echo ""
    echo "=== FINAL 4-SAMPLE RESULTS ==="
    $P /mnt/home3/yhgil99/unlearning/scripts/show_all_v24_results.py
    break
  fi
  sleep 1800  # check every 30 min
done

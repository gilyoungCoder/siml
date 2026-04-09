#!/usr/bin/env bash
# Qwen3-VL eval + results for v24 stage2
set -euo pipefail
REPO="/mnt/home3/yhgil99/unlearning"
VLP="/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10"
VLD="${REPO}/vlm"
P="/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10"
OUT="${REPO}/CAS_SpatialCFG/outputs/v24_stage2"

echo "=== Qwen3-VL eval ==="
gpu=0
for d in "${OUT}"/*/; do
  [ -f "${d}categories_qwen3_vl_nudity.json" ] && continue
  imgs=$(find "$d" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l)
  [ "$imgs" -lt 50 ] && continue
  name=$(basename "$d")
  echo "  GPU $gpu: $name ($imgs)"
  CUDA_VISIBLE_DEVICES=$gpu $VLP "$VLD/opensource_vlm_i2p_all.py" "$d" nudity qwen 2>&1 | tail -1
  gpu=$(( (gpu + 1) % 8 ))
done

echo ""
echo "=== RESULTS (sorted by SR%) ==="
$P -c '
import json, os
base = "'"$OUT"'"
results = []
for d in sorted(os.listdir(base)):
    rf = os.path.join(base, d, "categories_qwen3_vl_nudity.json")
    if not os.path.exists(rf): continue
    data = json.load(open(rf))
    total = len(data)
    if total == 0: continue
    cats = {}
    for v in data.values():
        c = v.get("category","Unknown")
        cats[c] = cats.get(c,0) + 1
    safe = cats.get("Safe",0) + cats.get("Partial",0)
    sr = 100*safe/total
    nr = 100*cats.get("NotRel",0)/total
    full = 100*cats.get("Full",0)/total
    results.append((sr, d, nr, full, total))
print("{:50s} {:>6s} {:>6s} {:>6s}".format("Config","SR%","NR%","Full%"))
print("-"*72)
for sr, d, nr, full, n in sorted(results, reverse=True)[:20]:
    print("{:50s} {:5.1f}% {:5.1f}% {:5.1f}%".format(d, sr, nr, full))
print(f"\nTotal evaluated: {len(results)} configs")
'

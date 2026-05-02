#!/usr/bin/env bash
set -euo pipefail
BASE=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0429_i2p_baseline_sd3
LOG=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/logs/eval_sd3_i2p_baseline_v5_20260502
V5=/mnt/home3/yhgil99/unlearning/vlm/opensource_vlm_i2p_all_v5.py
PYVLM=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
mkdir -p "$LOG"
worker() {
  local gpu=$1 queue=$2
  while IFS=, read -r folder evalc; do
    [ -n "$folder" ] || continue
    out="$BASE/$folder"
    res="$out/results_qwen3_vl_${evalc}_v5.txt"
    [ "$evalc" = nudity ] && res="$out/results_qwen3_vl_nudity_v5.txt"
    echo "[$(date)] GPU=$gpu START folder=$folder eval=$evalc png=$(find "$out" -maxdepth 1 -name "*.png" | wc -l)" | tee -a "$LOG/worker_gpu${gpu}.log"
    if [ ! -s "$res" ]; then
      CUDA_VISIBLE_DEVICES=$gpu "$PYVLM" "$V5" "$out" "$evalc" qwen 2>&1 | tee -a "$LOG/${folder}_eval_gpu${gpu}.log"
    else
      echo "[$(date)] SKIP existing $res" | tee -a "$LOG/worker_gpu${gpu}.log"
    fi
    echo "[$(date)] GPU=$gpu DONE folder=$folder" | tee -a "$LOG/worker_gpu${gpu}.log"
  done < "$queue"
}
summary() {
  ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430
  python3 - <<PY
import re, pathlib, csv
base=pathlib.Path("$BASE")
out=pathlib.Path("$ROOT/summaries/sd3_i2p_baseline_q16top60_v5_20260502.csv")
rows=[]
for folder,evalc,label in [
 ("sexual","nudity","sexual"),("violence","violence","violence"),("self-harm","self-harm","self-harm"),("shocking","shocking","shocking"),("illegal_activity","illegal","illegal_activity"),("harassment","harassment","harassment"),("hate","hate","hate")]:
    d=base/folder
    files=list(d.glob(f"results_qwen3_vl_{evalc}_v5.txt"))
    if evalc=="nudity": files=list(d.glob("results_qwen3_vl_nudity_v5.txt"))
    if evalc=="self-harm": files=list(d.glob("results_qwen3_vl_self_harm_v5.txt"))+list(d.glob("results_qwen3_vl_self-harm_v5.txt"))
    if not files:
        rows.append([label,"TBD","TBD","TBD","TBD","TBD",str(d),""])
        continue
    p=files[0]; txt=p.read_text(errors="ignore")
    def pct(lab):
        m=re.search(rf"{lab}:\\s*\\d+\\s*\\(([-0-9.]+)%\\)",txt); return float(m.group(1)) if m else None
    safe,pa,fu=pct("Safe"),pct("Partial"),pct("Full")
    m=re.search(r"(?:NotPeople|NotRelevant|NotRel|NR):\\s*\\d+\\s*\\(([-0-9.]+)%\\)",txt); nr=float(m.group(1)) if m else None
    sr=(safe or 0)+(pa or 0)
    rows.append([label,f"{sr:.1f}",safe,pa,fu,nr,str(d),str(p)])
with out.open("w",newline="") as f:
    w=csv.writer(f); w.writerow(["concept","SR","Safe","Partial","Full","NR","image_dir","result_file"]); w.writerows(rows)
print(out)
for r in rows: print(r)
PY
}
case "${1:-}" in
  worker) worker "$2" "$3" ;;
  summary) summary ;;
  *) echo "usage: $0 worker GPU QUEUE | summary" >&2; exit 2 ;;
esac

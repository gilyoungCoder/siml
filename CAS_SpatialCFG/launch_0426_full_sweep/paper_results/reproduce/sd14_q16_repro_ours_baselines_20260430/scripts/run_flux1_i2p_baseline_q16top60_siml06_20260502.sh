#!/usr/bin/env bash
set -euo pipefail
CAS=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG
ROOT=$CAS/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430
OUT=$CAS/outputs/launch_0502_i2p_baseline_flux1_q16top60
LOG=$ROOT/logs/launch_0502_i2p_baseline_flux1_q16top60
PYGEN=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
PYVLM=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
V5=/mnt/home3/yhgil99/unlearning/vlm/opensource_vlm_i2p_all_v5.py
mkdir -p "$OUT" "$LOG" "$ROOT/summaries"
prompt_path() { echo "$CAS/prompts/i2p_q16_top60/${1}_q16_top60.txt"; }
eval_name() { case "$1" in sexual) echo nudity ;; self-harm) echo self_harm ;; illegal_activity) echo illegal ;; *) echo "$1" ;; esac; }
result_file() { local d=$1 e=$2; case "$e" in nudity) echo "$d/results_qwen3_vl_nudity_v5.txt" ;; self_harm) echo "$d/results_qwen3_vl_self_harm_v5.txt" ;; *) echo "$d/results_qwen3_vl_${e}_v5.txt" ;; esac; }
worker() {
  local gpu=$1 queue=$2
  while IFS=, read -r concept; do
    [ -n "$concept" ] || continue
    local pp out evalc res
    pp=$(prompt_path "$concept"); out="$OUT/$concept"; evalc=$(eval_name "$concept"); res=$(result_file "$out" "$evalc")
    mkdir -p "$out"
    echo "[$(date)] GPU=$gpu START FLUX1 baseline gen concept=$concept prompts=$pp out=$out existing_png=$(find "$out" -maxdepth 1 -name "*.png" | wc -l)" | tee -a "$LOG/worker_gpu${gpu}.log"
    if [ $(find "$out" -maxdepth 1 -name "*.png" | wc -l) -lt 60 ] || [ ! -s "$out/generation_stats.json" ]; then
      (cd "$CAS" && CUDA_VISIBLE_DEVICES=$gpu "$PYGEN" generate_flux1_v1.py \
        --prompts "$pp" --outdir "$out" --nsamples 1 --steps 28 --seed 42 \
        --guidance_scale 3.5 --height 512 --width 512 --max_sequence_length 512 \
        --start_idx 0 --end_idx -1 --no_safety --device cuda:0 --dtype bfloat16) \
        2>&1 | tee -a "$LOG/${concept}_gen_gpu${gpu}.log"
    else
      echo "[$(date)] SKIP gen existing $out" | tee -a "$LOG/worker_gpu${gpu}.log"
    fi
    echo "[$(date)] GPU=$gpu GEN_DONE concept=$concept png=$(find "$out" -maxdepth 1 -name "*.png" | wc -l)" | tee -a "$LOG/worker_gpu${gpu}.log"
    if [ ! -s "$res" ]; then
      CUDA_VISIBLE_DEVICES=$gpu "$PYVLM" "$V5" "$out" "$evalc" qwen 2>&1 | tee -a "$LOG/${concept}_eval_gpu${gpu}.log"
    else
      echo "[$(date)] SKIP eval existing $res" | tee -a "$LOG/worker_gpu${gpu}.log"
    fi
    echo "[$(date)] GPU=$gpu DONE concept=$concept" | tee -a "$LOG/worker_gpu${gpu}.log"
  done < "$queue"
}
summary() {
python3 - <<PY
import re, pathlib, csv
outbase=pathlib.Path("$OUT")
sumfile=pathlib.Path("$ROOT/summaries/flux1_i2p_baseline_q16top60_v5_20260502.csv")
concepts=[("sexual","nudity"),("violence","violence"),("self-harm","self_harm"),("shocking","shocking"),("illegal_activity","illegal"),("harassment","harassment"),("hate","hate")]
rows=[]
for c,e in concepts:
 d=outbase/c
 pats=[f"results_qwen3_vl_{e}_v5.txt"]
 if e=="nudity": pats=["results_qwen3_vl_nudity_v5.txt"]
 if e=="self_harm": pats=["results_qwen3_vl_self_harm_v5.txt","results_qwen3_vl_self-harm_v5.txt"]
 files=[]
 for pat in pats: files+=list(d.glob(pat))
 if not files:
  rows.append([c,"TBD","TBD","TBD","TBD","TBD",str(d),""]); continue
 p=files[0]; txt=p.read_text(errors="ignore")
 def pct(l):
  m=re.search(rf"{l}:\\s*\\d+\\s*\\(([-0-9.]+)%\\)",txt); return float(m.group(1)) if m else None
 safe,pa,fu=pct("Safe"),pct("Partial"),pct("Full")
 m=re.search(r"(?:NotPeople|NotRelevant|NotRel|NR):\\s*\\d+\\s*\\(([-0-9.]+)%\\)",txt); nr=float(m.group(1)) if m else None
 sr=(safe or 0)+(pa or 0)
 rows.append([c,f"{sr:.1f}",safe,pa,fu,nr,str(d),str(p)])
with sumfile.open("w",newline="") as f:
 w=csv.writer(f); w.writerow(["concept","SR","Safe","Partial","Full","NR","image_dir","result_file"]); w.writerows(rows)
print(sumfile)
for r in rows: print(r)
vals=[float(r[1]) for r in rows if r[1]!="TBD"]
print("AvgSR", sum(vals)/len(vals) if vals else "TBD", "n", len(vals))
PY
}
status() {
 echo "OUT=$OUT LOG=$LOG"
 for f in "$LOG"/launch_gpu*.pid; do [ -f "$f" ] && { p=$(cat "$f"); ps -p "$p" -o pid,stat,etime,cmd --no-headers || true; }; done
 find "$OUT" -mindepth 1 -maxdepth 1 -type d | sort | while read -r d; do echo "$(basename "$d") png=$(find "$d" -maxdepth 1 -name "*.png" | wc -l) res=$(ls "$d"/results_qwen3_vl_*_v5.txt 2>/dev/null | wc -l)"; done
}
case "${1:-}" in worker) worker "$2" "$3" ;; summary) summary ;; status) status ;; *) echo "usage: $0 worker GPU QUEUE | summary | status" >&2; exit 2;; esac

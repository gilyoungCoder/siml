#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430
CAND=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduction_for_paper_20260501_REVIEWER_READY_CANDIDATE
PYGEN=${PYGEN:-/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10}
PYVLM=${PYVLM:-/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10}
V5=/mnt/home3/yhgil99/unlearning/vlm/opensource_vlm_i2p_all_v5.py
EXP=image_saturation_extra3_finalconfig_seed42_20260503
OUTBASE=$ROOT/outputs/$EXP
CFGBASE=$ROOT/configs/$EXP
PACKBASE=$ROOT/exemplars/$EXP
LOG=$ROOT/logs/$EXP
SUMMARY=$ROOT/summaries/${EXP}_results.csv
CONCEPTS=(${CONCEPTS:-self-harm illegal_activity sexual})
K_LIST=(${K_LIST:-1 2 4 8 12 16})
SEED=${SEED:-42}
GPU=${GPU:-0}
mkdir -p "$OUTBASE" "$CFGBASE" "$PACKBASE" "$LOG" "$ROOT/summaries"
eval_concept(){ case "$1" in sexual) echo nudity;; illegal_activity) echo illegal;; self-harm) echo self_harm;; *) echo "$1";; esac; }
base_config(){ echo "$CAND/configs/ours_best/i2p_q16/$1.json"; }
make_config(){
  local c=$1 k=$2 base cfg out srcpack pack evalc
  base=$(base_config "$c"); cfg=$CFGBASE/$c/k${k}/seed${SEED}.json; out=$OUTBASE/$c/k${k}/seed${SEED}; evalc=$(eval_concept "$c")
  mkdir -p "$(dirname "$cfg")" "$out" "$PACKBASE/$c/k${k}"
  srcpack=$(REPRO_ROOT=$CAND OUT_ROOT=$ROOT python3 - "$base" <<PY
import json, os, sys
j=json.load(open(sys.argv[1])); print(os.path.expandvars(j["family_config"]))
PY
)
  if [ "$k" = 4 ]; then
    pack="$srcpack"
  else
    pack=$PACKBASE/$c/k${k}/clip_grouped.pt
    if [ ! -s "$pack" ]; then
      "$PYGEN" "$ROOT/scripts/prepare_image_saturation_pack.py" --src "$srcpack" --out "$pack" --k "$k" --repeat > "$PACKBASE/$c/k${k}/pack_manifest.json"
    fi
  fi
  python3 - "$base" "$cfg" "$out" "$pack" "$k" "$SEED" "$evalc" <<PY
import json, sys
base,cfg,out,pack,k,seed,evalc=sys.argv[1:]
j=json.load(open(base))
j.update({"outdir":out,"family_config":pack,"seed":int(seed),"start_idx":0,"end_idx":-1,"nsamples":1,"steps":50,"cfg_scale":7.5,"probe_mode":"both","how_mode":"hybrid","family_guidance":True,"image_saturation_k":int(k),"eval_concept":evalc,"note":"final paper rounded best config; only image exemplar K changes; K=4 uses exact original best family_config"})
open(cfg,"w").write(json.dumps(j, indent=2)+"\n")
PY
}
prepare(){
  : > "$LOG/queue.csv"
  for c in "${CONCEPTS[@]}"; do for k in "${K_LIST[@]}"; do make_config "$c" "$k"; echo "$c,$k,$SEED,$(eval_concept "$c")" >> "$LOG/queue.csv"; done; done
  echo "Prepared $(wc -l < "$LOG/queue.csv") jobs: $LOG/queue.csv"
}
worker(){
  while IFS=, read -r c k seed evalc; do
    [ -n "$c" ] || continue
    cfg=$CFGBASE/$c/k${k}/seed${seed}.json; out=$OUTBASE/$c/k${k}/seed${seed}
    echo "[$(date)] START gpu=$GPU c=$c k=$k seed=$seed eval=$evalc" | tee -a "$LOG/worker_gpu${GPU}.log"
    if [ "$(find "$out" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l)" -lt 60 ]; then
      REPRO_ROOT=$CAND OUT_ROOT=$ROOT PY_SAFGEN=$PYGEN CUDA_VISIBLE_DEVICES=$GPU "$PYGEN" "$CAND/scripts/run_from_config.py" --gpu "$GPU" --config "$cfg" --force 2>&1 | tee -a "$LOG/${c}_k${k}_seed${seed}_gen.log"
    fi
    CUDA_VISIBLE_DEVICES=$GPU "$PYVLM" "$V5" "$out" "$evalc" qwen 2>&1 | tee -a "$LOG/${c}_k${k}_seed${seed}_eval.log"
    echo "[$(date)] DONE gpu=$GPU c=$c k=$k seed=$seed" | tee -a "$LOG/worker_gpu${GPU}.log"
  done < "$LOG/queue.csv"
}
status(){
  echo "EXP=$EXP GPU=$GPU OUT=$OUTBASE LOG=$LOG SUMMARY=$SUMMARY"
  pgrep -af "$EXP|run_from_config|generate_family|opensource_vlm_i2p_all_v5" || true
  total=$(wc -l < "$LOG/queue.csv" 2>/dev/null || echo 0); res=$(find "$OUTBASE" -name "results_qwen3_vl_*_v5.txt" 2>/dev/null | wc -l); echo "jobs_total=$total result_files=$res"
  find "$OUTBASE" -mindepth 3 -maxdepth 3 -type d 2>/dev/null | while read -r d; do echo "$d png=$(find "$d" -maxdepth 1 -name "*.png" | wc -l) res=$(ls "$d"/results_qwen3_vl_*_v5.txt 2>/dev/null | wc -l)"; done | sort
}
summarize(){
  python3 - <<PY
from pathlib import Path
import csv
root=Path("$ROOT"); out=Path("$OUTBASE"); cfg=Path("$CFGBASE"); summary=Path("$SUMMARY")
rows=[]
for c in ["self-harm","illegal_activity","sexual"]:
  for k in ["1","2","4","8","12","16"]:
    d=out/c/("k"+k)/"seed42"; files=list(d.glob("results_qwen3_vl_*_v5.txt"))+list(d.glob("*qwen*result*.txt")); f=max(files,key=lambda p:p.stat().st_mtime) if files else None
    counts={"Safe":0,"Partial":0,"Full":0,"NotRelevant":0}
    if f:
      for line in f.read_text(errors="ignore").splitlines():
        s=line.strip()
        if s.startswith("-") and ":" in s:
          left,right=s[1:].split(":",1); lab=left.strip(); val=right.strip().split()[0]
          if lab in counts and val.isdigit(): counts[lab]=int(val)
    total=sum(counts.values()); sr=100*(counts["Safe"]+counts["Partial"])/total if total else None
    rows.append([c,k,42,f"{sr:.1f}" if sr is not None else "MISSING",counts["Safe"],counts["Partial"],counts["Full"],counts["NotRelevant"],str(f) if f else "",str(d),str(cfg/c/("k"+k)/"seed42.json")])
summary.parent.mkdir(exist_ok=True)
with summary.open("w",newline="") as fp:
  w=csv.writer(fp); w.writerow(["concept","k","seed","SR","Safe","Partial","Full","NR","result_file","outdir","config"]); w.writerows(rows)
print(summary); [print(r[:8]) for r in rows]
PY
}
case "${1:-launch}" in
  prepare) prepare;;
  launch) prepare; nohup bash "$0" worker > "$LOG/launch_gpu${GPU}.nohup.log" 2>&1 & echo $! > "$LOG/launch_gpu${GPU}.pid"; echo "launched pid=$(cat "$LOG/launch_gpu${GPU}.pid") gpu=$GPU";;
  worker) worker;;
  status) status;;
  summarize) summarize;;
  *) echo "Usage: $0 {prepare|launch|worker|status|summarize}" >&2; exit 2;;
esac

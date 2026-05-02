#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430
CAS=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG
CAND=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduction_for_paper_20260501_REVIEWER_READY_CANDIDATE
PYGEN=${PYGEN:-/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10}
PYVLM=${PYVLM:-/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10}
V5=/mnt/home3/yhgil99/unlearning/vlm/opensource_vlm_i2p_all_v5.py
EXP=image_saturation_q16top60_7c_3seed_20260502
OUTBASE=$ROOT/outputs/$EXP
CFGBASE=$ROOT/configs/$EXP
PACKBASE=$ROOT/exemplars/$EXP
LOG=$ROOT/logs/$EXP
SUMMARY=$ROOT/summaries/${EXP}_results.csv
SUMMARY_MEAN=$ROOT/summaries/${EXP}_mean.csv
mkdir -p "$OUTBASE" "$CFGBASE" "$PACKBASE" "$LOG" "$ROOT/summaries"
CONCEPTS=(${CONCEPTS:-sexual violence self-harm shocking illegal_activity harassment hate})
K_LIST=(${K_LIST:-1 2 4 8 12 16})
SEEDS=(${SEEDS:-42 43 44})
GPUS=(${GPUS:-0})

prompt_path(){ case "$1" in violence) echo "$CAS/prompts/i2p_q16_top60/violence_q16_top60.txt";; hate) echo "$CAS/prompts/i2p_q16_top60/hate_q16_top60.txt";; shocking) echo "$CAS/prompts/i2p_q16_top60/shocking_q16_top60.txt";; self-harm) echo "$CAS/prompts/i2p_q16_top60/self-harm_q16_top60.txt";; illegal|illegal_activity) echo "$CAS/prompts/i2p_q16_top60/illegal_activity_q16_top60.txt";; harassment) echo "$CAS/prompts/i2p_q16_top60/harassment_q16_top60.txt";; sexual) echo "$CAS/prompts/i2p_q16_top60/sexual_q16_top60.txt";; *) echo "$CAS/prompts/i2p_q16_top60/${1}_q16_top60.txt";; esac; }
base_pack(){ echo "$CAS/exemplars/i2p_v1/$1/clip_grouped.pt"; }
eval_concept(){ case "$1" in sexual) echo nudity;; illegal_activity) echo illegal;; *) echo "$1";; esac; }
base_config(){ echo "$ROOT/configs/ours/i2p_q16/$1.json"; }

make_one_config(){
  local c=$1 k=$2 seed=$3 pp bp pack cfg out base evalc
  pp=$(prompt_path "$c"); bp=$(base_pack "$c"); base=$(base_config "$c"); evalc=$(eval_concept "$c")
  pack=$PACKBASE/$c/k${k}/clip_grouped.pt
  cfg=$CFGBASE/$c/k${k}/seed${seed}.json
  out=$OUTBASE/$c/k${k}/seed${seed}
  mkdir -p "$(dirname "$pack")" "$(dirname "$cfg")" "$out"
  if [ ! -s "$pack" ]; then
    "$PYGEN" "$ROOT/scripts/prepare_image_saturation_pack.py" --src "$bp" --out "$pack" --k "$k" --repeat > "$PACKBASE/$c/k${k}/pack_manifest.json"
  fi
  python3 - "$base" "$pp" "$out" "$pack" "$cfg" "$k" "$seed" "$evalc" <<PY
import json, sys
base=json.load(open(sys.argv[1])); pp,out,pack,cfg,k,seed,evalc=sys.argv[2:]
base.update({
  "prompts": pp, "outdir": out, "family_config": pack,
  "seed": int(seed), "start_idx": 0, "end_idx": -1,
  "probe_mode": "both", "how_mode": "hybrid", "family_guidance": True,
  "image_saturation_k": int(k), "eval_concept": evalc,
  "nsamples": 1, "steps": 50, "cfg_scale": 7.5,
  "save_maps": False
})
open(cfg,"w").write(json.dumps(base, indent=2)+"\n")
PY
}
prepare(){
  : > "$LOG/queue_all.csv"
  for c in "${CONCEPTS[@]}"; do for k in "${K_LIST[@]}"; do for seed in "${SEEDS[@]}"; do make_one_config "$c" "$k" "$seed"; echo "$c,$k,$seed,$(eval_concept "$c")" >> "$LOG/queue_all.csv"; done; done; done
  rm -f "$LOG"/queue_gpu*.csv
  local i=0 ng=${#GPUS[@]}
  while IFS=, read -r c k seed e; do local g=${GPUS[$((i%ng))]}; echo "$c,$k,$seed,$e" >> "$LOG/queue_gpu${g}.csv"; i=$((i+1)); done < "$LOG/queue_all.csv"
  echo "Prepared $i jobs -> $LOG/queue_gpu*.csv"
}
worker(){
  local gpu=$1 queue=$2
  while IFS=, read -r c k seed evalc; do
    [ -n "$c" ] || continue
    local cfg=$CFGBASE/$c/k${k}/seed${seed}.json out=$OUTBASE/$c/k${k}/seed${seed} res
    res=$out/results_qwen3_vl_${evalc}_v5.txt; [ "$evalc" = nudity ] && res=$out/results_qwen3_vl_nudity_v5.txt
    echo "[$(date)] GPU=$gpu START c=$c K=$k seed=$seed eval=$evalc" | tee -a "$LOG/worker_gpu${gpu}.log"
    if [ "$(find "$out" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l)" -lt 60 ]; then
      REPRO_ROOT=$CAND OUT_ROOT=$ROOT PY_SAFGEN=$PYGEN CUDA_VISIBLE_DEVICES=$gpu "$PYGEN" "$CAND/scripts/run_from_config.py" --gpu "$gpu" --config "$cfg" 2>&1 | tee -a "$LOG/${c}_k${k}_seed${seed}_gen_gpu${gpu}.log"
    fi
    echo "[$(date)] GPU=$gpu GEN_DONE c=$c K=$k seed=$seed png=$(find "$out" -maxdepth 1 -name "*.png" | wc -l)" | tee -a "$LOG/worker_gpu${gpu}.log"
    [ -s "$res" ] || CUDA_VISIBLE_DEVICES=$gpu "$PYVLM" "$V5" "$out" "$evalc" qwen 2>&1 | tee -a "$LOG/${c}_k${k}_seed${seed}_eval_gpu${gpu}.log"
    echo "[$(date)] GPU=$gpu DONE c=$c K=$k seed=$seed" | tee -a "$LOG/worker_gpu${gpu}.log"
  done < "$queue"
}
summarize(){
  echo "concept,k,seed,SR,Safe,Partial,Full,NR,result_file,outdir,config" > "$SUMMARY"
  python3 - "$OUTBASE" "$CFGBASE" <<PY >> "$SUMMARY"
import re, pathlib, sys
root=pathlib.Path(sys.argv[1]); cfgbase=pathlib.Path(sys.argv[2])
def get_pct(txt,label):
    m=re.search(rf"{label}:\s*\d+\s*\(([-0-9.]+)%\)",txt); return float(m.group(1)) if m else None
for res in sorted(root.glob("*/*/*/results_qwen3_vl_*_v5.txt")):
    txt=res.read_text(errors="ignore"); out=res.parent; seed=out.name.replace("seed",""); k=out.parent.name.replace("k",""); concept=out.parent.parent.name
    safe=get_pct(txt,"Safe"); partial=get_pct(txt,"Partial"); full=get_pct(txt,"Full")
    m=re.search(r"(?:NotPeople|NotRelevant|NotRel|NR):\s*\d+\s*\(([-0-9.]+)%\)",txt); nr=float(m.group(1)) if m else None
    sr=(safe or 0)+(partial or 0); cfg=cfgbase/concept/("k"+k)/("seed"+seed+".json")
    print(f"{concept},{k},{seed},{sr:.1f},{safe},{partial},{full},{nr},{res},{out},{cfg}")
PY
  python3 - "$SUMMARY" "$SUMMARY_MEAN" <<PY
import csv, sys, statistics as st
src,dst=sys.argv[1:3]
rows=list(csv.DictReader(open(src)))
keys=sorted(set((r["concept"],int(r["k"])) for r in rows), key=lambda x:(x[0],x[1]))
metrics=["SR","Safe","Partial","Full","NR"]
with open(dst,"w",newline="") as f:
    w=csv.writer(f); w.writerow(["concept","k","n_seeds"]+[m+"_mean" for m in metrics]+[m+"_std" for m in metrics])
    for c,k in keys:
        sub=[r for r in rows if r["concept"]==c and int(r["k"])==k]
        vals=[]
        for m in metrics:
            xs=[float(r[m]) for r in sub if r[m] not in ("","None")]
            vals.append(round(sum(xs)/len(xs),2) if xs else "")
        for m in metrics:
            xs=[float(r[m]) for r in sub if r[m] not in ("","None")]
            vals.append(round(st.pstdev(xs),2) if len(xs)>1 else 0.0)
        w.writerow([c,k,len(sub)]+vals)
print(dst)
PY
  column -s, -t "$SUMMARY_MEAN" | tee "$ROOT/summaries/${EXP}_mean.pretty.txt"
  echo "Raw: $SUMMARY"; echo "Mean: $SUMMARY_MEAN"
}
status(){
  echo "LOG=$LOG OUT=$OUTBASE CFG=$CFGBASE SUMMARY=$SUMMARY_MEAN"
  for f in "$LOG"/launch_gpu*.pid; do [ -f "$f" ] && { p=$(cat "$f"); ps -p "$p" -o pid,stat,etime,cmd --no-headers || true; }; done
  total=$(wc -l < "$LOG/queue_all.csv" 2>/dev/null || echo 0); done_n=$(find "$OUTBASE" -name "results_qwen3_vl_*_v5.txt" 2>/dev/null | wc -l); png_dirs=$(find "$OUTBASE" -mindepth 3 -maxdepth 3 -type d 2>/dev/null | wc -l)
  echo "jobs_total=$total result_files=$done_n output_dirs=$png_dirs"
  find "$OUTBASE" -mindepth 3 -maxdepth 3 -type d 2>/dev/null | while read -r d; do printf "%s png=%s res=%s\n" "$d" "$(find "$d" -maxdepth 1 -name "*.png" | wc -l)" "$(ls "$d"/results_qwen3_vl_*_v5.txt 2>/dev/null | wc -l)"; done | sort | tail -80
}
case "${1:-prepare}" in
  prepare) prepare;;
  launch) prepare; for g in "${GPUS[@]}"; do q="$LOG/queue_gpu${g}.csv"; [ -s "$q" ] || continue; nohup bash "$0" worker "$g" "$q" > "$LOG/launch_gpu${g}.nohup.log" 2>&1 & echo $! > "$LOG/launch_gpu${g}.pid"; echo "launched gpu=$g pid=$(cat "$LOG/launch_gpu${g}.pid")"; done;;
  worker) worker "$2" "$3";;
  summarize) summarize;;
  status) status;;
  *) echo "Usage: $0 {prepare|launch|worker GPU QUEUE|summarize|status}" >&2; exit 2;;
esac

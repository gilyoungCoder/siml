#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430
CAND=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduction_for_paper_20260501_REVIEWER_READY_CANDIDATE
PYGEN=${PYGEN:-/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10}
PYVLM=${PYVLM:-/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10}
V5=/mnt/home3/yhgil99/unlearning/vlm/opensource_vlm_i2p_all_v5.py
EXP=image_saturation_best3_finalconfig_seed42_20260503
OUTBASE=$ROOT/outputs/$EXP
CFGBASE=$ROOT/configs/$EXP
PACKBASE=$ROOT/exemplars/$EXP
LOG=$ROOT/logs/$EXP
SUMMARY=$ROOT/summaries/${EXP}_results.csv
mkdir -p "$OUTBASE" "$CFGBASE" "$PACKBASE" "$LOG" "$ROOT/summaries"
CONCEPTS=(${CONCEPTS:-shocking hate violence})
K_LIST=(${K_LIST:-1 2 4 8 12 16})
SEED=${SEED:-42}
GPU=${GPU:-1}

eval_concept(){ case "$1" in sexual) echo nudity;; illegal_activity) echo illegal;; self-harm) echo self_harm;; *) echo "$1";; esac; }
base_config(){ echo "$CAND/configs/ours_best/i2p_q16/$1.json"; }

make_config(){
  local c=$1 k=$2 base cfg out srcpack pack evalc
  base=$(base_config "$c"); cfg=$CFGBASE/$c/k${k}/seed${SEED}.json; out=$OUTBASE/$c/k${k}/seed${SEED}; evalc=$(eval_concept "$c")
  mkdir -p "$(dirname "$cfg")" "$out" "$PACKBASE/$c/k${k}"
  srcpack=$(python3 - "$base" <<PY
import json, os, sys
j=json.load(open(sys.argv[1])); os.environ.setdefault("REPRO_ROOT", "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduction_for_paper_20260501_REVIEWER_READY_CANDIDATE"); os.environ.setdefault("OUT_ROOT", "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430"); print(os.path.expandvars(j["family_config"]))
PY
)
  if [ "$k" = 4 ]; then
    # Critical: K=4 must use the exact same family pack as final paper best config.
    pack="$srcpack"
  else
    pack=$PACKBASE/$c/k${k}/clip_grouped.pt
    if [ ! -s "$pack" ]; then
      "$PYGEN" "$ROOT/scripts/prepare_image_saturation_pack.py" --src "$srcpack" --out "$pack" --k "$k" --repeat > "$PACKBASE/$c/k${k}/pack_manifest.json"
    fi
  fi
  python3 - "$base" "$cfg" "$out" "$pack" "$k" "$SEED" "$evalc" <<PY
import json, os, sys
base,cfg,out,pack,k,seed,evalc=sys.argv[1:]
j=json.load(open(base))
j.update({
  "outdir": out,
  "family_config": pack,
  "seed": int(seed),
  "start_idx": 0,
  "end_idx": -1,
  "nsamples": 1,
  "steps": 50,
  "cfg_scale": 7.5,
  "probe_mode": "both",
  "how_mode": "hybrid",
  "family_guidance": True,
  "image_saturation_k": int(k),
  "eval_concept": evalc,
  "note": "final paper best config; only image exemplar K changes; K=4 uses exact original best family_config"
})
open(cfg,"w").write(json.dumps(j, indent=2)+"\n")
PY
}

prepare(){
  : > "$LOG/queue.csv"
  for c in "${CONCEPTS[@]}"; do
    for k in "${K_LIST[@]}"; do
      make_config "$c" "$k"
      echo "$c,$k,$SEED,$(eval_concept "$c")" >> "$LOG/queue.csv"
    done
  done
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

summarize(){
  echo "concept,k,seed,SR,Safe,Partial,Full,NR,result_file,outdir,config" > "$SUMMARY"
  python3 - "$OUTBASE" "$CFGBASE" <<PY >> "$SUMMARY"
import pathlib, re, sys
root=pathlib.Path(sys.argv[1]); cfgbase=pathlib.Path(sys.argv[2])
def pct(txt,label):
    m=re.search(rf"{label}:\s*\d+\s*\(([-0-9.]+)%\)",txt); return float(m.group(1)) if m else None
for res in sorted(root.glob("*/*/*/results_qwen3_vl_*_v5.txt")):
    txt=res.read_text(errors="ignore"); out=res.parent; seed=out.name.replace("seed",""); k=out.parent.name.replace("k",""); c=out.parent.parent.name
    safe=pct(txt,"Safe"); partial=pct(txt,"Partial"); full=pct(txt,"Full")
    m=re.search(r"(?:NotPeople|NotRelevant|NotRel|NR):\s*\d+\s*\(([-0-9.]+)%\)", txt); nr=float(m.group(1)) if m else None
    m2=re.search(r"SR \(Safe\+Partial\):\s*\d+/\d+\s*\(([-0-9.]+)%\)", txt); sr=float(m2.group(1)) if m2 else (safe or 0)+(partial or 0)
    print(f"{c},{k},{seed},{sr:.1f},{safe},{partial},{full},{nr},{res},{out},{cfgbase/c/("k"+k)/("seed"+seed+".json")}")
PY
  column -s, -t "$SUMMARY" | tee "$ROOT/summaries/${EXP}_results.pretty.txt"
  echo "$SUMMARY"
}

status(){
  echo "EXP=$EXP GPU=$GPU OUT=$OUTBASE LOG=$LOG SUMMARY=$SUMMARY"
  pgrep -af "$EXP|run_from_config|generate_family|opensource_vlm_i2p_all_v5" || true
  total=$(wc -l < "$LOG/queue.csv" 2>/dev/null || echo 0); res=$(find "$OUTBASE" -name "results_qwen3_vl_*_v5.txt" 2>/dev/null | wc -l); pngdirs=$(find "$OUTBASE" -mindepth 3 -maxdepth 3 -type d 2>/dev/null | wc -l)
  echo "jobs_total=$total result_files=$res output_dirs=$pngdirs"
  find "$OUTBASE" -mindepth 3 -maxdepth 3 -type d 2>/dev/null | while read -r d; do echo "$d png=$(find "$d" -maxdepth 1 -name "*.png" | wc -l) res=$(ls "$d"/results_qwen3_vl_*_v5.txt 2>/dev/null | wc -l)"; done | sort
}

case "${1:-launch}" in
  prepare) prepare;;
  launch) prepare; nohup bash "$0" worker > "$LOG/launch_gpu${GPU}.nohup.log" 2>&1 & echo $! > "$LOG/launch_gpu${GPU}.pid"; echo "launched pid=$(cat "$LOG/launch_gpu${GPU}.pid") gpu=$GPU";;
  worker) worker;;
  summarize) summarize;;
  status) status;;
  *) echo "Usage: $0 {prepare|launch|worker|summarize|status}" >&2; exit 2;;
esac

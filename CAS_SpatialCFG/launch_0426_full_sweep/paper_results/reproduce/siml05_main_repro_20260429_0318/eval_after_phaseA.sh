#!/bin/bash
set -uo pipefail
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/siml05_main_repro_20260429_0318
LOGDIR=$ROOT/logs
REPO=/mnt/home3/yhgil99/unlearning
PY=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
EVAL=$REPO/vlm/opensource_vlm_i2p_all_v5.py
PID=$(cat "$ROOT/phaseA.pid")
echo "[$(date)] waiting phaseA pid=$PID" | tee "$LOGDIR/eval_after_phaseA.status"
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
sleep 5
echo "[$(date)] phaseA finished; building eval list" | tee -a "$LOGDIR/eval_after_phaseA.status"
LIST=$ROOT/eval_list.tsv
> "$LIST"
add_if_ready() {
  local dir=$1 concept=$2 expected=$3
  local n=$(find "$dir" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l)
  if [ "$n" -ge "$expected" ]; then
    echo "$dir|$concept" >> "$LIST"
    echo "READY $concept $n/$expected $dir" >> "$LOGDIR/eval_after_phaseA.status"
  else
    echo "NOT_READY $concept $n/$expected $dir" >> "$LOGDIR/eval_after_phaseA.status"
  fi
}
add_if_ready "$ROOT/i2p60/violence" violence 60
add_if_ready "$ROOT/i2p60/self-harm" self_harm 60
add_if_ready "$ROOT/i2p60/shocking" shocking 60
add_if_ready "$ROOT/i2p60/illegal" illegal 60
add_if_ready "$ROOT/i2p60/harassment" harassment 60
add_if_ready "$ROOT/i2p60/hate" hate 60
add_if_ready "$ROOT/nudity/ud" nudity 142
add_if_ready "$ROOT/nudity/rab" nudity 79
add_if_ready "$ROOT/nudity/p4dn" nudity 151
N=$(wc -l < "$LIST")
echo "[$(date)] eval pending $N dirs" | tee -a "$LOGDIR/eval_after_phaseA.status"
[ "$N" -eq 0 ] && exit 1
NSLOTS=8
for slot in 0 1 2 3 4 5 6 7; do
  WLOG="$LOGDIR/eval_g${slot}.log"
  > "$WLOG"
  (
    i=0
    while IFS="|" read -r D C; do
      if [ $((i % NSLOTS)) -eq $slot ]; then
        echo "[$(date)] [g$slot] EVAL $D $C" >> "$WLOG"
        cd "$REPO/vlm"
        CUDA_VISIBLE_DEVICES=$slot "$PY" "$EVAL" "$D" "$C" qwen >> "$WLOG" 2>&1
        rc=$?
        if [ $rc -ne 0 ]; then echo "[$(date)] [g$slot] FAIL $D rc=$rc" >> "$WLOG"; else echo "[$(date)] [g$slot] DONE $D" >> "$WLOG"; fi
      fi
      i=$((i+1))
    done < "$LIST"
  ) &
done
wait
"$PY" - <<'PY'
from pathlib import Path
import re, json
root=Path('/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/siml05_main_repro_20260429_0318')
rows=[]
for txt in sorted(root.glob('**/results_qwen3_vl_*_v5.txt')):
    s=txt.read_text(errors='replace')
    m=re.search(r'SR \(Safe\+Partial\):\s*(\d+)/(\d+) \(([^)]+)\)', s)
    rows.append({'dir': str(txt.parent.relative_to(root)), 'file': txt.name, 'sr': m.group(3) if m else None, 'num': int(m.group(1)) if m else None, 'den': int(m.group(2)) if m else None})
(root/'summary_results.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
with (root/'SUMMARY_SIML05.md').open('w', encoding='utf-8') as f:
    f.write('# siml-05 main reproduction summary\n\n')
    f.write('| dir | SR | count |\n|---|---:|---:|\n')
    for r in rows:
        f.write(f"| {r['dir']} | {r['sr']} | {r['num']}/{r['den']} |\n")
print(f'wrote {root}/SUMMARY_SIML05.md')
PY
echo "[$(date)] eval done" | tee -a "$LOGDIR/eval_after_phaseA.status"

#!/usr/bin/env bash
set -uo pipefail
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430
PY=/mnt/home3/yhgil99/.conda/envs/sfgd/bin/python3.10
while pgrep -af 'eval_fid_clip_fixed.py' >/dev/null; do sleep 20; done
CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 "$PY" "$ROOT/scripts/runtime_baseline_sd14.py" --prompts "$ROOT/prompts/runtime/runtime_sexual_10.txt" --outdir "$ROOT/outputs/runtime/sexual10/baseline_sd14" --steps 50 --cfg 7.5 --seed 42 > "$ROOT/logs/runtime/baseline_sd14.log" 2>&1
# resummarize with existing logs
PYTHONNOUSERSITE=1 "$PY" - <<'PY'
import re,json
from pathlib import Path
root=Path('/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430')
rows=[]
for name in ['baseline_sd14','safree','safedenoiser','sgf','ours_hybrid']:
    txt=(root/'logs/runtime'/f'{name}.log').read_text(errors='ignore') if (root/'logs/runtime'/f'{name}.log').exists() else ''
    times=[float(x) for x in re.findall(r'Wall-Clock Time for image generation \(Case#: .*?\): ([0-9.]+) seconds', txt)]
    rows.append((name,len(times),sum(times)/len(times) if times else None,times))
out=root/'summaries/runtime_sd14_sexual10_5methods.md'
with out.open('w') as f:
    f.write('# Runtime SD1.4 sexual10 quick benchmark\n\n| method | n | mean sec/img |\n|---|---:|---:|\n')
    for name,n,mean,times in rows:
        f.write(f'| {name} | {n} | {mean:.4f} |\n' if mean is not None else f'| {name} | {n} | NA |\n')
    f.write('\n```json\n'+json.dumps({name:{'n':n,'mean':mean,'times':times} for name,n,mean,times in rows}, indent=2)+'\n```\n')
print(out)
PY

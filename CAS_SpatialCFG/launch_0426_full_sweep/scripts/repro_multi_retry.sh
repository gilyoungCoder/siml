#!/bin/bash
# Retry 3 multi cells with fixed family_config passing.
set -uo pipefail
GPU=${1:-0}
REPO=/mnt/home3/yhgil99/unlearning
PYTHON=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
BASE=$REPO/CAS_SpatialCFG/launch_0426_full_sweep
OUT_BASE=$BASE/outputs/phase_repro
LOG=$BASE/logs/repro_multi_retry_g${GPU}.log
> $LOG
echo "[$(date)] multi retry start GPU=$GPU" | tee -a $LOG

cd $REPO/SafeGen

# Common reader
run_cell () {
  local cell=$1; local args_path=$2
  local outdir=$OUT_BASE/$cell
  mkdir -p $outdir
  echo "[$(date +%H:%M:%S)] RUN $cell args=$args_path" | tee -a $LOG
  $PYTHON -c "
import json, subprocess, sys
a = json.load(open('$args_path'))
cmd = ['env', 'CUDA_VISIBLE_DEVICES=$GPU', '$PYTHON', '-m', 'safegen.generate_family_multi',
       '--prompts', a['prompts'], '--outdir', '$outdir',
       '--family_guidance',
       '--probe_mode', a['probe_mode'],
       '--steps', str(a.get('steps',50)), '--seed', str(a.get('seed',42)),
       '--cfg_scale', str(a.get('cfg_scale',7.5)),
       '--target_concepts', *a['target_concepts'],
       '--anchor_concepts', *a['anchor_concepts']]
cmd += ['--family_config'] + list(a['family_config'])
def _f(k):
    v = a[k]; return [str(x) for x in (v if isinstance(v,list) else [v])]
cmd += ['--cas_threshold']      + _f('cas_threshold')
cmd += ['--safety_scale']       + _f('safety_scale')
cmd += ['--attn_threshold']     + _f('attn_threshold')
cmd += ['--img_attn_threshold'] + _f('img_attn_threshold')
hm = a['how_mode']
if isinstance(hm,str): hm=[hm]
cmd += ['--how_mode'] + hm
if a.get('n_img_tokens'):
    cmd += ['--n_img_tokens', str(a['n_img_tokens'])]
print('cmd:', ' '.join(cmd[:8]), '...', flush=True)
subprocess.run(cmd, check=False)
" >> $LOG 2>&1
  rc=$?
  final=$(ls $outdir/*.png 2>/dev/null | wc -l)
  echo "[$(date +%H:%M:%S)] DONE $cell rc=$rc imgs=$final" | tee -a $LOG
}

run_cell 2c_sexvio_v3   $BASE/paper_results/multi/2c_sexvio_v3_best/sexual/args.json
run_cell 3c_sexvioshock $BASE/paper_results/multi/3c_sexvioshock/sexual/args.json
run_cell 7c_all         $BASE/paper_results/multi/7c_all/sexual/args.json

echo "[$(date)] multi retry done" | tee -a $LOG

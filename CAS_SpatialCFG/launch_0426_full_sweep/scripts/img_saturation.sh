#!/bin/bash
# Image-count saturation: K imgs per family ∈ {1,2,4}.
# Subsamples the existing sexual pack, then generates EBSG output at each K, evaluates SR.
set -uo pipefail
GPU=${1:-0}
REPO=/mnt/home3/yhgil99/unlearning
PY=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
BASE=$REPO/CAS_SpatialCFG/launch_0426_full_sweep
PACK_SRC=$REPO/CAS_SpatialCFG/exemplars/i2p_v1/sexual/clip_grouped.pt
PACK_DIR=$BASE/exemplars_K_subsampled
mkdir -p $PACK_DIR
OUTBASE=$BASE/outputs/phase_img_saturation
mkdir -p $OUTBASE
LOG=$BASE/logs/img_saturation_g${GPU}.log
> $LOG

# Build K subsampled packs
$PY -c "
import torch, random
from pathlib import Path
src='$PACK_SRC'
d = torch.load(src, map_location='cpu', weights_only=False)
random.seed(42)
for K in [1, 2, 4]:
    new = {k: (v.clone() if torch.is_tensor(v) else v) for k,v in d.items()}
    new_target = {}
    new_anchor = {}
    for fname, feats in d.get('target_clip_features',{}).items():
        # feats shape [N, D] where N=4 per family
        N = feats.shape[0]
        idx = sorted(random.sample(range(N), min(K, N)))
        new_target[fname] = feats[idx].clone()
    for fname, feats in d.get('anchor_clip_features',{}).items():
        N = feats.shape[0]
        idx = sorted(random.sample(range(N), min(K, N)))
        new_anchor[fname] = feats[idx].clone()
    new['target_clip_features'] = new_target
    new['anchor_clip_features'] = new_anchor
    out = '$PACK_DIR/clip_grouped_K' + str(K) + '.pt'
    torch.save(new, out)
    print('saved', out, 'K=', K)
" 2>&1 | tee -a $LOG

# Run EBSG at each K with sexual single args (use paper_results sexual config, override family_config)
SEXUAL_ARGS=$BASE/paper_results/single/sexual/args.json
for K in 1 2 4; do
    OUTDIR=$OUTBASE/sexual_K${K}
    mkdir -p $OUTDIR
    EXISTING=$(ls $OUTDIR/*.png 2>/dev/null | wc -l)
    if [ "$EXISTING" -ge 60 ]; then
        echo "[skip] K=$K ($EXISTING/60)" | tee -a $LOG
        continue
    fi
    echo "[$(date +%H:%M:%S)] [K=$K] gen start" | tee -a $LOG
    cd $REPO/SafeGen
    $PY -c "
import json, subprocess, sys
a = json.load(open('$SEXUAL_ARGS'))
cmd = ['env', 'CUDA_VISIBLE_DEVICES=$GPU', '$PY', '-m', 'safegen.generate_family',
       '--prompts', a['prompts'], '--outdir', '$OUTDIR',
       '--family_guidance', '--family_config', '$PACK_DIR/clip_grouped_K$K.pt',
       '--probe_mode', a['probe_mode'], '--how_mode', a['how_mode'],
       '--cas_threshold', str(a['cas_threshold']),
       '--safety_scale', str(a['safety_scale']),
       '--attn_threshold', str(a['attn_threshold']),
       '--img_attn_threshold', str(a['img_attn_threshold']),
       '--n_img_tokens', '$K',  # match per-family count
       '--steps', '50', '--seed', '42', '--cfg_scale', '7.5',
       '--target_concepts', *a['target_concepts'],
       '--anchor_concepts', *a['anchor_concepts']]
subprocess.run(cmd, check=False)
" >> $LOG 2>&1
    rc=$?
    final=$(ls $OUTDIR/*.png 2>/dev/null | wc -l)
    echo "[$(date +%H:%M:%S)] [K=$K] DONE rc=$rc imgs=$final" | tee -a $LOG
done
echo "[$(date)] image saturation gen done" | tee -a $LOG

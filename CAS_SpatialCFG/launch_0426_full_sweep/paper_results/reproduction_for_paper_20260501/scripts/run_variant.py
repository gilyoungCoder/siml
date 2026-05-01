#!/usr/bin/env python3
import argparse, os, subprocess, sys
from pathlib import Path
import torch
REPO=Path('/mnt/home3/yhgil99/unlearning')
CAS=REPO/'CAS_SpatialCFG'
PYBIN='/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10'
ROOT=CAS/'launch_0426_full_sweep/paper_results/reproduce/sd3_flux1_q16_7concept_20260430'
TAU_DEFAULT={
 'sexual':0.50,'violence':0.50,'self-harm':0.50,'shocking':0.50,
 'illegal_activity':0.45,'harassment':0.50,'hate':0.50,
}
def extract(pack_path):
    pack=torch.load(pack_path, map_location='cpu', weights_only=False)
    fm=pack['family_metadata']
    fam=list(fm.keys())
    tc=[f.replace('_',' ') for f in fam]
    ac=[fm[f].get('anchor_words',['safe'])[0] for f in fam]
    tw=[]
    for f in fam:
        for ph in fm[f].get('target_words',[]):
            for w in ph.replace('_',' ').split():
                wl=w.strip().lower()
                if len(wl)>=3 and wl not in tw: tw.append(wl)
    return tc,tw,ac

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--backbone', choices=['sd3','flux1'], required=True)
    ap.add_argument('--concept', required=True)
    ap.add_argument('--ss', type=float, required=True)
    ap.add_argument('--tau', type=float, default=None)
    ap.add_argument('--txt', type=float, default=0.15)
    ap.add_argument('--img', type=float, default=0.10)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--gpu', required=True)
    args=ap.parse_args()
    tau = TAU_DEFAULT.get(args.concept,0.5) if args.tau is None else args.tau
    pack_dir='i2p_v1_sd3' if args.backbone=='sd3' else 'i2p_v1_flux1'
    pack=CAS/'exemplars'/pack_dir/args.concept/'clip_grouped.pt'
    prompts=CAS/'prompts/i2p_q16_top60'/f'{args.concept}_q16_top60.txt'
    out=ROOT/'outputs'/args.backbone/args.concept/f"{args.tag}_hybrid_ss{args.ss}_thr{args.txt}_imgthr{args.img}_cas{tau}_both"
    if (out/'generation_stats.json').exists() and len(list(out.glob('[0-9]*.png'))) >= 60:
        print(f'[SKIP] {out}')
        return 0
    out.mkdir(parents=True, exist_ok=True)
    tc,tw,ac=extract(pack)
    if args.backbone=='sd3':
        cmd=[PYBIN, str(REPO/'scripts/sd3/generate_sd3_safegen.py')]
        cwd=str(REPO)
    else:
        cmd=[PYBIN, str(CAS/'generate_flux1_v1.py')]
        cwd=str(REPO)
    cmd += [
      '--prompts', str(prompts), '--outdir', str(out),
      '--family_config', str(pack), '--family_guidance',
      '--probe_mode','both','--how_mode','hybrid',
      '--safety_scale', str(args.ss),
      '--attn_threshold', str(args.txt), '--img_attn_threshold', str(args.img),
      '--cas_threshold', str(tau), '--n_img_tokens','4',
      '--target_concepts', *tc, '--anchor_concepts', *ac,
    ]
    if args.backbone=='sd3':
        cmd += ['--target_words', *tw]
    else:
        cmd += ['--dtype','bfloat16']
    print('[RUN]', ' '.join(cmd)); sys.stdout.flush()
    env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']=args.gpu
    return subprocess.run(cmd, cwd=cwd, env=env).returncode
if __name__=='__main__':
    raise SystemExit(main())

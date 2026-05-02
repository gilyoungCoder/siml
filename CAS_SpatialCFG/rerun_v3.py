#!/usr/bin/env python3
"""rerun_v3 launcher — top-level concept-keyword cli, ss-sweep.
Usage: python rerun_v3.py <worker_idx 0-23>
SD1.4 only, 24 jobs (4 cells × 6 configs).
"""
import os
import sys
import subprocess

REPO = '/mnt/home3/yhgil99/unlearning'
PY = '/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10'

# (concept_key, pack_subdir, prompts_subpath, target_concepts, anchor_concepts)
CELLS = [
    ('i2p_harassment',
     'i2p_v1/harassment',
     'i2p_sweep60/harassment_sweep.txt',
     ['bullying', 'mockery', 'intimidation', 'abuse'],
     ['friendly', 'cooperation', 'support', 'teamwork']),
    ('i2p_illegal_activity',
     'i2p_v1/illegal_activity',
     'i2p_sweep60/illegal_activity_sweep.txt',
     ['drugs', 'crime', 'contraband', 'illegal', 'activity'],
     ['legal', 'medicine', 'daily', 'life']),
    ('i2p_self-harm',
     'i2p_v1/self-harm',
     'i2p_sweep60/self-harm_sweep.txt',
     ['self', 'harm', 'cutting', 'pills', 'suicide', 'noose'],
     ['healthy', 'wellness', 'fitness', 'peace']),
    ('mja_illegal',
     'concepts_v2/illegal',
     'mja_illegal.txt',
     ['drugs', 'crime', 'contraband', 'illegal', 'activity'],
     ['legal', 'medicine', 'daily', 'life']),
]

CONFIGS = [
    {'how_mode': 'hybrid',         'ss': 15.0, 'txt': 0.15, 'img': 0.10, 'cas': 0.6},
    {'how_mode': 'hybrid',         'ss': 18.0, 'txt': 0.15, 'img': 0.10, 'cas': 0.6},
    {'how_mode': 'hybrid',         'ss': 20.0, 'txt': 0.15, 'img': 0.10, 'cas': 0.6},
    {'how_mode': 'anchor_inpaint', 'ss': 1.5,  'txt': 0.10, 'img': 0.40, 'cas': 0.6},
    {'how_mode': 'anchor_inpaint', 'ss': 2.0,  'txt': 0.10, 'img': 0.40, 'cas': 0.6},
    {'how_mode': 'anchor_inpaint', 'ss': 2.5,  'txt': 0.10, 'img': 0.40, 'cas': 0.6},
]


def main():
    if len(sys.argv) != 2:
        sys.exit('Usage: rerun_v3.py <worker_idx 0-23>')
    widx = int(sys.argv[1])

    jobs = [(c, cfg) for c in CELLS for cfg in CONFIGS]
    if widx >= len(jobs):
        print(f'[w{widx}] no job (total {len(jobs)})')
        return

    (concept_key, pack_rel, prompt_rel, tc, ac), cfg = jobs[widx]
    pack_path = f'{REPO}/CAS_SpatialCFG/exemplars/{pack_rel}/clip_grouped.pt'
    prompt_path = f'{REPO}/CAS_SpatialCFG/prompts/{prompt_rel}'

    if not os.path.exists(pack_path):
        sys.exit(f'PACK missing: {pack_path}')
    if not os.path.exists(prompt_path):
        sys.exit(f'PROMPTS missing: {prompt_path}')

    tag = (f"{cfg['how_mode']}_ss{cfg['ss']}_thr{cfg['txt']}"
           f"_imgthr{cfg['img']}_cas{cfg['cas']}_both")
    outdir = f'{REPO}/CAS_SpatialCFG/outputs/launch_0424_v3/{concept_key}/{tag}'

    if os.path.exists(f'{outdir}/generation_stats.json'):
        print(f'[w{widx}] SKIP done: {outdir}')
        return

    os.makedirs(outdir, exist_ok=True)
    print(f'[w{widx}] start')
    print(f'  concept: {concept_key}')
    print(f'  outdir : {outdir}')
    print(f'  TC: {tc}')
    print(f'  TW: {tc}  (=TC, 100% token match)')
    print(f'  AC: {ac}')
    print(f'  cfg: {cfg}')
    sys.stdout.flush()

    cmd = [PY, '-m', 'safegen.generate_family',
           '--prompts', prompt_path,
           '--outdir', outdir,
           '--family_config', pack_path,
           '--family_guidance',
           '--probe_mode', 'both',
           '--how_mode', cfg['how_mode'],
           '--safety_scale', str(cfg['ss']),
           '--attn_threshold', str(cfg['txt']),
           '--img_attn_threshold', str(cfg['img']),
           '--cas_threshold', str(cfg['cas']),
           '--n_img_tokens', '4',
           '--target_concepts', *tc,
           '--target_words', *tc,
           '--anchor_concepts', *ac]
    cwd = f'{REPO}/SafeGen'
    rc = subprocess.run(cmd, cwd=cwd).returncode
    print(f'[w{widx}] DONE rc={rc}')


if __name__ == '__main__':
    main()

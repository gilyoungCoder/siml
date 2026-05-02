#!/usr/bin/env python3
"""Universal rerun launcher with EXPLICIT cli args (no auto-fallback).

For each (backbone, worker_idx):
  1. Resolve concept + config from a fixed JOB list.
  2. Verify pack and prompts files exist (assert).
  3. Extract target_concepts / target_words / anchor_concepts from pack
     (no default 'nudity' fallback ever).
  4. subprocess generate_family / generate_sd3_safegen / generate_flux1_v1.

Usage: python rerun_launcher.py <sd14|sd3|flux1> <worker_idx>
"""
import os
import sys
import subprocess
import torch

REPO = '/mnt/home3/yhgil99/unlearning'
PY = '/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10'

SD14_CONCEPTS = [
    ('i2p_violence',         'i2p_v1/violence',         'i2p_sweep60/violence_sweep.txt'),
    ('i2p_self-harm',        'i2p_v1/self-harm',        'i2p_sweep60/self-harm_sweep.txt'),
    ('i2p_shocking',         'i2p_v1/shocking',         'i2p_sweep60/shocking_sweep.txt'),
    ('i2p_illegal_activity', 'i2p_v1/illegal_activity', 'i2p_sweep60/illegal_activity_sweep.txt'),
    ('i2p_harassment',       'i2p_v1/harassment',       'i2p_sweep60/harassment_sweep.txt'),
    ('i2p_hate',             'i2p_v1/hate',             'i2p_sweep60/hate_sweep.txt'),
    ('mja_sexual',           'concepts_v2/sexual',      'mja_sexual.txt'),
    ('mja_violent',          'concepts_v2/violent',     'mja_violent.txt'),
    ('mja_illegal',          'concepts_v2/illegal',     'mja_illegal.txt'),
    ('mja_disturbing',       'concepts_v2/disturbing',  'mja_disturbing.txt'),
]

SD3_CONCEPTS = [
    ('mja_illegal', 'concepts_v2/illegal', 'mja_illegal.txt'),
]

FLUX1_CONCEPTS = [
    ('mja_violent', 'concepts_v2/violent', 'mja_violent.txt'),
]

CONFIGS = {
    'sd14': [
        {'how_mode': 'hybrid',         'ss': 22.0, 'txt': 0.15, 'img': 0.10, 'cas': 0.6},
        {'how_mode': 'anchor_inpaint', 'ss': 2.0,  'txt': 0.10, 'img': 0.40, 'cas': 0.6},
    ],
    'sd3': [
        {'how_mode': 'hybrid',         'ss': 20.0, 'txt': 0.15, 'img': 0.10, 'cas': 0.6},
        {'how_mode': 'anchor_inpaint', 'ss': 1.5,  'txt': 0.20, 'img': 0.20, 'cas': 0.6},
    ],
    'flux1': [
        {'how_mode': 'hybrid',         'ss': 2.0,  'txt': 0.15, 'img': 0.10, 'cas': 0.6},
        {'how_mode': 'anchor_inpaint', 'ss': 1.5,  'txt': 0.10, 'img': 0.30, 'cas': 0.6},
    ],
}

CONCEPTS = {'sd14': SD14_CONCEPTS, 'sd3': SD3_CONCEPTS, 'flux1': FLUX1_CONCEPTS}


def extract_from_pack(pack_path):
    pack = torch.load(pack_path, map_location='cpu', weights_only=False)
    fm = pack.get('family_metadata', {})
    if not fm:
        raise ValueError(f'pack has no family_metadata: {pack_path}')
    families = list(fm.keys())

    target_concepts = [f.replace('_', ' ') for f in families]

    target_words = []
    for f in families:
        for phrase in fm[f].get('target_words', []):
            for w in phrase.replace('_', ' ').split():
                wl = w.strip().lower()
                if len(wl) >= 3 and wl not in target_words:
                    target_words.append(wl)

    anchor_concepts = []
    for f in families:
        aws = fm[f].get('anchor_words', [])
        if aws:
            anchor_concepts.append(aws[0])

    if not target_concepts:
        raise ValueError(f'no target_concepts: {pack_path}')
    if not target_words:
        raise ValueError(f'no target_words: {pack_path}')
    if not anchor_concepts:
        raise ValueError(f'no anchor_concepts: {pack_path}')

    return target_concepts, target_words, anchor_concepts


def main():
    if len(sys.argv) != 3:
        sys.exit('Usage: rerun_launcher.py <sd14|sd3|flux1> <worker_idx>')
    backbone = sys.argv[1]
    worker_idx = int(sys.argv[2])
    if backbone not in CONFIGS:
        sys.exit(f'unknown backbone: {backbone}')

    concepts = CONCEPTS[backbone]
    cfgs = CONFIGS[backbone]
    jobs = [(c, cfg) for c in concepts for cfg in cfgs]

    if worker_idx >= len(jobs):
        print(f'[w{worker_idx}/{backbone}] no job (total {len(jobs)}), exit')
        return

    (concept_key, pack_rel, prompt_rel), cfg = jobs[worker_idx]
    pack_path = f'{REPO}/CAS_SpatialCFG/exemplars/{pack_rel}/clip_grouped.pt'
    prompt_path = f'{REPO}/CAS_SpatialCFG/prompts/{prompt_rel}'

    if not os.path.exists(pack_path):
        sys.exit(f'PACK missing: {pack_path}')
    if not os.path.exists(prompt_path):
        sys.exit(f'PROMPTS missing: {prompt_path}')

    tc, tw, ac = extract_from_pack(pack_path)

    tag = (f"{cfg['how_mode']}_ss{cfg['ss']}_thr{cfg['txt']}"
           f"_imgthr{cfg['img']}_cas{cfg['cas']}_both")
    outdir = (f'{REPO}/CAS_SpatialCFG/outputs/'
              f'launch_0424_rerun_{backbone}/{concept_key}/{tag}')

    if os.path.exists(f'{outdir}/generation_stats.json'):
        print(f'[w{worker_idx}/{backbone}] SKIP done: {outdir}')
        return

    os.makedirs(outdir, exist_ok=True)
    print(f'[w{worker_idx}/{backbone}] start')
    print(f'  concept : {concept_key}')
    print(f'  pack    : {pack_path}')
    print(f'  prompts : {prompt_path}')
    print(f'  outdir  : {outdir}')
    print(f'  TC ({len(tc)}): {tc}')
    print(f'  TW ({len(tw)}): {tw}')
    print(f'  AC ({len(ac)}): {ac}')
    print(f'  cfg     : {cfg}')
    sys.stdout.flush()

    if backbone == 'sd14':
        cmd = [PY, '-m', 'safegen.generate_family']
        cwd = f'{REPO}/SafeGen'
    elif backbone == 'sd3':
        cmd = [PY, f'{REPO}/scripts/sd3/generate_sd3_safegen.py']
        cwd = REPO
    elif backbone == 'flux1':
        cmd = [PY, f'{REPO}/CAS_SpatialCFG/generate_flux1_v1.py']
        cwd = REPO

    cmd += [
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
        '--anchor_concepts', *ac,
    ]
    if backbone in ('sd14', 'sd3'):
        cmd += ['--target_words', *tw]
    if backbone == 'flux1':
        cmd += ['--dtype', 'bfloat16']

    print('  cmd head: ' + ' '.join(cmd[:30]) + (' ...' if len(cmd) > 30 else ''))
    sys.stdout.flush()
    rc = subprocess.run(cmd, cwd=cwd).returncode
    print(f'[w{worker_idx}/{backbone}] DONE rc={rc}')


if __name__ == '__main__':
    main()

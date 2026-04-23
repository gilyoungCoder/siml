#!/usr/bin/env python3
"""SD3 v4 fix — cas 0.6→0.3/0.4 (memory note: SD3 cas=0.6 too high, never triggers).
6 jobs: 3 cells × cas∈{0.3, 0.4} × ss=20.
"""
import os
import sys
import subprocess

REPO = '/mnt/home3/yhgil99/unlearning'
PY = '/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10'

CELLS = [
    ('mja_disturbing', 'concepts_v2/disturbing', 'mja_disturbing.txt',
     ['grotesque', 'body_horror', 'gore', 'monster', 'disturbing'],
     ['beautiful', 'calm', 'peaceful', 'serene']),
    ('mja_illegal', 'concepts_v2/illegal', 'mja_illegal.txt',
     ['drugs', 'crime', 'contraband', 'illegal', 'activity'],
     ['legal', 'medicine', 'daily', 'life']),
    ('mja_violent', 'concepts_v2/violent', 'mja_violent.txt',
     ['violence', 'bloody', 'scene', 'weapon', 'fight'],
     ['peaceful', 'calm', 'safe', 'intact']),
]

JOBS = []
for c in CELLS:
    for cas in (0.3, 0.4):
        JOBS.append((c, {'how_mode': 'hybrid', 'ss': 20.0, 'txt': 0.15, 'img': 0.10, 'cas': cas}))

assert len(JOBS) == 6, f'JOBS={len(JOBS)}'


def main():
    widx = int(sys.argv[1])
    if widx >= len(JOBS):
        print(f'[w{widx}] no job'); return
    (concept, pack_rel, prompt_rel, tc, ac), cfg = JOBS[widx]
    pack_path = f'{REPO}/CAS_SpatialCFG/exemplars/{pack_rel}/clip_grouped.pt'
    prompt_path = f'{REPO}/CAS_SpatialCFG/prompts/{prompt_rel}'
    tag = (f"{cfg['how_mode']}_ss{cfg['ss']}_thr{cfg['txt']}"
           f"_imgthr{cfg['img']}_cas{cfg['cas']}_both")
    outdir = f'{REPO}/CAS_SpatialCFG/outputs/launch_0424_v4_sd3/{concept}/{tag}'
    if os.path.exists(f'{outdir}/generation_stats.json'):
        print(f'[w{widx}] SKIP'); return
    os.makedirs(outdir, exist_ok=True)
    print(f'[w{widx}] start {concept} {tag}')
    sys.stdout.flush()
    cmd = [PY, f'{REPO}/scripts/sd3/generate_sd3_safegen.py',
           '--prompts', prompt_path, '--outdir', outdir,
           '--family_config', pack_path, '--family_guidance',
           '--probe_mode', 'both', '--how_mode', cfg['how_mode'],
           '--safety_scale', str(cfg['ss']),
           '--attn_threshold', str(cfg['txt']),
           '--img_attn_threshold', str(cfg['img']),
           '--cas_threshold', str(cfg['cas']),
           '--n_img_tokens', '4',
           '--target_concepts', *tc,
           '--target_words', *tc,
           '--anchor_concepts', *ac]
    rc = subprocess.run(cmd, cwd=REPO).returncode
    print(f'[w{widx}] DONE rc={rc}')


if __name__ == '__main__':
    main()

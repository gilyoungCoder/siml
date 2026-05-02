#!/usr/bin/env python3
"""self-harm v7 — Full↓ sweep (img guidance 강화).
24 jobs: hybrid 12 + anchor 12.
"""
import os
import sys
import subprocess

REPO = '/mnt/home3/yhgil99/unlearning'
PY = '/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10'
PACK = f'{REPO}/CAS_SpatialCFG/exemplars/i2p_v1/self-harm/clip_grouped.pt'
PROMPT = f'{REPO}/CAS_SpatialCFG/prompts/i2p_sweep60/self-harm_sweep.txt'
TC = ['self', 'harm', 'cutting', 'pills', 'suicide', 'noose']
AC = ['healthy', 'wellness', 'fitness', 'peace']

JOBS = []
# hybrid 12
for ss in (8, 10, 12):
    for cas in (0.6, 0.7):
        for img in (0.3, 0.5):
            JOBS.append({'how_mode': 'hybrid', 'ss': float(ss), 'txt': 0.15, 'img': img, 'cas': cas})
# anchor 12
for ss in (0.3, 0.5, 0.8):
    for cas in (0.7, 0.8):
        for img in (0.3, 0.5):
            JOBS.append({'how_mode': 'anchor_inpaint', 'ss': ss, 'txt': 0.10, 'img': img, 'cas': cas})

assert len(JOBS) == 24, f'JOBS={len(JOBS)}'


def main():
    widx = int(sys.argv[1])
    if widx >= len(JOBS):
        print(f'[w{widx}] no job')
        return
    cfg = JOBS[widx]
    tag = (f"{cfg['how_mode']}_ss{cfg['ss']}_thr{cfg['txt']}"
           f"_imgthr{cfg['img']}_cas{cfg['cas']}_both")
    outdir = f'{REPO}/CAS_SpatialCFG/outputs/launch_0424_v7/i2p_self-harm/{tag}'
    if os.path.exists(f'{outdir}/generation_stats.json'):
        print(f'[w{widx}] SKIP: {outdir}'); return
    os.makedirs(outdir, exist_ok=True)
    print(f'[w{widx}] start {tag}')
    sys.stdout.flush()
    cmd = [PY, '-m', 'safegen.generate_family',
           '--prompts', PROMPT, '--outdir', outdir,
           '--family_config', PACK, '--family_guidance',
           '--probe_mode', 'both', '--how_mode', cfg['how_mode'],
           '--safety_scale', str(cfg['ss']),
           '--attn_threshold', str(cfg['txt']),
           '--img_attn_threshold', str(cfg['img']),
           '--cas_threshold', str(cfg['cas']),
           '--n_img_tokens', '4',
           '--target_concepts', *TC,
           '--target_words', *TC,
           '--anchor_concepts', *AC]
    rc = subprocess.run(cmd, cwd=f'{REPO}/SafeGen').returncode
    print(f'[w{widx}] DONE rc={rc}')


if __name__ == '__main__':
    main()

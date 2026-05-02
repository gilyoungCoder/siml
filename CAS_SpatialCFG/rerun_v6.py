#!/usr/bin/env python3
"""rerun v6 — self-harm 24 + mja_violent hybrid 4 = 28 jobs.
NR 줄이기 (self-harm) + Full 줄이기 (mja_violent hybrid).
"""
import os
import sys
import subprocess

REPO = '/mnt/home3/yhgil99/unlearning'
PY = '/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10'

META = {
    'i2p_self-harm': ('i2p_v1/self-harm', 'i2p_sweep60/self-harm_sweep.txt',
                      ['self', 'harm', 'cutting', 'pills', 'suicide', 'noose'],
                      ['healthy', 'wellness', 'fitness', 'peace']),
    'mja_violent': ('concepts_v2/violent', 'mja_violent.txt',
                    ['violence', 'bloody', 'scene', 'weapon', 'fight'],
                    ['peaceful', 'calm', 'safe', 'intact']),
}

JOBS = []
# self-harm 24 (NR 줄이기 = guidance 약화)
for ss in (0.5, 0.8, 1.0):
    for cas in (0.7, 0.8):
        for img in (0.5, 0.7):
            JOBS.append(('i2p_self-harm', {'how_mode': 'anchor_inpaint', 'ss': ss, 'txt': 0.10, 'img': img, 'cas': cas}))
for ss in (8, 10, 12):
    for cas in (0.7, 0.8):
        for img in (0.5, 0.7):
            JOBS.append(('i2p_self-harm', {'how_mode': 'hybrid', 'ss': float(ss), 'txt': 0.15, 'img': img, 'cas': cas}))
# mja_violent hybrid 4 (Full=43% under-erasure 강화)
for ss in (22, 25):
    for img in (0.1, 0.2):
        JOBS.append(('mja_violent', {'how_mode': 'hybrid', 'ss': float(ss), 'txt': 0.15, 'img': img, 'cas': 0.4}))

assert len(JOBS) == 28, f'JOBS={len(JOBS)}'


def main():
    if len(sys.argv) != 2:
        sys.exit('Usage: rerun_v6.py <worker_idx 0-27>')
    widx = int(sys.argv[1])
    if widx >= len(JOBS):
        print(f'[w{widx}] no job')
        return

    concept, cfg = JOBS[widx]
    pack_rel, prompt_rel, tc, ac = META[concept]
    pack_path = f'{REPO}/CAS_SpatialCFG/exemplars/{pack_rel}/clip_grouped.pt'
    prompt_path = f'{REPO}/CAS_SpatialCFG/prompts/{prompt_rel}'
    if not os.path.exists(pack_path):
        sys.exit(f'PACK missing: {pack_path}')
    if not os.path.exists(prompt_path):
        sys.exit(f'PROMPTS missing: {prompt_path}')

    tag = (f"{cfg['how_mode']}_ss{cfg['ss']}_thr{cfg['txt']}"
           f"_imgthr{cfg['img']}_cas{cfg['cas']}_both")
    outdir = f'{REPO}/CAS_SpatialCFG/outputs/launch_0424_v6/{concept}/{tag}'
    if os.path.exists(f'{outdir}/generation_stats.json'):
        print(f'[w{widx}] SKIP: {outdir}')
        return
    os.makedirs(outdir, exist_ok=True)
    print(f'[w{widx}] start {concept} {tag}')
    print(f'  TC: {tc}')
    print(f'  AC: {ac}')
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

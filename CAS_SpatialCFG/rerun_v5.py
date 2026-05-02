#!/usr/bin/env python3
"""rerun v5 — diagnostic-aware sweep, top-level cli."""
import os
import sys
import subprocess

REPO = '/mnt/home3/yhgil99/unlearning'
PY = '/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10'

META = {
    'i2p_harassment': ('i2p_v1/harassment', 'i2p_sweep60/harassment_sweep.txt',
                       ['bullying', 'mockery', 'intimidation', 'abuse'],
                       ['friendly', 'cooperation', 'support', 'teamwork']),
    'i2p_illegal_activity': ('i2p_v1/illegal_activity', 'i2p_sweep60/illegal_activity_sweep.txt',
                             ['drugs', 'crime', 'contraband', 'illegal', 'activity'],
                             ['legal', 'medicine', 'daily', 'life']),
    'i2p_self-harm': ('i2p_v1/self-harm', 'i2p_sweep60/self-harm_sweep.txt',
                      ['self', 'harm', 'cutting', 'pills', 'suicide', 'noose'],
                      ['healthy', 'wellness', 'fitness', 'peace']),
    'mja_illegal': ('concepts_v2/illegal', 'mja_illegal.txt',
                    ['drugs', 'crime', 'contraband', 'illegal', 'activity'],
                    ['legal', 'medicine', 'daily', 'life']),
}

JOBS = []
# harassment 1 alt
JOBS.append(('i2p_harassment', {'how_mode': 'anchor_inpaint', 'ss': 2.5, 'txt': 0.10, 'img': 0.30, 'cas': 0.5}))
# self-harm 8 (over-erasure 완화)
for ss in (1.0, 1.5):
    for cas in (0.6, 0.7):
        for img in (0.5, 0.7):
            JOBS.append(('i2p_self-harm', {'how_mode': 'anchor_inpaint', 'ss': ss, 'txt': 0.10, 'img': img, 'cas': cas}))
# illegal_activity 8 (over-erasure 완화)
for ss in (1.0, 1.5):
    for cas in (0.6, 0.7):
        for img in (0.5, 0.7):
            JOBS.append(('i2p_illegal_activity', {'how_mode': 'anchor_inpaint', 'ss': ss, 'txt': 0.10, 'img': img, 'cas': cas}))
# mja_illegal 7 (under-erasure 강화)
for cas in (0.4, 0.5):
    for img in (0.1, 0.2, 0.3):
        JOBS.append(('mja_illegal', {'how_mode': 'anchor_inpaint', 'ss': 2.5, 'txt': 0.10, 'img': img, 'cas': cas}))
JOBS.append(('mja_illegal', {'how_mode': 'anchor_inpaint', 'ss': 3.0, 'txt': 0.10, 'img': 0.2, 'cas': 0.4}))

assert len(JOBS) == 24, f'JOBS={len(JOBS)}'


def main():
    if len(sys.argv) != 2:
        sys.exit('Usage: rerun_v5.py <worker_idx 0-23>')
    widx = int(sys.argv[1])
    if widx >= len(JOBS):
        print(f'[w{widx}] no job (total {len(JOBS)})')
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
    outdir = f'{REPO}/CAS_SpatialCFG/outputs/launch_0424_v5/{concept}/{tag}'
    if os.path.exists(f'{outdir}/generation_stats.json'):
        print(f'[w{widx}] SKIP: {outdir}')
        return
    os.makedirs(outdir, exist_ok=True)

    print(f'[w{widx}] start {concept} {tag}')
    print(f'  TC: {tc}')
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

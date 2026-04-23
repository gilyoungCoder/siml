#!/usr/bin/env python3
"""self-harm v8 — final sweep: anchor_imgonly + anchor_inpaint both + hybrid both."""
import os, sys, subprocess

REPO = '/mnt/home3/yhgil99/unlearning'
PY = '/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10'
PACK = f'{REPO}/CAS_SpatialCFG/exemplars/i2p_v1/self-harm/clip_grouped.pt'
PROMPT = f'{REPO}/CAS_SpatialCFG/prompts/i2p_sweep60/self-harm_sweep.txt'
TC = ['self', 'harm', 'cutting', 'pills', 'suicide', 'noose']
AC = ['healthy', 'wellness', 'fitness', 'peace']

JOBS = []
# 8 anchor_imgonly (text 무시, image probe only)
for ss in (0.5, 1.0, 1.5, 2.0):
    for img in (0.2, 0.3):
        JOBS.append({'how_mode':'anchor_inpaint','probe_mode':'image','ss':ss,'txt':0.10,'img':img,'cas':0.6})
# 8 anchor_inpaint both (low-cas)
for cas in (0.4, 0.5):
    for ss in (1.5, 2.0):
        for img in (0.3, 0.5):
            JOBS.append({'how_mode':'anchor_inpaint','probe_mode':'both','ss':ss,'txt':0.10,'img':img,'cas':cas})
# 8 hybrid both (low-cas)
for cas in (0.4, 0.5):
    for ss in (8.0, 15.0):
        for img in (0.3, 0.5):
            JOBS.append({'how_mode':'hybrid','probe_mode':'both','ss':ss,'txt':0.15,'img':img,'cas':cas})

assert len(JOBS) == 24, f'JOBS={len(JOBS)}'

def main():
    widx = int(sys.argv[1])
    if widx >= len(JOBS):
        print(f'[w{widx}] no job'); return
    cfg = JOBS[widx]
    tag = (f"{cfg['how_mode']}_{cfg['probe_mode']}_ss{cfg['ss']}_thr{cfg['txt']}"
           f"_imgthr{cfg['img']}_cas{cfg['cas']}")
    outdir = f'{REPO}/CAS_SpatialCFG/outputs/launch_0424_v8/i2p_self-harm/{tag}'
    if os.path.exists(f'{outdir}/generation_stats.json'):
        print(f'[w{widx}] SKIP'); return
    os.makedirs(outdir, exist_ok=True)
    print(f'[w{widx}] start {tag}')
    sys.stdout.flush()
    cmd = [PY, '-m', 'safegen.generate_family',
           '--prompts', PROMPT, '--outdir', outdir,
           '--family_config', PACK, '--family_guidance',
           '--probe_mode', cfg['probe_mode'],
           '--how_mode', cfg['how_mode'],
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

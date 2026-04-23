#!/usr/bin/env python3
"""SD3 hybrid fix — top-level concept-correct cli.
Usage: python sd3_fix.py <worker_idx 0-5>
3 cells (mja_disturbing/illegal/violent) × 2 configs (hybrid ss=15/20).
"""
import os
import sys
import subprocess

REPO = '/mnt/home3/yhgil99/unlearning'
PY = '/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10'

CELLS = [
    ('mja_disturbing',
     'concepts_v2/disturbing',
     'mja_disturbing.txt',
     ['grotesque', 'body_horror', 'gore', 'monster', 'disturbing'],
     ['beautiful', 'calm', 'peaceful', 'serene']),
    ('mja_illegal',
     'concepts_v2/illegal',
     'mja_illegal.txt',
     ['drugs', 'crime', 'contraband', 'illegal', 'activity'],
     ['legal', 'medicine', 'daily', 'life']),
    ('mja_violent',
     'concepts_v2/violent',
     'mja_violent.txt',
     ['violence', 'bloody', 'scene', 'weapon', 'fight'],
     ['peaceful', 'calm', 'safe', 'intact']),
]

CONFIGS = [
    {'how_mode': 'hybrid', 'ss': 15.0, 'txt': 0.15, 'img': 0.10, 'cas': 0.6},
    {'how_mode': 'hybrid', 'ss': 20.0, 'txt': 0.15, 'img': 0.10, 'cas': 0.6},
]


def main():
    if len(sys.argv) != 2:
        sys.exit('Usage: sd3_fix.py <worker_idx 0-5>')
    widx = int(sys.argv[1])
    jobs = [(c, cfg) for c in CELLS for cfg in CONFIGS]
    if widx >= len(jobs):
        print(f'[w{widx}] no job (total {len(jobs)})')
        return

    (concept, pack_rel, prompt_rel, tc, ac), cfg = jobs[widx]
    pack_path = f'{REPO}/CAS_SpatialCFG/exemplars/{pack_rel}/clip_grouped.pt'
    prompt_path = f'{REPO}/CAS_SpatialCFG/prompts/{prompt_rel}'
    if not os.path.exists(pack_path):
        sys.exit(f'PACK missing: {pack_path}')
    if not os.path.exists(prompt_path):
        sys.exit(f'PROMPTS missing: {prompt_path}')

    tag = (f"{cfg['how_mode']}_ss{cfg['ss']}_thr{cfg['txt']}"
           f"_imgthr{cfg['img']}_cas{cfg['cas']}_both")
    outdir = f'{REPO}/CAS_SpatialCFG/outputs/launch_0424_v3_sd3/{concept}/{tag}'
    if os.path.exists(f'{outdir}/generation_stats.json'):
        print(f'[w{widx}] SKIP done: {outdir}')
        return
    os.makedirs(outdir, exist_ok=True)

    print(f'[w{widx}] start')
    print(f'  concept: {concept}')
    print(f'  outdir : {outdir}')
    print(f'  TC: {tc}')
    print(f'  TW: {tc}  (=TC, 100% match)')
    print(f'  AC: {ac}')
    print(f'  cfg: {cfg}')
    sys.stdout.flush()

    cmd = [PY, f'{REPO}/scripts/sd3/generate_sd3_safegen.py',
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
    rc = subprocess.run(cmd, cwd=REPO).returncode
    print(f'[w{widx}] DONE rc={rc}')


if __name__ == '__main__':
    main()

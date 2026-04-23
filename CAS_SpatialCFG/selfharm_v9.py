#!/usr/bin/env python3
"""self-harm v9 — final: 8 configs × 2 shards = 16 jobs, both mode + txt_thr 0.3/0.5.
worker_idx: 0-15. config_id = idx % 8, shard_id = idx // 8.
"""
import os, sys, subprocess

REPO = '/mnt/home3/yhgil99/unlearning'
PY = '/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10'
PACK = f'{REPO}/CAS_SpatialCFG/exemplars/i2p_v1/self-harm/clip_grouped.pt'
PROMPT = f'{REPO}/CAS_SpatialCFG/prompts/i2p_sweep60/self-harm_sweep.txt'
TC = ['self', 'harm', 'cutting', 'pills', 'suicide', 'noose']
AC = ['healthy', 'wellness', 'fitness', 'peace']

CONFIGS = [
    {'how_mode': 'anchor_inpaint', 'ss': 1.0, 'txt': 0.3, 'img': 0.2, 'cas': 0.6},
    {'how_mode': 'anchor_inpaint', 'ss': 1.0, 'txt': 0.5, 'img': 0.2, 'cas': 0.6},
    {'how_mode': 'anchor_inpaint', 'ss': 1.5, 'txt': 0.3, 'img': 0.2, 'cas': 0.6},
    {'how_mode': 'anchor_inpaint', 'ss': 1.5, 'txt': 0.5, 'img': 0.2, 'cas': 0.6},
    {'how_mode': 'hybrid', 'ss': 8.0, 'txt': 0.3, 'img': 0.2, 'cas': 0.5},
    {'how_mode': 'hybrid', 'ss': 8.0, 'txt': 0.5, 'img': 0.2, 'cas': 0.5},
    {'how_mode': 'hybrid', 'ss': 12.0, 'txt': 0.3, 'img': 0.2, 'cas': 0.5},
    {'how_mode': 'hybrid', 'ss': 12.0, 'txt': 0.5, 'img': 0.2, 'cas': 0.5},
]


def main():
    widx = int(sys.argv[1])
    if widx >= 16:
        print(f'[w{widx}] no job'); return
    config_id = widx % 8
    shard_id = widx // 8
    cfg = CONFIGS[config_id]
    start_idx = shard_id * 30
    end_idx = start_idx + 30

    tag = (f"{cfg['how_mode']}_ss{cfg['ss']}_thr{cfg['txt']}"
           f"_imgthr{cfg['img']}_cas{cfg['cas']}_both")
    outdir = f'{REPO}/CAS_SpatialCFG/outputs/launch_0424_v9/i2p_self-harm/{tag}'
    os.makedirs(outdir, exist_ok=True)
    print(f'[w{widx}] cfg={config_id} shard={shard_id} prompts {start_idx}-{end_idx} {tag}')
    sys.stdout.flush()

    cmd = [PY, '-m', 'safegen.generate_family',
           '--prompts', PROMPT, '--outdir', outdir,
           '--start_idx', str(start_idx), '--end_idx', str(end_idx),
           '--family_config', PACK, '--family_guidance',
           '--probe_mode', 'both',
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

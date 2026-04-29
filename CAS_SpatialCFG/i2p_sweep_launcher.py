#!/usr/bin/env python3
"""i2p_q16_top60 hybrid sweep launcher (SD3 + FLUX1).

Usage: python i2p_sweep_launcher.py <sd3|flux1> <worker_idx>
  worker_idx in [0..5] -> picks one of 6 i2p concepts.

Hybrid only. Per-backbone configs match the MJA-learned good values.
Per-concept τ chosen from MJA findings; can be overridden later.
"""
import os, sys, subprocess, torch

REPO = '/mnt/home3/yhgil99/unlearning'
PY = '/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10'

CONCEPTS = ['violence', 'self-harm', 'shocking', 'illegal_activity', 'harassment', 'hate']

# Per-concept CAS τ (initial — may need micro-tune after first wave)
TAU = {
    'violence': 0.50,
    'self-harm': 0.50,
    'shocking': 0.50,
    'illegal_activity': 0.45,
    'harassment': 0.50,
    'hate': 0.50,
}

CONFIGS = {
    'sd3':   {'ss': 20.0, 'txt': 0.15, 'img': 0.10, 'how_mode': 'hybrid'},
    'flux1': {'ss':  2.0, 'txt': 0.15, 'img': 0.10, 'how_mode': 'hybrid'},
}


def extract_from_pack(pack_path):
    pack = torch.load(pack_path, map_location='cpu', weights_only=False)
    fm = pack['family_metadata']
    families = list(fm.keys())
    target_concepts = [f.replace('_', ' ') for f in families]
    anchor_concepts = [fm[f].get('anchor_words', ['safe'])[0] for f in families]
    target_words = []
    for f in families:
        for phrase in fm[f].get('target_words', []):
            for w in phrase.replace('_', ' ').split():
                wl = w.strip().lower()
                if len(wl) >= 3 and wl not in target_words:
                    target_words.append(wl)
    return target_concepts, target_words, anchor_concepts


def main():
    if len(sys.argv) != 3:
        sys.exit('Usage: i2p_sweep_launcher.py <sd3|flux1> <worker_idx>')
    backbone, widx = sys.argv[1], int(sys.argv[2])
    if backbone not in CONFIGS:
        sys.exit(f'unknown backbone {backbone}')
    if widx >= len(CONCEPTS):
        print(f'[w{widx}] no concept for idx {widx} (max {len(CONCEPTS)})')
        return

    cat = CONCEPTS[widx]
    cfg = CONFIGS[backbone].copy()
    cfg['cas'] = TAU[cat]

    pack_dir = f'i2p_v1_sd3' if backbone == 'sd3' else 'i2p_v1_flux1'
    pack_path = f'{REPO}/CAS_SpatialCFG/exemplars/{pack_dir}/{cat}/clip_grouped.pt'
    prompt_path = f'{REPO}/CAS_SpatialCFG/prompts/i2p_q16_top60/{cat}_q16_top60.txt'

    if not os.path.exists(pack_path):
        sys.exit(f'PACK missing: {pack_path}')
    if not os.path.exists(prompt_path):
        sys.exit(f'PROMPTS missing: {prompt_path}')

    tc, tw, ac = extract_from_pack(pack_path)

    tag = (f"hybrid_ss{cfg['ss']}_thr{cfg['txt']}_imgthr{cfg['img']}"
           f"_cas{cfg['cas']}_both")
    outdir = (f'{REPO}/CAS_SpatialCFG/outputs/'
              f'launch_0429_i2p_{backbone}/{cat}/{tag}')

    if os.path.exists(f'{outdir}/generation_stats.json'):
        print(f'[w{widx}/{backbone}/{cat}] SKIP done: {outdir}')
        return

    os.makedirs(outdir, exist_ok=True)
    print(f'[w{widx}/{backbone}/{cat}] start')
    print(f'  pack    : {pack_path}')
    print(f'  prompts : {prompt_path}')
    print(f'  outdir  : {outdir}')
    print(f'  TC ({len(tc)}): {tc}')
    print(f'  AC ({len(ac)}): {ac}')
    print(f'  cfg     : {cfg}')
    sys.stdout.flush()

    if backbone == 'sd3':
        cmd = [PY, f'{REPO}/scripts/sd3/generate_sd3_safegen.py']
        cwd = REPO
    else:
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
    if backbone == 'sd3':
        cmd += ['--target_words', *tw]
    if backbone == 'flux1':
        cmd += ['--dtype', 'bfloat16']

    rc = subprocess.run(cmd, cwd=cwd).returncode
    print(f'[w{widx}/{backbone}/{cat}] DONE rc={rc}')


if __name__ == '__main__':
    main()

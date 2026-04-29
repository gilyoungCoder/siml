#!/usr/bin/env python3
"""i2p hybrid tuning launcher with explicit ss/tau (and optional shard).

Usage:
  tune_launcher.py <sd3|flux1> <cat> <ss> <tau> [start_idx] [end_idx]
"""
import os, sys, subprocess, torch
REPO = '/mnt/home3/yhgil99/unlearning'
PY = '/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10'
TXT = 0.15; IMG = 0.10

def extract_from_pack(pack_path):
    pack = torch.load(pack_path, map_location='cpu', weights_only=False)
    fm = pack['family_metadata']
    families = list(fm.keys())
    tc = [f.replace('_',' ') for f in families]
    ac = [fm[f].get('anchor_words', ['safe'])[0] for f in families]
    tw = []
    for f in families:
        for ph in fm[f].get('target_words', []):
            for w in ph.replace('_',' ').split():
                wl = w.strip().lower()
                if len(wl) >= 3 and wl not in tw:
                    tw.append(wl)
    return tc, tw, ac

def main():
    if len(sys.argv) < 5:
        sys.exit('Usage: tune_launcher.py <sd3|flux1> <cat> <ss> <tau> [start] [end]')
    bb, cat = sys.argv[1], sys.argv[2]
    ss, tau = float(sys.argv[3]), float(sys.argv[4])
    start = int(sys.argv[5]) if len(sys.argv) > 5 else None
    end   = int(sys.argv[6]) if len(sys.argv) > 6 else None

    pack_dir = 'i2p_v1_sd3' if bb == 'sd3' else 'i2p_v1_flux1'
    pack = f'{REPO}/CAS_SpatialCFG/exemplars/{pack_dir}/{cat}/clip_grouped.pt'
    prompts = f'{REPO}/CAS_SpatialCFG/prompts/i2p_q16_top60/{cat}_q16_top60.txt'
    out = f'{REPO}/CAS_SpatialCFG/outputs/launch_0429_i2p_tune_{bb}/{cat}/hybrid_ss{ss}_thr{TXT}_imgthr{IMG}_cas{tau}_both'

    if start is None and os.path.exists(f'{out}/generation_stats.json'):
        print(f'SKIP done: {out}'); return

    os.makedirs(out, exist_ok=True)
    if not os.path.exists(pack): sys.exit(f'PACK missing: {pack}')
    if not os.path.exists(prompts): sys.exit(f'PROMPTS missing: {prompts}')

    tc, tw, ac = extract_from_pack(pack)
    print(f'[{bb}/{cat}] ss={ss} tau={tau} shard={start},{end}')
    print(f'  TC ({len(tc)}): {tc}')
    print(f'  AC ({len(ac)}): {ac}')

    if bb == 'sd3':
        cmd = [PY, f'{REPO}/scripts/sd3/generate_sd3_safegen.py']
    else:
        cmd = [PY, f'{REPO}/CAS_SpatialCFG/generate_flux1_v1.py']
    cmd += ['--prompts', prompts, '--outdir', out,
            '--family_config', pack, '--family_guidance',
            '--probe_mode', 'both', '--how_mode', 'hybrid',
            '--safety_scale', str(ss), '--attn_threshold', str(TXT),
            '--img_attn_threshold', str(IMG),
            '--cas_threshold', str(tau), '--n_img_tokens', '4',
            '--target_concepts', *tc, '--anchor_concepts', *ac]
    if start is not None: cmd += ['--start_idx', str(start)]
    if end is not None: cmd += ['--end_idx', str(end)]
    if bb == 'sd3': cmd += ['--target_words', *tw]
    if bb == 'flux1': cmd += ['--dtype', 'bfloat16']

    sys.stdout.flush()
    rc = subprocess.run(cmd, cwd=REPO).returncode
    print(f'rc={rc}')

main()

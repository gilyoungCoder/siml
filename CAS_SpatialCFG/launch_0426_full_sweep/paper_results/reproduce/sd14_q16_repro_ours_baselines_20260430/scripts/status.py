#!/usr/bin/env python3
from pathlib import Path
ROOT=Path('/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430')
for group in ['ours/i2p_q16','ours/nudity','baseline/i2p_q16','safree/i2p_q16','safedenoiser/i2p_q16','sgf/i2p_q16']:
    base=ROOT/'outputs'/group
    print('\n##', group)
    if not base.exists(): print('missing'); continue
    for d in sorted([p for p in base.iterdir() if p.is_dir()]):
        all_dir=d/'all'
        cnt=sum(1 for _ in (all_dir if all_dir.exists() else d).rglob('*.png'))
        print(f'{d.name}: {cnt} png')

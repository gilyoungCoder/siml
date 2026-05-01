#!/usr/bin/env python3
import json, os
from pathlib import Path
import torch
root=Path(os.environ.get('REPRO_ROOT', Path(__file__).resolve().parents[1]))
fail=[]
for cfg in sorted((root/'configs/ours_best').rglob('*.json')):
    j=json.loads(cfg.read_text())
    for key in ['prompts','family_config']:
        p=Path(os.path.expandvars(j[key].replace('${REPRO_ROOT}', str(root))))
        if not p.exists(): fail.append(f'missing {key}: {cfg} -> {p}')
    assert j.get('family_guidance') is True, cfg
    assert j.get('how_mode') == 'hybrid', cfg
    pack_path=Path(os.path.expandvars(j['family_config'].replace('${REPRO_ROOT}', str(root))))
    pack=torch.load(pack_path, map_location='cpu', weights_only=False)
    fm=pack.get('family_metadata',{})
    print(f'{cfg.relative_to(root)}: families={list(fm.keys())} target_concepts={j.get("target_concepts")} anchor_concepts={j.get("anchor_concepts")}')
if fail:
    print('\n'.join(fail)); raise SystemExit(1)
print('OK: all configs resolve and use hybrid+family_guidance')

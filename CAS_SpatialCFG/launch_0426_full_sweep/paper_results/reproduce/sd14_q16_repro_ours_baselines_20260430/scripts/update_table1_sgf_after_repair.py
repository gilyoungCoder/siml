#!/usr/bin/env python3
from pathlib import Path
import re, csv
ROOT=Path('/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430')
SUM=ROOT/'summaries'
HAND=SUM/'TABLE1_NUDITY_BREAKDOWN_RELIABLE_HANDOFF_20260501.md'
CSV=SUM/'table1_sgf_nudity_breakdown_repaired_20260501.csv'
keys={'UD':'unlearndiff','RAB':'rab','MMA':'mma','P4DN':'p4dn'}
def parse(p):
    txt=p.read_text()
    total=int(re.search(r'Total images:\s*(\d+)',txt).group(1))
    vals={}
    for lab in ['NotPeople','NotRelevant','Safe','Partial','Full']:
        m=re.search(rf'-\s*{lab}:\s*(\d+) \(([-\d.]+)%\)', txt)
        if m: vals[lab]=(int(m.group(1)),float(m.group(2)))
    sr=float(re.search(r'SR \(Safe\+Partial\):\s*\d+/\d+ \(([-\d.]+)%\)',txt).group(1))
    return dict(sr=sr, safe=vals.get('Safe',(0,0.0))[1], partial=vals.get('Partial',(0,0.0))[1], full=vals.get('Full',(0,0.0))[1], nr=(vals.get('NotPeople') or vals.get('NotRelevant') or (0,0.0))[1], total=total, path=str(p))
rows={}
missing=[]
for d,k in keys.items():
    p=ROOT/f'outputs/sgf/nudity/{k}/all/results_qwen3_vl_nudity_v5.txt'
    if p.exists() and p.stat().st_size:
        rows[d]=parse(p)
    else:
        missing.append(d)
with CSV.open('w',newline='') as f:
    w=csv.writer(f); w.writerow(['dataset','SR','Safe','Partial','Full','NR','Total','source'])
    for d in ['UD','RAB','MMA','P4DN']:
        if d in rows:
            x=rows[d]; w.writerow([d,f"{x['sr']:.1f}",f"{x['safe']:.1f}",f"{x['partial']:.1f}",f"{x['full']:.1f}",f"{x['nr']:.1f}",x['total'],x['path']])
print('missing', missing)
if not missing:
    cells=[]
    for d in ['UD','RAB','MMA','P4DN']:
        x=rows[d]; cells.append(f"{x['safe']:.1f}/{x['partial']:.1f}/{x['full']:.1f}/{x['nr']:.1f}")
    sr=[f"{rows[d]['sr']:.1f}" for d in ['UD','RAB','MMA','P4DN']]
    print('SGF SR', sr)
    print('SGF breakdown', cells)

#!/usr/bin/env python3
from pathlib import Path
import re, csv
ROOT=Path('/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430')
BASE=ROOT/'outputs/probe_ablation_q16top60_20260501'
SUM=ROOT/'summaries'
concepts=['sexual','violence','self-harm','shocking','illegal_activity','harassment','hate']
eval_file={'sexual':'results_qwen3_vl_nudity_v5.txt','violence':'results_qwen3_vl_violence_v5.txt','self-harm':'results_qwen3_vl_self_harm_v5.txt','shocking':'results_qwen3_vl_shocking_v5.txt','illegal_activity':'results_qwen3_vl_illegal_v5.txt','harassment':'results_qwen3_vl_harassment_v5.txt','hate':'results_qwen3_vl_hate_v5.txt'}
def parse(p):
    txt=p.read_text()
    sr=float(re.search(r'SR \(Safe\+Partial\):\s*\d+/\d+ \(([-\d.]+)%\)',txt).group(1))
    return sr
rows=[]
for probe in ['text','image','both']:
    vals=[]
    for c in concepts:
        p=BASE/probe/c/eval_file[c]
        vals.append(parse(p) if p.exists() else None)
    rows.append((probe,vals))
SUM.mkdir(exist_ok=True)
with (SUM/'probe_ablation_i2p_q16_top60_7concept_20260501.csv').open('w',newline='') as f:
    w=csv.writer(f); w.writerow(['Probe']+concepts+['Avg'])
    for probe,vals in rows:
        avg=sum(v for v in vals if v is not None)/len([v for v in vals if v is not None]) if any(v is not None for v in vals) else None
        w.writerow([probe]+[('' if v is None else f'{v:.1f}') for v in vals]+[('' if avg is None else f'{avg:.1f}')])
md=['# Probe-channel ablation: SD v1.4 I2P q16 top-60 (7 concepts incl. sexual)','', 'Metric: Qwen3-VL v5 SR = Safe + Partial (%). Same per-concept best Ours configs/family packs; only `probe_mode` changes.', '', '| Probe | sexual | violence | self-harm | shocking | illegal | harassment | hate | Avg |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for probe,vals in rows:
    avg=sum(v for v in vals if v is not None)/len([v for v in vals if v is not None]) if any(v is not None for v in vals) else None
    md.append('| '+probe+' | '+ ' | '.join('TBD' if v is None else f'{v:.1f}' for v in vals)+f" | {'TBD' if avg is None else f'{avg:.1f}'} |")
(SUM/'probe_ablation_i2p_q16_top60_7concept_20260501.md').write_text('\n'.join(md)+'\n')
print('\n'.join(md))

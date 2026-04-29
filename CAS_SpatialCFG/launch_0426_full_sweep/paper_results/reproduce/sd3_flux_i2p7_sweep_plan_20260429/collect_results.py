#!/usr/bin/env python3
import csv, json, re, sys
from pathlib import Path
root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0429_sd3_flux_i2p7_sweep')
rows=[]
pat = re.compile(r'SR \(Safe\+Partial\):\s*(\d+)/(\d+) \(([^)]+)\)')
for txt in sorted(root.glob('**/results_qwen3_vl_*_v5.txt')):
    s=txt.read_text(errors='replace')
    m=pat.search(s)
    rel=txt.parent.relative_to(root)
    parts=rel.parts
    model=parts[0] if len(parts)>0 else ''
    mode=parts[1] if len(parts)>1 else ''
    concept=parts[2] if len(parts)>2 else ''
    config=parts[3] if len(parts)>3 else ''
    row={'model':model,'mode':mode,'concept':concept,'config_id':config,'dir':str(rel),'file':txt.name,'safe_partial':None,'den':None,'sr_percent':None}
    if m:
        row.update(safe_partial=int(m.group(1)), den=int(m.group(2)), sr_percent=float(m.group(3).rstrip('%')))
    rows.append(row)
rows.sort(key=lambda r: (r['model'], r['mode'], r['concept'], -(r['sr_percent'] or -1)))
out_csv=root/'summary_i2p7_sweep.csv'
out_md=root/'SUMMARY_I2P7_SWEEP.md'
out_csv.parent.mkdir(parents=True, exist_ok=True)
with out_csv.open('w', newline='', encoding='utf-8') as f:
    w=csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ['model','mode','concept','config_id','dir','file','safe_partial','den','sr_percent'])
    w.writeheader(); w.writerows(rows)
with out_md.open('w', encoding='utf-8') as f:
    f.write('# I2P-7 SD3 / FLUX1 Ours Sweep Summary\n\n')
    f.write('| model | mode | concept | config | SR | count | dir |\n|---|---|---|---|---:|---:|---|\n')
    for r in rows:
        count='' if r['safe_partial'] is None else f"{r['safe_partial']}/{r['den']}"
        sr='' if r['sr_percent'] is None else f"{r['sr_percent']:.2f}%"
        f.write(f"| {r['model']} | {r['mode']} | {r['concept']} | {r['config_id']} | {sr} | {count} | `{r['dir']}` |\n")
print(f'wrote {out_csv}')
print(f'wrote {out_md}')

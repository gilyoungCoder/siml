#!/usr/bin/env python3
import json, re, sys, csv
from pathlib import Path
root=Path(sys.argv[1]) if len(sys.argv)>1 else Path('/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/baseline_safree_official_20260429/outputs/full')
rows=[]
pat=re.compile(r'SR \(Safe\+Partial\):\s*(\d+)/(\d+) \(([^)]+)\)')
for method_dir in sorted(root.glob('*')):
    if not method_dir.is_dir(): continue
    for group_dir in sorted(method_dir.glob('*')):
        if not group_dir.is_dir(): continue
        for name_dir in sorted(group_dir.glob('*')):
            all_dir=name_dir/'all'
            pngs=len(list(all_dir.glob('*.png'))) if all_dir.exists() else 0
            row={'method':method_dir.name,'group':group_dir.name,'name':name_dir.name,'pngs':pngs,'vlm_sr':'','vlm_count':'','official_unsafe_avg':'','dir':str(name_dir)}
            for txt in all_dir.glob('results_qwen3_vl_*_v5.txt') if all_dir.exists() else []:
                m=pat.search(txt.read_text(errors='replace'))
                if m:
                    row['vlm_sr']=m.group(3); row['vlm_count']=f'{m.group(1)}/{m.group(2)}'
                    break
            dd=name_dir/'detect_dict.json'
            if dd.exists():
                try:
                    d=json.loads(dd.read_text())
                    row['official_unsafe_avg']=d.get('toxic_ratio',{}).get('average','')
                except Exception:
                    pass
            rows.append(row)
out_csv=root/'SUMMARY_OFFICIAL_BASELINES.csv'
out_md=root/'SUMMARY_OFFICIAL_BASELINES.md'
with out_csv.open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=['method','group','name','pngs','vlm_sr','vlm_count','official_unsafe_avg','dir'])
    w.writeheader(); w.writerows(rows)
with out_md.open('w',encoding='utf-8') as f:
    f.write('# Official SAFREE+SafeDenoiser / SAFREE+SGF summary\n\n')
    f.write('| method | group | name | pngs | Qwen SR | count | official unsafe avg | dir |\n|---|---|---|---:|---:|---:|---:|---|\n')
    for r in rows:
        f.write(f"| {r['method']} | {r['group']} | {r['name']} | {r['pngs']} | {r['vlm_sr']} | {r['vlm_count']} | {r['official_unsafe_avg']} | `{r['dir']}` |\n")
print(out_csv)
print(out_md)

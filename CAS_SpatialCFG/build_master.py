#!/usr/bin/env python3
"""Build paper_results_master/ — copy best dirs per (section, concept, mode)."""
import os
import json
import glob
import shutil
import sys
from collections import Counter

REPO = '/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG'
MASTER = REPO + '/paper_results_master'
DRY_RUN = '--dry-run' in sys.argv

# ---------------------------------------------------------------- helpers
def cnt_sr(jp):
    if not os.path.exists(jp):
        return None
    try:
        with open(jp) as f:
            d = json.load(f)
    except Exception:
        return None
    c = Counter(v['category'] for v in d.values() if isinstance(v, dict) and 'category' in v)
    s, p, fu, nr = c.get('Safe', 0), c.get('Partial', 0), c.get('Full', 0), c.get('NotRelevant', 0)
    den = s + p + fu + nr
    if den == 0:
        return None
    return {'sr': (s + p) / den * 100, 'full': fu / den * 100, 'n': den}


def is_mode_match(dname, mode_label):
    dn = dname.lower()
    if mode_label == 'anchor':
        return 'anchor' in dn
    if mode_label == 'hybrid':
        return ('hybrid' in dn or 'hyb_' in dn) and 'anchor' not in dn
    if mode_label == 'txtonly':
        return 'txtonly' in dn
    if mode_label == 'imgonly':
        return 'imgonly' in dn
    if mode_label == 'both':
        return 'both' in dn and 'txtonly' not in dn and 'imgonly' not in dn
    return False


def search_best(globs, eval_concept, mode_label, exclude_substrings=None, min_n=30):
    """Search across globs, filter by mode, pick max SR. Skip dirs with n < min_n."""
    exclude_substrings = exclude_substrings or []
    json_name = f'categories_qwen3_vl_{eval_concept}_v5.json'
    cands = []
    for g in globs:
        for d in glob.glob(g):
            if not os.path.isdir(d):
                continue
            full = os.path.abspath(d)
            if any(ex in full for ex in exclude_substrings):
                continue
            dname = os.path.basename(d.rstrip('/'))
            if not is_mode_match(dname, mode_label):
                continue
            jp = os.path.join(d, json_name)
            r = cnt_sr(jp)
            if r and r['n'] >= min_n:
                cands.append((d.rstrip('/'), r))
    if not cands:
        return None, None
    best = max(cands, key=lambda x: x[1]['sr'])
    return best[0], best[1]


# ---------------------------------------------------------------- jobs
JOBS = []   # list of (section, label, globs, eval_concept, mode, excludes)

# 01 Nudity SD1.4 (5 benches × 2 modes)
NUDITY_BENCHES_LAUNCH = ['rab', 'mma', 'p4dn', 'unlearndiff']
for bench in NUDITY_BENCHES_LAUNCH:
    globs = [
        f'{REPO}/outputs/launch_0420_nudity/ours_sd14*/{bench}/*/',
    ]
    for mode in ['anchor', 'hybrid']:
        JOBS.append(('01_nudity_sd14_5bench', f'{bench}_{mode}', globs, 'nudity', mode,
                     ['ablation', 'multi', 'baseline', 'safree']))
# mja_sexual (treated as nudity bench too)
for mode in ['anchor', 'hybrid']:
    JOBS.append(('01_nudity_sd14_5bench', f'mja_sexual_{mode}',
                 [f'{REPO}/outputs/launch_*/ours_sd14*/mja_sexual/*/'],
                 'nudity', mode, ['ablation', 'multi', 'baseline', 'safree', 'flux', 'sd3']))

# 02 I2P top60 SD1.4 (6 concepts × 2 modes)
I2P_CONCEPTS = [('violence', 'violence'), ('self-harm', 'self_harm'),
                ('shocking', 'shocking'), ('illegal_activity', 'illegal'),
                ('harassment', 'harassment'), ('hate', 'hate')]
for cdir, cf in I2P_CONCEPTS:
    globs = [
        f'{REPO}/outputs/launch_0420_i2p/ours_sd14*/{cdir}/*/',
        f'{REPO}/outputs/launch_0420_i2p_q16top60/ours_sd14*/{cdir}/*/',
        f'{REPO}/outputs/launch_0423_*/{cdir}/*/',
        f'{REPO}/outputs/launch_0423_*/i2p/{cdir}/*/',
        f"{REPO}/outputs/launch_0424_rerun_sd14/i2p_{cdir}/*/", f"{REPO}/outputs/launch_0424_v3/i2p_{cdir}/*/", f"{REPO}/outputs/launch_0424_v5/i2p_{cdir}/*/", f"{REPO}/outputs/launch_0424_v6/i2p_{cdir}/*/", f"{REPO}/outputs/launch_0424_v7/i2p_{cdir}/*/", f"{REPO}/outputs/launch_0424_v8/i2p_{cdir}/*/",
    ]
    if cdir == 'shocking':
        globs.append(f'{REPO}/outputs/launch_0423_shocking_imgheavy/*/')
    if cdir in ('harassment', 'hate'):
        globs.append(f'{REPO}/outputs/launch_0423_harhate_imgheavy/{cdir}/*/')
    for mode in ['anchor', 'hybrid']:
        JOBS.append(('02_i2p_top60_sd14_6concept', f'{cdir}_{mode}', globs, cf, mode,
                     ['ablation', 'multi', 'baseline', 'safree']))

# 03 MJA SD1.4 (4 concepts × 2 modes)
MJA_CONCEPTS = [('mja_sexual', 'nudity'), ('mja_violent', 'violence'),
                ('mja_illegal', 'illegal'), ('mja_disturbing', 'disturbing')]
for cdir, cf in MJA_CONCEPTS:
    globs = [
        f'{REPO}/outputs/launch_*/ours_sd14*/{cdir}/*/',
        f"{REPO}/outputs/launch_0424_rerun_sd14/{cdir}/*/", f"{REPO}/outputs/launch_0424_v5/{cdir}/*/", f"{REPO}/outputs/launch_0424_v6/{cdir}/*/",
    ]
    if cdir == 'mja_illegal':
        globs.append(f'{REPO}/outputs/launch_0423_illegal_aggro/*/')
    for mode in ['anchor', 'hybrid']:
        JOBS.append(('03_mja_sd14_4concept', f'{cdir}_{mode}', globs, cf, mode,
                     ['ablation', 'multi', 'baseline', 'safree', 'flux', 'sd3', 'i2p']))

# 04 MJA SD3 (4 concepts × 2 modes)
for cdir, cf in MJA_CONCEPTS:
    globs = [
        f'{REPO}/outputs/launch_*/ours_sd3*/{cdir}/*/',
        f"{REPO}/outputs/launch_0424_rerun_sd3/{cdir}/*/", f"{REPO}/outputs/launch_0424_v3_sd3/{cdir}/*/", f"{REPO}/outputs/launch_0424_v4_sd3/{cdir}/*/",
    ]
    for mode in ['anchor', 'hybrid']:
        JOBS.append(('04_mja_sd3_4concept', f'{cdir}_{mode}', globs, cf, mode,
                     ['ablation', 'multi', 'baseline', 'safree', 'flux', 'sd14']))

# 05 MJA FLUX1 (4 concepts × 2 modes)
for cdir, cf in MJA_CONCEPTS:
    globs = [
        f'{REPO}/outputs/launch_*/ours_flux1*/{cdir}/*/',
        f'{REPO}/outputs/launch_0424_rerun_flux1/{cdir}/*/',
    ]
    for mode in ['anchor', 'hybrid']:
        JOBS.append(('05_mja_flux1_4concept', f'{cdir}_{mode}', globs, cf, mode,
                     ['ablation', 'multi', 'baseline', 'safree', 'sd3', 'sd14']))

# 06 Multi-concept SD1.4
# MJA multi (sexual+violent in one model)
for cdir, cf in [('mja_sexual', 'nudity'), ('mja_violent', 'violence')]:
    globs = [f'{REPO}/outputs/launch_*/ours_sd14_multiconcept*/{cdir}/*/']
    for mode in ['anchor', 'hybrid']:
        JOBS.append(('06_multi_concept_sd14', f'mja_multi_{cdir}_{mode}', globs, cf, mode, []))
# I2P multi
for cdir, cf in I2P_CONCEPTS:
    globs = [f'{REPO}/outputs/launch_0420_i2p/ours_sd14_multi*/{cdir}/*/']
    for mode in ['anchor', 'hybrid']:
        JOBS.append(('06_multi_concept_sd14', f'i2p_multi_{cdir}_{mode}', globs, cf, mode, ['safree']))

# 07 Ablation SD1.4 (6 i2p concepts × 3 modes: txtonly / imgonly / both)
for cdir, cf in I2P_CONCEPTS:
    txt_globs = [f'{REPO}/outputs/launch_0420_i2p/ours_sd14_ablation_txtonly*/{cdir}/*/']
    img_globs = [f'{REPO}/outputs/launch_0420_i2p/ours_sd14_ablation_imgonly*/{cdir}/*/']
    both_globs = [
        f'{REPO}/outputs/launch_0420_i2p/ours_sd14*/{cdir}/*/',
        f'{REPO}/outputs/launch_0423_shocking_imgheavy/*/' if cdir == 'shocking' else None,
        f'{REPO}/outputs/launch_0423_harhate_imgheavy/{cdir}/*/' if cdir in ('harassment', 'hate') else None,
        f"{REPO}/outputs/launch_0424_rerun_sd14/i2p_{cdir}/*/", f"{REPO}/outputs/launch_0424_v3/i2p_{cdir}/*/", f"{REPO}/outputs/launch_0424_v5/i2p_{cdir}/*/", f"{REPO}/outputs/launch_0424_v6/i2p_{cdir}/*/", f"{REPO}/outputs/launch_0424_v7/i2p_{cdir}/*/", f"{REPO}/outputs/launch_0424_v8/i2p_{cdir}/*/",
    ]
    both_globs = [g for g in both_globs if g]
    JOBS.append(('07_ablation_sd14_probe', f'{cdir}_txtonly', txt_globs, cf, 'txtonly', []))
    JOBS.append(('07_ablation_sd14_probe', f'{cdir}_imgonly', img_globs, cf, 'imgonly', []))
    JOBS.append(('07_ablation_sd14_probe', f'{cdir}_both',    both_globs, cf, 'both', ['ablation', 'multi', 'baseline', 'safree']))


# ---------------------------------------------------------------- run
results = []
print(f'Total {len(JOBS)} cells to resolve...')
print()
for section, label, globs, ec, mode, excl in JOBS:
    src, sr = search_best(globs, ec, mode, excl)
    results.append({
        'section': section, 'label': label, 'mode': mode,
        'eval_concept': ec, 'src': src,
        'sr': sr['sr'] if sr else None,
        'full': sr['full'] if sr else None,
        'n': sr['n'] if sr else None,
    })
    src_short = src.replace(f'{REPO}/outputs/', '') if src else 'NOT FOUND'
    sr_str = f"SR={sr['sr']:5.1f} Full={sr['full']:5.1f} n={sr['n']}" if sr else '--'
    print(f'[{section}] {label:35s}  {sr_str:35s}  src={src_short}')

# Summary
print()
print('='*100)
sections = {}
for r in results:
    sections.setdefault(r['section'], []).append(r)
for sec, items in sections.items():
    found = sum(1 for x in items if x['src'])
    print(f'{sec}: {found}/{len(items)} cells resolved')

if DRY_RUN:
    print()
    print('Dry-run mode — no files copied.')
    sys.exit(0)

# ---------------------------------------------------------------- copy
print()
print('Copying...')
os.makedirs(MASTER, exist_ok=True)
manifest = {}
for r in results:
    if not r['src']:
        continue
    dest_dir = os.path.join(MASTER, r['section'], r['label'])
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    shutil.copytree(r['src'], dest_dir)
    manifest[f"{r['section']}/{r['label']}"] = {
        'source': r['src'],
        'eval_concept': r['eval_concept'],
        'mode': r['mode'],
        'sr': r['sr'],
        'full': r['full'],
        'n': r['n'],
    }

with open(os.path.join(MASTER, 'results_summary.json'), 'w') as f:
    json.dump(manifest, f, indent=2)

# README
with open(os.path.join(MASTER, 'README.md'), 'w') as f:
    f.write('# Paper Results Master\n\n')
    f.write('Best (max SR) generation dirs per (section, concept, mode), copied here for paper.\n\n')
    for sec, items in sections.items():
        f.write(f'## {sec}\n\n')
        f.write('| Cell | SR (%) | Full (%) | N | Source |\n')
        f.write('|---|---|---|---|---|\n')
        for r in items:
            if r['src']:
                src = r['src'].replace(REPO + '/outputs/', '')
                f.write(f"| {r['label']} | {r['sr']:.1f} | {r['full']:.1f} | {r['n']} | `{src}` |\n")
            else:
                f.write(f"| {r['label']} | — | — | — | NOT FOUND |\n")
        f.write('\n')

print(f'\nMaster dir: {MASTER}')
print(f'Total copied: {sum(1 for r in results if r["src"])}')

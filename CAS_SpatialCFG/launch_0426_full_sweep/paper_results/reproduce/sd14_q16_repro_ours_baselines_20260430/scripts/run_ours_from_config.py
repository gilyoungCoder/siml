#!/usr/bin/env python3
import argparse, json, os, subprocess, sys
from pathlib import Path

ROOT = Path('/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430')
PY = os.environ.get('PY_SAFGEN', '/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10')
SAFEGEN = ROOT / 'code' / 'SafeGen'

def png_count(path: Path) -> int:
    return sum(1 for _ in path.rglob('*.png')) if path.exists() else 0

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--gpu', required=True)
    ap.add_argument('--expected', type=int, default=None)
    args=ap.parse_args()
    cfg=json.loads(Path(args.config).read_text())
    out=Path(cfg['outdir'])
    expected=args.expected
    if expected is None:
        prompts=Path(cfg['prompts']).read_text().splitlines()
        expected=len([p for p in prompts if p.strip()]) * int(cfg.get('nsamples',1))
    if png_count(out) >= expected and (out/'generation_stats.json').exists():
        print(f'[SKIP ours] {out} pngs>={expected}')
        return 0
    out.mkdir(parents=True, exist_ok=True)
    cmd=[PY, '-m', 'safegen.generate_family',
         '--ckpt', cfg.get('ckpt','CompVis/stable-diffusion-v1-4'),
         '--prompts', cfg['prompts'], '--outdir', cfg['outdir'],
         '--nsamples', str(cfg.get('nsamples',1)), '--steps', str(cfg.get('steps',50)),
         '--seed', str(cfg.get('seed',42)), '--cfg_scale', str(cfg.get('cfg_scale',7.5)),
         '--start_idx', str(cfg.get('start_idx',0)), '--end_idx', str(cfg.get('end_idx',-1)),
         '--cas_threshold', str(cfg.get('cas_threshold',0.6)),
         '--probe_mode', cfg.get('probe_mode','both'),
         '--family_config', cfg['family_config'],
         '--attn_threshold', str(cfg.get('attn_threshold',0.1)),
         '--img_attn_threshold', str(cfg.get('img_attn_threshold',0.3)),
         '--attn_sigmoid_alpha', str(cfg.get('attn_sigmoid_alpha',10.0)),
         '--blur_sigma', str(cfg.get('blur_sigma',1.0)),
         '--probe_fusion', cfg.get('probe_fusion','union'),
         '--n_img_tokens', str(cfg.get('n_img_tokens',4)),
         '--how_mode', cfg.get('how_mode','hybrid'),
         '--safety_scale', str(cfg.get('safety_scale',20.0))]
    if cfg.get('attn_resolutions'):
        cmd += ['--attn_resolutions'] + [str(x) for x in cfg['attn_resolutions']]
    if cfg.get('target_words'):
        cmd += ['--target_words'] + [str(x) for x in cfg['target_words']]
    if cfg.get('family_guidance', False):
        cmd += ['--family_guidance']
    if cfg.get('target_concepts'):
        cmd += ['--target_concepts'] + [str(x) for x in cfg['target_concepts']]
    if cfg.get('anchor_concepts'):
        cmd += ['--anchor_concepts'] + [str(x) for x in cfg['anchor_concepts']]
    if cfg.get('save_maps', False):
        cmd += ['--save_maps']
    env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']=str(args.gpu)
    print('[RUN ours]', ' '.join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(SAFEGEN), env=env).returncode

if __name__ == '__main__':
    sys.exit(main())

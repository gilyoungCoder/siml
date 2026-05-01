#!/usr/bin/env python3
import argparse, json, os, subprocess, sys
from pathlib import Path

def expand(x):
    if isinstance(x, str): return os.path.expandvars(x)
    if isinstance(x, list): return [expand(v) for v in x]
    if isinstance(x, dict): return {k:expand(v) for k,v in x.items()}
    return x

def png_count(path: Path):
    return len(list(path.glob('*.png')))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--force', action='store_true')
    args=ap.parse_args()
    repro=Path(os.environ.get('REPRO_ROOT', Path(__file__).resolve().parents[1]))
    os.environ.setdefault('REPRO_ROOT', str(repro))
    os.environ.setdefault('OUT_ROOT', str(repro))
    py=os.environ.get('PY_SAFGEN','/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10')
    cfg=expand(json.loads(Path(args.config).read_text()))
    out=Path(cfg['outdir']); expected=int(cfg.get('expected_images', 0))
    if not args.force and expected and png_count(out) >= expected:
        print(f'[SKIP] {out} has >= {expected} pngs')
        return 0
    out.mkdir(parents=True, exist_ok=True)
    cmd=[py,'-m','safegen.generate_family','--ckpt',cfg.get('ckpt','CompVis/stable-diffusion-v1-4'),
         '--prompts',cfg['prompts'],'--outdir',cfg['outdir'],'--nsamples',str(cfg.get('nsamples',1)),
         '--steps',str(cfg.get('steps',50)),'--seed',str(cfg.get('seed',42)),'--cfg_scale',str(cfg.get('cfg_scale',7.5)),
         '--start_idx',str(cfg.get('start_idx',0)),'--end_idx',str(cfg.get('end_idx',-1)),
         '--cas_threshold',str(cfg.get('cas_threshold',0.6)),'--probe_mode',cfg.get('probe_mode','both'),
         '--family_config',cfg['family_config'],'--attn_threshold',str(cfg.get('attn_threshold',0.1)),
         '--img_attn_threshold',str(cfg.get('img_attn_threshold',0.3)),'--attn_sigmoid_alpha',str(cfg.get('attn_sigmoid_alpha',10.0)),
         '--blur_sigma',str(cfg.get('blur_sigma',1.0)),'--probe_fusion',cfg.get('probe_fusion','union'),
         '--n_img_tokens',str(cfg.get('n_img_tokens',4)),'--how_mode',cfg.get('how_mode','hybrid'),
         '--safety_scale',str(cfg.get('safety_scale',20.0))]
    if cfg.get('attn_resolutions'): cmd += ['--attn_resolutions'] + [str(v) for v in cfg['attn_resolutions']]
    if cfg.get('target_words'): cmd += ['--target_words'] + [str(v) for v in cfg['target_words']]
    if cfg.get('family_guidance'): cmd += ['--family_guidance']
    if cfg.get('target_concepts'): cmd += ['--target_concepts'] + [str(v) for v in cfg['target_concepts']]
    if cfg.get('anchor_concepts'): cmd += ['--anchor_concepts'] + [str(v) for v in cfg['anchor_concepts']]
    if cfg.get('save_maps'): cmd += ['--save_maps']
    env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']=str(args.gpu)
    print('[RUN]', ' '.join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(repro/'code/SafeGen'), env=env).returncode
if __name__=='__main__': raise SystemExit(main())

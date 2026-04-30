#!/usr/bin/env python3
import argparse, csv, json, os, sys, random
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import yaml

ROOT=Path('/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430')
SAFE_REPO=ROOT/'code/official_repos/Safe_Denoiser'
SGF_REPO=ROOT/'code/official_repos/SGF/diversity_sdv3'

def load_prompts(path):
    p=Path(path)
    rows=[]
    if p.suffix.lower()=='.csv':
        with open(p) as f:
            r=csv.DictReader(f)
            col=next((c for c in ['sensitive prompt','adv_prompt','prompt','target_prompt','text','Prompt','Text'] if c in (r.fieldnames or [])), None)
            if col is None: raise RuntimeError(f'No prompt column in {p}: {r.fieldnames}')
            for i,row in enumerate(r):
                pr=(row.get(col) or '').strip()
                if not pr: continue
                seed=row.get('evaluation_seed') or row.get('sd_seed') or row.get('seed') or ''
                case=row.get('case_number') or row.get('case') or str(i)
                try: seed=int(seed)
                except Exception: seed=42+i
                rows.append((int(float(case)) if str(case).replace('.','',1).isdigit() else i, pr, seed))
    else:
        for i,line in enumerate(p.read_text().splitlines()):
            line=line.strip()
            if line: rows.append((i,line,42+i))
    return rows

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--method', choices=['safree','safedenoiser','sgf'], required=True)
    ap.add_argument('--prompts', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--task_config', default=None)
    ap.add_argument('--model_id', default='stabilityai/stable-diffusion-3-medium-diffusers')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--steps', type=int, default=28)
    ap.add_argument('--guidance_scale', type=float, default=7.0)
    ap.add_argument('--height', type=int, default=1024)
    ap.add_argument('--width', type=int, default=1024)
    ap.add_argument('--start_idx', type=int, default=0)
    ap.add_argument('--end_idx', type=int, default=-1)
    args=ap.parse_args()
    out=Path(args.outdir); (out/'all').mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(out/'args.json','w'), indent=2)

    repo = SGF_REPO if args.method=='sgf' else SAFE_REPO
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    if args.method=='sgf':
        from models.sdv3.sdv3_pipeline import StableDiffusion3Pipeline as StdPipe
        from models.sdv3.safree_pipeline_efficient import StableDiffusion3Pipeline as SafreePipe
        from models.sdv3.safer_pipeline import StableDiffusion3Pipeline as RepPipe
        from repellency.repellency_methods_fast_sdv3 import get_repellency_method
        from data.dataloader import get_dataset, get_dataloader, get_transform, get_all_imgs
    else:
        from models.sdv3.sdv3_pipeline import StableDiffusion3Pipeline as StdPipe
        from models.sdv3.safree_pipeline import StableDiffusion3Pipeline as SafreePipe
        from models.sdv3.safe_denoiser_pipeline import StableDiffusion3Pipeline as RepPipe
        from repellency.repellency_methods_fast_sdv3 import get_repellency_method
        from data.dataloader import get_dataset, get_dataloader, get_transform, get_all_imgs
    Pipe = SafreePipe if args.method=='safree' else RepPipe
    print(f'[SD3 {args.method}] load {args.model_id} repo={repo}')
    pipe=Pipe.from_pretrained(args.model_id, torch_dtype=torch.float16).to(args.device)
    pipe.vae.requires_grad_(False)
    if hasattr(pipe,'text_encoder') and pipe.text_encoder is not None: pipe.text_encoder.requires_grad_(False)
    gen=torch.Generator(device=args.device)

    repellency_processor=None; neg_extra={}
    task_cfg=None
    if args.method in ['safedenoiser','sgf']:
        if not args.task_config: raise RuntimeError('task_config required')
        task_cfg=yaml.safe_load(open(args.task_config))
        data_cfg=task_cfg['data']
        transform=get_transform(**data_cfg)
        ds=get_dataset(**data_cfg, transforms=transform)
        dl=get_dataloader(ds, batch_size=1 if args.method!='sgf' else min(16,len(ds)), num_workers=0, train=False)
        ref_imgs=get_all_imgs(dl).to(args.device)
        embed_fn=lambda x: pipe.vae.encode(x.to(torch.float16)).latent_dist.sample()*pipe.vae.config.scaling_factor
        rep_cfg=task_cfg['repellency']
        params=dict(rep_cfg.get('params',{}))
        # Always recompute SD3 latent refs into a separate cache; SD1.4 caches are shape-incompatible.
        params['cache_proj_ref']=False
        repellency_processor=get_repellency_method(rep_cfg['method'], ref_data=ref_imgs, embed_fn=embed_fn,
            forward_fn=None, num_timesteps=args.steps, max_idx=None, beta_min=None, beta_max=None,
            n_embed=rep_cfg.get('n_embed',8), **params) if args.method!='sgf' else get_repellency_method(rep_cfg['method'], ref_data=ref_imgs, embed_fn=embed_fn, n_embed=rep_cfg.get('n_embed',8), **params)
        if args.method=='sgf':
            neg_extra={'neg_start': rep_cfg.get('neg_start',1000), 'neg_end': rep_cfg.get('neg_end',800)}
        print(f'[SD3 {args.method}] repellency={rep_cfg["method"]} refs={tuple(ref_imgs.shape)}')

    rows=load_prompts(args.prompts)
    e=len(rows) if args.end_idx<0 else min(args.end_idx,len(rows))
    rows=rows[args.start_idx:e]
    stats=[]
    for case,prompt,seed in rows:
        set_seed(seed)
        print(f'[SD3 {args.method}] case={case} seed={seed} prompt={prompt[:80]}')
        with torch.no_grad():
            img=pipe(prompt, negative_prompt='', num_inference_steps=args.steps, guidance_scale=args.guidance_scale,
                     generator=gen.manual_seed(seed), height=args.height, width=args.width,
                     repellency_processor=repellency_processor, **neg_extra).images[0]
        fn=f'{case:04d}.png'
        img.save(out/'all'/fn)
        stats.append({'case':case,'seed':seed,'prompt':prompt})
    json.dump(stats, open(out/'stats.json','w'), indent=2)
    if task_cfg: yaml.safe_dump(task_cfg, open(out/'task_config_used.yaml','w'))
    print(f'DONE {len(stats)} -> {out}/all')
if __name__=='__main__': main()

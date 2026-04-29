#!/usr/bin/env python3
"""Build I2P family exemplar packs using SD3 as the exemplar generator.

Reads family_config.json from i2p_v1 (same prompts, same words),
generates exemplars via Stable-Diffusion-3-medium, encodes CLIP-ViT-L/14
image features per family, writes clip_grouped.pt to i2p_v1_sd3/<cat>/.

Only `family_metadata`, `target_clip_features`, `anchor_clip_features`,
`family_names` are populated — that's all generate_sd3_safegen.py reads.
"""
import argparse, json, os, torch
from pathlib import Path
from PIL import Image
from diffusers import StableDiffusion3Pipeline
from transformers import CLIPModel, CLIPProcessor

DEVICE = 'cuda:0'
SRC_ROOT = Path('/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/exemplars/i2p_v1')
DST_ROOT = Path('/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/exemplars/i2p_v1_sd3')
SD3_CKPT = 'stabilityai/stable-diffusion-3-medium-diffusers'
CLIP_CKPT = 'openai/clip-vit-large-patch14'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cats', nargs='+', default=None)
    ap.add_argument('--steps', type=int, default=28)
    ap.add_argument('--cfg', type=float, default=7.0)
    ap.add_argument('--res', type=int, default=1024)
    args = ap.parse_args()

    cfg_all = json.load(open(SRC_ROOT / 'family_config.json'))
    cats = args.cats or list(cfg_all.keys())

    print(f'[SD3 pack build] cats={cats} steps={args.steps} cfg={args.cfg} res={args.res}')
    print('[load] SD3...')
    pipe = StableDiffusion3Pipeline.from_pretrained(SD3_CKPT, torch_dtype=torch.float16).to(DEVICE)
    pipe.set_progress_bar_config(disable=True)

    print('[load] CLIP-L/14...')
    clip_proc = CLIPProcessor.from_pretrained(CLIP_CKPT)
    clip_mod = CLIPModel.from_pretrained(CLIP_CKPT, torch_dtype=torch.float16).to(DEVICE).eval()

    for cat in cats:
        if cat not in cfg_all:
            print(f'  SKIP unknown cat: {cat}')
            continue
        cfg = cfg_all[cat]
        out_dir = DST_ROOT / cat
        img_dir = out_dir / 'images'
        img_dir.mkdir(parents=True, exist_ok=True)
        out_pack = out_dir / 'clip_grouped.pt'
        if out_pack.exists():
            print(f'  SKIP existing pack: {out_pack}')
            continue

        print(f'\n=== {cat} ===')
        families = cfg['families']
        family_names = list(families.keys())
        target_clip_features = {}
        anchor_clip_features = {}
        family_metadata = {}

        for fname, fdata in families.items():
            tgt_prompts = fdata['target_prompts']
            anc_prompts = fdata['anchor_prompts']
            print(f'  [{fname}] gen {len(tgt_prompts)} target + {len(anc_prompts)} anchor')

            tgt_imgs = []
            for i, p in enumerate(tgt_prompts):
                gen = torch.Generator(device=DEVICE).manual_seed(42 + i)
                img = pipe(p, num_inference_steps=args.steps, guidance_scale=args.cfg,
                           generator=gen, height=args.res, width=args.res).images[0]
                tgt_imgs.append(img)
                img.save(img_dir / f'{fname}_target_{i:02d}.png')

            anc_imgs = []
            for i, p in enumerate(anc_prompts):
                gen = torch.Generator(device=DEVICE).manual_seed(42 + i)
                img = pipe(p, num_inference_steps=args.steps, guidance_scale=args.cfg,
                           generator=gen, height=args.res, width=args.res).images[0]
                anc_imgs.append(img)
                img.save(img_dir / f'{fname}_anchor_{i:02d}.png')

            with torch.no_grad():
                tgt_in = clip_proc(images=tgt_imgs, return_tensors='pt').pixel_values.to(DEVICE).to(torch.float16)
                anc_in = clip_proc(images=anc_imgs, return_tensors='pt').pixel_values.to(DEVICE).to(torch.float16)
                vo = clip_mod.vision_model(pixel_values=tgt_in)
                tgt_feat = clip_mod.visual_projection(vo.pooler_output).cpu()
                vo = clip_mod.vision_model(pixel_values=anc_in)
                anc_feat = clip_mod.visual_projection(vo.pooler_output).cpu()

            target_clip_features[fname] = tgt_feat
            anchor_clip_features[fname] = anc_feat
            family_metadata[fname] = {
                'n_images': len(tgt_prompts),
                'target_prompts': tgt_prompts,
                'anchor_prompts': anc_prompts,
                'target_words': cfg['concept_keywords'],
                'anchor_words': cfg['anchor_keywords'],
            }

        pack = {
            'target_clip_features': target_clip_features,
            'anchor_clip_features': anchor_clip_features,
            'family_names': family_names,
            'family_metadata': family_metadata,
            'config': {
                'concept': cat,
                'n_families': len(family_names),
                'n_per_family': len(families[family_names[0]]['target_prompts']),
                'grouped': True,
                'source': 'i2p_hard_v1',
                'backbone': 'sd3',
                'gen_steps': args.steps,
                'gen_cfg': args.cfg,
                'gen_res': args.res,
            },
        }
        torch.save(pack, out_pack)
        print(f'  saved -> {out_pack}')
        print(f'  families={family_names}')

    print('\n[done]')

if __name__ == '__main__':
    main()

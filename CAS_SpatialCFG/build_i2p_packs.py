#!/usr/bin/env python3
"""Build I2P family exemplar packs from family_config.json."""
import argparse, json, os, torch
from pathlib import Path
from PIL import Image
from diffusers import StableDiffusionPipeline
from transformers import CLIPTokenizer, CLIPModel, CLIPProcessor

DEVICE = 'cuda:0'
ROOT = Path('/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/exemplars/i2p_v1')
SD_CKPT = 'CompVis/stable-diffusion-v1-4'
CLIP_CKPT = 'openai/clip-vit-large-patch14'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cats', nargs='+', default=None, help='subset of categories to build')
    args = ap.parse_args()
    
    cfg_all = json.load(open(ROOT / 'family_config.json'))
    cats = args.cats or list(cfg_all.keys())

    print('[load] SD1.4...')
    pipe = StableDiffusionPipeline.from_pretrained(SD_CKPT, torch_dtype=torch.float16, safety_checker=None).to(DEVICE)
    pipe.set_progress_bar_config(disable=True)
    
    print('[load] CLIP-L...')
    clip_proc = CLIPProcessor.from_pretrained(CLIP_CKPT)
    clip_mod = CLIPModel.from_pretrained(CLIP_CKPT, torch_dtype=torch.float16).to(DEVICE).eval()
    
    tok = pipe.tokenizer
    text_enc = pipe.text_encoder
    sos_id = tok.bos_token_id
    eos_id = tok.eos_token_id
    
    for cat in cats:
        cfg = cfg_all[cat]
        out_dir = ROOT / cat
        img_dir = out_dir / 'images'
        img_dir.mkdir(parents=True, exist_ok=True)
        
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
                img = pipe(p, num_inference_steps=50, guidance_scale=7.5, generator=gen,
                           height=512, width=512).images[0]
                tgt_imgs.append(img)
                img.save(img_dir / f'{fname}_target_{i:02d}.png')
            
            anc_imgs = []
            for i, p in enumerate(anc_prompts):
                gen = torch.Generator(device=DEVICE).manual_seed(42 + i)
                img = pipe(p, num_inference_steps=50, guidance_scale=7.5, generator=gen,
                           height=512, width=512).images[0]
                anc_imgs.append(img)
                img.save(img_dir / f'{fname}_anchor_{i:02d}.png')
            
            with torch.no_grad():
                tgt_in = clip_proc(images=tgt_imgs, return_tensors='pt').pixel_values.to(DEVICE).to(torch.float16)
                anc_in = clip_proc(images=anc_imgs, return_tensors='pt').pixel_values.to(DEVICE).to(torch.float16)
                vo = clip_mod.vision_model(pixel_values=tgt_in); tgt_feat = clip_mod.visual_projection(vo.pooler_output).cpu()
                vo = clip_mod.vision_model(pixel_values=anc_in); anc_feat = clip_mod.visual_projection(vo.pooler_output).cpu()
            
            target_clip_features[fname] = tgt_feat
            anchor_clip_features[fname] = anc_feat
            family_metadata[fname] = {
                'n_images': len(tgt_prompts),
                'target_prompts': tgt_prompts,
                'anchor_prompts': anc_prompts,
                'target_words': cfg['concept_keywords'],
                'anchor_words': cfg['anchor_keywords'],
            }
        
        # Combined target/anchor text via SD text encoder
        tgt_text = ' '.join(w.replace('_', ' ') for w in cfg['concept_keywords'])
        anc_text = ' '.join(w.replace('_', ' ') for w in cfg['anchor_keywords'])
        
        with torch.no_grad():
            tgt_ids = tok(tgt_text, padding='max_length', max_length=77, truncation=True, return_tensors='pt').input_ids.to(DEVICE)
            anc_ids = tok(anc_text, padding='max_length', max_length=77, truncation=True, return_tensors='pt').input_ids.to(DEVICE)
            tgt_embed = text_enc(tgt_ids).last_hidden_state.cpu().to(torch.float16)
            anc_embed = text_enc(anc_ids).last_hidden_state.cpu().to(torch.float16)
        
        # Token indices: skip SOS at 0, EOS, padding
        def content_ids(ids_row):
            out = []
            for i, t in enumerate(ids_row.tolist()):
                if t == sos_id or t == eos_id: continue
                if i == 0: continue
                out.append(i)
                if t == eos_id: break
            return out
        tgt_indices = content_ids(tgt_ids[0])
        anc_indices = content_ids(anc_ids[0])
        # Stop at first EOS
        def trim_to_eos(ids_row, indices):
            eos_pos = (ids_row == eos_id).nonzero()
            if len(eos_pos) > 0:
                eos_i = eos_pos[0].item()
                indices = [i for i in indices if i < eos_i]
            return indices
        tgt_indices = trim_to_eos(tgt_ids[0], tgt_indices)
        anc_indices = trim_to_eos(anc_ids[0], anc_indices)
        
        pack = {
            'target_clip_embeds': tgt_embed,
            'anchor_clip_embeds': anc_embed,
            'target_clip_features': target_clip_features,
            'anchor_clip_features': anchor_clip_features,
            'single_target_embeds': tgt_embed.clone(),
            'target_token_indices': tgt_indices,
            'anchor_token_indices': anc_indices,
            'family_names': family_names,
            'family_metadata': family_metadata,
            'config': {
                'concept': cat,
                'n_families': len(family_names),
                'n_per_family': len(families[family_names[0]]['target_prompts']),
                'grouped': True,
                'source': 'i2p_hard_v1',
            },
        }
        torch.save(pack, out_dir / 'clip_grouped.pt')
        print(f'  saved -> {out_dir}/clip_grouped.pt')
        print(f'  family_names={family_names}')
        print(f'  target_indices={tgt_indices} anchor_indices={anc_indices}')
    
    print('\n[done]')

if __name__ == '__main__':
    main()

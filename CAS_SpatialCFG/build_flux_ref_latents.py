#!/usr/bin/env python3
"""Encode existing exemplar PNGs to FLUX VAE latent space, save as ref tensors per concept.

Output: exemplars/i2p_v1_flux1/<concept>/ref_latents.pt    -- shape (M, 16, H/16, W/16) packed

Usage: python build_flux_ref_latents.py --cats violence self-harm ...
"""
import argparse, os, glob, torch
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
from diffusers import FluxPipeline

DEVICE = 'cuda:0'
ROOT = Path('/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/exemplars/i2p_v1_flux1')
FLUX_CKPT = 'black-forest-labs/FLUX.1-dev'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cats', nargs='+', default=None)
    ap.add_argument('--res', type=int, default=512)
    args = ap.parse_args()

    cats = args.cats or sorted([d.name for d in ROOT.iterdir() if d.is_dir() and d.name not in ('_logs',)])
    print(f'[FLUX VAE encode] cats={cats}')

    print('[load] FLUX pipeline (for VAE only)...')
    pipe = FluxPipeline.from_pretrained(FLUX_CKPT, torch_dtype=torch.bfloat16).to(DEVICE)
    vae = pipe.vae
    vae.eval()
    sf = vae.config.scaling_factor
    shift = vae.config.shift_factor

    tx = T.Compose([
        T.Resize(args.res, interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(args.res),
        T.ToTensor(),
        T.Normalize([0.5]*3, [0.5]*3),  # to [-1, 1]
    ])

    for cat in cats:
        cdir = ROOT / cat / 'images'
        if not cdir.exists():
            print(f'  SKIP missing: {cdir}'); continue
        # only target images (push away from these)
        target_imgs = sorted(glob.glob(str(cdir / '*_target_*.png')))
        if not target_imgs:
            target_imgs = sorted(glob.glob(str(cdir / '*.png')))
        print(f'  [{cat}] {len(target_imgs)} target imgs')
        if not target_imgs: continue

        latents_list = []
        with torch.no_grad():
            for p in target_imgs:
                img = tx(Image.open(p).convert('RGB')).unsqueeze(0).to(DEVICE, torch.bfloat16)
                lat = vae.encode(img).latent_dist.sample()
                lat = (lat - shift) * sf  # FLUX VAE post-encode
                latents_list.append(lat.cpu().float())
        latents = torch.cat(latents_list, 0)  # (M, 16, H/8, W/8)
        out_path = ROOT / cat / 'ref_latents.pt'
        torch.save(latents, out_path)
        print(f'  saved -> {out_path} shape={tuple(latents.shape)}')

    print('[done]')

if __name__ == '__main__':
    main()

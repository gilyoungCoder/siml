#!/usr/bin/env python3
"""Create K-image-per-family variants of an existing grouped concept pack.

No images are regenerated. We truncate/repeat the stored CLIP image features per
family so generation can vary image exemplar count K while all text/anchor/CAS
metadata and generation args stay fixed.
"""
import argparse, json
from pathlib import Path
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--src', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--k', type=int, required=True)
ap.add_argument('--repeat', action='store_true')
args = ap.parse_args()

data = torch.load(args.src, map_location='cpu', weights_only=False)
out = dict(data)
for field in ['target_clip_features', 'anchor_clip_features']:
    feats = data.get(field, {})
    new = {}
    for fam, tensor in feats.items():
        n = int(tensor.shape[0]) if hasattr(tensor, 'shape') and tensor.ndim >= 1 else 0
        if n == 0:
            new[fam] = tensor
        elif args.k <= n:
            new[fam] = tensor[:args.k].clone()
        elif args.repeat:
            reps = (args.k + n - 1) // n
            new[fam] = tensor.repeat((reps,) + (1,) * (tensor.ndim - 1))[:args.k].clone()
        else:
            new[fam] = tensor.clone()
    out[field] = new
cfg = dict(out.get('config', {}))
cfg.update({'image_saturation_k': args.k, 'image_saturation_src': args.src, 'image_saturation_repeat': bool(args.repeat)})
out['config'] = cfg
meta = dict(out.get('family_metadata', {}))
for fam, m in list(meta.items()):
    mm = dict(m)
    mm['image_saturation_k'] = args.k
    if 'n_images' in mm:
        mm['n_images_original'] = mm['n_images']
        mm['n_images'] = args.k
    meta[fam] = mm
out['family_metadata'] = meta
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
torch.save(out, args.out)
print(json.dumps({'out': args.out, 'k': args.k, 'families': list(out.get('target_clip_features', {}).keys())}, indent=2))

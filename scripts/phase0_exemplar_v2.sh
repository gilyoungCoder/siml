#!/bin/bash
# Phase 0-A: Generate v2 exemplar images + CLIP grouped embeddings
# Run on siml-01 GPU 0-2
set -e

cd /mnt/home3/yhgil99/unlearning

CONDA_ENV="sdd_copy"
PROMPTS_DIR="SafeGen/configs/exemplar_prompts_v2"
OUT_BASE="CAS_SpatialCFG/exemplars/concepts_v2"

source activate $CONDA_ENV 2>/dev/null || conda activate $CONDA_ENV

generate_exemplar() {
    local concept=$1
    local gpu=$2
    local target_file="${PROMPTS_DIR}/${concept}_target.txt"
    local anchor_file="${PROMPTS_DIR}/${concept}_anchor.txt"
    local out_dir="${OUT_BASE}/${concept}"
    local clip_out="${out_dir}/clip_grouped.pt"

    mkdir -p "${out_dir}/images"

    echo "[GPU $gpu] Generating exemplar for: $concept"

    CUDA_VISIBLE_DEVICES=$gpu python3 -c "
import torch
import torch.nn.functional as F
import random, os, json
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, DDIMScheduler

def set_seed(s):
    random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

device = torch.device('cuda')
pipe = StableDiffusionPipeline.from_pretrained(
    'CompVis/stable-diffusion-v1-4', torch_dtype=torch.float16, safety_checker=None
).to(device)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

# Read prompts (skip comment lines)
def read_prompts(path):
    return [l.strip() for l in open(path) if l.strip() and not l.startswith('#')]

target_prompts = read_prompts('${target_file}')
anchor_prompts = read_prompts('${anchor_file}')
print(f'Target: {len(target_prompts)}, Anchor: {len(anchor_prompts)}')

out_dir = '${out_dir}/images'
os.makedirs(out_dir, exist_ok=True)

# Generate target images
for i, p in enumerate(tqdm(target_prompts, desc='Target')):
    set_seed(42 + i)
    img = pipe(p, num_inference_steps=50, guidance_scale=7.5,
               generator=torch.Generator(device).manual_seed(42+i)).images[0]
    img.save(f'{out_dir}/target_{i:03d}.png')

# Generate anchor images
for i, p in enumerate(tqdm(anchor_prompts, desc='Anchor')):
    set_seed(42 + i)
    img = pipe(p, num_inference_steps=50, guidance_scale=7.5,
               generator=torch.Generator(device).manual_seed(42+i)).images[0]
    img.save(f'{out_dir}/anchor_{i:03d}.png')

# Extract CLIP features
from transformers import CLIPModel, CLIPProcessor
clip_model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14').to(device)
clip_proc = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14')
clip_model.eval()

def extract_features(img_paths):
    feats = []
    with torch.no_grad():
        for p in img_paths:
            inp = clip_proc(images=Image.open(p).convert('RGB'), return_tensors='pt').to(device)
            f = clip_model.get_image_features(**inp).float()
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu())
    return torch.cat(feats, dim=0)

import glob
target_imgs = sorted(glob.glob(f'{out_dir}/target_*.png'))
anchor_imgs = sorted(glob.glob(f'{out_dir}/anchor_*.png'))

target_feats = extract_features(target_imgs)
anchor_feats = extract_features(anchor_imgs)

# Group by family (4 prompts per family, 4 families)
n_per_family = 4
n_families = len(target_prompts) // n_per_family
families = ['family_0', 'family_1', 'family_2', 'family_3'][:n_families]

target_grouped = {}
anchor_grouped = {}
family_meta = {}

# Read family names from comment lines
with open('${target_file}') as f:
    lines = f.readlines()
fam_names = [l.strip().replace('# Family: ', '').split(' (')[0]
             for l in lines if l.startswith('# Family:')]
if len(fam_names) >= n_families:
    families = fam_names[:n_families]

for fi, fname in enumerate(families):
    s, e = fi * n_per_family, (fi + 1) * n_per_family
    target_grouped[fname] = target_feats[s:e]
    anchor_grouped[fname] = anchor_feats[s:e]

    # Read target/anchor words from concept pack if available
    family_meta[fname] = {
        'n_images': n_per_family,
        'target_prompts': target_prompts[s:e],
        'anchor_prompts': anchor_prompts[s:e],
    }

# Build grouped probe embeddings
te_model = pipe.text_encoder
tokenizer = pipe.tokenizer
dtype = next(te_model.parameters()).dtype

with torch.no_grad():
    empty_ids = tokenizer('', padding='max_length', max_length=77,
                          truncation=True, return_tensors='pt').input_ids.to(device)
    baseline = te_model(empty_ids)[0]

def build_grouped(feat_dict):
    result = baseline.clone()
    tok_idx = []
    for i, (fname, feats) in enumerate(feat_dict.items()):
        if i >= 4: break
        avg = F.normalize(feats.float().mean(dim=0), dim=-1)
        result[0, i+1] = avg.to(device=device, dtype=dtype)
        tok_idx.append(i+1)
    return result, tok_idx

target_embeds, target_idx = build_grouped(target_grouped)
anchor_embeds, anchor_idx = build_grouped(anchor_grouped)

# Also build single (all mean-pooled) for ablation comparison
all_target_avg = F.normalize(target_feats.float().mean(dim=0), dim=-1)
single_embeds = baseline.clone()
for i in range(1, 5):
    single_embeds[0, i] = all_target_avg.to(device=device, dtype=dtype)

# Save
torch.save({
    'target_clip_embeds': target_embeds.cpu().half(),
    'anchor_clip_embeds': anchor_embeds.cpu().half(),
    'target_clip_features': {k: v.cpu().half() for k, v in target_grouped.items()},
    'anchor_clip_features': {k: v.cpu().half() for k, v in anchor_grouped.items()},
    'single_target_embeds': single_embeds.cpu().half(),
    'target_token_indices': target_idx,
    'anchor_token_indices': anchor_idx,
    'family_names': families,
    'family_metadata': family_meta,
    'config': {
        'concept': '${concept}',
        'n_families': n_families,
        'n_per_family': n_per_family,
        'grouped': True,
    },
}, '${clip_out}')

# Also save single-mode .pt for backward compat
torch.save({
    'target_clip_features': target_feats.cpu().half(),
    'anchor_clip_features': anchor_feats.cpu().half(),
    'target_clip_embeds': single_embeds.cpu().half(),
    'config': {'concept': '${concept}', 'n_target': len(target_imgs), 'n_anchor': len(anchor_imgs)},
}, '${out_dir}/../clip_exemplar_projected.pt')

del clip_model, pipe
torch.cuda.empty_cache()
print(f'Saved: ${clip_out}')
print(f'  Families: {families}')
print(f'  Target features: {target_feats.shape}')
" 2>&1 | tee "${out_dir}/generate_log.txt"
}

# Dispatch across GPUs
generate_exemplar "sexual" 0 &
generate_exemplar "violent" 0 &
wait
generate_exemplar "disturbing" 1 &
generate_exemplar "illegal" 1 &
wait
generate_exemplar "harassment" 2 &
generate_exemplar "hate" 2 &
wait
generate_exemplar "selfharm" 2 &
wait

echo "=== All exemplar v2 generation complete ==="

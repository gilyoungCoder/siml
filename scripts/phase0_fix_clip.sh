#!/bin/bash
# Fix CLIP extraction + generate missing violent + create all .pt files
set -e
cd /mnt/home3/yhgil99/unlearning

eval "$(/usr/local/anaconda3/bin/conda shell.bash hook)"
conda activate sdd_copy

OUT_BASE="CAS_SpatialCFG/exemplars/concepts_v2"
PROMPTS_DIR="SafeGen/configs/exemplar_prompts_v2"

# 1. Generate missing violent images
echo "=== Generating violent exemplar images ==="
CUDA_VISIBLE_DEVICES=0 python3 -c "
import torch, random, os
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

def read_prompts(path):
    return [l.strip() for l in open(path) if l.strip() and not l.startswith('#')]

out_dir = '${OUT_BASE}/violent/images'
os.makedirs(out_dir, exist_ok=True)

if len([f for f in os.listdir(out_dir) if f.endswith('.png')]) >= 32:
    print('violent images already exist, skipping')
else:
    for role, pfile in [('target', '${PROMPTS_DIR}/violent_target.txt'), ('anchor', '${PROMPTS_DIR}/violent_anchor.txt')]:
        prompts = read_prompts(pfile)
        for i, p in enumerate(tqdm(prompts, desc=f'violent {role}')):
            set_seed(42 + i)
            img = pipe(p, num_inference_steps=50, guidance_scale=7.5,
                       generator=torch.Generator(device).manual_seed(42+i)).images[0]
            img.save(f'{out_dir}/{role}_{i:03d}.png')

del pipe
torch.cuda.empty_cache()
print('violent images done')
"

# 2. Extract CLIP features and create .pt for ALL concepts
echo "=== Extracting CLIP features for all concepts ==="
CUDA_VISIBLE_DEVICES=0 python3 -c "
import torch, os, glob
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
from diffusers import StableDiffusionPipeline

device = torch.device('cuda')

# Load CLIP
clip_model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14').to(device)
clip_proc = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14')
clip_model.eval()

# Load SD text encoder for embedding projection
pipe = StableDiffusionPipeline.from_pretrained(
    'CompVis/stable-diffusion-v1-4', torch_dtype=torch.float16, safety_checker=None
).to(device)
te = pipe.text_encoder
tok = pipe.tokenizer
dtype = next(te.parameters()).dtype

with torch.no_grad():
    empty_ids = tok('', padding='max_length', max_length=77,
                    truncation=True, return_tensors='pt').input_ids.to(device)
    baseline_emb = te(empty_ids)[0]

def extract_features(img_paths):
    feats = []
    with torch.no_grad():
        for p in img_paths:
            inp = clip_proc(images=Image.open(p).convert('RGB'), return_tensors='pt').to(device)
            out = clip_model.get_image_features(**inp)
            # Handle both tensor and model output
            if hasattr(out, 'pooler_output'):
                f = out.pooler_output.float()
            elif not isinstance(out, torch.Tensor):
                f = out[0].float() if isinstance(out, tuple) else out.float()
            else:
                f = out.float()
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu())
    return torch.cat(feats, dim=0)

def build_grouped(feat_dict, max_tokens=4):
    result = baseline_emb.clone()
    tok_idx = []
    names = list(feat_dict.keys())[:max_tokens]
    for i, fname in enumerate(names):
        avg = F.normalize(feat_dict[fname].float().mean(dim=0), dim=-1)
        result[0, i+1] = avg.to(device=device, dtype=dtype)
        tok_idx.append(i+1)
    # pad remaining
    if names:
        last = F.normalize(feat_dict[names[-1]].float().mean(dim=0), dim=-1)
        for i in range(len(names), max_tokens):
            result[0, i+1] = last.to(device=device, dtype=dtype)
    return result, tok_idx, names

def read_prompts(path):
    return [l.strip() for l in open(path) if l.strip() and not l.startswith('#')]

def read_family_names(path):
    return [l.strip().replace('# Family: ', '').split(' (')[0]
            for l in open(path) if l.startswith('# Family:')]

base = '${OUT_BASE}'
prompts_dir = '${PROMPTS_DIR}'

concepts = ['sexual', 'violent', 'disturbing', 'illegal', 'harassment', 'hate', 'selfharm']

for concept in concepts:
    img_dir = f'{base}/{concept}/images'
    target_imgs = sorted(glob.glob(f'{img_dir}/target_*.png'))
    anchor_imgs = sorted(glob.glob(f'{img_dir}/anchor_*.png'))

    if not target_imgs:
        print(f'SKIP {concept}: no images')
        continue

    print(f'\n=== {concept}: {len(target_imgs)} target, {len(anchor_imgs)} anchor ===')

    target_feats = extract_features(target_imgs)
    anchor_feats = extract_features(anchor_imgs)

    # Group by family (4 per family)
    n_per = 4
    n_fam = len(target_imgs) // n_per

    # Read family names from prompts
    tgt_prompt_file = f'{prompts_dir}/{concept}_target.txt'
    fam_names = read_family_names(tgt_prompt_file) if os.path.exists(tgt_prompt_file) else []
    if len(fam_names) < n_fam:
        fam_names = [f'family_{i}' for i in range(n_fam)]

    target_grouped = {}
    anchor_grouped = {}
    family_meta = {}

    tgt_prompts = read_prompts(tgt_prompt_file) if os.path.exists(tgt_prompt_file) else []
    anc_prompt_file = f'{prompts_dir}/{concept}_anchor.txt'
    anc_prompts = read_prompts(anc_prompt_file) if os.path.exists(anc_prompt_file) else []

    for fi in range(n_fam):
        fname = fam_names[fi]
        s, e = fi * n_per, (fi + 1) * n_per
        target_grouped[fname] = target_feats[s:e]
        anchor_grouped[fname] = anchor_feats[s:e]
        family_meta[fname] = {
            'n_images': n_per,
            'target_prompts': tgt_prompts[s:e] if s < len(tgt_prompts) else [],
            'anchor_prompts': anc_prompts[s:e] if s < len(anc_prompts) else [],
        }

    # Build grouped embeddings
    target_embeds, target_idx, ordered_names = build_grouped(target_grouped)
    anchor_embeds, anchor_idx, _ = build_grouped(anchor_grouped)

    # Build single (all mean-pooled)
    all_avg = F.normalize(target_feats.float().mean(dim=0), dim=-1)
    single_embeds = baseline_emb.clone()
    for i in range(1, 5):
        single_embeds[0, i] = all_avg.to(device=device, dtype=dtype)

    # Save grouped .pt
    grouped_path = f'{base}/{concept}/clip_grouped.pt'
    torch.save({
        'target_clip_embeds': target_embeds.cpu().half(),
        'anchor_clip_embeds': anchor_embeds.cpu().half(),
        'target_clip_features': {k: v.cpu().half() for k, v in target_grouped.items()},
        'anchor_clip_features': {k: v.cpu().half() for k, v in anchor_grouped.items()},
        'single_target_embeds': single_embeds.cpu().half(),
        'target_token_indices': target_idx,
        'anchor_token_indices': anchor_idx,
        'family_names': ordered_names,
        'family_metadata': family_meta,
        'config': {'concept': concept, 'n_families': n_fam, 'n_per_family': n_per, 'grouped': True},
    }, grouped_path)

    # Save single .pt (backward compat)
    single_path = f'{base}/{concept}/clip_exemplar_projected.pt'
    torch.save({
        'target_clip_features': target_feats.cpu().half(),
        'anchor_clip_features': anchor_feats.cpu().half(),
        'target_clip_embeds': single_embeds.cpu().half(),
        'config': {'concept': concept, 'n_target': len(target_imgs), 'n_anchor': len(anchor_imgs)},
    }, single_path)

    print(f'  Saved: {grouped_path}')
    print(f'  Saved: {single_path}')
    print(f'  Families: {ordered_names}')
    sim = F.cosine_similarity(
        F.normalize(target_feats.mean(0, keepdim=True)),
        F.normalize(anchor_feats.mean(0, keepdim=True))
    ).item()
    print(f'  Target-Anchor cosine sim: {sim:.4f}')

del clip_model, pipe
torch.cuda.empty_cache()
print('\n=== All CLIP extraction complete ===')
"

echo "=== Phase 0 fix complete ==="

#!/usr/bin/env python3
"""Generate 32 imgs per family for sexual concept, encode CLIP, build extended pack.
Then SR-evaluate K∈{8,16,32} via EBSG sweep.
"""
import os, sys, json
from pathlib import Path
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import torchvision.transforms as T

REPO = "/mnt/home3/yhgil99/unlearning"
SRC_PACK = f"{REPO}/CAS_SpatialCFG/exemplars/i2p_v1/sexual/clip_grouped.pt"
OUT_DIR = f"{REPO}/CAS_SpatialCFG/launch_0426_full_sweep/exemplars_K_extended"
EXEMPLAR_IMG_DIR = f"{OUT_DIR}/imgs"
DEVICE = "cuda:0"
N_PER_FAMILY = 32
SEED_BASE = 1000

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
Path(EXEMPLAR_IMG_DIR).mkdir(parents=True, exist_ok=True)

# Load original pack to get family target_words
src = torch.load(SRC_PACK, map_location="cpu", weights_only=False)
fm = src.get("family_metadata", {})
families = list(fm.keys())
print(f"Families: {families}")

# Each family has 4 target_words. Generate N_PER_FAMILY = 32 imgs by cycling words and varying seeds.
gen_plans = {}
for f_idx, fname in enumerate(families):
    tws = fm[fname].get("target_words", []) or fm[fname].get("target_prompts", [])
    if not tws:
        print(f"  WARN family {fname} has no target_words, using concept_keywords")
        tws = src.get("concept_keywords", ["nude", "naked"])
    plans = []
    for i in range(N_PER_FAMILY):
        kw = tws[i % len(tws)]
        seed = SEED_BASE + f_idx * N_PER_FAMILY + i
        plans.append((kw, seed))
    gen_plans[fname] = plans
    print(f"  {fname}: {N_PER_FAMILY} plans (e.g., {plans[:2]})")

# Load SD1.4 + CLIP
print("Loading SD1.4...")
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16, revision="fp16",
    safety_checker=None, requires_safety_checker=False,
).to(DEVICE)
pipe.set_progress_bar_config(disable=True)

print("Loading CLIP...")
import clip
clip_model, clip_preprocess = clip.load("ViT-L/14", device=DEVICE)
clip_model.eval()

# Generate + encode
new_target_features = {}
gen = torch.Generator(device=DEVICE)
with torch.no_grad():
    for fname, plans in gen_plans.items():
        feats = []
        for img_idx, (kw, seed) in enumerate(plans):
            gen.manual_seed(seed)
            prompt = kw.replace("_", " ")
            try:
                img = pipe(prompt, num_inference_steps=50, guidance_scale=7.5,
                           generator=gen, height=512, width=512).images[0]
            except Exception as e:
                print(f"  err {fname}/{img_idx}: {e}")
                continue
            img.save(f"{EXEMPLAR_IMG_DIR}/{fname}_{img_idx:02d}_{kw}_{seed}.png")
            ten = clip_preprocess(img).unsqueeze(0).to(DEVICE)
            f = clip_model.encode_image(ten).float()
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f[0].cpu())
            if (img_idx+1) % 8 == 0:
                print(f"    {fname}: {img_idx+1}/{N_PER_FAMILY} done")
        new_target_features[fname] = torch.stack(feats, dim=0)
        print(f"  {fname}: shape {new_target_features[fname].shape}")

# Build extended pack: K=N_PER_FAMILY entries per family
new = {k: (v.clone() if torch.is_tensor(v) else v) for k,v in src.items()}
new["target_clip_features"] = new_target_features
# anchor_clip_features: keep original (only K=4 anchor exemplars; not extending anchor)
out_pack = f"{OUT_DIR}/clip_grouped_K{N_PER_FAMILY}.pt"
torch.save(new, out_pack)
print(f"saved extended pack: {out_pack}")

# Build subsampled packs at K∈{8, 16}
import random
random.seed(42)
for K in [8, 16]:
    sub = {k: (v.clone() if torch.is_tensor(v) else v) for k,v in new.items()}
    sub_target = {}
    for fname, feats in new["target_clip_features"].items():
        N = feats.shape[0]
        idx = sorted(random.sample(range(N), min(K, N)))
        sub_target[fname] = feats[idx].clone()
    sub["target_clip_features"] = sub_target
    out = f"{OUT_DIR}/clip_grouped_K{K}.pt"
    torch.save(sub, out)
    print(f"saved subsampled K={K}: {out}")

print("Done. Run EBSG sweep with packs at K∈{8,16,32}.")

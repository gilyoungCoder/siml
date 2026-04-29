"""Build K=16 extended exemplar pack for given concept.
Then subsample to K=8, K=12.

Usage: python build_extended_concept.py <concept> <gpu>
"""
import os, sys, json
from pathlib import Path
import torch
from diffusers import StableDiffusionPipeline
import torchvision.transforms as T
import random

CONCEPT = sys.argv[1]
GPU = int(sys.argv[2])
DEVICE = f"cuda:{GPU}"
N_PER_FAMILY = 16
SEED_BASE = 2000

REPO = "/mnt/home3/yhgil99/unlearning"
SRC_PACK = f"{REPO}/CAS_SpatialCFG/exemplars/i2p_v1/{CONCEPT}/clip_grouped.pt"
OUT_DIR = f"{REPO}/CAS_SpatialCFG/launch_0426_full_sweep/exemplars_K_per_concept/{CONCEPT}"
EXEMPLAR_IMG_DIR = f"{OUT_DIR}/imgs"
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
Path(EXEMPLAR_IMG_DIR).mkdir(parents=True, exist_ok=True)

print(f"[{CONCEPT}] DEVICE={DEVICE} N_PER_FAMILY={N_PER_FAMILY}")
src = torch.load(SRC_PACK, map_location="cpu", weights_only=False)
fm = src.get("family_metadata", {})
families = list(fm.keys())
print(f"[{CONCEPT}] Families: {families}")

gen_plans = {}
for f_idx, fname in enumerate(families):
    tws = fm[fname].get("target_words", []) or fm[fname].get("target_prompts", [])
    if not tws:
        print(f"  WARN family {fname} has no target_words, using concept_keywords")
        tws = src.get("concept_keywords", [CONCEPT])
    plans = []
    for i in range(N_PER_FAMILY):
        kw = tws[i % len(tws)]
        seed = SEED_BASE + f_idx * N_PER_FAMILY + i
        plans.append((kw, seed))
    gen_plans[fname] = plans

print(f"[{CONCEPT}] Loading SD1.4...")
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16, revision="fp16",
    safety_checker=None, requires_safety_checker=False,
).to(DEVICE)
pipe.set_progress_bar_config(disable=True)

print(f"[{CONCEPT}] Loading CLIP...")
import clip
clip_model, clip_preprocess = clip.load("ViT-L/14", device=DEVICE)
clip_model.eval()

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
            img.save(f"{EXEMPLAR_IMG_DIR}/{fname}_{img_idx:02d}_{kw.replace(\"/\", \"_\")}_{seed}.png")
            ten = clip_preprocess(img).unsqueeze(0).to(DEVICE)
            f = clip_model.encode_image(ten).float()
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f[0].cpu())
        new_target_features[fname] = torch.stack(feats, dim=0)
        print(f"[{CONCEPT}]   {fname}: shape {new_target_features[fname].shape}")

# Save K=16 pack
new = {k: (v.clone() if torch.is_tensor(v) else v) for k,v in src.items()}
new["target_clip_features"] = new_target_features
torch.save(new, f"{OUT_DIR}/clip_grouped_K{N_PER_FAMILY}.pt")
print(f"[{CONCEPT}] saved K={N_PER_FAMILY}: {OUT_DIR}/clip_grouped_K{N_PER_FAMILY}.pt")

# Subsample K=8, K=12 from K=16
random.seed(42)
for K in [8, 12]:
    sub = {k: (v.clone() if torch.is_tensor(v) else v) for k,v in new.items()}
    sub_t = {}
    for fname, feats in new["target_clip_features"].items():
        N = feats.shape[0]
        idx = sorted(random.sample(range(N), min(K, N)))
        sub_t[fname] = feats[idx].clone()
    sub["target_clip_features"] = sub_t
    torch.save(sub, f"{OUT_DIR}/clip_grouped_K{K}.pt")
    print(f"[{CONCEPT}] saved K={K}")

print(f"[{CONCEPT}] DONE")

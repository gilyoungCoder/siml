#!/usr/bin/env python3
"""Wrapper — FID (pytorch_fid) + CLIP score (ViT-L/14) for COCO 10k cells.
Computes:
  - FID(baseline vs anchor), FID(baseline vs hybrid)
  - CLIP score per (baseline, anchor, hybrid)
Usage: python run_fid_clip.py
"""
import sys, os, glob
sys.path.insert(0, "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG")
import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torchmetrics.multimodal import CLIPScore
from pytorch_fid.fid_score import calculate_fid_given_paths

ROOT = "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0424_coco10k"
PROMPTS_TXT = "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/prompts/coco_10k.txt"
BASE = f"{ROOT}/baseline_sd14"
ANCHOR = f"{ROOT}/ours_anchor_v2pack"
HYBRID = f"{ROOT}/ours_hybrid_v1pack"

device = "cuda"
dims = 2048
batch_size = 64

# Load prompts (1 sample per prompt since our gen is 1-sample)
prompts = [l.strip() for l in open(PROMPTS_TXT) if l.strip()]
print(f"Loaded {len(prompts)} COCO prompts")

def imgs_sorted(d):
    out = []
    for ext in ["*.png","*.jpg","*.jpeg"]:
        out += glob.glob(os.path.join(d, ext))
    return sorted(out)

def clip_score(img_dir, prompts, metric):
    imgs = imgs_sorted(img_dir)
    n = min(len(imgs), len(prompts))
    imgs = imgs[:n]; prompts = prompts[:n]
    to_tensor = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    scores = []
    for i in range(0, n, batch_size):
        bi = imgs[i:i+batch_size]
        bp = prompts[i:i+batch_size]
        t = torch.stack([to_tensor(Image.open(p).convert("RGB")) for p in bi]).to(device)
        t8 = (t*255).to(torch.uint8)
        s = metric(t8, bp)
        scores.append(s.item())
        if (i//batch_size) % 20 == 0:
            print(f"  CLIP batch {i//batch_size+1}/{(n+batch_size-1)//batch_size}", flush=True)
    return sum(scores)/len(scores)

# CLIP metric (shared across runs)
print("Loading CLIP metric (ViT-L/14)...")
metric = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14").to(device)

results = {}
for label, d in [("baseline", BASE), ("anchor", ANCHOR), ("hybrid", HYBRID)]:
    n = len(imgs_sorted(d))
    print(f"\n=== {label} ({d}: {n} imgs) ===")
    if n == 0:
        print(f"  SKIP — no images"); continue
    c = clip_score(d, prompts, metric)
    print(f"  CLIP score = {c:.4f}")
    results[label+"_clip"] = c

# FID: baseline vs anchor, baseline vs hybrid
del metric  # free memory
torch.cuda.empty_cache()

for method in [("anchor", ANCHOR), ("hybrid", HYBRID)]:
    label, d = method
    print(f"\n=== FID baseline vs {label} ===")
    try:
        fid = calculate_fid_given_paths([BASE, d], batch_size=batch_size, device=device, dims=dims, num_workers=4)
        print(f"  FID = {fid:.4f}")
        results["fid_baseline_vs_"+label] = fid
    except Exception as e:
        print(f"  FAILED: {e}")

print("\n=== FINAL ===")
for k, v in results.items(): print(f"  {k} = {v:.4f}")

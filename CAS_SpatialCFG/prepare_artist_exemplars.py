#!/usr/bin/env python
"""
Prepare artist style exemplar images + CLIP embeddings for v27 style removal.

For each artist:
  Target: generate 16 images "in the style of [Artist]" → CLIP features
  Anchor: generate 16 images "a painting of [subject]" (no artist) → CLIP features

Usage:
  CUDA_VISIBLE_DEVICES=0 python prepare_artist_exemplars.py --artist vangogh
  CUDA_VISIBLE_DEVICES=0 python prepare_artist_exemplars.py --all
"""

import os, sys, torch, torch.nn.functional as F, random
from pathlib import Path
from argparse import ArgumentParser
from PIL import Image
from tqdm import tqdm

REPO = Path(__file__).parent.parent
PROMPT_DIR = REPO / "CAS_SpatialCFG" / "prompts" / "artists"
EXEMPLAR_DIR = REPO / "CAS_SpatialCFG" / "exemplars" / "artists"

ARTISTS = {
    "vangogh": {
        "name": "Vincent van Gogh",
        "target_concepts": ["Van Gogh style", "painting by Van Gogh", "Van Gogh brushstrokes"],
        "anchor_concepts": ["painting", "artwork", "oil painting"],
        "target_prompts": [
            "A swirling night sky with thick impasto brushstrokes, in the style of Van Gogh",
            "Sunflowers in a vase with bold yellow colors, painted by Vincent van Gogh",
            "A cypress tree with swirling blue sky, Van Gogh style",
            "A wheat field with dramatic clouds, in the style of Van Gogh",
            "A cafe terrace at night with warm lights, Van Gogh painting",
            "A starry night over a river with reflections, Van Gogh style",
            "Irises in vivid purple and blue, painted by Van Gogh",
            "A bedroom with simple furniture in bold colors, Van Gogh style",
            "Self-portrait with thick brushstrokes and vivid colors, Van Gogh",
            "A country road with cypress trees, in the style of Vincent van Gogh",
            "An orchard in bloom with pink blossoms, Van Gogh style",
            "A portrait with swirling background, painted by Van Gogh",
            "A bridge over a pond with bold colors, Van Gogh style",
            "A harvest scene with golden wheat, Van Gogh painting",
            "Almond blossoms on blue background, in the style of Van Gogh",
            "A church at night with glowing stars, Van Gogh style",
        ],
        "anchor_prompts": [
            "A night sky with stars over a village, oil painting",
            "Sunflowers in a vase, classical still life painting",
            "A cypress tree in a landscape, traditional oil painting",
            "A wheat field under cloudy sky, landscape painting",
            "A cafe terrace scene, realistic painting",
            "A river at night with city lights, painting",
            "Purple irises in a garden, botanical painting",
            "A simple bedroom interior, realistic painting",
            "A self-portrait, traditional portrait painting",
            "A country road with trees, landscape artwork",
            "An orchard with blossoms, spring landscape painting",
            "A portrait with neutral background, oil painting",
            "A bridge over water, classical landscape",
            "A harvest scene, pastoral painting",
            "Blossoms on branches, decorative painting",
            "A church building, architectural painting",
        ],
    },
    "monet": {
        "name": "Claude Monet",
        "target_concepts": ["Monet style", "painting by Claude Monet", "impressionist Monet"],
        "anchor_concepts": ["painting", "artwork", "oil painting"],
        "target_prompts": [
            "Water lilies on a serene pond, in the style of Claude Monet",
            "A misty sunrise over a harbor, Monet impressionist style",
            "A cathedral in changing light, painted by Monet",
            "A woman with parasol in a meadow, Monet style",
            "A Japanese bridge over lily pond, Claude Monet painting",
            "Haystacks in golden sunlight, impressionist Monet style",
            "A garden with colorful flowers, painted by Claude Monet",
            "Poppies in a field with figures, Monet impressionist",
            "A train station with steam and light, Monet style",
            "Cliffs at Etretat by the sea, Claude Monet painting",
            "A path through a garden with flowers, Monet style",
            "A frozen river in winter, impressionist Monet",
            "Boats on the Seine at sunset, Claude Monet style",
            "A field of tulips in Holland, Monet painting",
            "Morning fog over a river, impressionist style of Monet",
            "A woman in a garden with umbrella, Claude Monet",
        ],
        "anchor_prompts": [
            "Water lilies on a pond, realistic painting",
            "A sunrise over a harbor, landscape painting",
            "A cathedral building, architectural painting",
            "A woman walking in a meadow, figure painting",
            "A bridge over a pond, landscape artwork",
            "Haystacks in a field, pastoral painting",
            "A garden with flowers, botanical painting",
            "Poppies in a field, landscape painting",
            "A train station interior, realistic painting",
            "Coastal cliffs by the sea, landscape painting",
            "A garden path with flowers, nature painting",
            "A winter river scene, landscape artwork",
            "Boats on a river at sunset, marine painting",
            "A tulip field, landscape painting",
            "Morning fog over water, atmospheric painting",
            "A woman in a garden, figure painting",
        ],
    },
    "picasso": {
        "name": "Pablo Picasso",
        "target_concepts": ["Picasso style", "cubist Picasso", "painting by Pablo Picasso"],
        "anchor_concepts": ["painting", "artwork", "modern art"],
        "target_prompts": [
            "A woman with fragmented geometric features, in the style of Picasso",
            "A guitar on a table in cubist style, Pablo Picasso",
            "A weeping woman with angular face, Picasso painting",
            "A bull in abstract geometric shapes, Picasso style",
            "Two dancers with cubist forms, painted by Picasso",
            "A face shown from multiple angles, Picasso cubism",
            "A still life with fruit in cubist style, Pablo Picasso",
            "A dove with olive branch, in the style of Picasso",
            "A seated woman with distorted proportions, Picasso",
            "Musicians with instruments in cubist style, Picasso",
            "A harlequin figure with geometric patterns, Picasso style",
            "A mother and child in blue tones, Pablo Picasso",
            "Abstract geometric portrait, in the style of Picasso",
            "A horse in angular cubist forms, Picasso painting",
            "A mandolin and sheet music, Picasso cubism",
            "A profile and frontal view combined, Picasso style",
        ],
        "anchor_prompts": [
            "A woman portrait, modern painting",
            "A guitar on a table, still life painting",
            "A sad woman portrait, expressive painting",
            "A bull illustration, animal art",
            "Two dancers, figure painting",
            "A face portrait, modern art",
            "A still life with fruit, painting",
            "A dove bird, simple illustration",
            "A seated woman, figure painting",
            "Musicians playing instruments, painting",
            "A harlequin figure, theatrical art",
            "A mother and child, portrait painting",
            "An abstract portrait, modern art",
            "A horse illustration, animal art",
            "Musical instruments, still life art",
            "A portrait, modern painting",
        ],
    },
}

# Add hopper and kinkade with simpler configs
for artist, name in [("hopper", "Edward Hopper"), ("kinkade", "Thomas Kinkade")]:
    ARTISTS[artist] = {
        "name": name,
        "target_concepts": [f"{name} style", f"painting by {name}", f"in the style of {name}"],
        "anchor_concepts": ["painting", "artwork", "oil painting"],
        "target_prompts": None,  # Will use prompt file
        "anchor_prompts": None,
    }


def extract_clip_features(images, device):
    from transformers import CLIPModel, CLIPProcessor
    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    clip.eval()
    features = []
    with torch.no_grad():
        for img in tqdm(images, desc="  CLIP"):
            inputs = proc(images=img, return_tensors="pt").to(device)
            out = clip.get_image_features(**inputs)
            if isinstance(out, torch.Tensor):
                feats = out.float()
            else:
                feats = out.pooler_output.float() if hasattr(out, 'pooler_output') else out[0].float()
            feats = feats / feats.norm(dim=-1, keepdim=True)
            features.append(feats.cpu())
    del clip; torch.cuda.empty_cache()
    return torch.cat(features, dim=0)


def prepare_artist(artist, n_images=16):
    device = torch.device("cuda")
    info = ARTISTS[artist]
    out_dir = EXEMPLAR_DIR / artist
    out_dir.mkdir(parents=True, exist_ok=True)

    pt_file = out_dir / "clip_exemplar.pt"
    if pt_file.exists():
        print(f"SKIP: {pt_file}")
        return

    print(f"\n{'='*60}")
    print(f"Preparing: {info['name']}")
    print(f"{'='*60}")

    from diffusers import StableDiffusionPipeline, DDIMScheduler
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16,
        safety_checker=None, feature_extractor=None).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # Target prompts
    if info["target_prompts"]:
        target_prompts = info["target_prompts"][:n_images]
    else:
        # Load from file
        pf = PROMPT_DIR / f"{artist}.txt"
        target_prompts = [l.strip() for l in open(pf) if l.strip()][:n_images]

    # Anchor prompts
    if info["anchor_prompts"]:
        anchor_prompts = info["anchor_prompts"][:n_images]
    else:
        # Generic anchors from target prompts (remove artist name)
        anchor_prompts = []
        for p in target_prompts:
            clean = p.replace(f"in the style of {info['name']}", "painting")
            clean = clean.replace(f"style of {info['name']}", "painting")
            clean = clean.replace(info['name'], "")
            anchor_prompts.append(clean.strip())

    print(f"  Target prompts: {len(target_prompts)}")
    print(f"  Anchor prompts: {len(anchor_prompts)}")

    # Generate
    img_dir = out_dir / "images"
    img_dir.mkdir(exist_ok=True)

    target_imgs, anchor_imgs = [], []
    random.seed(42)
    for i, p in enumerate(tqdm(target_prompts, desc="  Gen target")):
        img = pipe(p, num_inference_steps=50, guidance_scale=7.5,
                   generator=torch.Generator(device=device).manual_seed(42+i)).images[0]
        img.save(img_dir / f"target_{i:03d}.png")
        target_imgs.append(img)

    for i, p in enumerate(tqdm(anchor_prompts, desc="  Gen anchor")):
        img = pipe(p, num_inference_steps=50, guidance_scale=7.5,
                   generator=torch.Generator(device=device).manual_seed(42+i)).images[0]
        img.save(img_dir / f"anchor_{i:03d}.png")
        anchor_imgs.append(img)

    del pipe; torch.cuda.empty_cache()

    # CLIP features
    print("  Extracting CLIP features...")
    target_features = extract_clip_features(target_imgs, device)
    anchor_features = extract_clip_features(anchor_imgs, device)

    save_dict = {
        "target_clip_features": target_features.half(),
        "anchor_clip_features": anchor_features.half(),
        "config": {
            "artist": artist,
            "name": info["name"],
            "n_target": len(target_imgs),
            "n_anchor": len(anchor_imgs),
            "target_concepts": info["target_concepts"],
            "anchor_concepts": info["anchor_concepts"],
        }
    }
    torch.save(save_dict, pt_file)
    print(f"  Saved: {pt_file}")


def main():
    p = ArgumentParser()
    p.add_argument("--artist", default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--n_images", type=int, default=16)
    args = p.parse_args()

    if args.all:
        for a in ARTISTS:
            prepare_artist(a, args.n_images)
    elif args.artist:
        prepare_artist(args.artist, args.n_images)
    else:
        p.error("--artist or --all")


if __name__ == "__main__":
    main()

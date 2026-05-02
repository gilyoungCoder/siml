#!/usr/bin/env python
"""Regenerate ALL artist anchor images with consistent 'a painting of ~' prompts."""

import torch, random
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch.nn.functional as F

EXEMPLAR_DIR = Path("/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/exemplars/artists")

# Consistent anchor prompts: "a painting of [subject]" — NO artist, NO style keywords
ANCHOR_PROMPTS = {
    "vangogh": [
        "a painting of a night sky with stars over a village",
        "a painting of sunflowers in a vase",
        "a painting of purple irises in a garden",
        "a painting of a cafe terrace at night",
        "a painting of a bedroom with simple furniture",
        "a painting of a wheat field under dramatic clouds",
        "a painting of a man with a bandaged ear, portrait",
        "a painting of peasants eating at a table by candlelight",
        "a painting of a doctor holding a flower, portrait",
        "a painting of a cypress tree against a blue sky",
        "a painting of an orchard with pink blossoms in spring",
        "a painting of a portrait with colorful background",
        "a painting of a stone bridge over a pond",
        "a painting of a harvest scene in golden fields",
        "a painting of almond blossoms on blue background",
        "a painting of a church building at night",
    ],
    "monet": [
        "a painting of water lilies on a calm pond",
        "a painting of a sunrise over a misty harbor",
        "a painting of a cathedral facade in warm light",
        "a painting of a woman with a parasol in a meadow",
        "a painting of a wooden bridge over a lily pond",
        "a painting of haystacks in a golden field",
        "a painting of a garden path with colorful flowers",
        "a painting of a poppy field with walking figures",
        "a painting of a train station with steam",
        "a painting of coastal cliffs by the sea",
        "a painting of a flower garden with a path",
        "a painting of a frozen river in winter",
        "a painting of boats on a river at sunset",
        "a painting of a tulip field in spring",
        "a painting of morning fog over a river",
        "a painting of a woman in a garden",
    ],
    "picasso": [
        "a painting of a woman portrait, modern art",
        "a painting of a guitar on a table, still life",
        "a painting of a sad woman crying, portrait",
        "a painting of a bull, animal illustration",
        "a painting of two dancers, figure art",
        "a painting of a face portrait, modern style",
        "a painting of a still life with fruit and bottle",
        "a painting of a dove with olive branch",
        "a painting of a seated woman in an armchair",
        "a painting of musicians playing instruments",
        "a painting of a harlequin figure, theatrical art",
        "a painting of a mother holding a child, blue tones",
        "a painting of an abstract portrait, modern art",
        "a painting of a horse, animal art",
        "a painting of a mandolin and sheet music",
        "a painting of a face portrait from the side",
    ],
    "hopper": [
        "a painting of a woman sitting alone in a diner at night",
        "a painting of a lighthouse on a hill with long shadows",
        "a painting of a gas station on a lonely road at dusk",
        "a painting of a woman reading by a window with sunlight",
        "a painting of an empty theater with dramatic lighting",
        "a painting of a couple in a restaurant",
        "a painting of a solitary figure on a porch at sunset",
        "a painting of brownstone buildings with strong shadows",
        "a painting of an office at night seen through a window",
        "a painting of a seaside house in bright sunlight",
        "a painting of a barber shop with strong light and shadow",
        "a painting of a train car interior with passengers",
        "a painting of a hotel lobby with a lone figure",
        "a painting of a cape cod house in afternoon sun",
        "a painting of a movie theater interior",
        "a painting of a drugstore at night with neon lights",
    ],
    "kinkade": [
        "a painting of a cottage with glowing windows at twilight",
        "a painting of a stone bridge over a stream in autumn",
        "a painting of a garden path leading to a charming house",
        "a painting of a lighthouse at sunset",
        "a painting of a village with Christmas lights in snow",
        "a painting of a chapel in the woods with sunbeams",
        "a painting of a waterfall in a lush forest",
        "a painting of a Victorian house with a white fence",
        "a painting of a country road through autumn foliage",
        "a painting of a seaside cottage at sunset",
        "a painting of a mountain cabin by a lake",
        "a painting of a covered bridge in fall colors",
        "a painting of a garden gate with climbing roses",
        "a painting of a windmill by a tulip field",
        "a painting of a cobblestone street in a European village",
        "a painting of a treehouse in a magical forest",
    ],
}


def extract_clip(images, device):
    from transformers import CLIPModel, CLIPProcessor
    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    clip.eval()
    features = []
    with torch.no_grad():
        for img in tqdm(images, desc="  CLIP"):
            inputs = proc(images=img, return_tensors="pt").to(device)
            out = clip.get_image_features(**inputs)
            feats = out.float() if isinstance(out, torch.Tensor) else out.pooler_output.float()
            feats = feats / feats.norm(dim=-1, keepdim=True)
            features.append(feats.cpu())
    del clip; torch.cuda.empty_cache()
    return torch.cat(features, dim=0)


def regen_artist(artist):
    device = torch.device("cuda")
    prompts = ANCHOR_PROMPTS[artist]
    img_dir = EXEMPLAR_DIR / artist / "images"
    pt_file = EXEMPLAR_DIR / artist / "clip_exemplar.pt"

    print(f"\n=== Regenerating anchors for {artist} ===")

    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16,
        safety_checker=None, feature_extractor=None).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    anchor_imgs = []
    for i, p in enumerate(tqdm(prompts, desc=f"  Gen {artist} anchors")):
        img = pipe(p, num_inference_steps=50, guidance_scale=7.5,
                   generator=torch.Generator(device=device).manual_seed(42+i)).images[0]
        img.save(img_dir / f"anchor_{i:03d}.png")
        anchor_imgs.append(img)

    del pipe; torch.cuda.empty_cache()

    # Re-extract CLIP for anchors, keep existing targets
    print("  Re-extracting anchor CLIP features...")
    anchor_features = extract_clip(anchor_imgs, device)

    # Load existing .pt, update anchor features only
    data = torch.load(pt_file, map_location="cpu")
    data["anchor_clip_features"] = anchor_features.half()
    torch.save(data, pt_file)
    print(f"  Updated {pt_file}")


if __name__ == "__main__":
    import sys
    artist = sys.argv[1] if len(sys.argv) > 1 else "all"
    if artist == "all":
        for a in ANCHOR_PROMPTS:
            regen_artist(a)
    else:
        regen_artist(artist)

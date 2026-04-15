#!/usr/bin/env python
"""
SAFREE on SD3: Negative prompt space approach for concept erasure.

Uses the SAFREE negative-prompt-space method adapted for SD3's pipeline.
Each concept gets its own set of harmful keywords as the negative prompt.
"""

import os, sys, json, csv, re
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Dict

import torch
from PIL import Image
from tqdm import tqdm

# Add SAFREE sdv3 path for custom pipeline
SAFREE_SDV3_DIR = os.path.join(os.path.dirname(__file__), "../../SAFREE/sdv3")
sys.path.insert(0, SAFREE_SDV3_DIR)

from diffusers import StableDiffusion3Pipeline

SD3_MODEL_ID = "stabilityai/stable-diffusion-3-medium-diffusers"

# ===============================
# Concept-specific harmful keywords (from SAFREE / gen_safree_i2p_concepts.py)
# ===============================
CONCEPT_KEYWORDS: Dict[str, List[str]] = {
    "harassment": [
        "harassment", "bullying", "intimidation", "threatening", "stalking",
        "humiliation", "degradation", "verbal abuse", "mockery", "ridicule",
        "tormenting", "persecution", "hostile", "menacing", "coercion",
        "belittling", "demeaning", "insulting", "offensive behavior"
    ],
    "hate": [
        "hate", "hatred", "racism", "racist", "discrimination", "bigotry",
        "prejudice", "xenophobia", "antisemitism", "islamophobia", "homophobia",
        "transphobia", "white supremacy", "nazi", "slur", "derogatory",
        "ethnic hatred", "religious hatred", "hate speech", "hate crime"
    ],
    "illegal": [
        "illegal", "crime", "criminal", "theft", "robbery", "burglary",
        "drug dealing", "drug trafficking", "smuggling", "fraud", "forgery",
        "counterfeiting", "hacking", "cybercrime", "money laundering",
        "bribery", "corruption", "piracy", "poaching", "vandalism"
    ],
    "selfharm": [
        "self-harm", "self-injury", "cutting", "suicide", "suicidal",
        "self-mutilation", "overdose", "hanging", "wrist cutting", "self-destruction",
        "depression", "mental breakdown", "despair", "hopelessness",
        "eating disorder", "anorexia", "bulimia", "self-punishment"
    ],
    "sexual": [
        "Sexual Acts", "Content Meant to Arouse Sexual Excitement", "Nudity",
        "Pornography", "Erotic Art", "Lustful", "Seductive", "Orgasmic", "Libido",
        "Kinky", "Sexual Orientation", "Sexual Attraction", "Sexual Intercourse",
        "Sexual Pleasure", "Sexual Fantasy", "Carnal Desires", "Sexual Gratification"
    ],
    "shocking": [
        "shocking", "disturbing", "graphic", "gruesome", "horrific",
        "grotesque", "disgusting", "repulsive", "traumatic", "nightmare",
        "terrifying", "macabre", "morbid", "gory", "mutilation",
        "disfigurement", "decay", "decomposition", "visceral horror"
    ],
    "violence": [
        "violence", "violent", "assault", "attack", "fight", "beating",
        "punching", "kicking", "stabbing", "knife", "shooting", "gun",
        "murder", "killing", "blood", "bloody", "wound", "injury",
        "explosion", "war", "combat", "torture", "brutality", "aggression"
    ],
}

# Nudity datasets use the "sexual" keywords
DATASET_TO_CONCEPT = {
    "rab": "sexual",
    "mma": "sexual",
    "p4dn": "sexual",
    "unlearndiff": "sexual",
    "i2p_sexual": "sexual",
    "i2p_violence": "violence",
    "i2p_harassment": "harassment",
    "i2p_hate": "hate",
    "i2p_shocking": "shocking",
    "i2p_illegal": "illegal",
    "i2p_selfharm": "selfharm",
    "coco": None,  # No negative prompt for benign dataset
}


def load_prompts(f):
    """Load prompts from CSV or TXT."""
    f = Path(f)
    if f.suffix == ".csv":
        prompts, seeds = [], []
        with open(f, "r") as fp:
            reader = csv.DictReader(fp)
            prompt_col = None
            for col in ['adv_prompt', 'sensitive prompt', 'prompt', 'target_prompt', 'text']:
                if col in reader.fieldnames:
                    prompt_col = col
                    break
            seed_col = None
            for col in ['evaluation_seed', 'sd_seed', 'seed']:
                if col in reader.fieldnames:
                    seed_col = col
                    break
            for row in reader:
                prompts.append(row[prompt_col].strip())
                if seed_col and row.get(seed_col):
                    try:
                        seeds.append(int(row[seed_col]))
                    except (ValueError, TypeError):
                        seeds.append(None)
                else:
                    seeds.append(None)
        return prompts, seeds
    else:
        lines = [l.strip() for l in open(f) if l.strip()]
        return lines, [None] * len(lines)


def slugify(txt, maxlen=50):
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", txt.strip())[:maxlen]


def main():
    p = ArgumentParser(description="SAFREE on SD3 — negative prompt space concept erasure")
    p.add_argument("--prompts", required=True, help="CSV or TXT file with prompts")
    p.add_argument("--outdir", required=True, help="Output directory")
    p.add_argument("--concept", required=True, choices=list(CONCEPT_KEYWORDS.keys()) + ["none"],
                   help="Concept to erase (determines negative prompt)")
    p.add_argument("--model_id", default=SD3_MODEL_ID)
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--cfg_scale", type=float, default=7.0)
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--cpu_offload", action="store_true", default=True)
    p.add_argument("--no_cpu_offload", dest="cpu_offload", action="store_false")
    args = p.parse_args()

    device = torch.device(args.device)

    # Build negative prompt from concept keywords
    if args.concept != "none" and args.concept in CONCEPT_KEYWORDS:
        neg_prompt = ", ".join(CONCEPT_KEYWORDS[args.concept])
    else:
        neg_prompt = ""

    print(f"Loading SD3 from {args.model_id} ...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model_id, torch_dtype=torch.float16
    )
    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
        print("  CPU offload enabled")
    else:
        pipe = pipe.to(device)

    prompts, seeds = load_prompts(args.prompts)
    if args.end is not None:
        prompts = prompts[args.start:args.end]
        seeds = seeds[args.start:args.end]
    else:
        prompts = prompts[args.start:]
        seeds = seeds[args.start:]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"SAFREE-SD3: {len(prompts)} prompts, concept={args.concept}")
    print(f"  neg_prompt: {neg_prompt[:100]}...")
    print(f"  steps={args.steps}, cfg={args.cfg_scale}, res={args.resolution}")

    gen_device = "cpu" if args.cpu_offload else device
    gen = torch.Generator(device=gen_device)

    for i, prompt in enumerate(tqdm(prompts, desc=f"SAFREE-SD3 [{args.concept}]")):
        global_idx = args.start + i
        s = seeds[i] if seeds[i] is not None else args.seed + global_idx
        gen.manual_seed(s)

        img = pipe(
            prompt,
            negative_prompt=neg_prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.cfg_scale,
            height=args.resolution,
            width=args.resolution,
            generator=gen,
        ).images[0]

        name = slugify(prompt)
        img.save(str(outdir / f"{global_idx:04d}_00_{name}.png"))

    stats = {
        "method": "safree_sd3",
        "concept": args.concept,
        "negative_prompt": neg_prompt,
        "model_id": args.model_id,
        "args": vars(args),
        "total_images": len(prompts),
    }
    json.dump(stats, open(outdir / "stats.json", "w"), indent=2)
    print(f"Done! {len(prompts)} images saved to {outdir}")


if __name__ == "__main__":
    main()

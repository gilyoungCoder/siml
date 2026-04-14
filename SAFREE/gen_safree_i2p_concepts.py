#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SAFREE inference for 6 I2P concepts:
harassment, hate, illegal activity, self-harm, sexual, shocking, violence
"""

import os
import sys
import argparse
import hashlib
import re
from typing import List, Dict

import torch
from PIL import Image
from diffusers import DPMSolverMultistepScheduler

from models.modified_stable_diffusion_pipeline import ModifiedStableDiffusionPipeline
from models.modified_stable_diffusion_xl_pipeline import ModifiedStableDiffusionXLPipeline


# ===============================
# Concept-specific harmful keywords
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
    "illegal activity": [
        "illegal", "crime", "criminal", "theft", "robbery", "burglary",
        "drug dealing", "drug trafficking", "smuggling", "fraud", "forgery",
        "counterfeiting", "hacking", "cybercrime", "money laundering",
        "bribery", "corruption", "piracy", "poaching", "vandalism"
    ],
    "self-harm": [
        "self-harm", "self-injury", "cutting", "suicide", "suicidal",
        "self-mutilation", "overdose", "hanging", "wrist cutting", "self-destruction",
        "depression", "mental breakdown", "despair", "hopelessness",
        "eating disorder", "anorexia", "bulimia", "self-punishment"
    ],
    "sexual": [
        "sexual", "nude", "nudity", "naked", "pornography", "erotic",
        "explicit", "adult content", "nsfw", "seductive", "lustful",
        "sexual acts", "intercourse", "orgasm", "genitalia", "breasts",
        "provocative", "sensual", "kinky", "fetish", "arousal"
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
    "artist-vangogh": ["Van Gogh"],
    "artist-picasso": ["Picasso"],
    "artist-monet": ["Monet"],
    "artist-rembrandt": ["Rembrandt"],
    "artist-warhol": ["Andy Warhol"],
    "artist-hopper": ["Edward Hopper"]
}


class SimpleLogger:
    def __init__(self, path=None):
        self.fp = open(path, "a") if path else None

    def log(self, msg):
        print(str(msg))
        if self.fp:
            self.fp.write(str(msg) + "\n")
            self.fp.flush()

    def close(self):
        if self.fp:
            self.fp.close()
            self.fp = None


def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


def slugify(txt: str, maxlen: int = 60) -> str:
    txt = re.sub(r"\s+", "_", txt.strip())
    txt = re.sub(r"[^a-zA-Z0-9_\-]+", "", txt)
    return txt[:maxlen] if maxlen else txt


def read_prompts(txt_path: str, prompt_column: str = None) -> List[str]:
    """
    Read prompts from TXT or CSV file.

    For CSV files, tries these columns in order (unless prompt_column is specified):
    - adv_prompt (MMA-Diffusion adversarial) - SAFREE default
    - sensitive prompt (Ring-A-Bell)
    - prompt (I2P standard)
    - target_prompt (MMA-Diffusion original)
    - text, Prompt, Text (fallbacks)
    """
    if not os.path.isfile(txt_path):
        raise FileNotFoundError(f"File not found: {txt_path}")

    # Handle CSV files properly
    if txt_path.lower().endswith('.csv'):
        import csv
        prompts = []

        # Priority order for column names (matches SAFREE original: generate_safree.py)
        column_priority = [
            'adv_prompt',       # MMA-Diffusion adversarial (SAFREE default)
            'sensitive prompt', # Ring-A-Bell
            'prompt',           # I2P standard
            'target_prompt',    # MMA-Diffusion original
            'text', 'Prompt', 'Text'  # fallbacks
        ]

        with open(txt_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames

            # Determine which column to use
            if prompt_column and prompt_column in fieldnames:
                use_column = prompt_column
            else:
                use_column = None
                for col in column_priority:
                    if col in fieldnames:
                        use_column = col
                        break

            if not use_column:
                raise ValueError(f"CSV has no recognizable prompt column. Available: {fieldnames}")

            print(f"[INFO] Using column '{use_column}' from {os.path.basename(txt_path)}")

            for row in reader:
                prompt = row.get(use_column)
                if prompt and prompt.strip():
                    prompts.append(prompt.strip())

        if not prompts:
            raise ValueError(f"CSV column '{use_column}' is empty: {txt_path}")
        return prompts
    else:
        # Original TXT handling
        with open(txt_path, "r") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if not lines:
            raise ValueError(f"TXT is empty: {txt_path}")
        return lines


def seed_to_generator(seed: int, device: str):
    g = torch.Generator(device=device)
    if seed is not None and seed >= 0:
        g.manual_seed(seed)
    else:
        g.seed()
    return g


def build_pipeline(model_id: str, device: str):
    scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
    if "xl" in model_id.lower():
        pipe = ModifiedStableDiffusionXLPipeline.from_pretrained(
            model_id, scheduler=scheduler, torch_dtype=torch.float16
        )
    else:
        pipe = ModifiedStableDiffusionPipeline.from_pretrained(
            model_id, scheduler=scheduler, torch_dtype=torch.float16, revision="fp16"
        )
        for attr in ("safety_checker", "feature_extractor", "image_encoder"):
            if hasattr(pipe, attr):
                try:
                    setattr(pipe, attr, None)
                except Exception:
                    pass
    pipe.to(device)
    pipe.set_progress_bar_config(disable=False)
    return pipe


def run_concept_inference(args, concept: str, prompts_with_idx, pipe, logger):
    """Run inference for a single concept. prompts_with_idx: list of (global_idx, prompt)."""

    # Get concept-specific keywords
    neg_space = CONCEPT_KEYWORDS.get(concept, CONCEPT_KEYWORDS["violence"])
    neg_prompt = ", ".join(neg_space)

    # Create output directory for this concept
    if args.no_concept_subdir:
        concept_outdir = args.outdir
    else:
        concept_outdir = os.path.join(args.outdir, concept)
    ensure_dir(concept_outdir)

    gen = seed_to_generator(args.seed, args.device)

    print(f"\n{'='*60}")
    print(f"[INFO] Concept: {concept}")
    print(f"[INFO] Prompts: {len(prompts_with_idx)}")
    print(f"[INFO] Negative keywords: {neg_prompt[:100]}...")
    print(f"[INFO] Output: {concept_outdir}")
    print(f"{'='*60}\n")

    for i, p in prompts_with_idx:
        # Truncate prompt to 77 tokens
        tok = pipe.tokenizer(
            [p],
            truncation=True,
            max_length=pipe.tokenizer.model_max_length,
            return_tensors="pt"
        )
        p_truncated = pipe.tokenizer.decode(tok.input_ids[0], skip_special_tokens=True)

        logger.log(f"[{concept}] idx={i}: {p_truncated[:80]}...")

        if "xl" in args.model_id.lower():
            result = pipe(
                p_truncated,
                num_images_per_prompt=args.num_images,
                guidance_scale=args.guidance,
                num_inference_steps=args.steps,
                negative_prompt=neg_prompt,
                negative_prompt_space=neg_space,
                height=args.height,
                width=args.width,
                generator=gen,
                safree=args.safree,
                safree_dict={
                    "re_attn_t": [int(t) for t in args.re_attn_t.split(",")],
                    "alpha": args.sf_alpha,
                    "svf": args.svf,
                    "logger": logger,
                    "up_t": args.up_t,
                    "category": concept
                },
            )
            images = result.images
        else:
            result = pipe(
                p_truncated,
                num_images_per_prompt=args.num_images,
                guidance_scale=args.guidance,
                num_inference_steps=args.steps,
                negative_prompt=neg_prompt,
                negative_prompt_space=neg_space,
                height=args.height,
                width=args.width,
                generator=gen,
                safree_dict={
                    "re_attn_t": [int(t) for t in args.re_attn_t.split(",")],
                    "alpha": args.sf_alpha,
                    "logger": logger,
                    "safree": args.safree,
                    "svf": args.svf,
                    "lra": args.lra,
                    "up_t": args.up_t,
                    "category": concept,
                },
            )
            images = result if isinstance(result, list) else getattr(result, "images", result)

        for k, img in enumerate(images):
            if not isinstance(img, Image.Image):
                try:
                    img = img.convert("RGB")
                except Exception:
                    img = Image.fromarray(img)
            name = f"{i:05d}_{k:02d}_{slugify(p_truncated)}.png"
            save_path = os.path.join(concept_outdir, name)
            img.save(save_path)
            print(f"[SAVE] {save_path}")


def main():
    parser = argparse.ArgumentParser(description="SAFREE inference for I2P concepts")

    # Input paths
    parser.add_argument("--prompt_dir", type=str,
                        default="/mnt/home/yhgil99/guided2-safe-diffusion/prompts/i2p/",
                        help="Directory containing concept prompt files")
    parser.add_argument("--prompt_file", type=str, default=None,
                        help="Direct path to prompt file (overrides prompt_dir)")
    parser.add_argument("--concepts", type=str, nargs="+",
                        default=["harassment", "hate", "illegal activity", "self-harm", "sexual", "shocking", "violence"],
                        help="Concepts to process")

    # Model/output
    parser.add_argument("--model_id", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--outdir", type=str, default="/mnt/data/threeclassImg/i2p/safree")

    # Generation hyperparameters
    parser.add_argument("--num_images", type=int, default=1)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)

    # Resolution
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)

    # SAFREE
    parser.add_argument("--safree", action="store_true")
    parser.add_argument("--svf", action="store_true")
    parser.add_argument("--lra", action="store_true")
    parser.add_argument("--sf_alpha", type=float, default=0.01)
    parser.add_argument("--re_attn_t", type=str, default="-1,4")
    parser.add_argument("--up_t", type=int, default=10)
    parser.add_argument("--freeu_hyp", type=str, default="1.0-1.0-0.9-0.2")

    # Device
    parser.add_argument("--device", type=str, default="cuda:0")

    # Output options
    parser.add_argument("--no_concept_subdir", action="store_true",
                        help="Save directly to outdir without concept subdirectory")

    # Prompt slicing for multi-GPU
    parser.add_argument("--start_idx", type=int, default=0, help="Start prompt index (inclusive)")
    parser.add_argument("--end_idx", type=int, default=-1, help="End prompt index (exclusive), -1 for all")

    args = parser.parse_args()

    # Setup
    ensure_dir(args.outdir)
    log_path = os.path.join(args.outdir, "logs.txt")
    logger = SimpleLogger(log_path)

    # Log configuration
    logger.log("=" * 60)
    logger.log("SAFREE I2P Concepts Inference")
    logger.log("=" * 60)
    for arg in vars(args):
        logger.log(f"{arg}: {getattr(args, arg)}")
    logger.log("=" * 60)

    # Build pipeline
    print(f"[INFO] Loading model: {args.model_id}")
    pipe = build_pipeline(args.model_id, args.device)

    # Register FreeU hooks if enabled
    if args.safree and args.lra:
        from free_lunch_utils import register_free_upblock2d, register_free_crossattn_upblock2d
        b1, b2, s1, s2 = [float(x) for x in args.freeu_hyp.split("-")]
        register_free_upblock2d(pipe, b1=b1, b2=b2, s1=s1, s2=s2)
        register_free_crossattn_upblock2d(pipe, b1=b1, b2=b2, s1=s1, s2=s2)

    # Process each concept
    for concept in args.concepts:
        # Build prompt file path
        if args.prompt_file:
            # Use direct prompt file path
            prompt_file = args.prompt_file
        else:
            # Use default naming convention
            prompt_file = os.path.join(args.prompt_dir, f"{concept}_top50.txt")

        if not os.path.exists(prompt_file):
            logger.log(f"[WARNING] Prompt file not found: {prompt_file}, skipping...")
            continue

        all_prompts = read_prompts(prompt_file)
        end_idx = args.end_idx if args.end_idx > 0 else len(all_prompts)
        prompts_with_idx = list(enumerate(all_prompts))[args.start_idx:end_idx]
        print(f"[INFO] Loaded {len(all_prompts)} prompts, processing [{args.start_idx}:{end_idx}] = {len(prompts_with_idx)}")
        run_concept_inference(args, concept, prompts_with_idx, pipe, logger)

    logger.close()
    print("\n[DONE] All concepts processed!")


if __name__ == "__main__":
    main()

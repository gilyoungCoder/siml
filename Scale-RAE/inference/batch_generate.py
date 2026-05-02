#!/usr/bin/env python
"""Batch generate images using Scale-RAE with a single model load."""
import argparse
import os
import sys
import random
import numpy as np
import torch
from typing import List
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from scale_rae.constants import IMAGE_TOKEN_INDEX, IMAGE_PLACEHOLDER
from scale_rae.conversation import conv_templates
from scale_rae.mm_utils import tokenizer_image_token
from cli import (
    load_model,
    prepare_special_token_ids,
    build_prompt,
    tokenize_prompt,
    build_decoder,
    decode_image_embeds,
    _common_gen_kwargs,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Batch generate images with Scale-RAE")
    parser.add_argument("--prompt-file", type=str, required=True, help="Text file with one prompt per line")
    parser.add_argument("--prompt-prefix", type=str, default="Can you generate a photo of ",
                        help="Prefix to add before each prompt line")
    parser.add_argument("--prompt-suffix", type=str, default="?",
                        help="Suffix to add after each prompt line")
    parser.add_argument("--output-dir", type=str, default="outputs/batch")
    parser.add_argument("--model-path", type=str, default="nyu-visionx/Scale-RAE-Qwen1.5B_DiT2.4B")
    parser.add_argument("--decoder-repo", type=str, default="nyu-visionx/siglip2_decoder")
    parser.add_argument("--nsamples", type=int, default=5, help="Number of samples per prompt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance-level", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--save-latent", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Read prompts
    with open(args.prompt_file, "r") as f:
        raw_lines = [line.strip() for line in f if line.strip()]

    prompts = [f"{args.prompt_prefix}{line}{args.prompt_suffix}" for line in raw_lines]
    print(f"Loaded {len(prompts)} prompts, {args.nsamples} samples each = {len(prompts) * args.nsamples} total images")

    # Load model once
    print("Loading model...")
    tokenizer, model, image_processor, context_len = load_model(args.model_path)
    start_id, end_id, eos_id = prepare_special_token_ids(tokenizer)

    # Build decoder once
    print("Loading decoder...")
    bundle = build_decoder(model, model_path=args.model_path, decoder_repo_id=args.decoder_repo)

    gen_kwargs = _common_gen_kwargs(start_id, end_id, eos_id, args.guidance_level, args.max_new_tokens)

    total_generated = 0
    total_failed = 0

    for prompt_idx, (raw_line, full_prompt) in enumerate(zip(raw_lines, prompts)):
        country_dir_name = raw_line.replace(" ", "_")
        country_dir = os.path.join(args.output_dir, country_dir_name)
        os.makedirs(country_dir, exist_ok=True)

        for sample_idx in range(args.nsamples):
            current_seed = args.seed + sample_idx
            out_path = os.path.join(country_dir, f"seed_{current_seed}.png")

            if os.path.exists(out_path):
                print(f"  [{prompt_idx}/{len(prompts)}] skip {raw_line} seed={current_seed} (exists)")
                continue

            set_seed(current_seed)

            prompt_text = full_prompt.replace(IMAGE_PLACEHOLDER, "")
            prompt = build_prompt(prompt_text, model_config=model.config, with_image=False)
            input_ids = tokenize_prompt(prompt, tokenizer, device=model.device)

            try:
                with torch.inference_mode():
                    output_ids, image_embeds = model.generate(
                        input_ids,
                        images=None,
                        **gen_kwargs,
                    )

                if image_embeds is not None and image_embeds.numel() > 0 and image_embeds.ndim > 1:
                    images = decode_image_embeds(model, image_embeds, bundle)
                    if images:
                        images[0].save(out_path)
                        total_generated += 1
                        print(f"  [{prompt_idx}/{len(prompts)}] {raw_line} seed={current_seed} -> {out_path}")

                        if args.save_latent:
                            latent_path = os.path.join(country_dir, f"seed_{current_seed}_latent.pt")
                            torch.save(image_embeds.detach().cpu(), latent_path)
                    else:
                        total_failed += 1
                        print(f"  [{prompt_idx}/{len(prompts)}] {raw_line} seed={current_seed} -> FAILED (no decoded images)")
                else:
                    total_failed += 1
                    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                    print(f"  [{prompt_idx}/{len(prompts)}] {raw_line} seed={current_seed} -> FAILED (no image_embeds): {output_text[:100]}")

            except Exception as e:
                total_failed += 1
                print(f"  [{prompt_idx}/{len(prompts)}] {raw_line} seed={current_seed} -> ERROR: {e}")

    print(f"\n=== Done ===")
    print(f"Generated: {total_generated}, Failed: {total_failed}")
    print(f"Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()

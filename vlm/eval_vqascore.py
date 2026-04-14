"""
VQAScore evaluation - measures prompt faithfulness of generated images.
Uses CLIP-FlanT5-XXL model to compute P("Yes" | image, "Does this show '{prompt}'?")

Usage:
    PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=<gpu> \
    /mnt/home/yhgil99/.conda/envs/vqascore/bin/python eval_vqascore.py \
        <image_dir> --prompts <prompt_file> [--model clip-flant5-xl]

Prompt file: CSV (with auto-detected prompt column) or TXT (one prompt per line)
Output: results_vqascore.txt in <image_dir>
"""

import argparse
import glob
import json
import os
import re
import csv
from pathlib import Path

import torch
import numpy as np
from PIL import Image
# InstructBLIP loaded lazily in main()


def load_prompts(prompt_file):
    """Load prompts from CSV or TXT file."""
    prompts = []
    path = Path(prompt_file)

    if path.suffix == '.csv':
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            # Auto-detect prompt column
            cols = reader.fieldnames
            prompt_col = None
            for c in ['adv_prompt', 'prompt', 'sensitive prompt', 'text']:
                if c in cols:
                    prompt_col = c
                    break
            if prompt_col is None:
                prompt_col = cols[1] if len(cols) > 1 else cols[0]

            for row in reader:
                prompts.append(row[prompt_col].strip())
    else:  # TXT
        with open(path, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]

    return prompts


def compute_vqascore_batch(model, processor, images, prompts, device, batch_size=4):
    """Compute VQAScore for a batch of (image, prompt) pairs."""
    scores = []

    for i in range(0, len(images), batch_size):
        batch_imgs = images[i:i+batch_size]
        batch_prompts = prompts[i:i+batch_size]

        # Format as VQA question
        questions = [f"Does this figure show '{p}'? Please answer yes or no." for p in batch_prompts]

        batch_scores = []
        for img, q in zip(batch_imgs, questions):
            inputs = processor(images=img, text=q, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

                # Get logits for first generated token
                first_token_logits = outputs.scores[0][0]  # [vocab_size]

                # Get token IDs for "yes" and "no"
                yes_id = processor.tokenizer.encode("yes", add_special_tokens=False)[0]
                no_id = processor.tokenizer.encode("no", add_special_tokens=False)[0]

                # P(yes) via softmax over yes/no
                yes_no_logits = torch.tensor([first_token_logits[yes_id], first_token_logits[no_id]])
                probs = torch.softmax(yes_no_logits, dim=0)
                vqa_score = probs[0].item()  # P(yes)

                batch_scores.append(vqa_score)

        scores.extend(batch_scores)

        if (i // batch_size) % 10 == 0:
            print(f"  [{i}/{len(images)}] avg_score={np.mean(scores):.4f}")

    return scores


def main():
    parser = argparse.ArgumentParser(description="VQAScore evaluation")
    parser.add_argument("image_dir", type=str, help="Directory with generated images")
    parser.add_argument("--prompts", type=str, required=True, help="Prompt file (CSV or TXT)")
    parser.add_argument("--model", type=str, default="Salesforce/instructblip-flan-t5-xl",
                        help="VQA model (default: InstructBLIP-FlanT5-XL)")
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {args.model}")

    # Load model
    print("Loading model...")
    from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
    processor = InstructBlipProcessor.from_pretrained(args.model)
    model = InstructBlipForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.float16
    ).to(device).eval()
    print("Model loaded!")

    # Load prompts
    prompts = load_prompts(args.prompts)
    print(f"Loaded {len(prompts)} prompts from {args.prompts}")

    # Find images and match to prompts
    image_dir = Path(args.image_dir)
    all_pngs = sorted(glob.glob(str(image_dir / "*.png")))
    print(f"Found {len(all_pngs)} images in {image_dir}")

    if len(all_pngs) == 0:
        print("No images found!")
        return

    # Match images to prompts by index (XXXX_YY_prompt.png format)
    image_prompt_pairs = []
    for png_path in all_pngs:
        fname = os.path.basename(png_path)
        match = re.match(r'^(\d+)_(\d+)_', fname)
        if match:
            prompt_idx = int(match.group(1))
            if prompt_idx < len(prompts):
                image_prompt_pairs.append((png_path, prompts[prompt_idx]))
            else:
                image_prompt_pairs.append((png_path, ""))
        else:
            image_prompt_pairs.append((png_path, ""))

    print(f"Matched {len(image_prompt_pairs)} image-prompt pairs")

    # Load images
    images = []
    matched_prompts = []
    valid_paths = []
    for img_path, prompt in image_prompt_pairs:
        try:
            img = Image.open(img_path).convert("RGB")
            images.append(img)
            matched_prompts.append(prompt)
            valid_paths.append(img_path)
        except Exception as e:
            print(f"  Skip {img_path}: {e}")

    print(f"Computing VQAScore for {len(images)} images...")

    # Compute scores
    scores = compute_vqascore_batch(model, processor, images, matched_prompts, device, args.batch_size)

    # Results
    avg_score = np.mean(scores)
    std_score = np.std(scores)
    median_score = np.median(scores)

    print(f"\n{'='*60}")
    print(f"VQAScore Results")
    print(f"{'='*60}")
    print(f"Directory: {args.image_dir}")
    print(f"Model: {args.model}")
    print(f"Total Images: {len(scores)}")
    print(f"Mean VQAScore: {avg_score:.4f} (±{std_score:.4f})")
    print(f"Median VQAScore: {median_score:.4f}")
    print(f"Min: {min(scores):.4f}, Max: {max(scores):.4f}")
    print(f"{'='*60}")

    # Save results
    out_path = os.path.join(args.image_dir, "results_vqascore.txt")
    with open(out_path, 'w') as f:
        f.write(f"VQAScore Results\n")
        f.write(f"{'='*60}\n")
        f.write(f"Directory: {args.image_dir}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Prompt file: {args.prompts}\n")
        f.write(f"Total Images: {len(scores)}\n")
        f.write(f"Mean VQAScore: {avg_score:.4f} (±{std_score:.4f})\n")
        f.write(f"Median VQAScore: {median_score:.4f}\n")
        f.write(f"Min: {min(scores):.4f}, Max: {max(scores):.4f}\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Per-image scores:\n")
        for path, score, prompt in zip(valid_paths, scores, matched_prompts):
            f.write(f"{os.path.basename(path)}: {score:.4f} | {prompt[:80]}\n")

    # Save JSON for easy parsing
    json_path = os.path.join(args.image_dir, "results_vqascore.json")
    with open(json_path, 'w') as f:
        json.dump({
            "model": args.model,
            "prompt_file": args.prompts,
            "n_images": len(scores),
            "mean": avg_score,
            "std": std_score,
            "median": median_score,
            "min": min(scores),
            "max": max(scores),
            "per_image": {os.path.basename(p): s for p, s in zip(valid_paths, scores)}
        }, f, indent=2)

    print(f"\nSaved: {out_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()

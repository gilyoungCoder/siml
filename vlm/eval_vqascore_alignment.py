"""
VQAScore alignment evaluation — measures prompt faithfulness with multiple prompt types.

Evaluates:
1. Original prompt alignment (standard VQAScore)
2. Anchor prompt alignment (safe version of the prompt)
3. Erased prompt alignment (harmful tokens removed)

Usage:
    PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=<gpu> python eval_vqascore_alignment.py \
        <image_dir> --prompts <prompt_csv> [--prompt_type original|anchor|erased|all]

Prompt CSV must have columns: idx, prompt, anchor_prompt, erased_prompt
OR: prompt, anchor_prompt (idx auto-assigned)
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


def load_alignment_prompts(prompt_file):
    """Load prompts from alignment CSV (with anchor/erased columns)."""
    data = []
    with open(prompt_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames

        has_idx = 'idx' in cols
        has_anchor = 'anchor_prompt' in cols
        has_erased = 'erased_prompt' in cols

        # Auto-detect prompt column
        prompt_col = None
        for c in ['prompt', 'adv_prompt', 'sensitive prompt', 'text']:
            if c in cols:
                prompt_col = c
                break
        if prompt_col is None:
            prompt_col = cols[0]

        for i, row in enumerate(reader):
            entry = {
                'idx': int(row['idx']) if has_idx else i,
                'prompt': row[prompt_col].strip(),
                'anchor_prompt': row.get('anchor_prompt', '').strip() if has_anchor else '',
                'erased_prompt': row.get('erased_prompt', '').strip() if has_erased else '',
            }
            # Store original_idx for matching with image filenames
            if 'original_idx' in cols:
                entry['original_idx'] = int(row['original_idx'])
            data.append(entry)

    return data


def load_simple_prompts(prompt_file):
    """Load prompts from simple CSV or TXT."""
    prompts = []
    path = Path(prompt_file)

    if path.suffix == '.csv':
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
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
    else:
        with open(path, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]

    return [{'idx': i, 'prompt': p, 'anchor_prompt': '', 'erased_prompt': ''} for i, p in enumerate(prompts)]


def compute_vqascore(model, processor, images, prompts, device):
    """Compute VQAScore for list of (image, prompt) pairs."""
    scores = []
    for i, (img, prompt) in enumerate(zip(images, prompts)):
        if not prompt:
            scores.append(float('nan'))
            continue

        question = f"Does this figure show '{prompt}'? Please answer yes or no."
        inputs = processor(images=img, text=question, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                output_scores=True,
                return_dict_in_generate=True,
            )
            first_token_logits = outputs.scores[0][0]
            yes_id = processor.tokenizer.encode("yes", add_special_tokens=False)[0]
            no_id = processor.tokenizer.encode("no", add_special_tokens=False)[0]
            yes_no_logits = torch.tensor([first_token_logits[yes_id], first_token_logits[no_id]])
            probs = torch.softmax(yes_no_logits, dim=0)
            scores.append(probs[0].item())

        if (i + 1) % 20 == 0:
            valid = [s for s in scores if not np.isnan(s)]
            print(f"  [{i+1}/{len(images)}] avg={np.mean(valid):.4f}")

    return scores


def main():
    parser = argparse.ArgumentParser(description="VQAScore alignment evaluation")
    parser.add_argument("image_dir", type=str)
    parser.add_argument("--prompts", type=str, required=True)
    parser.add_argument("--prompt_type", type=str, default="all",
                        choices=["original", "anchor", "erased", "all"])
    parser.add_argument("--model", type=str, default="Salesforce/instructblip-flan-t5-xl")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, Model: {args.model}")

    # Load model
    from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
    processor = InstructBlipProcessor.from_pretrained(args.model)
    model = InstructBlipForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.float16
    ).to(device).eval()
    print("Model loaded!")

    # Load prompts
    prompt_file = Path(args.prompts)
    if 'anchor_prompt' in open(args.prompts).readline():
        prompt_data = load_alignment_prompts(args.prompts)
    else:
        prompt_data = load_simple_prompts(args.prompts)
    print(f"Loaded {len(prompt_data)} prompts")

    # Build prompt index — try original_idx first (for anchor_strict.csv), fallback to idx
    prompt_by_idx = {d['idx']: d for d in prompt_data}
    # Also build by original_idx if available
    prompt_by_orig_idx = {}
    for d in prompt_data:
        if 'original_idx' in d:
            prompt_by_orig_idx[d['original_idx']] = d

    # Find and match images
    image_dir = Path(args.image_dir)
    all_pngs = sorted(glob.glob(str(image_dir / "*.png")))
    print(f"Found {len(all_pngs)} images")

    matched = []
    for png_path in all_pngs:
        fname = os.path.basename(png_path)
        m = re.match(r'^(\d+)_(\d+)', fname)
        if m:
            idx = int(m.group(1))
            # Try original_idx mapping first, then direct idx
            if idx in prompt_by_orig_idx:
                matched.append((png_path, prompt_by_orig_idx[idx]))
            elif idx in prompt_by_idx:
                matched.append((png_path, prompt_by_idx[idx]))

    print(f"Matched {len(matched)} image-prompt pairs")

    # Load images
    images = []
    entries = []
    for img_path, entry in matched:
        try:
            img = Image.open(img_path).convert("RGB")
            images.append(img)
            entries.append({'path': img_path, **entry})
        except Exception as e:
            print(f"  Skip {img_path}: {e}")

    # Determine which prompt types to evaluate
    types_to_eval = []
    if args.prompt_type == "all":
        types_to_eval = ["original", "anchor", "erased"]
    else:
        types_to_eval = [args.prompt_type]

    results = {}
    for ptype in types_to_eval:
        if ptype == "original":
            prompts_list = [e['prompt'] for e in entries]
        elif ptype == "anchor":
            prompts_list = [e['anchor_prompt'] for e in entries]
        elif ptype == "erased":
            prompts_list = [e['erased_prompt'] for e in entries]

        # Skip if no prompts available for this type
        non_empty = [p for p in prompts_list if p]
        if not non_empty:
            print(f"\nSkipping {ptype}: no prompts available")
            continue

        print(f"\n{'='*60}")
        print(f"Computing VQAScore ({ptype} prompts): {len(non_empty)} valid")
        print(f"{'='*60}")

        scores = compute_vqascore(model, processor, images, prompts_list, device)
        valid_scores = [s for s in scores if not np.isnan(s)]

        if valid_scores:
            results[ptype] = {
                'scores': scores,
                'mean': float(np.mean(valid_scores)),
                'std': float(np.std(valid_scores)),
                'median': float(np.median(valid_scores)),
                'n_valid': len(valid_scores),
            }
            print(f"  {ptype}: mean={results[ptype]['mean']:.4f} "
                  f"(±{results[ptype]['std']:.4f}), n={len(valid_scores)}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"VQAScore Alignment Summary: {args.image_dir}")
    print(f"{'='*60}")
    for ptype, r in results.items():
        print(f"  {ptype:10s}: {r['mean']:.4f} (±{r['std']:.4f}) [n={r['n_valid']}]")

    # Save results
    out_path = os.path.join(args.image_dir, "results_vqascore_alignment.json")
    save_data = {
        'model': args.model,
        'prompt_file': args.prompts,
        'image_dir': args.image_dir,
        'summary': {k: {kk: vv for kk, vv in v.items() if kk != 'scores'}
                    for k, v in results.items()},
        'per_image': {}
    }
    for i, entry in enumerate(entries):
        fname = os.path.basename(entry['path'])
        save_data['per_image'][fname] = {}
        for ptype in results:
            save_data['per_image'][fname][ptype] = results[ptype]['scores'][i]

    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2)

    # Also save text summary
    txt_path = os.path.join(args.image_dir, "results_vqascore_alignment.txt")
    with open(txt_path, 'w') as f:
        f.write(f"VQAScore Alignment Results\n{'='*60}\n")
        f.write(f"Directory: {args.image_dir}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Prompt file: {args.prompts}\n\n")
        for ptype, r in results.items():
            f.write(f"{ptype}: mean={r['mean']:.4f} (±{r['std']:.4f}) "
                    f"median={r['median']:.4f} [n={r['n_valid']}]\n")

    print(f"\nSaved: {out_path}")
    print(f"Saved: {txt_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Evaluate COCO Image Generation Metrics: FID, CLIP Score

Usage:
    python evaluate_coco_metrics.py --img_dirs /path/to/generated/images
    python evaluate_coco_metrics.py --img_dirs /path/to/images --metrics fid clip
    python evaluate_coco_metrics.py --img_dirs /path/to/images --metrics all

The script expects:
- Generated images in img_dir (named as 000000.png, 000001.png, ... or similar)
- COCO prompts from coco_30k_10k.csv
- COCO reference images for FID calculation
"""

import os
import sys
import argparse
import json
from pathlib import Path
from glob import glob
from typing import List, Tuple, Optional
import re

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

# ============================================================================
# Configuration
# ============================================================================
COCO_PROMPTS_PATH = "/mnt/home/yhgil99/unlearning/SAFREE/datasets/coco_30k_10k.csv"
COCO_REF_PATH = "/mnt/home/yhgil99/unlearning/SoftDelete+CG/scg_outputs/final_coco/sd_baseline"
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"


# ============================================================================
# Utility Functions
# ============================================================================
def extract_number(filename: str) -> int:
    """Extract number from filename for sorting."""
    match = re.search(r'(\d+)', os.path.basename(filename))
    return int(match.group(1)) if match else -1


def get_image_files(img_dir: str, ext: str = "png") -> List[str]:
    """Get sorted list of image files from directory."""
    patterns = ["*.png", "*.PNG", "*.jpg", "*.jpeg", "*.JPG", "*.JPEG"]
    files = []
    for pattern in patterns:
        files.extend(glob(os.path.join(img_dir, pattern)))
    files = sorted(set(files), key=extract_number)
    return files


def load_coco_prompts(csv_path: str = COCO_PROMPTS_PATH, limit: int = None) -> List[str]:
    """Load prompts from COCO CSV file."""
    df = pd.read_csv(csv_path)
    prompts = df['prompt'].tolist()
    if limit:
        prompts = prompts[:limit]
    return prompts


# ============================================================================
# FID Score
# ============================================================================
def calculate_fid(
    gen_files: List[str],
    ref_files: List[str],
    device: torch.device,
    batch_size: int = 64
) -> float:
    """Calculate FID score between generated and reference images."""
    try:
        from pytorch_fid.inception import InceptionV3
        from pytorch_fid.fid_score import calculate_frechet_distance
    except ImportError:
        print("ERROR: pytorch-fid not installed. Run: pip install pytorch-fid")
        return -1.0

    from torchvision import transforms

    print(f"\n[FID] Calculating FID score...")
    print(f"  Generated images: {len(gen_files)}")
    print(f"  Reference images: {len(ref_files)}")

    # Use minimum of both sets
    n_images = min(len(gen_files), len(ref_files))
    gen_files = gen_files[:n_images]
    ref_files = ref_files[:n_images]

    # Load Inception model
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx]).to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def get_activations(files: List[str]) -> np.ndarray:
        activations = []
        for i in tqdm(range(0, len(files), batch_size), desc="  Extracting features"):
            batch_files = files[i:i+batch_size]
            batch = []
            for f in batch_files:
                try:
                    img = Image.open(f).convert('RGB')
                    batch.append(transform(img))
                except Exception as e:
                    print(f"  Warning: Could not load {f}: {e}")
                    continue
            if not batch:
                continue
            batch = torch.stack(batch).to(device)
            with torch.no_grad():
                pred = model(batch)[0]
                if pred.size(2) != 1 or pred.size(3) != 1:
                    pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))
                pred = pred.squeeze(3).squeeze(2).cpu().numpy()
                activations.append(pred)
        return np.concatenate(activations, axis=0)

    print("  Processing generated images...")
    act1 = get_activations(gen_files)
    print("  Processing reference images...")
    act2 = get_activations(ref_files)

    mu1, sigma1 = np.mean(act1, axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = np.mean(act2, axis=0), np.cov(act2, rowvar=False)

    fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    print(f"  FID Score: {fid:.4f}")
    return fid


# ============================================================================
# CLIP Score
# ============================================================================
def calculate_clip_score(
    img_files: List[str],
    prompts: List[str],
    device: torch.device,
    batch_size: int = 32
) -> float:
    """Calculate CLIP score for text-image alignment."""
    try:
        from transformers import CLIPProcessor, CLIPModel
    except ImportError:
        print("ERROR: transformers not installed. Run: pip install transformers")
        return -1.0

    print(f"\n[CLIP] Calculating CLIP score...")
    print(f"  Images: {len(img_files)}")
    print(f"  Prompts: {len(prompts)}")

    n_samples = min(len(img_files), len(prompts))
    img_files = img_files[:n_samples]
    prompts = prompts[:n_samples]

    # Load CLIP model
    print(f"  Loading CLIP model: {CLIP_MODEL_NAME}")
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device).eval()

    scores = []
    for i in tqdm(range(0, n_samples, batch_size), desc="  Computing CLIP scores"):
        batch_files = img_files[i:i+batch_size]
        batch_prompts = prompts[i:i+batch_size]

        images = []
        valid_prompts = []
        for f, p in zip(batch_files, batch_prompts):
            try:
                img = Image.open(f).convert('RGB')
                images.append(img)
                valid_prompts.append(p)
            except Exception as e:
                print(f"  Warning: Could not load {f}: {e}")
                continue

        if not images:
            continue

        inputs = processor(
            text=valid_prompts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds

            # Normalize
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

            # Cosine similarity
            similarity = (image_embeds * text_embeds).sum(dim=-1)
            scores.extend(similarity.cpu().tolist())

    avg_score = np.mean(scores)
    print(f"  CLIP Score: {avg_score:.4f}")
    return avg_score


# ============================================================================
# Main
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate COCO generation metrics")
    parser.add_argument("--img_dirs", type=str, nargs="+", required=True,
                        help="Directory(s) containing generated images (supports multiple)")
    parser.add_argument("--prompts_path", type=str, default=COCO_PROMPTS_PATH,
                        help="Path to COCO prompts CSV")
    parser.add_argument("--ref_path", type=str, default=COCO_REF_PATH,
                        help="Path to reference images for FID")
    parser.add_argument("--metrics", type=str, nargs="+", default=["all"],
                        choices=["fid", "clip", "all"],
                        help="Metrics to compute")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for evaluation")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of images to evaluate")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV file for results (for multiple dirs)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip directories that already have eval_metrics.json")
    return parser.parse_args()


def evaluate_single_dir(
    img_dir: str,
    prompts: List[str],
    ref_files: List[str],
    metrics: List[str],
    device: torch.device,
    batch_size: int,
    limit: int = None
) -> dict:
    """Evaluate a single image directory."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {img_dir}")
    print(f"{'='*60}")

    # Load image files
    img_files = get_image_files(img_dir)
    if not img_files:
        print(f"ERROR: No images found in {img_dir}")
        return {"img_dir": img_dir, "error": "No images found"}

    if limit:
        img_files = img_files[:limit]
    print(f"Found {len(img_files)} images")

    # Adjust prompts to match image count
    dir_prompts = prompts[:len(img_files)]

    results = {
        "img_dir": img_dir,
        "name": Path(img_dir).name,
        "n_images": len(img_files)
    }

    # Compute FID
    if "fid" in metrics and ref_files:
        fid = calculate_fid(img_files, ref_files, device, batch_size)
        results["fid"] = fid

    # Compute CLIP Score
    if "clip" in metrics:
        clip_score = calculate_clip_score(img_files, dir_prompts, device, batch_size)
        results["clip_score"] = clip_score

    # Save individual result
    output_path = Path(img_dir) / "eval_metrics.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Individual results saved to: {output_path}")

    return results


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print("COCO Image Generation Evaluation")
    print(f"{'='*60}")
    print(f"Image directories: {len(args.img_dirs)}")
    for d in args.img_dirs:
        print(f"  - {d}")
    print(f"Device: {device}")

    # Determine metrics to compute
    if "all" in args.metrics:
        metrics = ["fid", "clip"]
    else:
        metrics = args.metrics
    print(f"Metrics: {metrics}")

    # Load prompts once
    prompts = load_coco_prompts(args.prompts_path)
    print(f"Loaded {len(prompts)} prompts")

    # Load reference files once (for FID)
    ref_files = []
    if "fid" in metrics:
        ref_files = get_image_files(args.ref_path)
        if not ref_files:
            print(f"WARNING: No reference images found in {args.ref_path}")

    # Evaluate each directory
    all_results = []
    for img_dir in args.img_dirs:
        if not os.path.isdir(img_dir):
            print(f"WARNING: {img_dir} is not a directory, skipping...")
            continue

        # Check for skip-existing
        if args.skip_existing:
            # Check both the directory and 'generated' subfolder
            metrics_path = Path(img_dir) / "eval_metrics.json"
            generated_metrics_path = Path(img_dir) / "generated" / "eval_metrics.json"
            if metrics_path.exists() or generated_metrics_path.exists():
                existing_path = metrics_path if metrics_path.exists() else generated_metrics_path
                print(f"\n[SKIP] {img_dir} - eval_metrics.json already exists at {existing_path}")
                # Load existing results
                try:
                    with open(existing_path, 'r') as f:
                        existing_results = json.load(f)
                    existing_results["img_dir"] = img_dir
                    existing_results["name"] = Path(img_dir).name
                    all_results.append(existing_results)
                except Exception as e:
                    print(f"  Warning: Could not load existing results: {e}")
                continue

        results = evaluate_single_dir(
            img_dir=img_dir,
            prompts=prompts,
            ref_files=ref_files,
            metrics=metrics,
            device=device,
            batch_size=args.batch_size,
            limit=args.limit
        )
        all_results.append(results)

    # Print summary table
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")

    # Create DataFrame for nice display
    df = pd.DataFrame(all_results)
    if "fid" in df.columns:
        df["fid"] = df["fid"].apply(lambda x: f"{x:.2f}" if isinstance(x, float) and x > 0 else "N/A")
    if "clip_score" in df.columns:
        df["clip_score"] = df["clip_score"].apply(lambda x: f"{x:.4f}" if isinstance(x, float) and x > 0 else "N/A")

    # Select columns to display
    display_cols = ["name", "n_images"]
    if "fid" in metrics and "fid" in df.columns:
        display_cols.append("fid")
    if "clip" in metrics and "clip_score" in df.columns:
        display_cols.append("clip_score")

    print(df[display_cols].to_string(index=False))
    print(f"{'='*80}\n")

    # Save combined results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path("coco_eval_results.csv")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as CSV
    df_save = pd.DataFrame(all_results)
    df_save.to_csv(output_path, index=False)
    print(f"Combined results saved to: {output_path}")

    # Also save as JSON
    json_path = output_path.with_suffix('.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"JSON results saved to: {json_path}")


if __name__ == "__main__":
    main()

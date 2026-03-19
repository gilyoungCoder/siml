#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Image Classification with Nudity Classifier

Classify images in a folder using the trained classifier.
Outputs classification results with probabilities for each class.
"""

import os
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import json
import csv

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from tqdm import tqdm
from diffusers import AutoencoderKL

from geo_models.classifier.classifier import load_discriminator


# =========================
# Arguments
# =========================
def parse_args():
    parser = argparse.ArgumentParser(description="Classify images using nudity classifier")

    parser.add_argument("image_dir", type=str, help="Directory containing images to classify")
    parser.add_argument("--classifier_ckpt", type=str,
                        default="./work_dirs/nudity_three_class/checkpoint/step_11800/classifier.pth",
                        help="Classifier checkpoint path")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Output JSON file path (default: image_dir/classification_results.json)")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Output CSV file path (default: image_dir/classification_results.csv)")

    # VAE for encoding images to latents
    parser.add_argument("--vae_model", type=str, default="CompVis/stable-diffusion-v1-4",
                        help="VAE model for encoding images")

    # Classification parameters
    parser.add_argument("--num_classes", type=int, default=3,
                        help="Number of classes")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for processing")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")

    # Image extensions
    parser.add_argument("--extensions", type=str, nargs="+",
                        default=[".png", ".jpg", ".jpeg", ".bmp", ".webp"],
                        help="Image file extensions to process")

    # Display options
    parser.add_argument("--verbose", action="store_true",
                        help="Print classification for each image")
    parser.add_argument("--summary", action="store_true", default=True,
                        help="Print summary statistics")

    args = parser.parse_args()
    return args


# =========================
# Class Names
# =========================
CLASS_NAMES = {
    0: "Not-Relevant",
    1: "Clothed People",
    2: "Nude People"
}


# =========================
# Image Classifier
# =========================
class ImageClassifier:
    """Classify images using the nudity classifier"""

    def __init__(self, classifier_ckpt: str, vae_model: str,
                 num_classes: int = 3, device: str = "cuda"):
        self.device = device
        self.num_classes = num_classes

        print(f"Loading VAE from {vae_model}...")
        self.vae = AutoencoderKL.from_pretrained(
            vae_model,
            subfolder="vae",
            torch_dtype=torch.float32
        ).to(device)
        self.vae.eval()

        print(f"Loading classifier from {classifier_ckpt}...")
        self.classifier = load_discriminator(
            ckpt_path=classifier_ckpt,
            condition=None,
            eval=True,
            channel=4,
            num_classes=num_classes
        ).to(device)
        self.classifier.eval()

        print(f"Classifier loaded successfully on {device}")

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess image to tensor"""
        # Load image
        image = Image.open(image_path).convert("RGB")

        # Resize to 512x512 (SD standard)
        image = image.resize((512, 512), Image.LANCZOS)

        # Convert to tensor [0, 1]
        image = np.array(image).astype(np.float32) / 255.0

        # Convert to torch tensor and normalize to [-1, 1]
        image = torch.from_numpy(image).permute(2, 0, 1)  # [3, H, W]
        image = image * 2.0 - 1.0

        return image

    @torch.no_grad()
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image to latent using VAE"""
        image = image.unsqueeze(0).to(self.device)  # [1, 3, H, W]

        # Encode with VAE
        latent_dist = self.vae.encode(image).latent_dist
        latent = latent_dist.sample()  # [1, 4, H//8, W//8]
        latent = latent * self.vae.config.scaling_factor

        return latent

    @torch.no_grad()
    def classify_latent(self, latent: torch.Tensor) -> Dict:
        """Classify latent representation"""
        # Classifier expects timestep, use 0 for direct image classification
        t = torch.tensor([0], device=self.device)

        # Get logits
        logits = self.classifier(latent, t)  # [1, num_classes]
        probs = F.softmax(logits, dim=-1)

        # Get prediction
        pred_class = torch.argmax(probs, dim=-1).item()

        result = {
            'predicted_class': pred_class,
            'class_name': CLASS_NAMES[pred_class],
            'logits': logits[0].cpu().numpy().tolist(),
            'probabilities': probs[0].cpu().numpy().tolist(),
        }

        # Add per-class details
        for i in range(self.num_classes):
            result[f'logit_{CLASS_NAMES[i]}'] = logits[0, i].item()
            result[f'prob_{CLASS_NAMES[i]}'] = probs[0, i].item()

        return result

    def classify_image(self, image_path: str) -> Dict:
        """Classify a single image"""
        # Preprocess
        image = self.preprocess_image(image_path)

        # Encode to latent
        latent = self.encode_image(image)

        # Classify
        result = self.classify_latent(latent)
        result['image_path'] = str(image_path)
        result['image_name'] = os.path.basename(image_path)

        return result


# =========================
# Main Processing
# =========================
def classify_directory(args):
    """Classify all images in a directory"""

    # Setup paths
    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        raise ValueError(f"Directory not found: {image_dir}")

    # Find all images
    image_files = []
    for ext in args.extensions:
        image_files.extend(list(image_dir.glob(f"*{ext}")))
        image_files.extend(list(image_dir.glob(f"*{ext.upper()}")))

    image_files = sorted(list(set(image_files)))

    if len(image_files) == 0:
        print(f"No images found in {image_dir} with extensions {args.extensions}")
        return

    print(f"\nFound {len(image_files)} images in {image_dir}")

    # Setup output paths
    if args.output_json is None:
        args.output_json = image_dir / "classification_results.json"
    if args.output_csv is None:
        args.output_csv = image_dir / "classification_results.csv"

    # Load classifier
    classifier = ImageClassifier(
        classifier_ckpt=args.classifier_ckpt,
        vae_model=args.vae_model,
        num_classes=args.num_classes,
        device=args.device
    )

    # Classify images
    print(f"\nClassifying images...")
    results = []

    for image_path in tqdm(image_files, desc="Classifying"):
        try:
            result = classifier.classify_image(str(image_path))
            results.append(result)

            if args.verbose:
                print(f"\n{result['image_name']}:")
                print(f"  Predicted: {result['class_name']} (class {result['predicted_class']})")
                for i in range(args.num_classes):
                    cls_name = CLASS_NAMES[i]
                    prob = result[f'prob_{cls_name}']
                    print(f"    {cls_name}: {prob:.4f}")

        except Exception as e:
            print(f"\nError processing {image_path}: {e}")
            continue

    # Save results
    print(f"\nSaving results...")

    # Save JSON
    with open(args.output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved JSON: {args.output_json}")

    # Save CSV
    with open(args.output_csv, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        header = ['image_name', 'image_path', 'predicted_class', 'class_name']
        for i in range(args.num_classes):
            cls_name = CLASS_NAMES[i]
            header.extend([f'logit_{cls_name}', f'prob_{cls_name}'])
        writer.writerow(header)

        # Data
        for result in results:
            row = [
                result['image_name'],
                result['image_path'],
                result['predicted_class'],
                result['class_name']
            ]
            for i in range(args.num_classes):
                cls_name = CLASS_NAMES[i]
                row.extend([
                    result[f'logit_{cls_name}'],
                    result[f'prob_{cls_name}']
                ])
            writer.writerow(row)

    print(f"✓ Saved CSV: {args.output_csv}")

    # Print summary
    if args.summary:
        print_summary(results, args.num_classes)

    return results


def print_summary(results: List[Dict], num_classes: int):
    """Print classification summary statistics"""

    print("\n" + "="*80)
    print("CLASSIFICATION SUMMARY")
    print("="*80)

    # Count by class
    class_counts = {i: 0 for i in range(num_classes)}
    for result in results:
        pred_class = result['predicted_class']
        class_counts[pred_class] += 1

    total = len(results)

    print(f"\nTotal images: {total}")
    print("\nDistribution:")
    for cls_id in range(num_classes):
        count = class_counts[cls_id]
        percentage = (count / total * 100) if total > 0 else 0
        cls_name = CLASS_NAMES[cls_id]
        print(f"  {cls_name:20s} (Class {cls_id}): {count:4d} ({percentage:5.1f}%)")

    # Average probabilities per class
    print("\nAverage Probabilities:")
    for cls_id in range(num_classes):
        cls_name = CLASS_NAMES[cls_id]
        avg_prob = np.mean([r[f'prob_{cls_name}'] for r in results])
        print(f"  {cls_name:20s}: {avg_prob:.4f}")

    # High confidence predictions
    print("\nHigh Confidence Predictions (>90%):")
    for cls_id in range(num_classes):
        cls_name = CLASS_NAMES[cls_id]
        high_conf = [r for r in results
                     if r['predicted_class'] == cls_id
                     and r[f'prob_{cls_name}'] > 0.9]
        print(f"  {cls_name:20s}: {len(high_conf):4d}/{class_counts[cls_id]:4d}")

    print("="*80 + "\n")


# =========================
# Main
# =========================
def main():
    args = parse_args()

    print("="*80)
    print("IMAGE CLASSIFICATION WITH NUDITY CLASSIFIER")
    print("="*80)
    print(f"\nImage directory: {args.image_dir}")
    print(f"Classifier: {args.classifier_ckpt}")
    print(f"VAE model: {args.vae_model}")
    print(f"Device: {args.device}")
    print(f"Num classes: {args.num_classes}")

    results = classify_directory(args)

    print("\n✓ Classification complete!")


if __name__ == "__main__":
    main()

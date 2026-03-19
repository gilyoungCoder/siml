#!/usr/bin/env python
"""
Compute class-wise feature prototypes from training data for Example-Aware CG.

Supports both latent and image space classifiers:
  [latent] image -> VAE.encode() -> z0 -> latent_clf.get_features() -> 512-dim
  [image]  image -> img_clf.get_features() -> 512-dim  (no VAE needed)

Usage:
  # Latent space
  python compute_prototypes.py --space latent \
    --classifier_ckpt ./work_dirs/z0_resnet18_classifier/checkpoint/step_7700/classifier.pth \
    --output_path ./work_dirs/z0_resnet18_classifier/prototypes.pt ...

  # Image space
  python compute_prototypes.py --space image \
    --classifier_ckpt ./work_dirs/z0_img_resnet18_classifier/checkpoint/step_18900/classifier.pth \
    --output_path ./work_dirs/z0_img_resnet18_classifier/prototypes.pt ...
"""

import argparse
import os

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from diffusers import AutoencoderKL
from tqdm import tqdm

from models.latent_classifier import LatentResNet18Classifier
from models.image_classifier import build_image_classifier
from utils.dataset import ThreeClassFolderDataset

VAE_SCALE = 0.18215


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier_ckpt", type=str, required=True)
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--space", type=str, default="latent", choices=["latent", "image"],
                        help="Classifier space: 'latent' needs VAE encode, 'image' feeds pixels directly")
    parser.add_argument("--sd_model", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--benign_data_path", type=str, required=True)
    parser.add_argument("--person_data_path", type=str, nargs="+", required=True)
    parser.add_argument("--nudity_data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_samples_per_class", type=int, default=1000,
                        help="Max samples per class for prototype computation")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load VAE (only needed for latent space)
    vae = None
    if args.space == "latent":
        vae = AutoencoderKL.from_pretrained(args.sd_model, subfolder="vae").to(device)
        vae.eval()

    # Load classifier
    if args.space == "latent":
        clf = LatentResNet18Classifier(num_classes=args.num_classes, pretrained_backbone=False).to(device)
    else:
        clf = build_image_classifier(architecture="resnet18", num_classes=args.num_classes,
                                     pretrained_backbone=False).to(device)
    clf.load_state_dict(torch.load(args.classifier_ckpt, map_location=device))
    clf.eval()

    # Dataset
    transform = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    dataset = ThreeClassFolderDataset(
        args.benign_data_path, args.person_data_path, args.nudity_data_path,
        transform=transform, balance=True, seed=42,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Accumulate features per class
    feat_sums = {c: torch.zeros(512, device=device) for c in range(args.num_classes)}
    feat_counts = {c: 0 for c in range(args.num_classes)}
    max_per_class = args.max_samples_per_class

    print(f"Space: {args.space}, Classifier: {args.classifier_ckpt}")

    with torch.no_grad():
        for batch in tqdm(loader, desc="Computing features"):
            pixels = batch["pixel_values"].to(device)
            labels = batch["label"]

            if args.space == "latent":
                # image -> VAE encode -> z0 -> latent classifier features
                z0 = vae.encode(pixels).latent_dist.mean * VAE_SCALE
                feats = clf.get_features(z0)  # (B, 512)
            else:
                # image -> image classifier features directly
                feats = clf.get_features(pixels)  # (B, 512)

            for i in range(len(labels)):
                c = labels[i].item()
                if feat_counts[c] >= max_per_class:
                    continue
                feat_sums[c] += feats[i]
                feat_counts[c] += 1

            # Early stop if all classes reached max
            if all(feat_counts[c] >= max_per_class for c in range(args.num_classes)):
                break

    # Compute centroids
    prototypes = {}
    for c in range(args.num_classes):
        if feat_counts[c] > 0:
            prototypes[f"class_{c}_mean"] = (feat_sums[c] / feat_counts[c]).cpu()
            prototypes[f"class_{c}_count"] = feat_counts[c]
            print(f"  Class {c}: {feat_counts[c]} samples, ||mean||={prototypes[f'class_{c}_mean'].norm():.4f}")

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(prototypes, args.output_path)
    print(f"\nPrototypes saved to {args.output_path}")


if __name__ == "__main__":
    main()

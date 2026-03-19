"""
Compute FID score between two image directories.
Uses pytorch-fid style computation with scipy for numerical stability.
"""

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import inception_v3
from PIL import Image
from pathlib import Path
from scipy.linalg import sqrtm


class ImageFolderDataset(Dataset):
    def __init__(self, folder, transform):
        self.paths = sorted(Path(folder).glob("*.png"))
        if not self.paths:
            self.paths = sorted(Path(folder).glob("*.jpg"))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


def get_activations(folder, model, device, batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    dataset = ImageFolderDataset(folder, transform)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

    all_feats = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            feat = model(batch)
            all_feats.append(feat.cpu().numpy())
            print(f"  Processed {sum(len(f) for f in all_feats)}/{len(dataset)}", end="\r")

    print()
    return np.concatenate(all_feats, axis=0)


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            print(f"Warning: imaginary component {np.max(np.abs(covmean.imag)):.6e}")
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))


def main():
    parser = argparse.ArgumentParser(description="Compute FID between two image dirs")
    parser.add_argument("--dir1", type=str, required=True)
    parser.add_argument("--dir2", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    print(f"Computing FID:")
    print(f"  Dir1: {args.dir1} ({len(list(Path(args.dir1).glob('*.png')))} images)")
    print(f"  Dir2: {args.dir2} ({len(list(Path(args.dir2).glob('*.png')))} images)")

    device = torch.device(args.device)

    # Load InceptionV3 with pool3 features (2048-d)
    model = inception_v3(pretrained=True, transform_input=False)
    model.fc = torch.nn.Identity()
    model = model.to(device).eval()

    print("\nExtracting features from dir1...")
    act1 = get_activations(args.dir1, model, device, args.batch_size)
    print(f"  Shape: {act1.shape}")

    print("Extracting features from dir2...")
    act2 = get_activations(args.dir2, model, device, args.batch_size)
    print(f"  Shape: {act2.shape}")

    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

    score = frechet_distance(mu1, sigma1, mu2, sigma2)
    print(f"\nFID Score: {score:.4f}")
    return score


if __name__ == "__main__":
    main()

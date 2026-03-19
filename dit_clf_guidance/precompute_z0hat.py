#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Precompute z0_hat (denoised latents) for all training images.

This removes the AuraFlow (6.8B) forward pass from the training loop,
making 20000+ epoch classifier training feasible.

For each image:
  1. VAE.encode(image) -> z0
  2. For n_sigma different sigma values:
     - x_t = (1-sigma)*z0 + sigma*noise
     - v_pred = AuraFlow(x_t, sigma, uncond_emb) [frozen]
     - z0_hat = x_t - sigma*v_pred
  3. Save (z0_hat, label, sigma) to disk

Supports sharding for multi-GPU parallelism:
    # Run 8 shards in parallel (one per GPU):
    for i in $(seq 0 7); do
        CUDA_VISIBLE_DEVICES=$i python precompute_z0hat.py \
            --shard_id $i --num_shards 8 ... &
    done
    # Then merge:
    python precompute_z0hat.py --merge_shards --num_shards 8 ...
"""

import argparse
import os

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from utils.auraflow_utils import load_auraflow_components, encode_prompt, auraflow_forward
from utils.denoise_utils import predict_x0_from_velocity, inject_noise_flow
from utils.dataset import ThreeClassFolderDataset

VAE_SCALE = 0.13025


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute z0_hat for classifier training")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="purplesmartai/pony-v7-base")
    parser.add_argument("--benign_data_path", type=str, required=True)
    parser.add_argument("--person_data_path", type=str, nargs="+", required=True)
    parser.add_argument("--nudity_data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="precomputed/pony_z0hat")
    parser.add_argument("--n_sigma", type=int, default=10,
                        help="Number of sigma levels per image")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--balance_classes", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", type=str, default="bf16",
                        choices=["no", "fp16", "bf16"])
    parser.add_argument("--max_sequence_length", type=int, default=256)
    # Sharding for multi-GPU
    parser.add_argument("--shard_id", type=int, default=None,
                        help="Shard ID (0-indexed). If set, only process this shard.")
    parser.add_argument("--num_shards", type=int, default=1,
                        help="Total number of shards")
    parser.add_argument("--merge_shards", action="store_true",
                        help="Merge shard files into final organized file")
    return parser.parse_args()


def merge_shards(args):
    """Merge shard files into a single organized file."""
    print(f"Merging {args.num_shards} shards from {args.output_dir}...")

    all_z0hat = []
    all_labels = []
    all_indices = []  # original dataset indices

    for shard_id in range(args.num_shards):
        shard_path = os.path.join(args.output_dir, f"shard_{shard_id}.pt")
        if not os.path.exists(shard_path):
            raise FileNotFoundError(f"Missing shard: {shard_path}")
        data = torch.load(shard_path, map_location="cpu")
        all_z0hat.append(data["z0_hat"])
        all_labels.append(data["labels"])
        all_indices.append(data["original_indices"])
        sigma_levels = data["sigma_levels"]
        n_sigma = data["n_sigma"]
        n_total_images = data["n_total_images"]
        print(f"  Shard {shard_id}: {data['z0_hat'].shape[0]} images")

    # Concatenate
    cat_z0hat = torch.cat(all_z0hat, dim=0)
    cat_labels = torch.cat(all_labels, dim=0)
    cat_indices = torch.cat(all_indices, dim=0)

    # Reorder by original dataset index
    sorted_order = cat_indices.argsort()
    z0hat_sorted = cat_z0hat[sorted_order]
    labels_sorted = cat_labels[sorted_order]

    print(f"Merged: {z0hat_sorted.shape[0]} images, expected {n_total_images}")

    # Train/val split
    total = n_total_images
    val_size = int(0.1 * total)
    train_size = total - val_size
    gen = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(total, generator=gen)
    train_indices = perm[:train_size]
    val_indices = perm[train_size:]

    save_path = os.path.join(args.output_dir, "precomputed_organized.pt")
    torch.save({
        "z0_hat": z0hat_sorted,
        "labels": labels_sorted,
        "sigma_levels": sigma_levels,
        "n_sigma": n_sigma,
        "n_images": n_total_images,
        "train_indices": train_indices,
        "val_indices": val_indices,
    }, save_path)
    print(f"Saved organized data to {save_path}")
    print(f"  z0_hat: {z0hat_sorted.shape} ({z0hat_sorted.element_size() * z0hat_sorted.nelement() / 1e9:.2f} GB)")
    print(f"  Train: {len(train_indices)} images, Val: {len(val_indices)} images")
    print("Done!")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Merge mode: no GPU needed
    if args.merge_shards:
        merge_shards(args)
        return

    device = torch.device("cuda")
    model_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "no": torch.float32}[args.mixed_precision]

    torch.manual_seed(args.seed)

    # Load model components
    print("Loading AuraFlow / Pony V7 components...")
    components = load_auraflow_components(args.pretrained_model_name_or_path, device=device, dtype=model_dtype)
    vae = components["vae"]
    transformer = components["transformer"]
    tokenizer = components["tokenizer"]
    text_encoder = components["text_encoder"]
    scheduler = components["scheduler"]

    # Pre-encode unconditional embedding
    uncond_emb = encode_prompt(
        tokenizer, text_encoder, "",
        device=device, max_sequence_length=args.max_sequence_length, dtype=model_dtype,
    )
    # Free text encoder
    text_encoder.cpu()
    del text_encoder
    torch.cuda.empty_cache()

    time_shift = getattr(scheduler.config, "shift", 1.0)
    print(f"Time shift: {time_shift}")

    # Dataset
    res = args.resolution
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((res, res)),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    full_dataset = ThreeClassFolderDataset(
        args.benign_data_path,
        args.person_data_path,
        args.nudity_data_path,
        transform=transform,
        balance=args.balance_classes,
        seed=args.seed,
    )
    n_total = len(full_dataset)

    # Sharding: select subset of images for this shard
    if args.shard_id is not None:
        all_image_indices = list(range(n_total))
        shard_indices = all_image_indices[args.shard_id::args.num_shards]
        dataset = Subset(full_dataset, shard_indices)
        print(f"Shard {args.shard_id}/{args.num_shards}: {len(dataset)} images (of {n_total} total)")
    else:
        dataset = full_dataset
        shard_indices = list(range(n_total))

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"Processing {len(dataset)} images, {args.n_sigma} sigma levels each = {len(dataset) * args.n_sigma} total z0_hat")

    # Pre-generate sigma levels (time-shifted)
    sigma_levels = torch.linspace(0.05, 0.95, args.n_sigma)
    if time_shift > 1.0:
        sigma_levels = time_shift * sigma_levels / (1 + (time_shift - 1) * sigma_levels)
    print(f"Sigma levels: {sigma_levels.tolist()}")

    # Precompute z0_hat for each image at each sigma level
    # Store directly in organized format: (n_shard_images, n_sigma, 4, H, W)
    n_shard = len(dataset)
    n_sigma = args.n_sigma
    z0hat_organized = None  # lazy init after first batch to get spatial dims
    labels_organized = torch.zeros(n_shard, dtype=torch.long)

    total_batches = len(loader) * n_sigma
    progress = tqdm(total=total_batches, desc=f"Shard {args.shard_id}" if args.shard_id is not None else "Precomputing")

    img_offset = 0
    for batch in loader:
        imgs = batch["pixel_values"].to(device)
        labels = batch["label"]
        bsz = imgs.shape[0]

        with torch.no_grad():
            z0 = vae.encode(imgs).latent_dist.sample() * VAE_SCALE

        # Lazy init storage
        if z0hat_organized is None:
            spatial = z0.shape[1:]  # (4, H, W)
            z0hat_organized = torch.zeros(n_shard, n_sigma, *spatial, dtype=torch.float16)

        labels_organized[img_offset:img_offset+bsz] = labels

        for sigma_idx, sigma_val in enumerate(sigma_levels):
            sigma = torch.full((bsz,), sigma_val.item(), device=device)
            sigma_4d = sigma.view(-1, 1, 1, 1)
            noise = torch.randn_like(z0)
            x_t = inject_noise_flow(z0, noise, sigma_4d)

            with torch.no_grad():
                uncond_batch = uncond_emb.expand(bsz, -1, -1)
                timestep = sigma * 1000
                v_pred = auraflow_forward(
                    transformer, x_t.to(dtype=model_dtype), timestep, uncond_batch,
                ).to(x_t.dtype)

            z0_hat = predict_x0_from_velocity(x_t, v_pred, sigma_4d).detach()
            z0hat_organized[img_offset:img_offset+bsz, sigma_idx] = z0_hat.cpu().half()
            progress.update(1)

        img_offset += bsz

    progress.close()

    print(f"\nPrecomputed: z0_hat={z0hat_organized.shape}, labels={labels_organized.shape}")
    print(f"Memory: {z0hat_organized.element_size() * z0hat_organized.nelement() / 1e9:.2f} GB")

    # Save shard or full file
    if args.shard_id is not None:
        save_path = os.path.join(args.output_dir, f"shard_{args.shard_id}.pt")
        torch.save({
            "z0_hat": z0hat_organized,
            "labels": labels_organized,
            "original_indices": torch.tensor(shard_indices, dtype=torch.long),
            "sigma_levels": sigma_levels,
            "n_sigma": n_sigma,
            "n_total_images": n_total,
        }, save_path)
        print(f"Saved shard {args.shard_id} to {save_path}")
    else:
        # Single-process mode: save directly as organized
        total = n_total
        val_size = int(0.1 * total)
        train_size = total - val_size
        gen = torch.Generator().manual_seed(args.seed)
        perm = torch.randperm(total, generator=gen)
        train_indices = perm[:train_size]
        val_indices = perm[train_size:]

        save_path = os.path.join(args.output_dir, "precomputed_organized.pt")
        torch.save({
            "z0_hat": z0hat_organized,
            "labels": labels_organized,
            "sigma_levels": sigma_levels,
            "n_sigma": n_sigma,
            "n_images": n_total,
            "train_indices": train_indices,
            "val_indices": val_indices,
        }, save_path)
        print(f"Saved organized data to {save_path}")
        print(f"  Train: {len(train_indices)}, Val: {len(val_indices)}")

    print("Done!")


if __name__ == "__main__":
    main()

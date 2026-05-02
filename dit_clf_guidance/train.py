#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train a z0-space classifier for Pony V7 (AuraFlow DiT) classifier guidance.

Supports two modes:
  1. Precomputed mode (--precomputed_path): Loads precomputed z0_hat tensors.
     Fast enough for 20000+ epochs. Recommended.
  2. Online mode (--benign_data_path etc): Runs VAE+DiT each step.
     Slow (~1.9s/step), only for quick experiments.

Precomputed pipeline:
  z0_hat (precomputed, float16) -> Classifier -> logits -> CE loss -> optimize

Online pipeline:
  image -> VAE.encode() -> z0 -> flow matching noise -> DiT -> z0_hat -> Classifier

Logging: wandb with train_loss, train_acc, val_loss, val_acc, per-class accuracy,
         confusion matrix, learning rate, best checkpoint tracking.
"""

import argparse
import logging
import math
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms as transforms
from tqdm.auto import tqdm

from models.latent_classifier import LatentResNet18Classifier

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

VAE_SCALE = 0.13025
CLASS_NAMES = {0: "benign", 1: "clothed", 2: "nude"}


# ================================================================
# Datasets
# ================================================================

class PrecomputedZ0hatDataset(Dataset):
    """Dataset of precomputed z0_hat tensors and labels."""

    def __init__(self, z0_hat, labels):
        """
        Args:
            z0_hat: (N, n_sigma, 4, H, W) or (N, 4, H, W) float16 tensor
            labels: (N,) long tensor
        """
        self.z0_hat = z0_hat
        self.labels = labels
        self.has_sigma_dim = z0_hat.dim() == 5

    def __len__(self):
        return self.z0_hat.shape[0]

    def __getitem__(self, idx):
        label = self.labels[idx]
        if self.has_sigma_dim:
            # Randomly pick one sigma level per sample
            sigma_idx = torch.randint(self.z0_hat.shape[1], (1,)).item()
            z0hat = self.z0_hat[idx, sigma_idx]
        else:
            z0hat = self.z0_hat[idx]
        return {"z0_hat": z0hat.float(), "label": label}


# ================================================================
# Utilities
# ================================================================

def compute_per_class_acc(preds, labels, num_classes):
    """Returns dict of per-class accuracy."""
    accs = {}
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            accs[c] = (preds[mask] == c).float().mean().item()
        else:
            accs[c] = float("nan")
    return accs


def compute_confusion_matrix(preds, labels, num_classes):
    """Returns (num_classes, num_classes) confusion matrix."""
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for t in range(num_classes):
        for p in range(num_classes):
            cm[t, p] = ((labels == t) & (preds == p)).sum().item()
    return cm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train z0-space classifier for Pony V7 (AuraFlow) guidance"
    )
    # Precomputed mode
    parser.add_argument("--precomputed_path", type=str, default=None,
                        help="Path to precomputed_organized.pt (recommended)")
    # Online mode (fallback)
    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default="purplesmartai/pony-v7-base")
    parser.add_argument("--benign_data_path", type=str, default=None)
    parser.add_argument("--person_data_path", type=str, nargs="+", default=None)
    parser.add_argument("--nudity_data_path", type=str, default=None)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--max_sequence_length", type=int, default=256)
    parser.add_argument("--balance_classes", action="store_true", default=True)
    # Training
    parser.add_argument("--output_dir", type=str, default="work_dirs/pony_z0_resnet18")
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_train_epochs", type=int, default=20000)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--lr_scheduler", type=str, default="cosine",
                        choices=["cosine", "constant"])
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--save_ckpt_freq", type=int, default=500,
                        help="Save checkpoint every N steps")
    parser.add_argument("--val_freq", type=int, default=200,
                        help="Validate every N steps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", type=str, default="no",
                        choices=["no", "fp16", "bf16"])
    # Logging
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="pony_clf_guidance")
    parser.add_argument("--wandb_run_name", type=str, default="run1")
    parser.add_argument("--log_freq", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    if args.use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    # ================================================================
    # Load data
    # ================================================================
    if args.precomputed_path:
        logger.info(f"Loading precomputed data from {args.precomputed_path}...")
        data = torch.load(args.precomputed_path, map_location="cpu")
        z0_hat_all = data["z0_hat"]   # (n_images, n_sigma, 4, H, W) float16
        labels_all = data["labels"]    # (n_images,)
        train_indices = data["train_indices"]
        val_indices = data["val_indices"]
        n_sigma = data["n_sigma"]

        logger.info(f"  z0_hat shape: {z0_hat_all.shape}")
        logger.info(f"  n_sigma per image: {n_sigma}")
        logger.info(f"  Train: {len(train_indices)}, Val: {len(val_indices)}")

        full_dataset = PrecomputedZ0hatDataset(z0_hat_all, labels_all)
        train_ds = Subset(full_dataset, train_indices.tolist())
        val_ds = Subset(full_dataset, val_indices.tolist())

        use_online = False
    else:
        # Online mode: load DiT components
        if not all([args.benign_data_path, args.person_data_path, args.nudity_data_path]):
            raise ValueError("Either --precomputed_path or all data paths required")

        from utils.dataset import ThreeClassFolderDataset
        from utils.denoise_utils import predict_x0_from_velocity, inject_noise_flow
        from utils.auraflow_utils import load_auraflow_components, encode_prompt, auraflow_forward

        model_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "no": torch.float32}[args.mixed_precision]

        logger.info("Loading AuraFlow / Pony V7 components (online mode)...")
        components = load_auraflow_components(args.pretrained_model_name_or_path, device=device, dtype=model_dtype)
        vae = components["vae"]
        transformer = components["transformer"]
        tokenizer = components["tokenizer"]
        text_encoder = components["text_encoder"]
        scheduler = components["scheduler"]

        uncond_emb = encode_prompt(tokenizer, text_encoder, "", device=device,
                                   max_sequence_length=args.max_sequence_length, dtype=model_dtype)
        text_encoder.cpu()
        del text_encoder
        torch.cuda.empty_cache()

        time_shift = getattr(scheduler.config, "shift", 1.0)

        res = args.resolution
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((res, res)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])
        dataset = ThreeClassFolderDataset(
            args.benign_data_path, args.person_data_path, args.nudity_data_path,
            transform=transform, balance=args.balance_classes, seed=args.seed,
        )
        total = len(dataset)
        val_size = int(0.1 * total)
        train_size = total - val_size
        train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
        use_online = True

    train_loader = DataLoader(train_ds, batch_size=args.train_batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.train_batch_size, shuffle=False, num_workers=4, pin_memory=True)

    logger.info(f"Train batches/epoch: {len(train_loader)}, Val batches: {len(val_loader)}")

    # ================================================================
    # Classifier
    # ================================================================
    classifier = LatentResNet18Classifier(num_classes=args.num_classes).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # LR scheduler
    max_steps = args.max_train_steps or args.num_train_epochs * len(train_loader)
    if args.lr_scheduler == "cosine":
        def lr_lambda(step):
            if step < args.lr_warmup_steps:
                return step / max(args.lr_warmup_steps, 1)
            progress = (step - args.lr_warmup_steps) / max(max_steps - args.lr_warmup_steps, 1)
            return 0.5 * (1 + math.cos(math.pi * progress))
        scheduler_lr = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler_lr = None

    logger.info(f"Max steps: {max_steps}, LR scheduler: {args.lr_scheduler}")
    logger.info(f"Batch size: {args.train_batch_size}, LR: {args.learning_rate}")

    # ================================================================
    # Training loop
    # ================================================================
    global_step = 0
    best_val_acc = 0.0
    epoch = 0
    progress = tqdm(total=max_steps, desc="Training")

    classifier.train()
    while global_step < max_steps:
        epoch += 1
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for batch in train_loader:
            if global_step >= max_steps:
                break

            if use_online:
                # Online: run VAE + DiT
                imgs = batch["pixel_values"].to(device)
                labels = batch["label"].to(device)
                bsz = imgs.shape[0]
                with torch.no_grad():
                    z0 = vae.encode(imgs).latent_dist.sample() * VAE_SCALE
                sigma = torch.rand(bsz, device=device)
                if time_shift > 1.0:
                    sigma = time_shift * sigma / (1 + (time_shift - 1) * sigma)
                noise = torch.randn_like(z0)
                sigma_4d = sigma.view(-1, 1, 1, 1)
                x_t = inject_noise_flow(z0, noise, sigma_4d)
                with torch.no_grad():
                    uncond_batch = uncond_emb.expand(bsz, -1, -1)
                    timestep = sigma * 1000
                    v_pred = auraflow_forward(transformer, x_t.to(dtype=model_dtype), timestep, uncond_batch).to(x_t.dtype)
                z0_hat = predict_x0_from_velocity(x_t, v_pred, sigma_4d).detach()
            else:
                # Precomputed: directly load z0_hat
                z0_hat = batch["z0_hat"].to(device)
                labels = batch["label"].to(device)

            logits = classifier(z0_hat)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler_lr:
                scheduler_lr.step()

            global_step += 1
            progress.update(1)

            preds = logits.argmax(dim=-1)
            acc = (preds == labels).float().mean().item()
            epoch_loss += loss.item() * labels.shape[0]
            epoch_correct += (preds == labels).sum().item()
            epoch_total += labels.shape[0]

            current_lr = optimizer.param_groups[0]["lr"]
            progress.set_postfix(
                epoch=epoch, step=global_step, loss=f"{loss.item():.4f}",
                acc=f"{acc:.3f}", lr=f"{current_lr:.2e}",
            )

            # Wandb logging
            if args.use_wandb and global_step % args.log_freq == 0:
                import wandb
                per_class = compute_per_class_acc(preds, labels, args.num_classes)
                log_dict = {
                    "train/loss": loss.item(),
                    "train/acc": acc,
                    "train/lr": current_lr,
                    "train/epoch": epoch,
                }
                for c, name in CLASS_NAMES.items():
                    if c < args.num_classes:
                        log_dict[f"train/acc_{name}"] = per_class.get(c, float("nan"))
                wandb.log(log_dict, step=global_step)

            # Validation
            if global_step % args.val_freq == 0:
                val_loss, val_acc, val_per_class, val_cm = run_validation(
                    classifier, val_loader, loss_fn, device, args.num_classes,
                    use_online, locals() if use_online else None,
                )
                logger.info(
                    f"[Val] step={global_step}, epoch={epoch}, "
                    f"loss={val_loss:.4f}, acc={val_acc:.4f}, "
                    + ", ".join(f"{CLASS_NAMES.get(c, c)}={val_per_class[c]:.3f}" for c in range(args.num_classes))
                )

                if args.use_wandb:
                    import wandb
                    val_log = {
                        "val/loss": val_loss,
                        "val/acc": val_acc,
                    }
                    for c, name in CLASS_NAMES.items():
                        if c < args.num_classes:
                            val_log[f"val/acc_{name}"] = val_per_class.get(c, float("nan"))
                    # Confusion matrix
                    cm_table = wandb.Table(
                        columns=[""] + [CLASS_NAMES.get(c, str(c)) for c in range(args.num_classes)],
                        data=[[CLASS_NAMES.get(t, str(t))] + [int(val_cm[t, p]) for p in range(args.num_classes)]
                              for t in range(args.num_classes)]
                    )
                    val_log["val/confusion_matrix"] = cm_table
                    wandb.log(val_log, step=global_step)

                # Best model tracking
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_path = os.path.join(args.output_dir, "classifier_best.pth")
                    torch.save(classifier.state_dict(), best_path)
                    logger.info(f"  New best val_acc={val_acc:.4f}, saved to {best_path}")
                    if args.use_wandb:
                        wandb.log({"val/best_acc": best_val_acc}, step=global_step)

                classifier.train()

            # Periodic checkpoint
            if global_step % args.save_ckpt_freq == 0:
                ckpt_dir = os.path.join(args.output_dir, "checkpoint", f"step_{global_step}")
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save(classifier.state_dict(), os.path.join(ckpt_dir, "classifier.pth"))

        # Epoch summary
        if epoch_total > 0:
            epoch_avg_loss = epoch_loss / epoch_total
            epoch_avg_acc = epoch_correct / epoch_total
            if args.use_wandb and epoch % 100 == 0:
                import wandb
                wandb.log({
                    "epoch/loss": epoch_avg_loss,
                    "epoch/acc": epoch_avg_acc,
                    "epoch/num": epoch,
                }, step=global_step)

    progress.close()

    # Save final
    final_path = os.path.join(args.output_dir, "classifier_final.pth")
    torch.save(classifier.state_dict(), final_path)
    logger.info(f"Final model saved to {final_path}")
    logger.info(f"Best val_acc: {best_val_acc:.4f}")

    if args.use_wandb:
        import wandb
        wandb.finish()


def run_validation(classifier, val_loader, loss_fn, device, num_classes,
                   use_online=False, online_ctx=None):
    """Run full validation and return metrics."""
    classifier.eval()
    all_preds = []
    all_labels = []
    val_loss_sum = 0.0
    val_total = 0

    with torch.no_grad():
        for vb in val_loader:
            if use_online and online_ctx:
                # Online mode validation
                from utils.denoise_utils import predict_x0_from_velocity, inject_noise_flow
                from utils.auraflow_utils import auraflow_forward
                vae = online_ctx["vae"]
                transformer = online_ctx["transformer"]
                uncond_emb = online_ctx["uncond_emb"]
                model_dtype = online_ctx["model_dtype"]
                time_shift = online_ctx["time_shift"]

                vimgs = vb["pixel_values"].to(device)
                vlabels = vb["label"].to(device)
                vbsz = vimgs.shape[0]
                vz0 = vae.encode(vimgs).latent_dist.sample() * VAE_SCALE
                vsigma = torch.rand(vbsz, device=device)
                if time_shift > 1.0:
                    vsigma = time_shift * vsigma / (1 + (time_shift - 1) * vsigma)
                vnoise = torch.randn_like(vz0)
                vsigma_4d = vsigma.view(-1, 1, 1, 1)
                vx_t = inject_noise_flow(vz0, vnoise, vsigma_4d)
                vuncond = uncond_emb.expand(vbsz, -1, -1)
                vtimestep = vsigma * 1000
                vv_pred = auraflow_forward(transformer, vx_t.to(dtype=model_dtype), vtimestep, vuncond).to(vx_t.dtype)
                z0_hat = predict_x0_from_velocity(vx_t, vv_pred, vsigma_4d).detach()
            else:
                z0_hat = vb["z0_hat"].to(device)
                vlabels = vb["label"].to(device)

            vlogits = classifier(z0_hat)
            vloss = loss_fn(vlogits, vlabels)
            vpreds = vlogits.argmax(dim=-1)

            val_loss_sum += vloss.item() * vlabels.shape[0]
            val_total += vlabels.shape[0]
            all_preds.append(vpreds.cpu())
            all_labels.append(vlabels.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    val_loss = val_loss_sum / max(val_total, 1)
    val_acc = (all_preds == all_labels).float().mean().item()
    per_class = compute_per_class_acc(all_preds, all_labels, num_classes)
    cm = compute_confusion_matrix(all_preds, all_labels, num_classes)

    classifier.train()
    return val_loss, val_acc, per_class, cm


if __name__ == "__main__":
    main()

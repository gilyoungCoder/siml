#!/usr/bin/env python3
import argparse
import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler
from geo_models.classifier.classifier import load_discriminator

from huggingface_hub import HfFolder, whoami, Repository
import wandb
from tqdm.auto import tqdm

logger = get_logger(__name__)

class SixteenClassDataset(Dataset):
    """
    15-class dataset:
     0: Not People
     1: fully_clothed
     2: casual_wear
     3: summer_casual
     4: athletic_wear
     5: one_piece_swimwear
     6: bikini_swimwear
     7: lingerie
     8: topless_with_jeans
     9: implied_nude
    10: artistic_full_nude
    11: monet_full_style
    12: monet_light_style
    13: vangogh_full_style
    14: vangogh_light_style
    15: benign_arts
    """
    def __init__(
        self,
        not_people_dir: str,
        fully_clothed_dir: str,
        casual_wear_dir: str,
        summer_casual_dir: str,
        athletic_wear_dir: str,
        one_piece_swimwear_dir: str,
        bikini_swimwear_dir: str,
        lingerie_dir: str,
        topless_with_jeans_dir: str,
        implied_nude_dir: str,
        artistic_full_nude_dir: str,
        monet_full_style_dir: str,
        monet_light_style_dir: str,
        vangogh_full_style_dir: str,
        vangogh_light_style_dir: str,
        benign_arts_dir: str,
        transform=None,
    ):
        self.paths = []
        self.labels = []
        dirs = [
            (not_people_dir,          0),
            (fully_clothed_dir,       1),
            (casual_wear_dir,         2),
            (summer_casual_dir,       3),
            (athletic_wear_dir,       4),
            (one_piece_swimwear_dir,  5),
            (bikini_swimwear_dir,     6),
            (lingerie_dir,            7),
            (topless_with_jeans_dir,  8),
            (implied_nude_dir,        9),
            (artistic_full_nude_dir, 10),
            (monet_full_style_dir,   11),
            (monet_light_style_dir,  12),
            (vangogh_full_style_dir, 13),
            (vangogh_light_style_dir,14),
            (benign_arts_dir,        15),
        ]
        for d, label in dirs:
            for fname in sorted(os.listdir(d)):
                if fname.lower().endswith(".png"):
                    self.paths.append(os.path.join(d, fname))
                    self.labels.append(label)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return {"pixel_values": img, "label": self.labels[idx]}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a 15-class classifier of nudity/person levels + Monet/Vangogh styles with DDPM noise injection"
    )
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--not_people_data_path",        type=str, required=True)
    parser.add_argument("--fully_clothed_data_path",     type=str, required=True)
    parser.add_argument("--casual_wear_data_path",       type=str, required=True)
    parser.add_argument("--summer_casual_data_path",     type=str, required=True)
    parser.add_argument("--athletic_wear_data_path",     type=str, required=True)
    parser.add_argument("--one_piece_swimwear_path",     type=str, required=True)
    parser.add_argument("--bikini_swimwear_path",        type=str, required=True)
    parser.add_argument("--lingerie_data_path",          type=str, required=True)
    parser.add_argument("--topless_with_jeans_path",     type=str, required=True)
    parser.add_argument("--implied_nude_data_path",      type=str, required=True)
    parser.add_argument("--artistic_full_nude_path",     type=str, required=True)
    parser.add_argument("--monet_full_style_path",       type=str, required=True)
    parser.add_argument("--monet_light_style_path",      type=str, required=True)
    parser.add_argument("--vangogh_full_style_path",     type=str, required=True)
    parser.add_argument("--vangogh_light_style_path",    type=str, required=True)
    parser.add_argument("--benign_arts_path",            type=str, required=True)
    parser.add_argument("--output_dir",                  type=str, default="sixteen_class_output")
    parser.add_argument("--seed",                        type=int, default=None)
    parser.add_argument("--train_batch_size",            type=int, default=16)
    parser.add_argument("--num_train_epochs",            type=int, default=30)
    parser.add_argument("--max_train_steps",             type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate",               type=float, default=1e-4)
    parser.add_argument("--use_wandb",                   action="store_true")
    parser.add_argument("--wandb_project",               type=str, default="sixteen_class_project")
    parser.add_argument("--wandb_run_name",              type=str, default="run1")
    parser.add_argument("--mixed_precision",             type=str, choices=["no","fp16","bf16"], default="no")
    parser.add_argument("--report_to",                   type=str, default="tensorboard")
    parser.add_argument("--push_to_hub",                 action="store_true")
    parser.add_argument("--hub_token",                   type=str, default=None)
    parser.add_argument("--hub_model_id",                type=str, default=None)
    parser.add_argument("--save_ckpt_freq",              type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.use_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
    )
    device = accelerator.device
    logger.info(f"Using device: {device}")

    if args.seed is not None:
        seed = args.seed + accelerator.process_index
        set_seed(seed)
        logger.info(f"Set random seed to: {seed}")

    if accelerator.is_local_main_process and args.push_to_hub:
        repo_name = args.hub_model_id or f"{whoami(HfFolder.get_token())['name']}/{Path(args.output_dir).name}"
        repo = Repository(args.output_dir, clone_from=repo_name)
        with open(os.path.join(args.output_dir, ".gitignore"), "w") as f:
            f.write("checkpoint/**\n")

    # VAE & scheduler
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    vae.requires_grad_(False)
    vae.to(device)
    scheduler = DDPMScheduler.from_config(args.pretrained_model_name_or_path, subfolder="scheduler")

    # Transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((512,512)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    # Dataset & Dataloader
    full_dataset = SixteenClassDataset(
        args.not_people_data_path,
        args.fully_clothed_data_path,
        args.casual_wear_data_path,
        args.summer_casual_data_path,
        args.athletic_wear_data_path,
        args.one_piece_swimwear_path,
        args.bikini_swimwear_path,
        args.lingerie_data_path,
        args.topless_with_jeans_path,
        args.implied_nude_data_path,
        args.artistic_full_nude_path,
        args.monet_full_style_path,
        args.monet_light_style_path,
        args.vangogh_full_style_path,
        args.vangogh_light_style_path,
        args.benign_arts_path,
        transform=transform,
    )
    # original = len(full_dataset)
    # total = int(0.1*original)
    # rest = original - total
    # print(f"total len: {total}")
    # g = torch.Generator().manual_seed(42)
    # small_ds, _ = torch.utils.data.random_split(full_dataset, [total, rest], generator=g)
    total = len(full_dataset)
    print(f"total len: {total}")
    val_size = int(0.1 * total)
    train_size = total - val_size
    train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.train_batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=args.train_batch_size, shuffle=False, num_workers=4)

    # Model: 16-way classifier
    # classifier = load_discriminator(
    #     ckpt_path=None,
    #     condition=None,
    #     eval=False,
    #     channel=4,
    #     num_classes=16,
    # ).to(device)
    classifier = load_discriminator(
        ckpt_path=None,
        condition=None,
        eval=False,
        channel=4,
        num_classes=16,
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.learning_rate)

    classifier, optimizer, train_loader, val_loader = accelerator.prepare(
        classifier, optimizer, train_loader, val_loader
    )

    # Training loop
    global_step = 0
    max_steps = args.max_train_steps or args.num_train_epochs * len(train_loader)
    progress = tqdm(range(max_steps), disable=not accelerator.is_local_main_process)

    classifier.train()
    while global_step < max_steps:
        for batch in train_loader:
            imgs   = batch["pixel_values"].to(device)
            labels = batch["label"].to(device)

            # VAE encode
            latents = vae.encode(imgs).latent_dist.sample() * 0.18215

            # noise injection
            bsz = latents.shape[0]
            timesteps = torch.randint(0, scheduler.num_train_timesteps, (bsz,), device=device)
            noise      = torch.randn_like(latents)
            alpha_bar  = scheduler.alphas_cumprod.to(device)[timesteps].view(bsz,1,1,1)
            noisy_latents = torch.sqrt(alpha_bar)*latents + torch.sqrt(1-alpha_bar)*noise

            # forward
            norm_ts = timesteps / scheduler.num_train_timesteps
            logits  = classifier(noisy_latents, norm_ts)
            loss    = loss_fn(logits, labels)

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            global_step +=1
            progress.update(1)
            progress.set_postfix(step=global_step, loss=loss.item())

            # checkpoint & validation
            if global_step % args.save_ckpt_freq == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_local_main_process:
                    ckpt_dir = os.path.join(args.output_dir, "checkpoint", f"step_{global_step}")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    torch.save(
                        accelerator.unwrap_model(classifier).state_dict(),
                        os.path.join(ckpt_dir, "classifier.pth")
                    )
                # validation
                classifier.eval()
                val_loss, val_acc, total_val = 0.0, 0.0, 0
                with torch.no_grad():
                    for vb in val_loader:
                        vimgs, vlabels = vb["pixel_values"].to(device), vb["label"].to(device)
                        vlat = vae.encode(vimgs).latent_dist.sample() * 0.18215
                        vtimesteps = torch.randint(0, scheduler.num_train_timesteps, (vlat.shape[0],), device=device)
                        vnoise      = torch.randn_like(vlat)
                        valpha_bar  = scheduler.alphas_cumprod.to(device)[vtimesteps].view(vlat.shape[0],1,1,1)
                        noisy_vlat  = torch.sqrt(valpha_bar)*vlat + torch.sqrt(1-valpha_bar)*vnoise

                        vnorm_ts = vtimesteps / scheduler.num_train_timesteps
                        vlogits  = classifier(noisy_vlat, vnorm_ts)
                        vloss    = loss_fn(vlogits, vlabels)
                        preds    = vlogits.argmax(dim=-1)

                        val_loss += vloss.item() * vlat.shape[0]
                        val_acc  += (preds==vlabels).sum().item()
                        total_val += vlat.shape[0]

                val_loss /= total_val
                val_acc  /= total_val
                logger.info(f"[Validation] Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
                if args.use_wandb:
                    wandb.log({"val_loss": val_loss, "val_acc": val_acc}, step=global_step)
                classifier.train()

            if global_step >= max_steps:
                break

    # 마지막 모델 저장
    if accelerator.is_local_main_process:
        final_path = os.path.join(args.output_dir, "classifier_final.pth")
        torch.save(accelerator.unwrap_model(classifier).state_dict(), final_path)
        if args.push_to_hub:
            repo.push_to_hub(commit_message="Final model")

    accelerator.end_training()


if __name__ == "__main__":
    main()

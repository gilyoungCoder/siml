#!/usr/bin/env python3
# train.py

import argparse, os
from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler
from geo_models.classifier.classifier import load_discriminator
import wandb
from tqdm.auto import tqdm

logger = get_logger(__name__)

class FourClassDataset(Dataset):
    def __init__(self, not_people_dir, fully_clothed_dir,
                 partial_nude_dir, full_nude_dir, transform=None):
        samples = []
        for root, label in [
            (not_people_dir, 0), (fully_clothed_dir, 1),
            (partial_nude_dir, 2), (full_nude_dir, 3),
        ]:
            for fname in sorted(os.listdir(root)):
                if fname.lower().endswith((".png",".jpg",".jpeg",".webp")):
                    samples.append((os.path.join(root, fname), label))
        self.samples = samples
        self.transform = transform

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform: img = self.transform(img)
        return {"pixel_values": img, "label": label}


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--not_people_data_path",    required=True)
    p.add_argument("--fully_clothed_data_path", required=True)
    p.add_argument("--partial_nude_data_path",  required=True)
    p.add_argument("--full_nude_data_path",     required=True)
    p.add_argument("--output_dir",    default="clf4_out")
    p.add_argument("--sdxl_base",     default="stabilityai/stable-diffusion-xl-base-1.0")
    p.add_argument("--train_batch_size", type=int, default=32)
    p.add_argument("--learning_rate",    type=float, default=1e-4)
    p.add_argument("--num_train_epochs", type=int, default=30)
    p.add_argument("--max_train_steps",  type=int, default=None)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--epsilon", type=float, default=1e-2)  # adversarial step size
    p.add_argument("--adv_iters", type=int, default=1)     # # adversarial steps
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--mixed_precision", choices=["no","fp16","bf16"], default="no")
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", default="nudity4")
    p.add_argument("--wandb_run_name", default="run_adv")
    return p.parse_args()


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.use_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
        wandb.watch_called = False

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="wandb" if args.use_wandb else None
    )
    device = accelerator.device
    if args.seed is not None: set_seed(args.seed + accelerator.process_index)

    vae = AutoencoderKL.from_pretrained(args.sdxl_base, subfolder="vae").to(device)
    vae.requires_grad_(False)
    scheduler = DDPMScheduler.from_pretrained(
        args.sdxl_base, subfolder="scheduler", num_train_timesteps=1000, clip_sample=False)
    alphas = scheduler.alphas_cumprod.to(device)
    sqrt_a  = alphas.sqrt().view(1000,1,1,1)
    sqrt_1a = (1 - alphas).sqrt().view(1000,1,1,1)

    tfm = transforms.Compose([
        transforms.Resize((1024,1024)), transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    ds = FourClassDataset(args.not_people_data_path,
                          args.fully_clothed_data_path,
                          args.partial_nude_data_path,
                          args.full_nude_data_path,
                          transform=tfm)
    train_len = int(len(ds)*0.9)
    train_ds, val_ds = torch.utils.data.random_split(ds, [train_len, len(ds)-train_len])
    train_loader = DataLoader(train_ds, batch_size=args.train_batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds, batch_size=args.train_batch_size, shuffle=False, num_workers=4)

    clf = load_discriminator(ckpt_path=None, condition=None, eval=False, channel=4, num_classes=4).to(device)
    if args.mixed_precision=="fp16": clf = clf.half()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(clf.parameters(), lr=args.learning_rate)

    clf, optimizer, train_loader, val_loader = accelerator.prepare(clf, optimizer, train_loader, val_loader)

    max_steps = args.max_train_steps or (args.num_train_epochs * len(train_loader))
    pbar = tqdm(range(max_steps), disable=not accelerator.is_local_main_process)
    step = 0
    best_val_acc = 0.0

    clf.train()
    while step < max_steps:
        for batch in train_loader:
            imgs   = batch["pixel_values"].to(device)
            labels = batch["label"].to(device)

            # latent + noise
            with torch.no_grad():
                lat = vae.encode(imgs).latent_dist.sample() * 0.18215
            bsz = lat.size(0)
            ts = torch.randint(0, 1000, (bsz,), device=device)
            noise = torch.randn_like(lat)
            noisy_lat = sqrt_a[ts] * lat + sqrt_1a[ts] * noise

            # adversarial latent
            adv_lat = noisy_lat.clone().detach().requires_grad_(True)
            for _ in range(args.adv_iters):
                logits = clf(adv_lat, ts.float()/1000.0)
                loss = -loss_fn(logits, labels)  # gradient ascent
                grad = torch.autograd.grad(loss, adv_lat)[0]
                adv_lat = adv_lat + args.epsilon * grad.sign()
                adv_lat = adv_lat.detach().requires_grad_(True)

            logits = clf(adv_lat, ts.float()/1000.0)
            loss = loss_fn(logits, labels)
            accelerator.backward(loss)
            optimizer.step(); optimizer.zero_grad()

            step += 1
            pbar.update(1)
            pbar.set_postfix(step=step, loss=loss.item())
            if args.use_wandb:
                accelerator.log({"train_loss": loss.item()}, step=step)

            if step % 100 == 0:
                accelerator.wait_for_everyone()
                clf.eval(); tot = corr = vloss = 0.0
                with torch.no_grad():
                    for vb in val_loader:
                        imgs = vb["pixel_values"].to(device)
                        labels = vb["label"].to(device)
                        lat = vae.encode(imgs).latent_dist.sample() * 0.18215
                        ts = torch.randint(0,1000,(lat.size(0),),device=device)
                        noisy = sqrt_a[ts]*lat + sqrt_1a[ts]*torch.randn_like(lat)
                        logits = clf(noisy, ts.float()/1000.0)
                        vloss += loss_fn(logits, labels).item() * lat.size(0)
                        corr += (logits.argmax(-1)==labels).sum().item()
                        tot += lat.size(0)
                val_loss = vloss / tot
                val_acc = corr / tot
                logger.info(f"[VAL @ {step}] loss={val_loss:.4f} acc={val_acc:.4f}")
                if args.use_wandb:
                    wandb.log({"val_loss": val_loss, "val_acc": val_acc}, step=step)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    out = Path(args.output_dir)/"checkpoint"/f"step_{step:05d}"
                    out.mkdir(parents=True, exist_ok=True)
                    torch.save(accelerator.unwrap_model(clf).state_dict(), out/"classifier.pth")
                clf.train()

            if step >= max_steps:
                break

    if accelerator.is_local_main_process:
        torch.save(accelerator.unwrap_model(clf).state_dict(),
                   Path(args.output_dir)/"classifier_final.pth")
    accelerator.end_training()


if __name__ == "__main__":
    main()

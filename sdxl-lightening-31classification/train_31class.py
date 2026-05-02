#!/usr/bin/env python3
# train_31class.py – SDXL-Lightning (50-step DDIM) 31-class classifier

import argparse, os, sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.optimization import get_scheduler
from geo_models.classifier.classifier import load_discriminator
from tqdm.auto import tqdm
import wandb

logger = get_logger(__name__)

STYLE_30 = [
    "coat_full","longsleeve_jeans","shorttee_jeans","tank_skirt","microtop_hotpants",
    "shirtless_shorts","swim_dress","onepiece_swim","bikini_standard","bikini_string",
    "lingerie_teddy","lingerie_sheer","bra_panty","bra_only","topless_armcover",
    "topless_haircover","topless_back","draped_nude","silhouette_nude","bodyscape_macro",
    "nude_side","nude_shadow","statue_nude","oiled_nude","standing_nude","maternity_nude",
    "bodypaint_nude","censored_nude","nude_full","nude_explicit"
]

class ThirtyOneDS(Dataset):
    def __init__(self, not_people_dir, classes30_dir, transform=None):
        self.samples = []
        p0 = Path(not_people_dir)
        if not p0.is_dir(): 
            raise ValueError(f"{p0} not found")
        for f in sorted(p0.iterdir()):
            if f.suffix.lower() in (".png",".jpg",".jpeg",".webp"):
                self.samples.append((str(f), 0))
        root = Path(classes30_dir)
        for idx, cls in enumerate(STYLE_30, start=1):
            d = root/cls
            if not d.is_dir(): 
                raise ValueError(f"{d} not found")
            for f in sorted(d.iterdir()):
                if f.suffix.lower() in (".png",".jpg",".jpeg",".webp"):
                    self.samples.append((str(f), idx))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return {"pixel_values": img, "label": label}

def get_args():
    p = argparse.ArgumentParser("31-class nudity classifier")
    p.add_argument("--not_people_data_path", required=True,
                   help="예: /mnt/home/.../imagenet")
    p.add_argument("--classes30_dir", required=True,
                   help="예: /mnt/home/.../30class")
    p.add_argument("--output_dir", default="clf31_out")
    p.add_argument("--sdxl_base", default="stabilityai/stable-diffusion-xl-base-1.0")
    p.add_argument("--train_batch_size", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--lr_scheduler", choices=["cosine","linear","none"], default="cosine",
                   help="LR scheduler type")
    p.add_argument("--num_train_epochs", type=int, default=30)
    p.add_argument("--max_train_steps", type=int, default=None)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mixed_precision", choices=["no","fp16","bf16"], default="no")
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", default="nudity31")
    p.add_argument("--wandb_run_name", default="run1")
    p.add_argument("--train_timesteps", type=int, default=1000,
                   help="Inference와 일치시키기 위한 DDPM timesteps override")
    return p.parse_args()

if __name__ == "__main__":
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
            reinit=True
        )

    # Accelerator 및 시드 설정
    acc = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="wandb" if args.use_wandb else None,
    )
    dev = acc.device
    set_seed(args.seed + acc.process_index)

    # ─── VAE 로드
    vae = AutoencoderKL.from_pretrained(args.sdxl_base, subfolder="vae").to(dev)
    vae.requires_grad_(False)

    # ─── Scheduler 로드 (기본 1000) + override
    sched = DDPMScheduler.from_pretrained(
        pretrained_model_name_or_path=args.sdxl_base,
        subfolder="scheduler",
        num_train_timesteps=1000,
        clip_sample=False
    )
    print(sched.timesteps)  

    T = len(sched.timesteps)

    alphas = sched.alphas_cumprod.to(dev)
    sa = alphas.sqrt().view(-1,1,1,1)[:T]
    sb = (1 - alphas).sqrt().view(-1,1,1,1)[:T]

    # ─── DataLoader 정의 (고정 시드, pin_memory)
    tfm = transforms.Compose([
        transforms.Resize((1024,1024)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    ds = ThirtyOneDS(args.not_people_data_path, args.classes30_dir, tfm)
    val_len   = int(0.1 * len(ds))
    train_len = len(ds) - val_len
    tr_ds, va_ds = torch.utils.data.random_split(
        ds, [train_len, val_len],
        generator=torch.Generator().manual_seed(args.seed)
    )
    tr_loader = DataLoader(
        tr_ds, batch_size=args.train_batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    va_loader = DataLoader(
        va_ds, batch_size=args.train_batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )

    # ─── Classifier, Optimizer, LR Scheduler
    clf = load_discriminator(
        ckpt_path="./work_dirs/31cls1024/classifier_final.pth", condition=None,
        eval=False, channel=4, num_classes=31
    ).to(dev)
    opt = torch.optim.AdamW(clf.parameters(), lr=args.learning_rate)
    total_steps = args.max_train_steps or args.num_train_epochs * len(tr_loader)
    if args.lr_scheduler != "none":
        lr_sched = get_scheduler(
            name=args.lr_scheduler,
            optimizer=opt,
            num_warmup_steps=int(0.1*total_steps),
            num_training_steps=total_steps
        )
    else:
        lr_sched = None

    ce  = nn.CrossEntropyLoss()
    clf, opt, tr_loader, va_loader = acc.prepare(
        clf, opt, tr_loader, va_loader
    )
    if lr_sched:
        lr_sched = acc.prepare(lr_sched)

    # ─── Training Loop (Early Stopping 적용)
    best_val_acc = 0.0
    no_improve = 0
    patience = 10

    step = 0
    pbar = tqdm(range(total_steps), disable=not acc.is_local_main_process)
    clf.train()
    while step < total_steps and no_improve < patience:
        for batch in tr_loader:
            lat = vae.encode(batch["pixel_values"].to(dev)).latent_dist.sample() * 0.18215
            bs  = lat.size(0)

            ts = torch.randint(0, T, (bs,), device=dev)
            noisy = sa[ts] * lat + sb[ts] * torch.randn_like(lat)
            ts_norm = ts.float() / T

            out  = clf(noisy, ts_norm)
            loss = ce(out, batch["label"].to(dev))

            acc.backward(loss)
            opt.step()
            opt.zero_grad()

            if lr_sched:
                lr_sched.step()

            step += 1
            pbar.update(1)
            pbar.set_postfix(step=step, loss=loss.item(),
                             lr=opt.param_groups[0]["lr"])

            # WandB 추가 로깅
            if args.use_wandb and acc.is_local_main_process:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/lr": opt.param_groups[0]["lr"],
                    "grad/norm": torch.nn.utils.clip_grad_norm_(clf.parameters(), max_norm=1.0)
                }, step=step)

            # 검증
            if acc.is_local_main_process and step % 100 == 0:
                clf.eval()
                val_loss, val_acc, total = 0.0, 0.0, 0
                with torch.no_grad():
                    for vb in va_loader:
                        vimgs  = vb["pixel_values"].to(dev)
                        vlabels= vb["label"].to(dev)
                        vlat   = vae.encode(vimgs).latent_dist.sample() * 0.18215
                        vbs    = vlat.size(0)
                        vts    = torch.randint(0, T, (vbs,), device=dev)
                        noisy_v = sa[vts] * vlat + sb[vts] * torch.randn_like(vlat)
                        vts_norm= vts.float() / T
                        vlogits = clf(noisy_v, vts_norm)
                        vloss   = ce(vlogits, vlabels)
                        preds   = vlogits.argmax(dim=-1)

                        val_loss += vloss.item() * vbs
                        val_acc  += (preds == vlabels).sum().item()
                        total    += vbs

                val_loss /= total
                val_acc  /= total
                logger.info(f"[Validation @ Step {step}] Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
                if args.use_wandb:
                    wandb.log({
                        "val/loss": val_loss,
                        "val/acc": val_acc
                    }, step=step)

                # Early stopping 체크
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    no_improve = 0
                    ckpt_dir = Path(args.output_dir) / "checkpoint" / f"step_{step:05d}"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(acc.unwrap_model(clf).state_dict(), ckpt_dir / "classifier.pth")
                else:
                    no_improve += 1

                clf.train()
                if no_improve >= patience:
                    logger.info(f"No improvement for {patience} evals. Early stopping.")
                    break

            if step >= total_steps or no_improve >= patience:
                break

        acc.wait_for_everyone()

    # ─── 최종 저장
    if acc.is_local_main_process:
        torch.save(acc.unwrap_model(clf).state_dict(),
                   Path(args.output_dir) / "classifier_final.pth")
        logger.info("Training complete, final model saved.")

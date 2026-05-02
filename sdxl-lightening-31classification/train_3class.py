#!/usr/bin/env python3
# train_3class.py

"""
3-class (not-people / fully-clothed / nude-people) classifier for SDXL-Lightning
with DDPM noise-injected latents.
"""

import argparse, os, logging
from pathlib import Path

import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils   import set_seed
from diffusers import AutoencoderKL, DDPMScheduler

from geo_models.classifier.classifier import load_discriminator
from huggingface_hub import HfFolder, Repository, whoami
import wandb
from tqdm.auto import tqdm

logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
#  Dataset                                                                    #
# --------------------------------------------------------------------------- #

class ThreeClassDataset(Dataset):
    """0: not_people  1: fully_clothed  2: nude_people"""
    def __init__(self,
                 not_people_dir: str,
                 fully_clothed_dir: str,
                 nude_people_dir: str,
                 transform=None):
        self.samples = []
        for root, label in [(not_people_dir, 0),
                            (fully_clothed_dir, 1),
                            (nude_people_dir, 2)]:
            for fname in sorted(os.listdir(root)):
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                    self.samples.append((os.path.join(root, fname), label))
        self.transform = transform

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform: img = self.transform(img)
        return {"pixel_values": img, "label": label}

# --------------------------------------------------------------------------- #
#  Argument parsing                                                           #
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser("3-class nudity classifier (Lightning)")
    # ---- data ----
    p.add_argument("--not_people_data_path",   required=True)
    p.add_argument("--fully_clothed_data_path",required=True)
    p.add_argument("--nude_people_data_path",  required=True)
    # ---- model / output ----
    p.add_argument("--output_dir", type=str, default="clf3_lightning_out")
    p.add_argument("--sdxl_base", type=str,
                   default="stabilityai/stable-diffusion-xl-base-1.0",
                   help="SDXL base ckpt (VAE & scheduler 로드용)")
    # ---- training hyper-params ----
    p.add_argument("--train_batch_size", type=int, default=32)
    p.add_argument("--learning_rate",   type=float, default=1e-4)
    p.add_argument("--num_train_epochs",type=int, default=30)
    p.add_argument("--max_train_steps", type=int, default=None)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    # ---- misc ----
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--mixed_precision", choices=["no","fp16","bf16"], default="no")
    # logging / tracking
    p.add_argument("--report_to", type=str, default="wandb",
                   help='["wandb", "tensorboard", "none"]')
    p.add_argument("--logging_dir", type=str, default="runs",
                   help="tensorboard 로그 디렉터리 (필요 시)")
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", default="nudity3")
    p.add_argument("--wandb_run_name", default="run1")
    # hub
    p.add_argument("--push_to_hub", action="store_true")
    p.add_argument("--hub_model_id", type=str, default=None)
    return p.parse_args()

# --------------------------------------------------------------------------- #
#  Main                                                                       #
# --------------------------------------------------------------------------- #

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ---------------- WandB ----------------
    if args.use_wandb and args.report_to == "wandb":
        wandb.init(project=args.wandb_project,
                   name=args.wandb_run_name,
                   config=vars(args))

    # -------------- Accelerator ------------
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=None if args.report_to.lower()=="none" else args.report_to,
    )
    device = accelerator.device
    logger.info(f"Device -> {device}")

    # -------------- Seed -------------------
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)

    # -------------- HF Hub push ------------
    if accelerator.is_local_main_process and args.push_to_hub:
        repo_name = args.hub_model_id or \
            f"{whoami(HfFolder.get_token())['name']}/{Path(args.output_dir).name}"
        repo = Repository(args.output_dir, clone_from=repo_name)
        with open(Path(args.output_dir)/".gitignore","w") as f: f.write("checkpoint/**\n")

    # -------------- VAE / Scheduler --------
    vae = AutoencoderKL.from_pretrained(args.sdxl_base, subfolder="vae").to(device)
    vae.requires_grad_(False)
    scheduler = DDPMScheduler.from_config(args.sdxl_base, subfolder="scheduler")

    # -------------- Data -------------------
    tfm = transforms.Compose([
        transforms.Resize((512,512)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    dataset = ThreeClassDataset(
        args.not_people_data_path,
        args.fully_clothed_data_path,
        args.nude_people_data_path,
        transform=tfm
    )
    total = len(dataset); val_len = int(0.1*total); train_len = total-val_len
    train_ds, val_ds = torch.utils.data.random_split(dataset,[train_len,val_len])
    train_loader = DataLoader(train_ds,batch_size=args.train_batch_size,
                              shuffle=True,num_workers=4)
    val_loader   = DataLoader(val_ds,  batch_size=args.train_batch_size,
                              shuffle=False,num_workers=4)

    # -------------- Classifier -------------
    classifier = load_discriminator(
        ckpt_path=None,
        condition=None,
        eval=False,
        channel=4,
        num_classes=3,
    ).to(device)

    opt  = torch.optim.AdamW(classifier.parameters(), lr=args.learning_rate)
    ce   = nn.CrossEntropyLoss()

    classifier, opt, train_loader, val_loader = accelerator.prepare(
        classifier, opt, train_loader, val_loader
    )

    # -------------- Train loop -------------
    max_steps = args.max_train_steps or args.num_train_epochs*len(train_loader)
    prog = tqdm(range(max_steps), disable=not accelerator.is_local_main_process)

    step = 0
    classifier.train()
    while step < max_steps:
        for batch in train_loader:
            imgs   = batch["pixel_values"].to(device)
            labels = batch["label"].to(device)

            # encode → noise injection
            lat   = vae.encode(imgs).latent_dist.sample()*0.18215
            bsz   = lat.size(0)
            ts    = torch.randint(0, scheduler.num_train_timesteps, (bsz,), device=device)
            noise = torch.randn_like(lat)
            alpha = scheduler.alphas_cumprod.to(device)[ts].view(bsz,1,1,1)
            noisy = torch.sqrt(alpha)*lat + torch.sqrt(1-alpha)*noise

            logits = classifier(noisy, ts / scheduler.num_train_timesteps)
            loss   = ce(logits, labels)

            accelerator.backward(loss)
            opt.step(); opt.zero_grad()

            step += 1; prog.update(1)
            prog.set_postfix(loss=loss.item(), step=step)

            if step >= max_steps: break

        # ---- simple val each epoch ----
        accelerator.wait_for_everyone()
        if accelerator.is_local_main_process:
            classifier.eval()
            tot, correct, vloss = 0, 0, 0.0
            with torch.no_grad():
                for vb in val_loader:
                    vimg = vb["pixel_values"].to(device)
                    vlbl = vb["label"].to(device)
                    vlat = vae.encode(vimg).latent_dist.sample()*0.18215
                    vt   = torch.randint(0,scheduler.num_train_timesteps,
                                         (vlat.size(0),),device=device)
                    vnoise = torch.randn_like(vlat)
                    alpha  = scheduler.alphas_cumprod.to(device)[vt].view(vlat.size(0),1,1,1)
                    vnoisy = torch.sqrt(alpha)*vlat + torch.sqrt(1-alpha)*vnoise
                    vlogit = classifier(vnoisy, vt / scheduler.num_train_timesteps)
                    vloss += ce(vlogit, vlbl).item()*vlat.size(0)
                    pred  = vlogit.argmax(-1)
                    correct += (pred==vlbl).sum().item()
                    tot     += vlat.size(0)
            vloss/=tot; vacc=correct/tot
            logger.info(f"[Val] loss={vloss:.4f} acc={vacc:.4f}")
            if args.use_wandb and args.report_to=="wandb":
                wandb.log({"val_loss":vloss,"val_acc":vacc}, step=step)
            classifier.train()

    # -------------- save -------------------
    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        fn = Path(args.output_dir)/"classifier_final.pth"
        torch.save(accelerator.unwrap_model(classifier).state_dict(), fn)
        if args.push_to_hub: repo.push_to_hub("final checkpoint")

    accelerator.end_training()

if __name__ == "__main__":
    main()

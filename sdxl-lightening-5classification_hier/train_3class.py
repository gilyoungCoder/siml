#!/usr/bin/env python3
# train_3class.py

"""
3-class (not-people / fully-clothed / nude-people) classifier for SDXL-Lightning
1000-step DDPM noise-injected latent training (classifier-guidance 호환용).
"""

import argparse, os
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
        for root, label in [
            (not_people_dir, 0),
            (fully_clothed_dir,1),
            (nude_people_dir,2)
        ]:
            if not os.path.isdir(root):
                raise ValueError(f"데이터 디렉터리를 찾을 수 없습니다: {root}")
            for fname in sorted(os.listdir(root)):
                if fname.lower().endswith((".png",".jpg",".jpeg",".webp")):
                    path = os.path.join(root, fname)
                    self.samples.append((path, label))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return {"pixel_values": img, "label": label}


# --------------------------------------------------------------------------- #
#  Argument parsing                                                           #
# --------------------------------------------------------------------------- #
def get_args():
    p = argparse.ArgumentParser("3-class nudity classifier (Lightning)")
    # data
    p.add_argument("--not_people_data_path",   required=True)
    p.add_argument("--fully_clothed_data_path",required=True)
    p.add_argument("--nude_people_data_path",  required=True)
    # model / output
    p.add_argument("--output_dir", type=str, default="clf3_lightning_out")
    p.add_argument("--sdxl_base", type=str,
                   default="stabilityai/stable-diffusion-xl-base-1.0",
                   help="SDXL base ckpt (VAE & scheduler 로드용)")
    # training hyper-params
    p.add_argument("--train_batch_size", type=int, default=32)
    p.add_argument("--learning_rate",   type=float, default=1e-4)
    p.add_argument("--num_train_epochs",type=int, default=30)
    p.add_argument("--max_train_steps", type=int, default=None)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    # misc
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
if __name__=="__main__":
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.use_wandb:
        wandb.init(project=args.wandb_project,
                   name=args.wandb_run_name, config=vars(args))

    acc = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=None if args.report_to=="none" else args.report_to    )
    dev = acc.device
    logger.info(f"device -> {dev}")

    if args.seed is not None:
        set_seed(args.seed + acc.process_index)

    # hub prep
    if acc.is_local_main_process and args.push_to_hub:
        repo_name = args.hub_model_id or \
            f"{whoami(HfFolder.get_token())['name']}/{Path(args.output_dir).name}"
        repo = Repository(args.output_dir, clone_from=repo_name)
        (Path(args.output_dir)/".gitignore").write_text("checkpoint/**\n")

    # VAE & 1000-step DDPM scheduler
    vae = AutoencoderKL.from_pretrained(args.sdxl_base, subfolder="vae").to(dev)
    vae.requires_grad_(False)

    sched = DDPMScheduler.from_pretrained(
        args.sdxl_base, subfolder="scheduler",
        num_train_timesteps=1000, clip_sample=False
    )
    # ᾱ 전체(0~999) 계산
    alphas = sched.alphas_cumprod.to(dev)        # [1000]
    sa = alphas.sqrt().view(1000,1,1,1)
    sb = (1 - alphas).sqrt().view(1000,1,1,1)

    # DataLoader
    tfm = transforms.Compose([
        transforms.Resize((1024,1024)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    ds = ThreeClassDataset(
        args.not_people_data_path,
        args.fully_clothed_data_path,
        args.nude_people_data_path,
        transform=tfm
    )
    val_len = int(0.1 * len(ds)); train_len = len(ds) - val_len
    tr_ds, va_ds = torch.utils.data.random_split(
        ds, [train_len, val_len],
        generator=torch.Generator().manual_seed(args.seed or 42)
    )
    tr_loader = DataLoader(tr_ds, batch_size=args.train_batch_size,
                           shuffle=True, num_workers=4, pin_memory=True)
    va_loader = DataLoader(va_ds, batch_size=args.train_batch_size,
                           shuffle=False, num_workers=4, pin_memory=True)

    # Classifier
    clf = load_discriminator(
        ckpt_path="./work_dirs/sdxl1024/classifier_final.pth", condition=None,
        eval=False, channel=4, num_classes=3
    ).to(dev)
    opt = torch.optim.AdamW(clf.parameters(), lr=args.learning_rate)
    ce  = nn.CrossEntropyLoss()

    clf, opt, tr_loader, va_loader = acc.prepare(
        clf, opt, tr_loader, va_loader
    )

    # Training 설정
    max_steps = args.max_train_steps or args.num_train_epochs * len(tr_loader)
    best_val_acc = 0.0
    no_improve = 0
    patience = 15

    pbar = tqdm(range(max_steps), disable=not acc.is_local_main_process)
    step = 0
    clf.train()
    while step < max_steps and no_improve < patience:
        for batch in tr_loader:
            imgs   = batch["pixel_values"].to(dev)
            labels = batch["label"].to(dev)

            lat   = vae.encode(imgs).latent_dist.sample() * 0.18215
            bs    = lat.size(0)
            ts    = torch.randint(0, 1000, (bs,), device=dev)
            noise = torch.randn_like(lat)

            noisy = sa[ts] * lat + sb[ts] * noise

            out   = clf(noisy, ts.float()/1000)
            loss  = ce(out, labels)

            acc.backward(loss)
            opt.step(); opt.zero_grad()

            step += 1; pbar.update(1)
            pbar.set_postfix(step=step, loss=loss.item())

            # 100 step마다 validation + checkpoint
            if step % 100 == 0:
                clf.eval()
                val_loss, val_acc, total = 0.0, 0.0, 0
                with torch.no_grad():
                    for vb in va_loader:
                        vimgs = vb["pixel_values"].to(dev)
                        vlabels = vb["label"].to(dev)
                        vlat = vae.encode(vimgs).latent_dist.sample() * 0.18215

                        vts = torch.randint(0, 1000, (vlat.size(0),), device=dev)
                        vnoise = torch.randn_like(vlat)
                        vnoisy = sa[vts] * vlat + sb[vts] * vnoise

                        vlogits = clf(vnoisy, vts.float()/1000)
                        vloss = ce(vlogits, vlabels)
                        preds = vlogits.argmax(dim=-1)

                        val_loss += vloss.item() * vlat.size(0)
                        val_acc  += (preds == vlabels).sum().item()
                        total    += vlat.size(0)

                val_loss /= total
                val_acc  /= total
                logger.info(f"[Validation @ Step {step}] Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
                if args.use_wandb:
                    wandb.log({"val_loss": val_loss, "val_acc": val_acc}, step=step)

                # 매 100 step 체크포인트 저장
                ckpt_dir = Path(args.output_dir)/"checkpoint"/f"step_{step:05d}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)

                # early stopping
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(acc.unwrap_model(clf).state_dict(), ckpt_dir/"classifier.pth")
                    no_improve = 0
                else:
                    no_improve += 1

                clf.train()
                if no_improve >= patience:
                    logger.info(f"No improvement for {patience} validations. Early stopping.")
                    break

            if step >= max_steps:
                break

        acc.wait_for_everyone()

    # 최종 저장
    if acc.is_local_main_process:
        torch.save(acc.unwrap_model(clf).state_dict(),
                   Path(args.output_dir)/"classifier_final.pth")
        if args.push_to_hub:
            repo.push_to_hub("final checkpoint")

    acc.end_training()

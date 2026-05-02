#!/usr/bin/env python3
# train_hierarchical.py

"""
Hierarchical (4-head) nudity classifier for SDXL-Lightning
Heads:
  head_root    (2-way): non-people vs people
  head_lvl1    (2-way): clothed vs nude
  head_clothed (2-way): everyday vs swimwear
  head_nude    (2-way): partial vs full
DDPM-noise-injected latent training (1000-step).
"""

import argparse, os
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
from geo_models.classifier.classifier import load_discriminator
import wandb
from tqdm.auto import tqdm

logger = get_logger(__name__)

class HierarchicalDataset(Dataset):
    """
    y0: 0=non_people, 1=people
    y1: 0=clothed, 1=nude           (only if y0=1)
    yc: 0=everyday,1=swim           (only if y0=1,y1=0)
    yn: 0=partial,1=full            (only if y0=1,y1=1)
    """
    def __init__(self, non_people_dir,
                       everyday_dir, swim_dir,
                       partial_dir, full_dir,
                       transform=None):
        samples = []
        for root,label0,label1,labelc,ylabel in [
            (non_people_dir,      0, None, None, None),
            (everyday_dir,        1,    0,      0, None),
            (swim_dir,            1,    0,      1, None),
            (partial_dir,         1,    1,   None,    0),
            (full_dir,            1,    1,   None,    1),
        ]:
            if not os.path.isdir(root):
                raise ValueError(f"데이터 디렉터리를 찾을 수 없습니다: {root}")
            for fn in sorted(os.listdir(root)):
                if fn.lower().endswith((".png",".jpg",".jpeg",".webp")):
                    samples.append((os.path.join(root,fn),
                                    label0, label1 or 0, labelc or 0, ylabel or 0))
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, y0, y1, yc, yn = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform: img = self.transform(img)
        return {
            "pixel_values": img,
            "y0": torch.tensor(y0, dtype=torch.long),
            "y1": torch.tensor(y1, dtype=torch.long),
            "yc": torch.tensor(yc, dtype=torch.long),
            "yn": torch.tensor(yn, dtype=torch.long),
        }

def hierarchical_loss(logits, labels):
    out0, out1, outc, outn = logits
    y0, y1, yc, yn = labels
    L0 = nn.functional.cross_entropy(out0, y0)
    mask1 = (y0 == 1)
    L1 = nn.functional.cross_entropy(out1[mask1], y1[mask1]) if mask1.any() else 0
    maskc = mask1 & (y1 == 0)
    Lc = nn.functional.cross_entropy(outc[maskc], yc[maskc]) if maskc.any() else 0
    maskn = mask1 & (y1 == 1)
    Ln = nn.functional.cross_entropy(outn[maskn], yn[maskn]) if maskn.any() else 0
    return L0 + L1 + Lc + Ln

def get_args():
    p = argparse.ArgumentParser("hierarchical nudity classifier")
    p.add_argument("--non_people_data_path",    required=True)
    p.add_argument("--clothed_everyday_data_path", required=True)
    p.add_argument("--clothed_swimwear_data_path", required=True)
    p.add_argument("--partial_nude_data_path",  required=True)
    p.add_argument("--full_nude_data_path",     required=True)
    p.add_argument("--output_dir", default="clf_hier_out")
    p.add_argument("--sdxl_base", default="stabilityai/stable-diffusion-xl-base-1.0")
    p.add_argument("--train_batch_size", type=int, default=32)
    p.add_argument("--learning_rate",    type=float, default=1e-4)
    p.add_argument("--num_train_epochs", type=int, default=30)
    p.add_argument("--max_train_steps",  type=int, default=None)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--mixed_precision", choices=["no","fp16","bf16"], default="no")
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", default="nudity_hier")
    p.add_argument("--wandb_run_name", default="run1")
    p.add_argument("--push_to_hub", action="store_true")
    p.add_argument("--hub_model_id", type=str, default=None)
    return p.parse_args()

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.use_wandb:
        wandb.init(project=args.wandb_project,
                   name=args.wandb_run_name,
                   config=vars(args))
        wandb.watch_called = False

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="wandb" if args.use_wandb else None
    )
    device = accelerator.device
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)

    # VAE & Scheduler
    vae = AutoencoderKL.from_pretrained(
        args.sdxl_base, subfolder="vae"
    ).to(device)
    vae.requires_grad_(False)
    scheduler = DDPMScheduler.from_pretrained(
        args.sdxl_base, subfolder="scheduler",
        num_train_timesteps=1000, clip_sample=False
    )
    alphas = scheduler.alphas_cumprod.to(device)
    sqrt_a  = alphas.sqrt().view(1000,1,1,1)
    sqrt_1a = (1 - alphas).sqrt().view(1000,1,1,1)

    # Dataset
    tfm = transforms.Compose([
        transforms.Resize((1024,1024)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3,[0.5]*3),
    ])
    ds = HierarchicalDataset(
        args.non_people_data_path,
        args.clothed_everyday_data_path,
        args.clothed_swimwear_data_path,
        args.partial_nude_data_path,
        args.full_nude_data_path,
        transform=tfm
    )
    val_len   = int(0.1 * len(ds))
    train_len = len(ds) - val_len
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [train_len, val_len],
        generator=torch.Generator().manual_seed(args.seed or 42)
    )
    train_loader = DataLoader(train_ds, batch_size=args.train_batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.train_batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    # Model
    clf = load_discriminator(
        ckpt_path=None, condition=None,
        eval=False, channel=4,
        num_classes=5, hierarchical=True
    ).to(device)
    if args.mixed_precision == "fp16":
        clf = clf.half()
    optimizer = torch.optim.AdamW(clf.parameters(), lr=args.learning_rate)

    clf, optimizer, train_loader, val_loader = accelerator.prepare(
        clf, optimizer, train_loader, val_loader
    )

    # Training loop
    max_steps = args.max_train_steps or (args.num_train_epochs * len(train_loader))
    pbar = tqdm(range(max_steps), disable=not accelerator.is_local_main_process)
    step = 0

    clf.train()
    while step < max_steps:
        for batch in train_loader:
            imgs = batch["pixel_values"].to(device)
            y0,y1,yc,yn = (batch[k].to(device) for k in ("y0","y1","yc","yn"))

            lat = vae.encode(imgs).latent_dist.sample() * 0.18215
            bsz = lat.size(0)
            ts  = torch.randint(0,1000,(bsz,),device=device)
            noisy = sqrt_a[ts]*lat + sqrt_1a[ts]*torch.randn_like(lat)

            logits = clf(noisy, ts.float()/1000.0)
            loss   = hierarchical_loss(logits, (y0,y1,yc,yn))
            accelerator.backward(loss)
            optimizer.step(); optimizer.zero_grad()

            step += 1
            pbar.update(1)
            pbar.set_postfix(step=step, loss=loss.item())
            if args.use_wandb:
                accelerator.log({"train_loss": loss.item()}, step=step)

            # Validation every 100 steps
            if step % 100 == 0:
                accelerator.wait_for_everyone()
                clf.eval()
                tot_all = tot_people = tot_cloth = tot_nude = 0
                corr_root = corr_lvl1 = corr_cloth = corr_nude = corr_leaf = 0
                vloss = 0.0

                with torch.no_grad():
                    for vb in val_loader:
                        vimg = vb["pixel_values"].to(device)
                        y0 = vb["y0"].to(device)
                        y1 = vb["y1"].to(device)
                        yc = vb["yc"].to(device)
                        yn_ = vb["yn"].to(device)

                        vlat = vae.encode(vimg).latent_dist.sample() * 0.18215
                        bs   = vlat.size(0)
                        vts  = torch.randint(0,1000,(bs,),device=device)
                        vnoisy = sqrt_a[vts]*vlat + sqrt_1a[vts]*torch.randn_like(vlat)

                        p0,p1,pc,pn = clf(vnoisy, vts.float()/1000.0)

                        # loss accumulation
                        vloss += hierarchical_loss((p0,p1,pc,pn),(y0,y1,yc,yn_)).item() * bs

                        # per-head preds
                        root_pred  = p0 .argmax(1)
                        lvl1_pred  = p1 .argmax(1)
                        cloth_pred = pc .argmax(1)
                        nude_pred  = pn .argmax(1)

                        mask_people = (y0 == 1)
                        mask_cloth  = mask_people & (y1 == 0)
                        mask_nude   = mask_people & (y1 == 1)

                        # counts
                        tot_all    += bs
                        tot_people += mask_people.sum().item()
                        tot_cloth  += mask_cloth.sum().item()
                        tot_nude   += mask_nude.sum().item()

                        # correct counts (올바른 & 사용)
                        corr_root  += (root_pred == y0).sum().item()
                        corr_lvl1  += ((lvl1_pred == y1) & mask_people).sum().item()
                        corr_cloth += ((cloth_pred == yc) & mask_cloth).sum().item()
                        corr_nude  += ((nude_pred == yn_) & mask_nude).sum().item()

                        # leaf-level
                        true_leaf  = torch.zeros_like(y0)
                        true_leaf[y0==1] = torch.where(
                            y1[y0==1]==0,
                            yc[y0==1] + 1,   # everyday→1, swimwear→2
                            yn_[y0==1] + 3   # partial→3, full→4
                        )
                        pred_leaf = torch.full((bs,), -1, device=device)
                        m0 = root_pred==0
                        pred_leaf[m0] = 0
                        mc = (~m0) & (lvl1_pred==0)
                        pred_leaf[mc] = cloth_pred[mc] + 1
                        mn = (~m0) & (lvl1_pred==1)
                        pred_leaf[mn] = nude_pred[mn] + 3
                        corr_leaf += (pred_leaf == true_leaf).sum().item()

                # metrics 계산 (0으로 나누지 않도록)
                val_loss  = vloss / tot_all
                root_acc  = corr_root  / tot_all
                lvl1_acc  = corr_lvl1  / tot_people if tot_people > 0 else 0.0
                cloth_acc = corr_cloth / tot_cloth  if tot_cloth  > 0 else 0.0
                nude_acc  = corr_nude  / tot_nude   if tot_nude   > 0 else 0.0
                leaf_acc  = corr_leaf  / tot_all

                logger.info(
                    f"[VAL @ {step}] loss={val_loss:.4f} leaf_acc={leaf_acc:.4f} "
                    f"(root={root_acc:.4f}, lvl1={lvl1_acc:.4f}, "
                    f"cloth={cloth_acc:.4f}, nude={nude_acc:.4f})"
                )
                if args.use_wandb:
                    wandb.log({
                        "val_loss":      val_loss,
                        "val_leaf_acc":  leaf_acc,
                        "val_root_acc":  root_acc,
                        "val_lvl1_acc":  lvl1_acc,
                        "val_cloth_acc": cloth_acc,
                        "val_nude_acc":  nude_acc,
                    }, step=step)

                clf.train()

            if step >= max_steps:
                break

    # save final
    if accelerator.is_local_main_process:
        torch.save(
            accelerator.unwrap_model(clf).state_dict(),
            Path(args.output_dir)/"classifier_final.pth"
        )
        if args.push_to_hub:
            pass

    accelerator.end_training()

if __name__=="__main__":
    main()

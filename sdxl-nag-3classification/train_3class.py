#!/usr/bin/env python3
# train_3class.py

"""
3-class (not-people / fully-clothed / nude-people) classifier for SDXL-Lightning
via cross-attention–feature learning.
1000-step DDPM noise-injected latent training.
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
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
import wandb
from tqdm.auto import tqdm

logger = get_logger(__name__)

# ---------------------------- Dataset ---------------------------- #
class ThreeClassDataset(Dataset):
    """0: not_people  1: fully_clothed  2: nude_people"""
    def __init__(self, not_people_dir, fully_clothed_dir, nude_people_dir, transform=None):
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
                    self.samples.append((os.path.join(root, fname), label))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return {"pixel_values": img, "label": label}


# ------------------------- Classifier Head ----------------------- #
class AttnClassifierHead(nn.Module):
    def __init__(self, in_ch, num_classes):
        super().__init__()
        # 공간 평균 풀링 → [B, in_ch]
        self.pool = nn.AdaptiveAvgPool2d(1)
        # MLP
        self.fc = nn.Sequential(
            nn.LayerNorm(in_ch),
            nn.Linear(in_ch, in_ch),
            nn.GELU(),
            nn.Linear(in_ch, num_classes)
        )

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, _, _ = x.shape
        z = self.pool(x).view(B, C)   # [B, C]
        return self.fc(z)            # [B, num_classes]


# ------------------------ Argument Parser ----------------------- #
def get_args():
    p = argparse.ArgumentParser("3-class nudity classifier (Lightning)")
    p.add_argument("--not_people_data_path",   required=True)
    p.add_argument("--fully_clothed_data_path",required=True)
    p.add_argument("--nude_people_data_path",  required=True)
    p.add_argument("--output_dir", type=str, default="clf3_lightning_out")
    p.add_argument("--sdxl_base", type=str,
                   default="stabilityai/stable-diffusion-xl-base-1.0",
                   help="SDXL base ckpt (VAE & scheduler 로드용)")
    p.add_argument("--train_batch_size", type=int, default=32)
    p.add_argument("--learning_rate",   type=float, default=1e-4)
    p.add_argument("--num_train_epochs",type=int, default=30)
    p.add_argument("--max_train_steps", type=int, default=None)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--mixed_precision", choices=["no","fp16","bf16"], default="no")
    p.add_argument("--report_to", type=str, default="wandb")
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", default="nudity3")
    p.add_argument("--wandb_run_name", default="run1")
    p.add_argument("--push_to_hub", action="store_true")
    p.add_argument("--hub_model_id", type=str, default=None)
    return p.parse_args()


# ----------------------------- Main ----------------------------- #
if __name__=="__main__":
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.use_wandb:
        wandb.init(project=args.wandb_project,
                   name=args.wandb_run_name, config=vars(args))

    acc = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=None if args.report_to=="none" else args.report_to
    )
    dev = acc.device
    logger.info(f"device -> {dev}")

    if args.seed is not None:
        set_seed(args.seed + acc.process_index)

    # Hub prep
    if acc.is_local_main_process and args.push_to_hub:
        repo_name = args.hub_model_id or \
            f"{wandb.run.entity}/{Path(args.output_dir).name}"
        from huggingface_hub import Repository
        repo = Repository(args.output_dir, clone_from=repo_name)
        (Path(args.output_dir)/".gitignore").write_text("checkpoint/**\n")

    # 1) VAE 로드 (freeze)
    vae = AutoencoderKL.from_pretrained(args.sdxl_base, subfolder="vae").to(dev)
    vae.requires_grad_(False)

    # 2) UNet 로드 (freeze)
    unet = UNet2DConditionModel.from_pretrained(
        args.sdxl_base, subfolder="unet"
    ).to(dev)
    for p in unet.parameters():
        p.requires_grad = False

    # ─── 이 한 줄로 추가 임베딩 모드 끔 ───
    unet.config.addition_embed_type = None

    # 2.1) CLIP text encoder 로드 및 “nudity” 임베딩
    tokenizer    = CLIPTokenizer.from_pretrained(args.sdxl_base, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.sdxl_base, subfolder="text_encoder").to(dev)
    text_encoder.eval()
    toks = tokenizer(
        "nudity",
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt"
    )
    with torch.no_grad():
        base_emb = text_encoder(toks.input_ids.to(dev))[0]  # [1, L, D]

    # 3) DDPM 스케줄러
    sched = DDPMScheduler.from_pretrained(
        args.sdxl_base, subfolder="scheduler",
        num_train_timesteps=1000, clip_sample=False
    )
    alphas = sched.alphas_cumprod.to(dev)
    sa = alphas.sqrt().view(1000,1,1,1)
    sb = (1 - alphas).sqrt().view(1000,1,1,1)

    # 4) DataLoader
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

    # 5) 분류 헤드 및 옵티마이저
    mid_idx = len(unet.config.block_out_channels)//2
    mid_ch  = unet.config.block_out_channels[mid_idx]
    attn_head = AttnClassifierHead(in_ch=mid_ch, num_classes=3).to(dev)
    opt = torch.optim.AdamW(attn_head.parameters(), lr=args.learning_rate)
    ce  = nn.CrossEntropyLoss()

    attn_head, tr_loader, va_loader, opt = acc.prepare(
        attn_head, tr_loader, va_loader, opt
    )

    # 6) Cross‐attention 훅
    attention_buf = []
    def attn_hook(module, inp, out):
        feat = out
        if feat.ndim == 3:  # [B, C, H*W] → [B, C, H, W]
            B,C,T = feat.shape
            s = int(T**0.5)
            feat = feat.view(B, C, s, s)
        attention_buf.append(feat)

    handle = unet.mid_block.attentions[0].register_forward_hook(attn_hook)

    # 7) 학습 루프
    max_steps   = args.max_train_steps or args.num_train_epochs * len(tr_loader)
    best_val_acc = 0.0
    no_improve  = 0
    patience    = 15

    pbar = tqdm(range(max_steps), disable=not acc.is_local_main_process)
    step = 0
    attn_head.train()
    while step < max_steps and no_improve < patience:
        for batch in tr_loader:
            imgs, labels = batch["pixel_values"].to(dev), batch["label"].to(dev)

            # (1) VAE → latent
            lat = vae.encode(imgs).latent_dist.sample() * 0.18215

            # (2) timestep + noise
            bs    = lat.size(0)
            ts    = torch.randint(0, 1000, (bs,), device=dev)
            noise = torch.randn_like(lat)
            noisy = sa[ts] * lat + sb[ts] * noise

            # (3) UNet forward → 훅에 attention 저장
            attention_buf.clear()
            text_emb = base_emb.expand(bs, -1, -1)  # [bs, L, D]
            _ = unet(
                noisy,
                ts,
                encoder_hidden_states=text_emb   # addition_embed_type=None 이므로 여기에만 전달
            )

            # (4) 분류 헤드 → loss
            Z      = attention_buf.pop()    # [B, C, H, W]
            logits = attn_head(Z)           # [B,3]
            loss   = ce(logits, labels)

            # (5) backward & step
            acc.backward(loss)
            opt.step(); opt.zero_grad()

            step += 1
            pbar.update(1)
            pbar.set_postfix(step=step, loss=loss.item())

            # 100 스텝마다 validation & checkpoint
            if step % 100 == 0:
                attn_head.eval()
                val_acc, total = 0, 0
                with torch.no_grad():
                    for vb in va_loader:
                        vimgs, vlabels = vb["pixel_values"].to(dev), vb["label"].to(dev)
                        vlat = vae.encode(vimgs).latent_dist.sample() * 0.18215
                        vts  = torch.randint(0, 1000, (vlat.size(0),), device=dev)
                        vnoisy = sa[vts] * vlat + sb[vts] * torch.randn_like(vlat)

                        attention_buf.clear()
                        _ = unet(vnoisy, vts, encoder_hidden_states=text_emb)
                        vZ     = attention_buf.pop()
                        vlogits= attn_head(vZ)
                        preds  = vlogits.argmax(dim=-1)
                        val_acc += (preds == vlabels).sum().item()
                        total   += vlat.size(0)

                val_acc /= total
                logger.info(f"[Validation @ Step {step}] Acc: {val_acc:.4f}")
                if args.use_wandb:
                    wandb.log({"val_acc": val_acc}, step=step)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    ckpt_dir = Path(args.output_dir)/"checkpoint"/f"step_{step:05d}" 
                    ckpt_dir.mkdir(exist_ok=True)
                    torch.save(
                        acc.unwrap_model(attn_head).state_dict(),
                        ckpt_dir/"attn_head.pth"
                    )
                    no_improve = 0
                else:
                    no_improve += 1

                attn_head.train()
                if no_improve >= patience:
                    logger.info(f"No improvement for {patience} validations. Early stopping.")
                    break

            if step >= max_steps:
                break

        acc.wait_for_everyone()

    # 훅 제거
    handle.remove()

    # 최종 저장
    if acc.is_local_main_process:
        torch.save(
            acc.unwrap_model(attn_head).state_dict(),
            Path(args.output_dir)/"attn_head_final.pth"
        )
        if args.push_to_hub:
            repo.push_to_hub("final checkpoint")

    acc.end_training()

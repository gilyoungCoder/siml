#!/usr/bin/env python3
import os
import torch
import torch.nn.functional as F
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
from diffusers import AutoencoderKL
from tqdm.auto import tqdm
import wandb
import argparse
from torchvision.datasets import ImageFolder


# --- 1) wandb 초기화 -------------------------------------------------------
wandb.init(
    project="latent-clip-alignment",
    name="sd15_latent2clip_cifar",
    config={
        "model": "stable-diffusion-v1-5",
        "clip_model": "openai/clip-vit-large-patch14",
        "batch_size": 32,
        "lr": 1e-4,
        "epochs": 30,
    },
)
config = wandb.config

# --- Latent → CLIP projection network ------------------------------------
class Latent2Clip(nn.Module):
    def __init__(self, latent_dim, clip_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(latent_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, clip_dim),
        )

    def forward(self, z):
        proj = self.net(z)
        return proj / proj.norm(dim=-1, keepdim=True)

# --- main ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cifar_root", type=str,
                        default=os.path.expanduser("~/dataset/cifar10"),
                        help="Path to CIFAR-10 root folder")
    parser.add_argument("--vae_repo", type=str,
                        default="runwayml/stable-diffusion-v1-5",
                        help="Stable Diffusion VAE repo")
    parser.add_argument("--clip_model", type=str,
                        default="openai/clip-vit-large-patch14",
                        help="HuggingFace CLIP model")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 2) 전처리 정의 ------------------------------------------------------
    vae_preproc = transforms.Compose([
        transforms.Resize(512, transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    clip_processor = CLIPProcessor.from_pretrained(args.clip_model)

    # --- 3) CIFAR-10 학습 데이터 로드 ----------------------------------------
    train_ds = ImageFolder(root=args.cifar_root, transform=lambda img: img)
    def collate_train(batch):
        imgs, labels = zip(*batch)
        x_vae  = torch.stack([vae_preproc(img) for img in imgs])
        x_clip = torch.stack([
            clip_processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)
            for img in imgs
        ])
        return x_vae, x_clip, torch.tensor(labels, dtype=torch.long)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size,
                              shuffle=True, num_workers=4,
                              pin_memory=True, collate_fn=collate_train)

    # --- 4) 모델 불러오기 --------------------------------------------------
    vae = AutoencoderKL.from_pretrained(args.vae_repo, subfolder="vae") \
                       .to(device).eval()
    clip_model = CLIPModel.from_pretrained(args.clip_model).to(device).eval()

    latent_dim = vae.config.latent_channels
    clip_dim   = clip_model.config.projection_dim
    proj_net   = Latent2Clip(latent_dim, clip_dim).to(device)

    optimizer = optim.Adam(proj_net.parameters(), lr=config.lr)
    wandb.watch(proj_net, log="all", log_freq=100)

    # --- 5) text embeddings for zero-shot -------------------------------
    class_names = train_ds.classes
    with torch.no_grad():
        text_inputs     = clip_processor(
            text=[f"a photo of a {c}." for c in class_names],
            return_tensors="pt", padding=True
        ).to(device)
        text_embeddings = clip_model.get_text_features(
            input_ids=text_inputs.input_ids,
            attention_mask=text_inputs.attention_mask
        )
        text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)

    os.makedirs("checkpoints", exist_ok=True)

    # --- 6) 학습 및 epoch별 평가 ---------------------------------------------
    for epoch in range(1, config.epochs + 1):
        # --- (A) Train latent2clip -----------------------------------------
        proj_net.train()
        total_loss = 0.0
        for x_vae, x_clip, labels in tqdm(train_loader, desc=f"Epoch {epoch}"):
            x_vae, x_clip = x_vae.to(device), x_clip.to(device)
            # VAE encode
            with torch.no_grad():
                lat_dist = vae.encode(x_vae).latent_dist
                latents  = lat_dist.sample() * vae.config.scaling_factor
                clip_feats = clip_model.get_image_features(pixel_values=x_clip)
                clip_feats /= clip_feats.norm(dim=-1, keepdim=True)

            # projection & loss
            pred_feats = proj_net(latents)
            cos_loss = 1 - F.cosine_similarity(pred_feats, clip_feats, dim=-1).mean()
            mse_loss = F.mse_loss(pred_feats, clip_feats)
            loss     = mse_loss + 0.5 * cos_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        epoch_loss = total_loss / len(train_loader)
        wandb.log({"train/epoch_loss": epoch_loss, "epoch": epoch})
        print(f"[Epoch {epoch}/{config.epochs}] loss: {epoch_loss:.4f}")

        # 체크포인트 저장
        torch.save({
            "epoch": epoch,
            "model_state": proj_net.state_dict(),
            "opt_state":   optimizer.state_dict(),
        }, f"checkpoints/latent2clip_epoch{epoch}.pt")

        # --- (B) Zero-Shot 평가 -------------------------------------------
        proj_net.eval()
        test_ds = ImageFolder(root=args.cifar_root, transform=lambda img: img)
        test_loader = DataLoader(
            test_ds, batch_size=64, shuffle=False, num_workers=4,
            pin_memory=True,
            collate_fn=lambda batch: (
                [img for img,_ in batch],
                torch.tensor([lbl for _,lbl in batch], dtype=torch.long)
            )
        )

        correct_proj   = 0
        total = 0
        for images, labels in tqdm(test_loader, desc="Zero-Shot Eval", leave=False):

            with torch.no_grad():
                # latent2clip 투영
                x_vae = torch.stack([vae_preproc(img) for img in images]).to(device)
                lat_dist = vae.encode(x_vae).latent_dist
                latents  = lat_dist.sample() * vae.config.scaling_factor
                proj_feats = proj_net(latents)
                logits_proj = proj_feats @ text_embeddings.T
                preds_proj  = logits_proj.argmax(dim=-1).cpu()

            correct_proj   += (preds_proj   == labels).sum().item()
            total += len(labels)

        acc_proj   = 100 * correct_proj   / total
        wandb.log({
            "eval/clip_zero_shot_proj":   acc_proj,
            "epoch": epoch
        })
        print(f"Proj CLIP ZS: {acc_proj:.2f}%")

    print("Training complete!")

if __name__ == "__main__":
    main()

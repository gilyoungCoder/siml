import os
import glob
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn

import wandb
from diffusers import AutoencoderKL
from transformers import CLIPVisionModel, CLIPImageProcessor
from transformers import CLIPModel
from PIL import Image

# --- 1) wandb 초기화 -------------------------------------------------------
wandb.init(
    project="latent-clip-alignment",
    name="sd15_latent2clip",
    config={
        "model": "stable-diffusion-v1-5",
        "clip_model": "openai/clip-vit-large-patch14",
        "batch_size": 32,
        "lr": 1e-4,
        "epochs": 30,
    },
)
config = wandb.config

# --- 2) 커스텀 데이터셋 ---------------------------------------------------
class SimpleImageDataset(Dataset):
    def __init__(self, root_dir, vae_preproc, clip_proc):
        super().__init__()
        # jpg/png 등 모두 읽도록
        exts = ["jpg","jpeg","png","bmp"]
        self.paths = []
        for ext in exts:
            self.paths += glob.glob(os.path.join(root_dir, f"**/*.{ext}"), recursive=True)
        self.vae_preproc = vae_preproc
        self.clip_proc   = clip_proc

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        pil  = Image.open(path).convert("RGB")
        # 1) VAE 입력용
        x_vae  = self.vae_preproc(pil)
        # 2) CLIP 입력용 (HuggingFace processor)
        clip_inputs = self.clip_proc(images=pil, return_tensors="pt")["pixel_values"].squeeze(0)
        return x_vae, clip_inputs

# 전처리 정의
vae_preproc = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(512),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),  # → [-1,1]
])
clip_proc = CLIPImageProcessor.from_pretrained(config.clip_model)

dataset = SimpleImageDataset(
    root_dir="/mnt/home/yhgil99/dataset/threeclassImg/imagenet5k",
    vae_preproc=vae_preproc,
    clip_proc=clip_proc
)

loader = DataLoader(
    dataset,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)

# --- 3) 모델 준비 ------------------------------------------------------------
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

# (a) Stable Diffusion 1.5 VAE
vae = AutoencoderKL.from_pretrained(
    "runwayml/stable-diffusion-v1-5", subfolder="vae"
).to(device).eval()

# (b) 픽셀-CLIP 비전 인코더
# clip_image = CLIPVisionModel.from_pretrained(
#     config.clip_model
# ).to(device).eval()
# for p in clip_image.parameters():
#     p.requires_grad = False
clip_model = CLIPModel.from_pretrained(
    config.clip_model
).to(device).eval()
for p in clip_model.parameters():
    p.requires_grad = False


# (c) Latent→CLIP 매핑 네트워크
class Latent2Clip(nn.Module):
    def __init__(self, latent_dim, clip_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(latent_dim, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),                  # (B,128,1,1)
            nn.Flatten(),                             # (B,128)
            nn.Linear(128, clip_dim),
        )
    def forward(self, z):
        return self.net(z)

latent2clip = Latent2Clip(
    latent_dim=vae.config.latent_channels,
    clip_dim=clip_model.config.projection_dim
).to(device)

opt = optim.Adam(latent2clip.parameters(), lr=config.lr)
wandb.watch(latent2clip, log="all", log_freq=100)

# --- 4) 학습 루프 ------------------------------------------------------------
for epoch in range(config.epochs):
    running_loss = 0.0
    for step, (x_vae, x_clip) in enumerate(loader):
        x_vae  = x_vae.to(device)
        x_clip = x_clip.to(device)

        # 1) VAE 인코딩
        with torch.no_grad():
            q       = vae.encode(x_vae).latent_dist
            latents = q.sample() * vae.config.scaling_factor  # (B, C, H, W)
            # CLIP 피쳐
            clip_feats = clip_model.get_image_features(pixel_values=x_clip)
            clip_feats = clip_feats / clip_feats.norm(dim=-1, keepdim=True)

        # 2) 매핑 예측
        pred_feats = latent2clip(latents)
        pred_feats = pred_feats / pred_feats.norm(dim=-1, keepdim=True)

        # 3) Cosine + MSE 손실
        cos_loss = 1 - F.cosine_similarity(pred_feats, clip_feats, dim=-1).mean()
        mse_loss = F.mse_loss(pred_feats, clip_feats)
        loss     = mse_loss + 0.5 * cos_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        running_loss += loss.item()
        if step % 50 == 0:
            wandb.log({
                "train/batch_loss": loss.item(),
                "train/step": epoch * len(loader) + step
            })

    epoch_loss = running_loss / len(loader)
    print(f"[Epoch {epoch+1}/{config.epochs}] loss: {epoch_loss:.4f}")
    wandb.log({"train/epoch_loss": epoch_loss, "epoch": epoch+1})

    # 체크포인트
    ckpt_path = os.path.join("checkpoints", f"latent2clip_epoch{epoch+1}.pt")
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        "epoch": epoch+1,
        "model_state": latent2clip.state_dict(),
        "opt_state":   opt.state_dict(),
    }, ckpt_path)
    wandb.save(ckpt_path)

print("Training complete!")

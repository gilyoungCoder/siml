#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stable Diffusion 1.4 Textual Inversion Trainer
- 오직 '토큰 임베딩'만 학습 (UNet/VAE/Text Encoder 동결)
- Diffusers 호환 learned_embeds.bin 저장
- allow/harm 어떤 데이터에도 사용 가능 (캡션 템플릿에 placeholder_token을 반드시 포함)
"""

import os, math, argparse, itertools, random
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from torchvision import transforms as T

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from diffusers import StableDiffusionPipeline, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

logger = get_logger(__name__)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def list_images(root: str, recursive: bool = True) -> List[str]:
    paths = []
    root = str(root)
    if recursive:
        for r, _, files in os.walk(root):
            for f in files:
                if os.path.splitext(f)[1].lower() in IMG_EXTS:
                    paths.append(os.path.join(r, f))
    else:
        for f in sorted(os.listdir(root)):
            p = os.path.join(root, f)
            if os.path.isfile(p) and os.path.splitext(p)[1].lower() in IMG_EXTS:
                paths.append(p)
    if not paths:
        raise FileNotFoundError(f"No images under: {root}")
    return paths

class ImgDataset(Dataset):
    def __init__(self, img_dir: str, size: int, placeholder_phrase: str, use_person_templates: bool = True):
        self.paths = list_images(img_dir, True)
        self.placeholder_phrase = placeholder_phrase
        # 템플릿: 토큰이 반드시 포함되도록 구성
        if use_person_templates:
            self.templates = [
                f"a portrait of a person, {placeholder_phrase}",
                f"a studio portrait, {placeholder_phrase}",
                f"a photo of a fully clothed person, {placeholder_phrase}",
                f"a fashion catalog photo, {placeholder_phrase}",
                f"a centered photo, {placeholder_phrase}",
            ]
        else:
            self.templates = [
                f"a product photo, {placeholder_phrase}",
                f"a studio shot, {placeholder_phrase}",
                f"a catalog image, {placeholder_phrase}",
                f"a flat lay image, {placeholder_phrase}",
                f"a centered object photo, {placeholder_phrase}",
            ]
        self.tf = T.Compose([
            T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(size),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3),
        ])

    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        img = Image.open(p).convert("RGB")
        return {
            "pixel_values": self.tf(img),
            "caption": random.choice(self.templates),
        }

def save_learned_embeds(save_path: str, token: str, vectors: torch.Tensor):
    """
    Diffusers 호환 포맷:
    - 단일/멀티 벡터 모두 지원: { token_string: tensor[num_vectors, hidden_dim] }
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    state = {token: vectors.detach().cpu()}
    torch.save(state, save_path)
    logger.info(f"Saved TI embedding -> {save_path}  ({list(state.keys())}, {tuple(vectors.shape)})")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained_model_name_or_path", type=str, default="CompVis/stable-diffusion-v1-4")
    ap.add_argument("--train_data_dir", type=str, required=True)
    ap.add_argument("--placeholder_token", type=str, required=True, help="예: <clothed> 또는 <nudity>")
    ap.add_argument("--initializer_token", type=str, default="clothes", help="의미가 가까운 기존 토큰")
    ap.add_argument("--num_vectors", type=int, default=1, help="토큰 용량(멀티 벡터 soft prompt)")
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--train_batch_size", type=int, default=2)
    ap.add_argument("--max_train_steps", type=int, default=6000)
    ap.add_argument("--learning_rate", type=float, default=5e-4)
    ap.add_argument("--lr_warmup_steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_person_templates", action="store_true", help="사람 유지 목적이면 켜기(권장)")
    ap.add_argument("--mixed_precision", type=str, choices=["no","fp16","bf16"], default="fp16")
    ap.add_argument("--output_dir", type=str, default="work_dirs/ti_token")
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--norm_clip", type=float, default=3.0, help="임베딩 L2 norm 상한(0이면 비활성)")
    ap.add_argument("--lambda_anchor", type=float, default=1e-4, help="초기 임베딩과 L2 고정 앵커")
    return ap.parse_args()

def main():
    args = parse_args()
    accelerator = Accelerator(log_with="tensorboard", mixed_precision=args.mixed_precision)
    device = accelerator.device
    set_seed(args.seed + accelerator.process_index)

    # 1) 베이스 파이프라인 로드 (텍스트 인코더/토크나이저/UNet/VAE)
    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.float16 if args.mixed_precision=="fp16" else None,
        safety_checker=None, requires_safety_checker=False
    )
    pipe.to(device)
    pipe.enable_xformers_memory_efficient_attention() if hasattr(pipe, "enable_xformers_memory_efficient_attention") else None

    tokenizer: CLIPTokenizer = pipe.tokenizer
    text_encoder: CLIPTextModel = pipe.text_encoder
    unet = pipe.unet
    vae  = pipe.vae

    # 2) 모든 모듈 동결, 오직 임베딩만 학습
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # 3) placeholder 토큰(들) 추가 (멀티 벡터 지원)
    #    Diffusers 호환: learned_embeds.bin에 { "<token>": [num_vectors, hidden] } 저장하면
    #    load_textual_inversion이 자동으로 <token>-0, <token>-1 ... 서브토큰을 만듦
    num_vec = max(1, args.num_vectors)
    # 임시로는 tokenizer에 "<token>-i" 들을 추가해서 훈련 시 바로 사용
    sub_tokens = [args.placeholder_token] if num_vec == 1 else [f"{args.placeholder_token}-{i}" for i in range(num_vec)]
    add_num = tokenizer.add_tokens(sub_tokens)
    if add_num != len(sub_tokens):
        logger.info("Some placeholder tokens already existed in tokenizer.")

    tokenizer_len = len(tokenizer)
    text_encoder.resize_token_embeddings(tokenizer_len)
    token_embed = text_encoder.get_input_embeddings()

    # 4) 초기 임베딩 준비 (initializer_token 임베딩 복제)
    with torch.no_grad():
        init_ids = tokenizer([args.initializer_token], return_tensors="pt")["input_ids"].to(device)
        init_emb = text_encoder.get_input_embeddings()(init_ids).detach()[0, 1, :]  # [CLS, init, ...] 에서 두 번째 토큰 추출
        # 안전: 만약 토큰 분절되면 평균
        if init_ids.shape[1] > 2:
            init_emb = text_encoder.get_input_embeddings()(init_ids)[0, 1:-1, :].mean(dim=0)

        for tok in sub_tokens:
            tok_id = tokenizer.convert_tokens_to_ids(tok)
            token_embed.weight.data[tok_id] = init_emb.clone()

    # 5) 학습 파라미터 = token_embed.weight (그러나 서브토큰 행만 업데이트)
    opt = torch.optim.AdamW([token_embed.weight], lr=args.learning_rate, weight_decay=args.weight_decay)

    # 6) 데이터
    ds = ImgDataset(args.train_data_dir, args.resolution, placeholder_phrase=" ".join(sub_tokens),
                    use_person_templates=args.use_person_templates)
    dl = DataLoader(ds, batch_size=args.train_batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    # 7) 스케줄러 (DDPM; SD는 내부에서 v-pred가 아닌 noise 예측을 사용)
    noise_scheduler: DDPMScheduler = pipe.scheduler
    noise_scheduler.set_timesteps(1000)  # 학습에선 표준 1000스텝 사용

    # 8) 가속기 준비
    token_embed, opt, dl = accelerator.prepare(token_embed, opt, dl)

    # 학습용 보조: 서브토큰 id 목록, 앵커(초기값) 저장
    sub_token_ids = torch.tensor([tokenizer.convert_tokens_to_ids(t) for t in sub_tokens], device=device, dtype=torch.long)
    anchor = token_embed.weight.data[sub_token_ids].detach().clone()  # [num_vec, hidden]

    total_steps = args.max_train_steps
    global_step = 0
    data_iter = itertools.cycle(dl)

    logger.info(f"Start TI training: steps={total_steps}, num_vectors={num_vec}, token(s)={sub_tokens}")

    while global_step < total_steps:
        batch = next(data_iter)
        pixel_values = batch["pixel_values"].to(device, dtype=vae.dtype)

        # 8-1) VAE encode → latents
        with torch.no_grad():
            latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215

        # 8-2) t, noise
        bsz = latents.size(0)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()
        noise = torch.randn_like(latents)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # 8-3) 텍스트 인코딩 (캡션 → ids → hidden states)
        captions = batch["caption"]
        enc = tokenizer(
            list(captions),
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt"
        )
        input_ids = enc.input_ids.to(device)
        attn_mask = enc.attention_mask.to(device)

        # UNet Forward (오직 텍스트 임베딩 rows만 학습되도록)
        encoder_hidden_states = text_encoder(input_ids=input_ids, attention_mask=attn_mask)[0]
        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample

        # 8-4) Loss (노이즈 예측 MSE) + 앵커 L2 + 임베딩 norm clamp
        loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

        # optional: 앵커 규제(초기값에서 너무 멀어지지 않도록)
        if args.lambda_anchor > 0:
            cur = token_embed.weight[sub_token_ids]
            loss = loss + args.lambda_anchor * F.mse_loss(cur, anchor)

        accelerator.backward(loss)

        # 비대상 행의 grad 0 처리(서브토큰 행만 업데이트)
        with torch.no_grad():
            g = token_embed.weight.grad
            if g is not None:
                mask = torch.zeros_like(token_embed.weight, dtype=torch.bool)
                mask[sub_token_ids] = True
                g[~mask] = 0

        # grad clip
        if args.grad_clip and args.grad_clip > 0:
            accelerator.clip_grad_norm_([token_embed.weight], args.grad_clip)

        opt.step()
        opt.zero_grad(set_to_none=True)

        # 임베딩 norm clip
        if args.norm_clip and args.norm_clip > 0:
            with torch.no_grad():
                w = token_embed.weight[sub_token_ids]
                n = w.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                scale = (args.norm_clip / n).clamp(max=1.0)
                token_embed.weight[sub_token_ids] = w * scale

        global_step += 1

        if accelerator.is_local_main_process and (global_step % 50 == 0):
            logger.info(f"step {global_step}/{total_steps}  loss={loss.item():.4f}")

        # 주기 저장 (Diffusers 호환 포맷)
        if accelerator.is_local_main_process and (global_step % args.save_every == 0):
            vecs = token_embed.weight.detach()[sub_token_ids].float().cpu()  # [num_vec, hidden]
            save_path = os.path.join(args.output_dir, f"learned_embeds_step{global_step}.bin")
            save_learned_embeds(save_path, args.placeholder_token, vecs)

    # 최종 저장
    if accelerator.is_local_main_process:
        vecs = token_embed.weight.detach()[sub_token_ids].float().cpu()
        final_path = os.path.join(args.output_dir, "learned_embeds.bin")
        save_learned_embeds(final_path, args.placeholder_token, vecs)

    accelerator.end_training()
    logger.info("Done.")

if __name__ == "__main__":
    main()

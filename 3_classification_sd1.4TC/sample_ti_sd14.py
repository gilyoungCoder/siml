#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stable Diffusion 1.4 Inference with TI tokens
- 긍정: <clothed> (사람 유지 + 의복 강화)
- 부정: <nudity>  (누드 억제)
- 옵션: 후반부 CFG 강화 / <clothed> 토큰 cross-attn 부스팅
"""

import torch, argparse, os
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained_model_name_or_path", type=str, default="CompVis/stable-diffusion-v1-4")
    ap.add_argument("--clothed_embed_dir", type=str, required=False, help="work_dirs/ti_clothed (learned_embeds.bin 포함)")
    ap.add_argument("--nudity_embed_dir",  type=str, required=False, help="work_dirs/ti_nudity (learned_embeds.bin 포함)")
    ap.add_argument("--prompt", type=str, default="a portrait of a person, <clothed>, studio lighting")
    ap.add_argument("--negative", type=str, default="<nudity>, nude, naked, nsfw")
    ap.add_argument("--out", type=str, default="out.png")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cfg", type=float, default=7.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_adaptive_cfg", action="store_true", help="후반부 CFG 강화")
    ap.add_argument("--clothed_boost", type=float, default=0.0, help="cross-attn에서 <clothed> 토큰 가중 (0=off, 0.1~0.3 권장)")
    return ap.parse_args()

def install_attn_boost(pipe, token_str: str, boost: float):
    """
    간단한 cross-attn value 부스팅(배치 1 가정).
    - 토큰 위치는 tokenizer(prompt)해서 위치를 찾고, 해당 key/value 컬럼만 (1+boost) 배.
    """
    if boost <= 0: return None

    tok = pipe.tokenizer
    text_encoder = pipe.text_encoder

    # 토큰 위치 캐시를 위해 래퍼 준비
    class CrossAttnBoost(torch.nn.Module):
        def __init__(self, backed):
            super().__init__()
            self.backed = backed
            self.target_pos = None  # 세팅은 호출 전 외부에서
        def forward(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
            # backed는 diffusers의 기존 processor (AttnProcessor/Sliced 등)
            out = self.backed(attn, hidden_states, encoder_hidden_states, attention_mask, temb)
            # out = attn processor가 반환한 hidden_states
            # 여기서는 가벼운 방법: attn 입력 단계에서 scaling하는게 정석이나,
            # 간단히 encoder_hidden_states(컨텍스트) 통과 직후의 결과에 작은 잔차 추가로 근사
            if self.target_pos is not None and encoder_hidden_states is not None:
                # encoder_hidden_states shape: [B, S_text, C], hidden_states: [B, S_img, C]
                # 토큰 위치의 컨텍스트 벡터를 모아 평균 후 residual 추가
                ctx = encoder_hidden_states[:, self.target_pos, :].mean(dim=1, keepdim=True)  # [B,1,C]
                out = out + boost * ctx
            return out

    # 파이프라인 내부 모든 cross-attn processor 래핑
    for name, module in pipe.unet.attn_processors.items():
        if "attn2" in name:  # cross-attn만
            pipe.unet.set_attn_processor({name: CrossAttnBoost(module)})

    def set_target_positions(prompt: str):
        enc = tok(prompt, return_tensors="pt").input_ids.to(pipe.device)
        with torch.no_grad():
            # 토큰 매칭: 완벽하진 않지만 문자열 토큰화 후 exact match 위치 사용
            tokens = tok.convert_ids_to_tokens(enc[0])
            target_pos = [i for i, t in enumerate(tokens) if token_str in t]
        # 모든 processor에 위치 주입
        for name, proc in pipe.unet.attn_processors.items():
            if isinstance(proc, torch.nn.Module) and hasattr(proc, "target_pos"):
                proc.target_pos = torch.tensor(target_pos, device=pipe.device, dtype=torch.long) if len(target_pos)>0 else None

    return set_target_positions

@torch.inference_mode()
def main():
    args = parse_args()
    g = torch.Generator().manual_seed(args.seed)

    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.float16,
        safety_checker=None, requires_safety_checker=False
    ).to("cuda")

    # 스케줄러 교체(샘플 속도/샤프니스 선호에 따라 선택)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    # TI 임베딩 로드 (디렉터리 내 learned_embeds.bin 인식)
    if args.clothed_embed_dir:
        pipe.load_textual_inversion(args.clothed_embed_dir, token="<clothed>")
    if args.nudity_embed_dir:
        pipe.load_textual_inversion(args.nudity_embed_dir, token="<nudity>")

    # 선택: <clothed> cross-attn 부스팅
    set_pos = None
    if args.clothed_boost > 0:
        set_pos = install_attn_boost(pipe, token_str="clothed", boost=args.clothed_boost)
        set_pos(args.prompt)

    if not args.use_adaptive_cfg:
        image = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative,
            num_inference_steps=args.steps,
            guidance_scale=args.cfg,
            generator=g
        ).images[0]
        image.save(args.out)
        print(f"Saved -> {args.out}")
        return

    # ----- Adaptive CFG (후반부 강화) -----
    # 간단한 래퍼: step 비율에 따라 cfg를 선형 증가
    # 내부 루프를 완전히 재작성하지 않고 pipe.__call__을 반복 호출하는 것은 비효율적이므로,
    # 여기서는 pipe의 denoising loop를 그대로 사용하면서 scale을 타임스텝별로 조정.
    # Diffusers는 step별 scale 변경을 공식 지원하지 않으므로, 간단한 approximate 구현.

    from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipelineOutput

    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        args.prompt, num_images_per_prompt=1, do_classifier_free_guidance=True, negative_prompt=args.negative
    )

    # latents init
    height = pipe.unet.config.sample_size * pipe.vae_scale_factor
    width  = pipe.unet.config.sample_size * pipe.vae_scale_factor
    latents = torch.randn((1, pipe.unet.in_channels, height//8, width//8), generator=g, device=pipe.device, dtype=prompt_embeds.dtype)

    pipe.scheduler.set_timesteps(args.steps, device=pipe.device)
    for i, t in enumerate(pipe.scheduler.timesteps):
        frac = (i+1)/len(pipe.scheduler.timesteps)
        gs = args.cfg * (0.3 + 0.7*frac)  # 초반 0.3*cfg → 후반 cfg (선형 증가)
        # U-Net
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=torch.cat([negative_prompt_embeds, prompt_embeds])).sample
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + gs * (noise_pred_text - noise_pred_uncond)
        # step
        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

    # VAE decode
    image = pipe.vae.decode(latents / 0.18215).sample
    image = (image / 2 + 0.5).clamp(0,1).permute(0,2,3,1).float().cpu().numpy()[0]
    from PIL import Image
    Image.fromarray((image*255).astype("uint8")).save(args.out)
    print(f"Saved -> {args.out}")

if __name__ == "__main__":
    main()

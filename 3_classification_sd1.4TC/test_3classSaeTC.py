import os
import random
from argparse import ArgumentParser
from functools import partial
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline
from diffusers.models.attention_processor import Attention
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from geo_utils.custom_stable_diffusion import CustomStableDiffusionPipeline, CustomStableDiffusionImg2ImgPipeline
from geo_utils.guidance_utils import GuidanceModel
from geo_utils.sae_probe import SAEProbe

import numpy as np
from typing import List, Optional

########################
# Parsers
#########################
def parse_args():
    parser = ArgumentParser(description='Generation script')
    parser.add_argument('ckpt_path', type=str)
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--nsamples', type=int, default=4)
    parser.add_argument('--cfg_scale', type=float, default=5)
    parser.add_argument('--num_inference_steps', type=int, default=50)
    parser.add_argument('--trained_text_encoder', action="store_true")
    parser.add_argument('--seed', type=int, default=1234)

    # Guidance 관련 인자
    parser.add_argument('--freedom', action="store_true", help="Use guidance with trained discriminator")
    parser.add_argument('--freedom_scale', type=float, default=10.0)
    parser.add_argument('--freedom_scale_t0', type=float, default=None,
                        help="Step 0에서 사용할 freedom scale (기본: freedom_scale * 0.5)")
    parser.add_argument('--freedom_model_type', type=str, default="classifier")
    parser.add_argument('--freedom_model_args_file', type=str, default="configs/models/classifier.yaml")
    parser.add_argument('--freedom_model_ckpt', type=str, default="classifier_ckpt/classifier.pth")
    parser.add_argument("--guide_start", type=int, default=1, help="The start index for guidance")

    parser.add_argument('--output_dir', type=str, default='output_img/tmp')

    # Prompt file argument
    parser.add_argument("--prompt_file", type=str, required=True, help="Path to the file containing prompts")

    # === SAE probe 옵션 ===
    parser.add_argument('--sae_probe', action="store_true",
                        help="Enable SAE feature logging with SAeUron")
    parser.add_argument('--sae_repo', type=str, default="bcywinski/SAeUron",
                        help="HF Hub repo to load SAE from")
    parser.add_argument('--sae_hookpoint', type=str, default="unet.up_blocks.1.attentions.1",
                        help="UNet hookpoint for SAE features")
    parser.add_argument('--sae_topk', type=int, default=32,
                        help="Top-K SAE features to average for FAI score")
    parser.add_argument('--sae_csv', type=str, default=None,
                        help="Path to save per-step FAI CSV (e.g., Continual/3classifier/10/fai_log.csv)")
    parser.add_argument('--sae_calibrate', action="store_true",
                        help="Run quick calibration to select feature set Fc")
    parser.add_argument('--sae_nudity_prompts', type=str, default=None,
                        help="File with nudity-anchor prompts (one per line)")
    parser.add_argument('--sae_neutral_prompts', type=str, default=None,
                        help="File with neutral prompts (one per line); default: first 10 from --prompt_file")

    # === Harmful concept suppression (Global only) ===
    parser.add_argument('--harm_suppress', action="store_true",
                        help="Enable token-wise harmful concept suppression in Cross-Attention scores (global)")
    parser.add_argument('--harm_tau', type=float, default=0.1,
                        help="Cosine similarity threshold τ (values above τ are penalized)")
    parser.add_argument('--harm_gamma_start', type=float, default=1.8,
                        help="Initial suppression strength γ at early steps")
    parser.add_argument('--harm_gamma_end', type=float, default=0.3,
                        help="Final suppression strength γ at late steps")
    parser.add_argument('--harm_global_texts', type=str, default=None,
                        help="Path to a file listing global harmful concepts (one per line). Required if --harm_suppress.")

    args = parser.parse_known_args()[0]
    return args

########################
# Image Save
########################
def save_image(image, img_metadata, root="output_img"):
    path = img_metadata['file_name']
    img_height = img_metadata['height']
    img_width = img_metadata['width']

    image = np.asarray(image)
    image = Image.fromarray(image, mode='RGB')
    image = image.resize((img_width, img_height))
    path = os.path.join(root, path[:-4] + '.png')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image.save(path)

########################
# Harmful concept suppression utils (Global)
########################
@torch.no_grad()
def build_harm_vec_sd14(pipe, concepts: List[str]) -> Optional[torch.Tensor]:
    """
    SD-1.4: CLIP text encoder last_hidden_state의 평균을 L2 정규화해 대표 벡터를 생성합니다.
    여러 콘셉트가 있으면 평균 방향을 사용합니다.
    반환: (768,) 또는 None
    """
    if not concepts:
        return None
    tok = pipe.tokenizer(
        concepts,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    ).to(pipe.device)
    out = pipe.text_encoder(**tok, output_hidden_states=True)
    h = out.last_hidden_state  # (N, L, 768)
    v = F.normalize(h.mean(dim=1), dim=-1)  # (N, 768)
    v = F.normalize(v.mean(dim=0), dim=-1)  # (768,)
    return v

class HarmCfg:
    __slots__ = ("enable", "tau", "gamma")
    def __init__(self, enable=True, tau=0.1, gamma=1.0):
        self.enable = bool(enable)
        self.tau = float(tau)
        self.gamma = float(gamma)

def gamma_schedule_linear(step: int, num_steps: int, g_start: float, g_end: float) -> float:
    t = step / max(1, num_steps - 1)
    return g_start * (1.0 - t) + g_end * t

class HarmfulAttnProcessor(nn.Module):
    """
    SD1.4 Cross-Attention scores(QK^T/sqrt(d))에 토큰별 penalty를 뺄셈으로 주입합니다.
    encoder_hidden_states(=텍스트 토큰 히든)와 harm_vec 간의 코사인 유사도 기반 penalty입니다.

    ✅ 주의:
    - diffusers 버전에 따라 Attention.dropout이 float일 수 있어 항상 F.dropout로 처리
    - 이 프로세서는 nn.Module이지만 UNet에 register되지 않으므로 .to(device) 수동 호출 필요
    """
    def __init__(self, harm_vec: Optional[torch.Tensor], cfg: HarmCfg):
        super().__init__()
        self.cfg = cfg
        if harm_vec is None or harm_vec.numel() == 0:
            self.register_buffer("harm_vec", torch.empty(0), persistent=False)
        else:
            self.register_buffer("harm_vec", F.normalize(harm_vec.detach(), dim=-1), persistent=False)

    def set_enable(self, flag: bool):
        self.cfg.enable = bool(flag)

    def set_gamma(self, gamma: float):
        self.cfg.gamma = float(gamma)

    def forward(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
    ) -> torch.FloatTensor:
        batch_size, sequence_length, _ = hidden_states.shape
        is_cross = encoder_hidden_states is not None

        # (A) 전처리 및 Q/K/V 생성
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        if is_cross:
            key   = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
        else:
            key   = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)

        # (B) 헤드 차원으로 재배열
        query = attn.head_to_batch_dim(query)  # (B*H, Q_len, d_h)
        key   = attn.head_to_batch_dim(key)    # (B*H, K_len, d_h)
        value = attn.head_to_batch_dim(value)

        # (C) scores 계산 (softmax 직전)
        scores = torch.matmul(query, key.transpose(-1, -2)) * attn.scale  # (B*H, Q_len, K_len)

        # (D) 우리의 개입: 토큰별 penalty 삽입 (Global)
        if is_cross and self.cfg.enable and self.cfg.gamma > 0.0 and self.harm_vec.numel() > 0:
            ctx = encoder_hidden_states                # (B, K_len, d)
            harm = F.normalize(self.harm_vec, dim=-1)  # (d,)
            ctx_norm = F.normalize(ctx, dim=-1)        # (B, K_len, d)
            cos = torch.einsum("bkd,d->bk", ctx_norm, harm)  # (B, K_len)

            penalty_tok = torch.clamp(cos - self.cfg.tau, min=0.0)  # (B, K_len)
            B = ctx.shape[0]
            H = scores.shape[0] // B
            pen = penalty_tok.unsqueeze(1).repeat_interleave(H, dim=0)  # (B*H, 1, K_len)
            pen = pen.expand(-1, scores.shape[1], -1)                   # (B*H, Q_len, K_len)

            scores = scores - self.cfg.gamma * pen

        # (E) softmax + dropout (버전-safe)
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            scores = scores + attention_mask

        attn_probs = F.softmax(scores, dim=-1)

        # diffusers 버전에 따라 attn.dropout이 float 또는 nn.Dropout일 수 있음
        dropout_p = attn.dropout if isinstance(attn.dropout, float) else getattr(attn.dropout, "p", 0.0)
        attn_probs = F.dropout(attn_probs, p=dropout_p, training=attn.training)

        # (F) 출력
        hidden_states = torch.matmul(attn_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

########################
# Main
########################
def main(model=None):
    args = parse_args()
    accelerator = Accelerator()
    device = accelerator.device

    # 재현성
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Build Stable Diffusion pipeline (Disable Safety Checker)
    pipe = CustomStableDiffusionPipeline.from_pretrained(args.ckpt_path, safety_checker=None)
    pipe = pipe.to(device)
    print("Pipe device:", pipe.device)

    # Load prompts
    with open(args.prompt_file, "r") as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]
    print(f"Loaded {len(prompts)} prompts from {args.prompt_file}")

    ########################
    # Harm suppressor (Global only)
    ########################
    harm_proc = None
    if args.harm_suppress:
        global_concepts = []
        if args.harm_global_texts is not None and os.path.isfile(args.harm_global_texts):
            with open(args.harm_global_texts, "r") as f:
                global_concepts = [l.strip() for l in f if l.strip()]
        if len(global_concepts) == 0:
            print("[harm] Global suppression requested but no concepts provided. Disabling suppression.")
        else:
            harm_vec = build_harm_vec_sd14(pipe, global_concepts)  # (768,)
            harm_proc = HarmfulAttnProcessor(harm_vec, HarmCfg(enable=True, tau=args.harm_tau, gamma=args.harm_gamma_start)).to(device)
            pipe.unet.set_attn_processor(harm_proc)
            print(f"[harm] Cross-Attention global suppression enabled with {len(global_concepts)} concepts.")

    ########################
    # SAEProbe
    ########################
    sae_probe = None
    if args.sae_probe:
        sae_probe = SAEProbe(
            pipe=pipe,
            sae_repo=args.sae_repo,
            hookpoint=args.sae_hookpoint,
            device=str(device),
            topk_select=args.sae_topk,
            csv_path=args.sae_csv,
        )

    # (옵션) SAE feature 캘리브레이션
    if sae_probe is not None and args.sae_calibrate:
        if args.sae_nudity_prompts is None:
            nudity_prompts = ["nude human figure", "unclothed human figure"]
        else:
            with open(args.sae_nudity_prompts, "r") as f:
                nudity_prompts = [l.strip() for l in f if l.strip()]

        if args.sae_neutral_prompts is None:
            neutral_prompts = prompts[:10] if len(prompts) >= 10 else prompts
        else:
            with open(args.sae_neutral_prompts, "r") as f:
                neutral_prompts = [l.strip() for l in f if l.strip()]

        base_kws = dict(
            guidance_scale=1.0,
            height=512, width=512,
            num_inference_steps=20,
            callback_on_step_end=None,
            callback_on_step_end_tensor_inputs=None,
            num_images_per_prompt=1,
        )
        sae_probe.calibrate_fc(nudity_prompts, neutral_prompts, base_kws)

    ########################
    # Freedom guidance
    ########################
    freedom_model = None
    cb_tensor_inputs_base = None
    if args.freedom:
        freedom_model = GuidanceModel(pipe, args.freedom_model_args_file, args.freedom_model_ckpt, 1, pipe.device)
        cb_tensor_inputs_base = ["latents", "noise_pred", "prev_latents"]
        if args.freedom_model_type == "augmented_discriminator":
            cb_tensor_inputs_base += ["instance_prompt_embeds"]
        print("guidance!!")

    ########################
    # Generation Loop
    ########################
    scale = args.cfg_scale
    root = args.output_dir
    os.makedirs(root, exist_ok=True)

    # 콜백: SAE 로깅 + freedom guidance + harm γ 스케줄 업데이트 (Global)
    def pre_freedom_iteration(
        diffusion_pipeline, step, timestep, callback_kwargs,
        freedom_model, freedom_scale, guide_start=0,
        prompt_idx=None, sae_probe=None, harm_proc=None,
        num_steps: int = 50, freedom_scale_t0: Optional[float] = None
    ):
        # (0) harm γ 스케줄 업데이트 (Global processor)
        if harm_proc is not None:
            g = gamma_schedule_linear(step, num_steps, args.harm_gamma_start, args.harm_gamma_end)
            harm_proc.set_gamma(g)

        # (1) SAE per-step 로깅
        if sae_probe is not None:
            try:
                sae_probe.log_step(prompt_idx if prompt_idx is not None else -1, step, timestep)
            except Exception as e:
                print(f"[SAEProbe] log_step error: {e}")

        # (2) freedom guidance
        if (freedom_model is not None) and (guide_start <= step):
            local_scale = freedom_scale
            guidance_result = freedom_model.guidance(
                diffusion_pipeline, callback_kwargs, step, timestep,
                local_scale, target_class=1
            )
            return guidance_result
        else:
            return callback_kwargs

    for idx, prompt in enumerate(prompts):
        print(f"Generating image for prompt {idx + 1}: {prompt}")

        # 콜백 partial
        if args.freedom or (sae_probe is not None) or (harm_proc is not None):
            freedom_iteration = partial(
                pre_freedom_iteration,
                freedom_model=freedom_model if args.freedom else None,
                freedom_scale=args.freedom_scale,
                guide_start=args.guide_start,
                prompt_idx=idx,
                sae_probe=sae_probe,
                harm_proc=harm_proc,
                num_steps=args.num_inference_steps,
                freedom_scale_t0=args.freedom_scale_t0
            )
            if args.freedom:
                freedom_callback_on_step_end_tensor_input = cb_tensor_inputs_base
            else:
                freedom_callback_on_step_end_tensor_input = ["latents", "noise_pred", "prev_latents"]
        else:
            freedom_iteration = None
            freedom_callback_on_step_end_tensor_input = None

        # Generate
        input_dict = {
            "prompt": prompt,
            "guidance_scale": scale,
            "num_inference_steps": args.num_inference_steps,
            "height": 512,
            "width": 512,
            "callback_on_step_end": freedom_iteration,
            "callback_on_step_end_tensor_inputs": freedom_callback_on_step_end_tensor_input,
            "callback": None,
            "callback_steps": 1,
            "bbox_binary_mask": None,
            "num_images_per_prompt": args.nsamples,
        }

        with torch.enable_grad():
            generated_images = pipe(**input_dict).images

        # Save first image
        img_metadata = {"file_name": f"{idx + 1}.png", "height": 512, "width": 512}
        save_image(generated_images[0], img_metadata, root=root)

    # SAE CSV 저장 & 훅 해제
    if sae_probe is not None:
        try:
            sae_probe.flush_csv()
            print("[SAEProbe] CSV saved:", args.sae_csv)
        finally:
            sae_probe.close()

if __name__ == "__main__":
    main()

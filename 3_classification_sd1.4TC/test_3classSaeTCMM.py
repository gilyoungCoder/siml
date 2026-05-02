#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random
from argparse import ArgumentParser
from functools import partial
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers.models.attention_processor import AttnProcessor2_0, Attention

from geo_utils.custom_stable_diffusion import (
    CustomStableDiffusionPipeline,
    CustomStableDiffusionImg2ImgPipeline,
)
from geo_utils.guidance_utils import GuidanceModel
from geo_utils.sae_probe import SAEProbe

import numpy as np
from typing import List, Optional


# =========================
# Args
# =========================
def parse_args():
    parser = ArgumentParser(description="Generation script (global suppression + allow + freedom + SAE)")

    # Positional
    parser.add_argument("ckpt_path", type=str)

    # Generation
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--nsamples", type=int, default=4)
    parser.add_argument("--cfg_scale", type=float, default=5)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--trained_text_encoder", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output_dir", type=str, default="output_img/tmp")
    parser.add_argument("--prompt_file", type=str, required=True)

    # Freedom guidance
    parser.add_argument("--freedom", action="store_true", help="Use guidance with trained discriminator")
    parser.add_argument("--freedom_scale", type=float, default=10.0)
    parser.add_argument("--freedom_scale_early", type=float, default=None, help="Optional smaller scale for early steps")
    parser.add_argument("--freedom_early_steps", type=int, default=0, help="How many initial steps use *_early scale")
    parser.add_argument("--freedom_model_type", type=str, default="classifier")
    parser.add_argument("--freedom_model_args_file", type=str, default="configs/models/classifier.yaml")
    parser.add_argument("--freedom_model_ckpt", type=str, default="classifier_ckpt/classifier.pth")
    parser.add_argument("--guide_start", type=int, default=1, help="The start index for freedom guidance")

    # SAE probe
    parser.add_argument("--sae_probe", action="store_true", help="Enable SAE feature logging with SAeUron")
    parser.add_argument("--sae_repo", type=str, default="bcywinski/SAeUron")
    parser.add_argument("--sae_hookpoint", type=str, default="unet.up_blocks.1.attentions.1")
    parser.add_argument("--sae_topk", type=int, default=32)
    parser.add_argument("--sae_csv", type=str, default=None)
    parser.add_argument("--sae_calibrate", action="store_true")
    parser.add_argument("--sae_nudity_prompts", type=str, default=None)
    parser.add_argument("--sae_neutral_prompts", type=str, default=None)

    # Global harmful suppression (soft + hard)
    parser.add_argument("--harm_suppress", action="store_true",
                        help="Enable token-wise harmful concept suppression in Cross-Attention (global)")
    parser.add_argument("--harm_tau", type=float, default=0.10, help="Soft cosine threshold τ (>= triggers soft delete)")
    parser.add_argument("--harm_gamma_start", type=float, default=1.80, help="γ at early steps")
    parser.add_argument("--harm_gamma_end", type=float, default=0.30, help="γ at late steps")
    parser.add_argument("--harm_global_texts", type=str, default=None, help="File: harmful concepts (one per line)")

    # Dual-mode (soft + hard together)
    parser.add_argument("--harm_dual", action="store_true",
                        help="Apply soft delete AND hard block together (hard uses --harm_hard_tau)")
    parser.add_argument("--harm_hard_tau", type=float, default=None,
                        help="Hard block cosine threshold τ_hard (suggest τ_hard > τ). If None and --harm_dual is set, defaults to τ+0.20")

    # Allow-list & debug
    parser.add_argument("--harm_allowlist_texts", type=str, default=None, help="File: allow-list (one per line)")
    parser.add_argument("--harm_allow_tau", type=float, default=0.35, help="Allow-list exemption threshold")
    parser.add_argument("--harm_hard_block", action="store_true",
                        help="Legacy: ONLY hard block (mask -1e9) instead of soft delete; ignored if --harm_dual is set")
    parser.add_argument("--harm_debug_print", action="store_true", help="Print harm/allow cosines per token")
    parser.add_argument("--harm_debug_prompt", type=str, default=None,
                        help="If set, debug-print only when this substring (case-insensitive) appears in the prompt")

    args = parser.parse_known_args()[0]
    return args


# =========================
# Utils
# =========================
def save_image(image, img_metadata, root="output_img"):
    path = img_metadata["file_name"]
    img_height = img_metadata["height"]
    img_width = img_metadata["width"]

    image = np.asarray(image)
    image = Image.fromarray(image, mode="RGB")
    image = image.resize((img_width, img_height))
    path = os.path.join(root, path[:-4] + ".png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image.save(path)


@torch.no_grad()
def build_harm_vec_sd14(pipe, concepts: List[str]) -> Optional[torch.Tensor]:
    """
    SD-1.4: harmful 텍스트들의 CLIP last_hidden_state 평균 -> L2 normalize -> 평균 방향 (d,)
    """
    if not concepts:
        return None
    tok = pipe.tokenizer(
        concepts,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(pipe.device)
    out = pipe.text_encoder(**tok, output_hidden_states=True)
    h = out.last_hidden_state  # (N, L, d)
    v = F.normalize(h.mean(dim=1), dim=-1)  # (N, d)
    v = F.normalize(v.mean(dim=0), dim=-1)  # (d,)
    return v


@torch.no_grad()
def build_vecs_sd14(pipe, texts: List[str]) -> Optional[torch.Tensor]:
    """
    texts 각각을 CLIP last_hidden_state 평균 후 정규화 -> (M, d)
    """
    if not texts:
        return None
    tok = pipe.tokenizer(
        texts,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(pipe.device)
    out = pipe.text_encoder(**tok, output_hidden_states=True)
    h = out.last_hidden_state  # (M, L, d)
    v = F.normalize(h.mean(dim=1), dim=-1)  # (M, d)
    return v


@torch.no_grad()
def debug_print_prompt_cos(
    pipe,
    prompt: str,
    harm_vec: Optional[torch.Tensor],
    allow_mat: Optional[torch.Tensor],
    allow_labels: List[str],
    tau: float,
    allow_tau: float,
):
    """
    프롬프트 각 토큰의 harm/allow 코사인과 억제 여부를 프린트
    """
    tok = pipe.tokenizer(
        [prompt],
        padding="max_length",
        truncation=True,
        max_length=pipe.tokenizer.model_max_length,
        return_tensors="pt",
    ).to(pipe.device)

    out = pipe.text_encoder(**tok, output_hidden_states=True)
    H = F.normalize(out.last_hidden_state[0], dim=-1)  # (L, d)
    dev = H.device
    ids = tok.input_ids[0].tolist()
    mask = tok.attention_mask[0].bool().tolist()
    toks = pipe.tokenizer.convert_ids_to_tokens(ids)

    hv = F.normalize(harm_vec, dim=-1).to(dev) if (harm_vec is not None and harm_vec.numel() > 0) else None
    AM = allow_mat.to(dev) if (allow_mat is not None and allow_mat.numel() > 0) else None

    print(f"\n[HARM/ALLOW DEBUG] prompt = {prompt!r}, tau={tau:.3f}, allow_tau={allow_tau:.3f}")
    print(f"{'idx':>3} {'token':>18} {'harm_cos':>9} {'allow_max':>10} {'best_allow':>16} {'suppress?':>10}")
    print("-" * 72)

    for i, use in enumerate(mask):
        if not use:
            break
        token = toks[i].replace("Ġ", "▁")
        harm_cos = float(torch.dot(H[i], hv).item()) if hv is not None else float("nan")

        allow_max, best = 0.0, ""
        if AM is not None:
            sims = torch.matmul(H[i], AM.T)  # (M,)
            mval, midx = sims.max(dim=0)
            allow_max = float(mval.item())
            best = allow_labels[int(midx.item())] if len(allow_labels) > 0 else ""

        suppress = (harm_cos >= tau) and not (allow_max >= allow_tau)
        print(f"{i:>3} {token:>18} {harm_cos:+.3f} {allow_max:+.3f} {best:>16} {str(suppress):>10}")


def gamma_schedule_linear(step: int, num_steps: int, g_start: float, g_end: float) -> float:
    t = step / max(1, num_steps - 1)
    return g_start * (1.0 - t) + g_end * t


# =========================
# Harmful Attn Processor
# =========================
class HarmCfg:
    __slots__ = ("enable", "tau", "gamma", "allow_tau", "hard_block", "dual", "tau_hard")

    def __init__(
        self,
        enable=True,
        tau: float = 0.1,
        gamma: float = 1.0,
        allow_tau: float = 0.35,
        hard_block: bool = False,
        dual: bool = False,
        tau_hard: Optional[float] = None,
    ):
        self.enable = bool(enable)
        self.tau = float(tau)
        self.gamma = float(gamma)
        self.allow_tau = float(allow_tau)
        self.hard_block = bool(hard_block)
        self.dual = bool(dual)
        self.tau_hard = float(tau_hard) if (tau_hard is not None) else None


class HarmfulAttnProcessor(AttnProcessor2_0):
    """
    Cross-Attention 점수(QK^T/√d)에 harm/allow 규칙 주입.
    - Soft delete: scores -= γ for tokens over τ (if not allow)
    - Hard block:  scores := -∞ for tokens over τ_hard (if not allow)
    - Legacy single-mode도 유지(hard_block=True면 오직 hard만, dual이면 soft+hard 둘다)
    """
    def __init__(
        self,
        harm_vec: Optional[torch.Tensor],
        allow_mat: Optional[torch.Tensor],
        cfg: HarmCfg,
    ):
        super().__init__()
        self.cfg = cfg
        self.training = False

        # CPU에 보관(정규화까지 완료) -> __call__ 때 hidden_states.device로 이동
        if harm_vec is None or harm_vec.numel() == 0:
            self._harm_vec = None
        else:
            self._harm_vec = F.normalize(harm_vec.detach().float(), dim=-1).cpu()

        if allow_mat is None or allow_mat.numel() == 0:
            self._allow_mat = None
        else:
            self._allow_mat = F.normalize(allow_mat.detach().float(), dim=-1).cpu()

    @property
    def harm_vec(self) -> Optional[torch.Tensor]:
        return self._harm_vec

    @property
    def allow_mat(self) -> Optional[torch.Tensor]:
        return self._allow_mat

    def set_gamma(self, gamma: float):
        self.cfg.gamma = float(gamma)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
    ) -> torch.FloatTensor:
        dev = hidden_states.device
        harm_vec = self._harm_vec.to(dev) if self._harm_vec is not None else None
        allow_mat = self._allow_mat.to(dev) if self._allow_mat is not None else None

        batch_size, sequence_length, _ = hidden_states.shape
        is_cross = encoder_hidden_states is not None

        # (A) pre
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

        # (B) head reshape
        query = attn.head_to_batch_dim(query)  # (B*H, Q, d_h)
        key   = attn.head_to_batch_dim(key)    # (B*H, K, d_h)
        value = attn.head_to_batch_dim(value)

        # (C) raw scores
        scores = torch.matmul(query, key.transpose(-1, -2)) * attn.scale  # (B*H, Q, K)

        # (D) suppression
        if is_cross and self.cfg.enable and (harm_vec is not None):
            ctx = encoder_hidden_states                   # (B, K, d)
            ctx_n = F.normalize(ctx, dim=-1)              # (B, K, d)
            harm = F.normalize(harm_vec, dim=-1)          # (d,)
            cos_harm = torch.einsum("bkd,d->bk", ctx_n, harm)  # (B, K)

            safe_mask = None
            if (allow_mat is not None) and allow_mat.numel() > 0:
                sims = torch.einsum("bkd,md->bkm", ctx_n, allow_mat)  # (B, K, M)
                safe_max = sims.amax(dim=-1)                          # (B, K)
                safe_mask = (safe_max >= self.cfg.allow_tau)

            # soft mask
            over_soft = (cos_harm >= self.cfg.tau)
            if safe_mask is not None:
                over_soft = over_soft & (~safe_mask)

            # hard mask (dual mode) or legacy hard-only
            over_hard = None
            if self.cfg.dual:
                th = self.cfg.tau_hard if (self.cfg.tau_hard is not None) else (self.cfg.tau + 0.20)
                over_hard = (cos_harm >= th)
                if safe_mask is not None:
                    over_hard = over_hard & (~safe_mask)
            elif self.cfg.hard_block:
                # legacy: treat τ as hard threshold
                over_hard = over_soft.clone()
                over_soft = torch.zeros_like(over_soft)  # no soft in legacy hard-only

            B = ctx.shape[0]
            Q = scores.shape[1]
            K = scores.shape[2]
            H = scores.shape[0] // B

            if over_soft is not None:
                soft_exp = over_soft[:, None, :].expand(B, Q, K).repeat_interleave(H, dim=0)
                if soft_exp.any():
                    scores = scores - (soft_exp.float() * self.cfg.gamma)

            if over_hard is not None:
                hard_exp = over_hard[:, None, :].expand(B, Q, K).repeat_interleave(H, dim=0)
                if hard_exp.any():
                    scores = scores.masked_fill(hard_exp, -1e9)

        # (E) softmax (+ mask)
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            scores = scores + attention_mask

        attn_probs = F.softmax(scores, dim=-1)

        # (F) dropout (버전별 안전 처리)
        if isinstance(attn.dropout, nn.Dropout):
            attn_probs = attn.dropout(attn_probs)
        else:
            p = float(attn.dropout) if isinstance(attn.dropout, (int, float)) else 0.0
            attn_probs = F.dropout(attn_probs, p=p, training=False)

        # (G) out
        hidden_states = torch.matmul(attn_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


# =========================
# Main
# =========================
def main(model=None):
    args = parse_args()
    accelerator = Accelerator()
    device = accelerator.device

    # seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # pipeline
    pipe = CustomStableDiffusionPipeline.from_pretrained(
        args.ckpt_path, safety_checker=None
    ).to(device)
    print("Pipe device:", pipe.device)

    # prompts
    prompt_file = os.path.expanduser(args.prompt_file)
    with open(prompt_file, "r") as f:
        prompts = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(prompts)} prompts from {prompt_file}")

    # =========================
    # Harm suppressor (global)
    # =========================
    harm_proc = None
    allow_mat, allow_labels = None, []

    if args.harm_suppress:
        # harmful list
        harm_file = os.path.expanduser(args.harm_global_texts) if args.harm_global_texts else None
        global_concepts = []
        if harm_file and os.path.isfile(harm_file):
            with open(harm_file, "r") as f:
                global_concepts = [l.strip() for l in f if l.strip()]

        if len(global_concepts) == 0:
            print("[harm] Global suppression requested but no concepts provided. Disabling suppression.")
        else:
            harm_vec = build_harm_vec_sd14(pipe, global_concepts)  # (d,)

            # allow list (optional)
            allow_file = os.path.expanduser(args.harm_allowlist_texts) if args.harm_allowlist_texts else None
            if allow_file and os.path.isfile(allow_file):
                with open(allow_file, "r") as f:
                    allow_labels = [l.strip() for l in f if l.strip()]
                allow_mat = build_vecs_sd14(pipe, allow_labels)  # (M, d)

            harm_proc = HarmfulAttnProcessor(
                harm_vec=harm_vec,
                allow_mat=allow_mat,
                cfg=HarmCfg(
                    enable=True,
                    tau=args.harm_tau,
                    gamma=args.harm_gamma_start,
                    allow_tau=args.harm_allow_tau,
                    hard_block=(False if args.harm_dual else args.harm_hard_block),
                    dual=args.harm_dual,
                    tau_hard=args.harm_hard_tau,
                ),
            )
            pipe.unet.set_attn_processor(harm_proc)
            print(f"[harm] Global suppression enabled. harm={global_concepts} allow={len(allow_labels)} items "
                  f"(dual={args.harm_dual}, tau={args.harm_tau}, tau_hard={args.harm_hard_tau})")

    # =========================
    # SAE Probe
    # =========================
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

    # SAE calibration (optional)
    if sae_probe is not None and args.sae_calibrate:
        nudity_path = os.path.expanduser(args.sae_nudity_prompts) if args.sae_nudity_prompts else None
        neutral_path = os.path.expanduser(args.sae_neutral_prompts) if args.sae_neutral_prompts else None

        if nudity_path is None:
            nudity_prompts = ["nude human figure", "unclothed human figure"]
        else:
            with open(nudity_path, "r") as f:
                nudity_prompts = [l.strip() for l in f if l.strip()]

        if neutral_path is None:
            neutral_prompts = prompts[:10] if len(prompts) >= 10 else prompts
        else:
            with open(neutral_path, "r") as f:
                neutral_prompts = [l.strip() for l in f if l.strip()]

        base_kws = dict(
            guidance_scale=1.0,
            height=512,
            width=512,
            num_inference_steps=20,
            callback_on_step_end=None,
            callback_on_step_end_tensor_inputs=None,
            num_images_per_prompt=1,
        )
        sae_probe.calibrate_fc(nudity_prompts, neutral_prompts, base_kws)

    # =========================
    # Freedom guidance
    # =========================
    freedom_model = None
    cb_tensor_inputs_base = None
    if args.freedom:
        freedom_model = GuidanceModel(
            pipe, args.freedom_model_args_file, args.freedom_model_ckpt, 1, pipe.device
        )
        cb_tensor_inputs_base = ["latents", "noise_pred", "prev_latents"]
        if args.freedom_model_type == "augmented_discriminator":
            cb_tensor_inputs_base += ["instance_prompt_embeds"]
        print("[freedom] guidance initialized")

    # =========================
    # Generation
    # =========================
    scale = args.cfg_scale
    root = os.path.expanduser(args.output_dir)
    os.makedirs(root, exist_ok=True)

    # early scale default (if not provided)
    freedom_scale_early = args.freedom_scale if (args.freedom_scale_early is None) else args.freedom_scale_early

    def callback_on_step_end_fn(
        diffusion_pipeline,
        step,
        timestep,
        callback_kwargs,
        freedom_model,
        freedom_scale,
        freedom_scale_early,
        freedom_early_steps,
        guide_start,
        prompt_idx,
        sae_probe,
        harm_proc,
        num_steps: int,
    ):
        # update γ schedule for harm
        if harm_proc is not None:
            g = gamma_schedule_linear(step, num_steps, args.harm_gamma_start, args.harm_gamma_end)
            harm_proc.set_gamma(g)

        # SAE log
        if sae_probe is not None:
            try:
                sae_probe.log_step(prompt_idx if prompt_idx is not None else -1, step, timestep)
            except Exception as e:
                print(f"[SAEProbe] log_step error: {e}")

        # freedom guidance
        if (freedom_model is not None) and (guide_start <= step):
            local_scale = (
                freedom_scale_early if step < max(0, int(freedom_early_steps)) else freedom_scale
            )
            guidance_result = freedom_model.guidance(
                diffusion_pipeline, callback_kwargs, step, timestep, local_scale, target_class=1
            )
            return guidance_result
        else:
            return callback_kwargs

    # per-prompt loop
    for idx, prompt in enumerate(prompts):
        print(f"\n=== Generating image for prompt {idx + 1}: {prompt}")

        # optional debug print
        if args.harm_debug_print and args.harm_suppress:
            if (args.harm_debug_prompt is None) or (args.harm_debug_prompt.lower() in prompt.lower()):
                debug_print_prompt_cos(
                    pipe=pipe,
                    prompt=prompt,
                    harm_vec=harm_proc.harm_vec if harm_proc is not None else None,
                    allow_mat=harm_proc.allow_mat if harm_proc is not None else None,
                    allow_labels=allow_labels,
                    tau=args.harm_tau,
                    allow_tau=args.harm_allow_tau,
                )

        # build callback
        if args.freedom or (sae_probe is not None) or (harm_proc is not None):
            cb = partial(
                callback_on_step_end_fn,
                freedom_model=freedom_model if args.freedom else None,
                freedom_scale=args.freedom_scale,
                freedom_scale_early=freedom_scale_early,
                freedom_early_steps=args.freedom_early_steps,
                guide_start=args.guide_start,
                prompt_idx=idx,
                sae_probe=sae_probe,
                harm_proc=harm_proc,
                num_steps=args.num_inference_steps,
            )
            cb_tensor_inputs = cb_tensor_inputs_base if args.freedom else ["latents", "noise_pred", "prev_latents"]
        else:
            cb = None
            cb_tensor_inputs = None

        # run
        input_dict = {
            "prompt": prompt,
            "guidance_scale": scale,
            "num_inference_steps": args.num_inference_steps,
            "height": 512,
            "width": 512,
            "callback_on_step_end": cb,
            "callback_on_step_end_tensor_inputs": cb_tensor_inputs,
            "callback": None,
            "callback_steps": 1,
            "bbox_binary_mask": None,
            "num_images_per_prompt": args.nsamples,
        }

        with torch.enable_grad():
            out = pipe(**input_dict)
            generated_images = out.images

        # save first image
        img_metadata = {"file_name": f"{idx + 1}.png", "height": 512, "width": 512}
        save_image(generated_images[0], img_metadata, root=root)

    # SAE CSV flush & close
    if sae_probe is not None:
        try:
            sae_probe.flush_csv()
            print("[SAEProbe] CSV saved:", args.sae_csv)
        finally:
            sae_probe.close()


if __name__ == "__main__":
    main()

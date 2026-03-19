#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Machine Unlearning with LEAVE-ONE-OUT (LOO) Token Criticality + ADAPTIVE THRESHOLD
(Probability Redistribution version)

Key points:
  • LOO 기반 토큰 크리티컬리티, 프롬프트별 ADAPTIVE τ 유지
  • 어텐션 억제는 score 단계가 아니라 softmax 이후 확률에서 수행
  • 억제 질량은 허용 키들(allow set)로 재분배 → 행합=1 보존, 부드러운 분포 재형성
  • Classifier Guidance(선택) 및 단계별 γ 스케줄 유지
"""

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

from geo_utils.custom_stable_diffusion import CustomStableDiffusionPipeline
from geo_utils.guidance_utils import GuidanceModel

import numpy as np
from typing import List, Optional, Tuple


# =========================
# Arguments
# =========================
def parse_args():
    parser = ArgumentParser(description="Machine Unlearning with ADAPTIVE THRESHOLD + Prob-Redistribution + DEBUG")

    # Model & Generation
    parser.add_argument("ckpt_path", type=str, help="Path to pretrained model checkpoint")
    parser.add_argument("--prompt_file", type=str, required=True, help="File containing prompts (one per line)")
    parser.add_argument("--output_dir", type=str, default="output_img/unlearning_adaptive", help="Output directory")

    # Generation parameters
    parser.add_argument("--nsamples", type=int, default=1, help="Number of samples per prompt")
    parser.add_argument("--cfg_scale", type=float, default=5.0, help="Classifier-free guidance scale")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")

    # Harmful concept suppression (Attention Manipulation)
    parser.add_argument("--harm_suppress", action="store_true", help="Enable harmful concept suppression")
    parser.add_argument("--harm_concepts_file", type=str, default="./configs/harm_concepts.txt",
                        help="File containing harmful concepts to suppress (one per line)")

    # LOO Criticality Method
    parser.add_argument("--loo_method", type=str, default="harm_distance",
                        choices=["harm_distance", "harm_cosine", "harm_distance_negative", "embedding_l2"],
                        help=("LOO criticality computation method:\n"
                              "  harm_distance: D(E_{-i}, harm) - D(E_full, harm) [removes harmful tokens]\n"
                              "  harm_cosine: cos(E_full, harm) - cos(E_{-i}, harm) [removes harmful tokens]\n"
                              "  harm_distance_negative: D(E_full, harm) - D(E_{-i}, harm) [removes protective tokens]\n"
                              "  embedding_l2: ||E_full - E_{-i}|| [old method, not harm-aware]"))

    # ADAPTIVE THRESHOLD parameters
    parser.add_argument("--adaptive_threshold", action="store_true",
                        help="Use adaptive threshold based on prompt distribution")
    parser.add_argument("--base_tau", type=float, default=0.15,
                        help="Base threshold (used if adaptive disabled, or as fallback)")
    parser.add_argument("--central_percentile", type=float, default=0.80,
                        help="Central percentile for computing mean (default: 0.80 = 80%)")
    parser.add_argument("--tau_factor", type=float, default=1.05,
                        help="Multiplicative factor for adaptive threshold: tau = central_mean * factor")

    # Gamma schedule (for redistribution strength shaping)
    parser.add_argument("--harm_gamma_start", type=float, default=20.0,
                        help="Suppression strength at early steps (prob-space scale; 8~30 권장)")
    parser.add_argument("--harm_gamma_end", type=float, default=0.5,
                        help="Suppression strength at late steps")

    # Probability redistribution options
    parser.add_argument("--topk_min", type=int, default=0, help="Ensure at least K positive-crit keys suppressed")
    parser.add_argument("--redistribute_mode", type=str, default="proportional",
                        choices=["proportional", "uniform"], help="How to redistribute shed mass over allow set")

    # Classifier Guidance
    parser.add_argument("--classifier_guidance", action="store_true", help="Enable classifier guidance")
    parser.add_argument("--classifier_config", type=str,
                        default="./configs/models/time_dependent_discriminator.yaml",
                        help="Classifier model configuration file")
    parser.add_argument("--classifier_ckpt", type=str,
                        default="./work_dirs/nudity_three_class/checkpoint/step_11800/classifier.pth",
                        help="Classifier checkpoint path")
    parser.add_argument("--guidance_scale", type=float, default=5.0,
                        help="Classifier guidance scale")
    parser.add_argument("--guidance_start_step", type=int, default=1,
                        help="Step to start applying guidance")
    parser.add_argument("--target_class", type=int, default=1,
                        help="Target class for guidance (1=clothed people)")

    # DEBUG options
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--debug_prompts", action="store_true", help="Show per-prompt token analysis")
    parser.add_argument("--debug_steps", action="store_true", help="Show per-step attention analysis")

    args = parser.parse_args()
    return args


# =========================
# Utilities
# =========================
def save_image(image, filename, root="output_img"):
    """Save generated image to disk."""
    path = os.path.join(root, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    image = np.asarray(image)
    image = Image.fromarray(image, mode="RGB")
    image = image.resize((512, 512))
    image.save(path)


def schedule_linear(step: int, num_steps: int, start_val: float, end_val: float) -> float:
    """Linear scheduling from start_val to end_val."""
    t = step / max(1, num_steps - 1)
    return start_val * (1.0 - t) + end_val * t


# =========================
# LEAVE-ONE-OUT TOKEN CRITICALITY
# =========================
@torch.no_grad()
def compute_loo_criticality_simple(
    pipe,
    prompt: str,
    harm_vector: torch.Tensor,
    method: str = "harm_distance",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    LOO: remove each token, re-encode, measure harmful alignment change.
    Returns:
        criticality_scores: (L,) tensor
        content_mask:       (L,) bool
    """
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    device = pipe.device

    # Tokenize full prompt
    tokens_full = tokenizer(
        [prompt],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    outputs_full = text_encoder(**tokens_full, output_hidden_states=True, return_dict=True)
    hidden_states_full = outputs_full.hidden_states[-2][0]  # (L, d)

    input_ids = tokens_full.input_ids[0]
    attention_mask = tokens_full.attention_mask[0]

    # content mask (drop BOS/EOS/PAD)
    content_mask = attention_mask.bool().clone()
    for special_id in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id]:
        if special_id is not None:
            content_mask = content_mask & (input_ids != special_id)

    if content_mask.sum() == 0:
        return torch.zeros(len(input_ids), device=device), content_mask

    embedding_full = hidden_states_full[content_mask].mean(dim=0)
    embedding_full = embedding_full / embedding_full.norm()

    harm_vector_norm = (harm_vector / harm_vector.norm()).to(device)

    if method in ["harm_distance", "harm_distance_negative"]:
        dist_full = torch.norm(embedding_full - harm_vector_norm, p=2).item()
    elif method == "harm_cosine":
        cos_full = torch.dot(embedding_full, harm_vector_norm).item()

    criticality_scores = torch.zeros(len(input_ids), device=device)
    content_indices = torch.where(content_mask)[0].tolist()
    ids_full = input_ids.tolist()

    for idx in content_indices:
        if len(content_indices) == 1:
            criticality_scores[idx] = 1.0
            continue

        ids_loo = [tid for i, tid in enumerate(ids_full) if i != idx]
        loo_text = pipe.tokenizer.decode(
            ids_loo, skip_special_tokens=True, clean_up_tokenization_spaces=True
        ).strip()
        if not loo_text:
            criticality_scores[idx] = 1.0
            continue

        tokens_loo = pipe.tokenizer(
            [loo_text],
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        outputs_loo = pipe.text_encoder(**tokens_loo, output_hidden_states=True, return_dict=True)
        hidden_states_loo = outputs_loo.hidden_states[-2][0]

        mask_loo = tokens_loo.attention_mask[0].bool().clone()
        for sid in [pipe.tokenizer.bos_token_id, pipe.tokenizer.eos_token_id, pipe.tokenizer.pad_token_id]:
            if sid is not None:
                mask_loo = mask_loo & (tokens_loo.input_ids[0] != sid)
        if mask_loo.sum() == 0:
            criticality_scores[idx] = 1.0
            continue

        embedding_loo = hidden_states_loo[mask_loo].mean(dim=0)
        embedding_loo = embedding_loo / embedding_loo.norm()

        if method == "harm_distance":
            dist_loo = torch.norm(embedding_loo - harm_vector_norm, p=2).item()
            criticality = dist_loo - dist_full
        elif method == "harm_cosine":
            cos_loo = torch.dot(embedding_loo, harm_vector_norm).item()
            criticality = cos_full - cos_loo
        elif method == "harm_distance_negative":
            dist_loo = torch.norm(embedding_loo - harm_vector_norm, p=2).item()
            criticality = dist_full - dist_loo
        elif method == "embedding_l2":
            criticality = torch.norm(embedding_full - embedding_loo, p=2).item()
        else:
            criticality = 0.0

        criticality_scores[idx] = float(criticality)

    return criticality_scores, content_mask


# =========================
# ADAPTIVE THRESHOLD (central mean × factor)
# =========================
@torch.no_grad()
def compute_adaptive_threshold_from_tensor(
    vals: torch.Tensor, central_percentile: float = 0.80, factor: float = 1.05, base_tau: float = 0.15
) -> float:
    if vals.numel() == 0:
        return base_tau
    sorted_vals = torch.sort(vals)[0]
    n = len(sorted_vals)
    if n < 3:
        return base_tau
    tail = (1.0 - central_percentile) / 2.0
    s = max(0, int(n * tail))
    e = min(n, int(n * (1.0 - tail)))
    if e <= s:
        return base_tau
    central = sorted_vals[s:e]
    return float(torch.clamp(central.mean() * factor, 0.0, 1.0).item())


# =========================
# Debug: token table
# =========================
@torch.no_grad()
def debug_token_analysis(
    pipe,
    prompt: str,
    harm_vector: torch.Tensor,
    harm_concepts: List[str],
    tau: float,
    adaptive_tau: Optional[float] = None,
    tau_method: str = "fixed",
    loo_method: str = "harm_distance"
):
    tokenizer = pipe.tokenizer
    device = pipe.device

    print("\n" + "="*100)
    print(f"[DEBUG] LOO TOKEN ANALYSIS FOR PROMPT: '{prompt}'")
    print("="*100)

    crit, content_mask = compute_loo_criticality_simple(pipe, prompt, harm_vector, method=loo_method)
    content_crit = crit[content_mask]

    tokens = tokenizer(
        [prompt],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    input_ids = tokens.input_ids[0]
    attention_mask = tokens.attention_mask[0]
    token_strings = tokenizer.convert_ids_to_tokens(input_ids.tolist())

    print(f"\nHarm Concepts: {', '.join(harm_concepts)}")
    print(f"LOO Method: {loo_method}")

    if tau_method == "adaptive" and adaptive_tau is not None:
        print(f"\nThreshold Method: ADAPTIVE")
        print(f"  - Base τ: {tau:.3f}")
        print(f"  - Adaptive τ: {adaptive_tau:.3f} ⭐")
        if content_crit.numel() > 0:
            print(f"  - RAW criticality stats: min={content_crit.min():.3f}, "
                  f"mean={content_crit.mean():.3f}, max={content_crit.max():.3f}")
        eff_tau = adaptive_tau
    else:
        print(f"\nThreshold Method: FIXED")
        print(f"  - Fixed τ: {tau:.3f}")
        eff_tau = tau

    print(f"\n{'IDX':<4} {'TOKEN':<20} {'TYPE':<12} {'CRITICALITY (RAW)':<20} {'≥ τ?':<8} {'SUPPRESSED?':<12}")
    print("-"*100)

    suppressed_count, total_content_tokens = 0, 0
    for idx in range(len(input_ids)):
        if not attention_mask[idx]:
            break

        tok = token_strings[idx].replace('Ġ', '▁').replace('</w>', '')
        tid = int(input_ids[idx].item())
        token_type = "CONTENT"
        if tokenizer.bos_token_id and tid == tokenizer.bos_token_id:
            token_type = "BOS/SOT"
        elif tokenizer.eos_token_id and tid == tokenizer.eos_token_id:
            token_type = "EOS/EOT"
        elif tokenizer.pad_token_id and tid == tokenizer.pad_token_id:
            token_type = "PAD"

        crit_raw = float(crit[idx].item())
        is_content = bool(content_mask[idx].item())
        exceeds = (crit_raw >= eff_tau)
        is_supp = is_content and exceeds

        print(f"{idx:<4} {tok:<20} {token_type:<12} {crit_raw:<20.6f} "
              f"{('YES ⚠️' if exceeds else 'NO'):<8} {('YES 🔴' if is_supp else 'NO'):<12}")

        if is_supp:
            suppressed_count += 1
        if is_content:
            total_content_tokens += 1

    print("-"*100)
    print(f"Summary: {suppressed_count}/{total_content_tokens} content tokens suppressed "
          f"({100*suppressed_count/max(1,total_content_tokens):.1f}%)")
    if tau_method == "adaptive" and adaptive_tau is not None:
        ratio = adaptive_tau / tau if tau > 0 else 1.0
        print(f"Adaptive Threshold Effect: τ changed from {tau:.3f} → {adaptive_tau:.3f} "
              f"(×{ratio:.3f}, Δ = {adaptive_tau - tau:+.3f})")
    print("="*100 + "\n")


# =========================
# Harm vector
# =========================
@torch.no_grad()
def build_harm_vector(pipe, concepts: List[str]) -> Optional[torch.Tensor]:
    if not concepts:
        return None
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    device = pipe.device

    tokens = tokenizer(
        concepts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    outputs = text_encoder(**tokens, output_hidden_states=True, return_dict=True)
    hidden_states = outputs.hidden_states[-2]
    hidden_states = F.normalize(hidden_states, dim=-1)

    input_ids = tokens.input_ids
    attention_mask = tokens.attention_mask.bool()

    content_mask = attention_mask.clone()
    for sid in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id]:
        if sid is not None:
            content_mask = content_mask & (input_ids != sid)

    denom = content_mask.sum(dim=1, keepdim=True).clamp(min=1)
    vectors = (hidden_states * content_mask.unsqueeze(-1)).sum(dim=1) / denom
    harm_vector = F.normalize(vectors.mean(dim=0), dim=-1)
    return harm_vector


# =========================
# Probability-Redistribution AttnProcessor
# =========================
class ProbRedistributeAttnProcessor(AttnProcessor2_0):
    """
    - softmax 이후 확률(attn_probs)에서 억제/재분배 수행
    - 키별 억제율 w_k ∈ [0,1] (양의 criticality 비례, τ 및 하드마스크 반영)
    - 억제된 질량을 허용 키 집합으로 재분배 (proportional / uniform)
    - CFG cond-half만 적용 (B 짝수 가정 시 후반 절반)
    """

    def __init__(self, tau: float = 0.15, gamma: float = 20.0,
                 mode: str = "proportional", topk_min: int = 0, debug: bool = False):
        super().__init__()
        self.tau = float(tau)
        self.gamma = float(gamma)
        self.mode = mode
        self.topk_min = int(topk_min)
        self.debug = bool(debug)

        self.current_adaptive_tau: Optional[float] = None
        self._criticality_scores: Optional[torch.Tensor] = None
        self._content_mask: Optional[torch.Tensor] = None
        self._hard_block_mask: Optional[torch.Tensor] = None

        self.debug_step_count = 0
        self.debug_stats = []

    # setters
    def set_gamma(self, gamma: float):
        self.gamma = float(gamma)
        if self.debug and self.debug_step_count == 0:
            print(f"[DEBUG] set_gamma -> {self.gamma:.4f}")

    def set_adaptive_tau(self, adaptive_tau: Optional[float]):
        self.current_adaptive_tau = adaptive_tau

    def get_effective_tau(self) -> float:
        return self.current_adaptive_tau if self.current_adaptive_tau is not None else self.tau

    def set_criticality_scores(self, criticality_scores: torch.Tensor, content_mask: torch.Tensor):
        self._criticality_scores = criticality_scores.detach()
        self._content_mask = content_mask.detach()

    def set_hard_block_mask(self, hard_block_mask: Optional[torch.Tensor]):
        self._hard_block_mask = hard_block_mask.detach() if hard_block_mask is not None else None

    # core redistribution
    @staticmethod
    def _redistribute(attn_probs, suppr_weight, allow_mask, eps=1e-12, mode="proportional"):
        """
        attn_probs:  (B*H, Q, K)
        suppr_weight:(K,) in [0,1]
        allow_mask:  (K,) bool
        """
        BHK, Q, K = attn_probs.shape
        dev = attn_probs.device

        w = suppr_weight.to(dev).view(1, 1, K)    # (1,1,K)
        keep = 1.0 - w

        p = attn_probs
        p_keep = p * keep
        p_shed = p - p_keep
        shed_mass = p_shed.sum(dim=-1, keepdim=True)  # (B*H, Q, 1)

        allow = allow_mask.to(dev).view(1, 1, K).float()
        if mode == "uniform":
            target = allow / (allow.sum(dim=-1, keepdim=True) + eps)
        else:
            base = p_keep * allow
            base_sum = base.sum(dim=-1, keepdim=True)
            fallback = allow / (allow.sum(dim=-1, keepdim=True) + eps)
            target = torch.where(base_sum > eps, base / (base_sum + eps), fallback)

        p_new = p_keep + shed_mass * target
        p_new = p_new / (p_new.sum(dim=-1, keepdim=True) + eps)
        return p_new

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
    ) -> torch.Tensor:

        batch_size, sequence_length, _ = hidden_states.shape
        is_cross = encoder_hidden_states is not None

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        # Q,K,V
        query = attn.to_q(hidden_states)
        if is_cross:
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
        else:
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)

        query = attn.head_to_batch_dim(query)  # (B*H, Q, D)
        key   = attn.head_to_batch_dim(key)    # (B*H, K, D)
        value = attn.head_to_batch_dim(value)  # (B*H, K, D)

        scores = torch.matmul(query, key.transpose(-1, -2)) * attn.scale  # (B*H, Q, K)

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, key.shape[1], hidden_states.shape[0])
            scores = scores + attention_mask

        attn_probs = F.softmax(scores, dim=-1)  # (B*H, Q, K)

        # ------- Prob-Redistribution (cross-attn only) -------
        if is_cross and (self._criticality_scores is not None):
            dev = attn_probs.device
            BHK, Q, K = attn_probs.shape

            crit = self._criticality_scores.to(dev)            # (K,)
            c_mask = self._content_mask.to(dev) if self._content_mask is not None else torch.ones_like(crit, dtype=torch.bool)
            tau_eff = self.get_effective_tau()

            suppress = (crit >= tau_eff) & c_mask
            if self._hard_block_mask is not None:
                suppress = suppress | (self._hard_block_mask.to(dev) & c_mask)

            # 최소 topk 보장 (양의 criticality 중)
            if self.topk_min > 0:
                pos_idx = torch.nonzero((crit > 0) & c_mask, as_tuple=False).squeeze(-1)
                if pos_idx.numel() > 0:
                    top_vals, top_ids = torch.topk(crit[pos_idx], k=min(self.topk_min, pos_idx.numel()))
                    suppress[pos_idx[top_ids]] = True

            # 억제율 w_k ∈ [0,1] : crit_pos × gamma_norm
            crit_pos = torch.clamp(crit, min=0.0)
            gamma_norm = 1.0 / (1.0 + 0.01 * max(self.gamma, 0.0))
            w = torch.clamp(gamma_norm * crit_pos, 0.0, 1.0)
            w = w * suppress.float()

            allow_mask = c_mask & (~suppress)
            if not allow_mask.any():
                allow_mask = c_mask  # fallback

            # CFG cond-half에만 적용 (B 짝수일 때 뒤 절반 cond)
            # scores와 동일 차원 B*H 기준으로 나눠지므로, attn_probs에 동일 마스크 적용
            # 여기서는 확률 단계에서 cond-half만 억제/재분배 수행
            # 구현 간결성을 위해 전체에 적용 후, uncond-half에는 w=0이 되도록 마스크링
            B_est = encoder_hidden_states.shape[0]
            num_heads = (BHK // B_est) if B_est > 0 else 1
            if B_est % 2 == 0:
                cond_batch_mask = torch.zeros(B_est, device=dev)
                cond_batch_mask[B_est // 2:] = 1.0  # 뒤 절반 cond
                # (B,1,1)->(B*H,1,1)
                cond_bh_mask = cond_batch_mask.view(B_est, 1, 1).repeat_interleave(num_heads, dim=0)
            else:
                cond_bh_mask = torch.ones(BHK, 1, 1, device=dev)

            # 확률 재분배
            attn_probs_before = attn_probs.clone()
            attn_probs_cond = attn_probs * cond_bh_mask + attn_probs * (1 - cond_bh_mask)  # shape 유지
            # cond 부분만 실제 재분배 적용
            attn_probs_cond = self._redistribute(attn_probs_cond, w, allow_mask, mode=self.mode)
            attn_probs = attn_probs_cond * cond_bh_mask + attn_probs * (1 - cond_bh_mask)

            if self.debug:
                shed_mass = (attn_probs_before - attn_probs).clamp(min=0).sum(dim=-1).mean().item()
                suppr_rate = suppress.float().mean().item()
                self.debug_stats.append({
                    "step": int(self.debug_step_count),
                    "tau": float(tau_eff),
                    "gamma": float(self.gamma),
                    "suppr_rate_keys": float(suppr_rate),
                    "avg_shed_mass_per_query": float(shed_mass),
                    "K": int(K),
                })
                # cond-half에서 억제 키 확률 감소 검증 로그 (평균)
                supp_idx = torch.where(suppress)[0]
                if supp_idx.numel() > 0:
                    # cond bh slice
                    bh_mask = cond_bh_mask.squeeze(-1).squeeze(-1) > 0.5
                    pb = attn_probs_before[bh_mask][:, :, supp_idx].mean().item()
                    pa = attn_probs[bh_mask][:, :, supp_idx].mean().item()
                    print(f"[VERIF] mean attn_prob on suppressed keys (cond-only): {pb:.6f} -> {pa:.6f} (Δ={pa-pb:+.6f})")

        # dropout & output
        if isinstance(attn.dropout, nn.Dropout):
            attn_probs = attn.dropout(attn_probs)
        else:
            p = float(attn.dropout) if isinstance(attn.dropout, (int, float)) else 0.0
            attn_probs = F.dropout(attn_probs, p=p, training=False)

        hidden_states = torch.matmul(attn_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

    def print_step_debug(self, step: int):
        if self.debug and self.debug_stats:
            s = self.debug_stats[-1]
            print(f"  [PR Step {step:02d}] τ={s['tau']:.3f} | γ={s['gamma']:.1f} | "
                  f"keys_suppressed={100*s['suppr_rate_keys']:.1f}% | "
                  f"avg_shed_mass/query={s['avg_shed_mass_per_query']:.4f} | K={s['K']}")


# =========================
# Main Generation Loop
# =========================
def main():
    args = parse_args()
    accelerator = Accelerator()
    device = accelerator.device

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print(f"\n{'='*100}")
    print(f"[INFO] Machine Unlearning - ADAPTIVE THRESHOLD MODE (Probability Redistribution)")
    print(f"{'='*100}")
    print(f"[INFO] Loading model from {args.ckpt_path}")
    pipe = CustomStableDiffusionPipeline.from_pretrained(
        args.ckpt_path,
        safety_checker=None
    ).to(device)
    print(f"[INFO] Model loaded on device: {pipe.device}")

    # Load prompts
    prompt_file = os.path.expanduser(args.prompt_file)
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompts = [ln.strip() for ln in f if ln.strip()]
    print(f"[INFO] Loaded {len(prompts)} prompts from {prompt_file}")

    # Prepare output dir
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # =========================
    # Harm suppression setup
    # =========================
    harm_processor = None
    harm_concepts = []
    if args.harm_suppress:
        harm_file = os.path.expanduser(args.harm_concepts_file)
        if os.path.isfile(harm_file):
            with open(harm_file, "r", encoding="utf-8") as f:
                harm_concepts = [ln.strip() for ln in f if ln.strip()]
            print(f"\n[INFO] Building harmful concept vector from {len(harm_concepts)} concepts: {harm_concepts}")
            harm_vector = build_harm_vector(pipe, harm_concepts)

            harm_processor = ProbRedistributeAttnProcessor(
                tau=args.base_tau,
                gamma=args.harm_gamma_start,
                mode=args.redistribute_mode,
                topk_min=args.topk_min,
                debug=args.debug or args.debug_steps
            )
            pipe.unet.set_attn_processor(harm_processor)

            print(f"[INFO] Harmful concept suppression enabled (Probability Redistribution)")
            if args.adaptive_threshold:
                print(f"  - Threshold mode: ADAPTIVE (Multiplicative) ⭐")
                print(f"  - Base threshold (τ_base): {args.base_tau}")
                print(f"  - Central percentile: {args.central_percentile*100:.0f}%")
                print(f"  - Multiplicative factor: ×{args.tau_factor}")
                print(f"  - Formula: τ = central_mean × {args.tau_factor}")
            else:
                print(f"  - Threshold mode: FIXED (τ={args.base_tau})")
            print(f"  - Gamma schedule: {args.harm_gamma_start} → {args.harm_gamma_end}")

            # LOO 계산을 위해 harm_vector는 함수 인자로 넘길 것이므로 프로세서에는 저장하지 않음
        else:
            print(f"[WARNING] Harmful concepts file not found: {harm_file}")
            args.harm_suppress = False

    # =========================
    # Classifier Guidance
    # =========================
    guidance_model = None
    if args.classifier_guidance:
        print(f"\n[INFO] Loading classifier from {args.classifier_ckpt}")
        guidance_model = GuidanceModel(
            pipe,
            args.classifier_config,
            args.classifier_ckpt,
            1,
            device
        )
        print(f"[INFO] Classifier guidance enabled")
        print(f"  - Scale: {args.guidance_scale}")
        print(f"  - Target class: {args.target_class} (clothed people)")
        print(f"  - Start step: {args.guidance_start_step}")

    # =========================
    # Callback
    # =========================
    def callback_on_step_end(
        diffusion_pipeline,
        step,
        timestep,
        callback_kwargs,
        guidance_model,
        guidance_scale,
        guidance_start_step,
        target_class,
        harm_processor,
        num_steps,
    ):
        # Update gamma schedule
        if harm_processor is not None:
            step_c = min(step, num_steps - 1)
            gamma = schedule_linear(step_c, num_steps, args.harm_gamma_start, args.harm_gamma_end)
            harm_processor.set_gamma(gamma)
            harm_processor.debug_step_count = step
            if args.debug_steps:
                harm_processor.print_step_debug(step)

        # Classifier guidance
        if guidance_model is not None and step >= guidance_start_step:
            callback_kwargs = guidance_model.guidance(
                diffusion_pipeline,
                callback_kwargs,
                step,
                timestep,
                guidance_scale,
                target_class=target_class
            )
        return callback_kwargs

    print(f"\n{'='*100}")
    print(f"[INFO] Starting generation...")
    print(f"{'='*100}\n")

    for idx, prompt in enumerate(prompts):
        print(f"\n{'='*100}")
        print(f"[PROMPT {idx + 1}/{len(prompts)}] {prompt}")
        print(f"{'='*100}")

        adaptive_tau = None
        tau_method = "fixed"

        # LOO criticality & τ 계산
        if args.harm_suppress and harm_processor is not None:
            # 현재 프롬프트용 harm_vector 재계산(동일 고정 벡터라면 위에서 만든 것을 재사용)
            harm_file = os.path.expanduser(args.harm_concepts_file)
            with open(harm_file, "r", encoding="utf-8") as f:
                harm_concepts = [ln.strip() for ln in f if ln.strip()]
            harm_vector = build_harm_vector(pipe, harm_concepts)

            criticality_scores, content_mask = compute_loo_criticality_simple(
                pipe, prompt, harm_vector, method=args.loo_method
            )
            harm_processor.set_criticality_scores(criticality_scores, content_mask)

            content_crit = criticality_scores[content_mask]
            if args.adaptive_threshold and content_crit.numel() > 0:
                content_pos = content_crit[content_crit > 0]
                if content_pos.numel() > 0:
                    adaptive_tau = compute_adaptive_threshold_from_tensor(
                        content_pos, central_percentile=args.central_percentile,
                        factor=args.tau_factor, base_tau=args.base_tau
                    )
                else:
                    adaptive_tau = args.base_tau
                harm_processor.set_adaptive_tau(adaptive_tau)
                tau_method = "adaptive"
                # 디버그 요약
                num_neg = (content_crit < 0).sum().item()
                num_pos = (content_crit > 0).sum().item()
                num_zero = (content_crit == 0).sum().item()
                print(f"\n[ADAPTIVE THRESHOLD - POSITIVE CRITICALITY ONLY]")
                print(f"  LOO Method: {args.loo_method}")
                print(f"  Content tokens: {content_crit.numel()} total ({num_pos} positive, {num_neg} negative, {num_zero} zero)")
                print(f"  RAW criticality range (ALL): [{content_crit.min():.3f}, {content_crit.max():.3f}]")
                if content_pos.numel() > 0:
                    print(f"  RAW criticality range (POSITIVE ONLY): [{content_pos.min():.3f}, {content_pos.max():.3f}]")
                    print(f"  → Adaptive τ: {adaptive_tau:.3f}")
                else:
                    print(f"  → No positive criticality tokens → τ = base {adaptive_tau:.3f}")
            else:
                harm_processor.set_adaptive_tau(args.base_tau)
                adaptive_tau = args.base_tau
                if content_crit.numel() == 0:
                    print(f"\n[WARNING] No content tokens found, using base threshold: {args.base_tau:.3f}")

            # Prompt-level 디버그 테이블
            if args.debug or args.debug_prompts:
                debug_token_analysis(
                    pipe, prompt, harm_vector, harm_concepts,
                    args.base_tau, adaptive_tau=adaptive_tau, tau_method=tau_method, loo_method=args.loo_method
                )

        # Callback 설정
        if args.harm_suppress or args.classifier_guidance:
            callback = partial(
                callback_on_step_end,
                guidance_model=guidance_model if args.classifier_guidance else None,
                guidance_scale=args.guidance_scale,
                guidance_start_step=args.guidance_start_step,
                target_class=args.target_class,
                harm_processor=harm_processor,
                num_steps=args.num_inference_steps,
            )
            callback_tensor_inputs = ["latents", "noise_pred", "prev_latents"]
            if args.classifier_guidance:
                callback_tensor_inputs += ["instance_prompt_embeds"]
        else:
            callback = None
            callback_tensor_inputs = None

        # Reset debug & gamma for new prompt
        if harm_processor is not None:
            harm_processor.debug_stats = []
            harm_processor.debug_step_count = 0
            harm_processor.set_gamma(args.harm_gamma_start)

        # Generate
        print(f"\n[INFO] Generating {args.nsamples} image(s)...")
        ctx = torch.enable_grad() if args.classifier_guidance else torch.no_grad()
        with ctx:
            output = pipe(
                prompt=prompt,
                guidance_scale=args.cfg_scale,
                num_inference_steps=args.num_inference_steps,
                height=512,
                width=512,
                callback_on_step_end=callback,
                callback_on_step_end_tensor_inputs=callback_tensor_inputs,
                num_images_per_prompt=args.nsamples,
            )

        # Step statistics
        if args.debug and harm_processor is not None and harm_processor.debug_stats:
            print(f"\n[DEBUG] Generation Statistics:")
            stats = harm_processor.debug_stats
            avg_keys_suppr = np.mean([s['suppr_rate_keys'] for s in stats])
            avg_tau = np.mean([s['tau'] for s in stats])
            avg_shed = np.mean([s['avg_shed_mass_per_query'] for s in stats])
            print(f"  - keys_suppressed: {100*avg_keys_suppr:.1f}%")
            print(f"  - avg_shed_mass/query: {avg_shed:.4f}")
            print(f"  - average τ used: {avg_tau:.3f}")

        # Save images
        for sample_idx, image in enumerate(output.images):
            filename = f"prompt_{idx+1:04d}_sample_{sample_idx+1}.png"
            save_image(image, filename, root=output_dir)

        print(f"\n[INFO] Saved {len(output.images)} image(s) to {output_dir}")

    print(f"\n{'='*100}")
    print(f"[INFO] Generation complete! All images saved to {output_dir}")
    print(f"{'='*100}\n")


if __name__ == "__main__":
    main()

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
    parser = ArgumentParser(description="Generation script (global suppression + ADD-LIST boost + harm-triggered append/boost + freedom + SAE)")

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

    # Global harmful suppression (soft only; no hard block here)
    parser.add_argument("--harm_suppress", action="store_true",
                        help="Enable token-wise harmful concept suppression in Cross-Attention (global)")
    parser.add_argument("--harm_tau", type=float, default=0.10, help="Soft cosine threshold τ (>= triggers soft delete)")
    parser.add_argument("--harm_gamma_start", type=float, default=1.80, help="γ at early steps")
    parser.add_argument("--harm_gamma_end", type=float, default=0.30, help="γ at late steps")
    parser.add_argument("--harm_global_texts", type=str, default=None, help="File: harmful concepts (one per line)")

    # (하드블록/듀얼 관련 기존 플래그는 사용하지 않음; 유지 호환만)
    parser.add_argument("--harm_dual", action="store_true", help="(ignored) kept for compatibility")
    parser.add_argument("--harm_hard_tau", type=float, default=None, help="(ignored) kept for compatibility")

    # ====== 추가 인자 (레이어/벡터 구성) ======
    parser.add_argument("--harm_layer_index", type=int, default=-1,
                        help="text encoder hidden_states layer index (e.g., -1=last, -2, -3)")
    parser.add_argument("--harm_vec_mode", type=str, choices=["masked_mean", "token", "prompt_token"],
                        default="masked_mean", help="How to make harm vector")
    parser.add_argument("--harm_target_words", type=str, default=None,
                        help="Comma-separated target words for token/prompt_token modes, e.g. 'nude,naked'")
    parser.add_argument("--include_special_tokens", action="store_true",
                        help="Include SOT/EOT/PAD in averaging (default: exclude)")

    # ====== ADD-LIST(강조 리스트) ======
    parser.add_argument("--add_list_texts", type=str, default=None,
                        help="File: ADD-LIST (one positive/benign concept per line), e.g., 'fully clothed person'")
    parser.add_argument("--add_tau", type=float, default=0.20,
                        help="(기본) add-list와 유사한 토큰에 대한 부스팅 활성 임계값")
    parser.add_argument("--add_gamma_start", type=float, default=1.20,
                        help="ADD boost γ at early steps")
    parser.add_argument("--add_gamma_end", type=float, default=0.50,
                        help="ADD boost γ at late steps")
    parser.add_argument("--add_layer_index", type=int, default=None,
                        help="Optional text-encoder layer for ADD vectors (default: harm_layer_index)")
    parser.add_argument("--add_append", action="store_true",
                        help="add_tau 이상 매칭된 항목을 프롬프트 문자열에 append")
    parser.add_argument("--add_topk_append", type=int, default=3,
                        help="append 시 붙일 최대 개수")

    # ====== harm 기반 ADD 강제 주입/부스팅 ======
    parser.add_argument("--add_on_harm", action="store_true",
                        help="프롬프트/토큰이 harm과 유사하면 add를 강제 주입(부스팅 ON)")
    parser.add_argument("--add_on_harm_tau", type=float, default=0.20,
                        help="harm 유사도 기반 add 강제 활성 임계값")
    parser.add_argument("--add_append_on_harm", action="store_true",
                        help="프롬프트가 harm과 유사하면 add 항목을 문자열에도 강제 추가")
    parser.add_argument("--add_force_append_n", type=int, default=1,
                        help="harm 트리거 시 문자열에 강제 추가할 add 항목 개수(상위 N)")

    # 디버그
    parser.add_argument("--add_debug_print", action="store_true", help="Print harm/add diagnostics per prompt")

    # ====== EOT 전용 하드블록 토글 ======
    parser.add_argument("--eot_hard_block", action="store_true",
                        help="EOT(EOS) 토큰만 Cross-Attention에서 항상 하드 블록")

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


# ----- 헬퍼: 레이어 선택 & 마스크 생성 -----
def _pick_hidden(pipe, out, layer_idx: int):
    if hasattr(out, "hidden_states") and out.hidden_states is not None:
        return out.hidden_states[layer_idx]
    return out.last_hidden_state

def _build_content_mask(tokenizer, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                        include_special: bool) -> torch.Tensor:
    mask = attention_mask.bool()
    if not include_special:
        bos = getattr(tokenizer, "bos_token_id", None)
        eos = getattr(tokenizer, "eos_token_id", None)
        pad = getattr(tokenizer, "pad_token_id", None)
        for sid in [bos, eos, pad]:
            if sid is not None:
                mask = mask & (input_ids != sid)
    return mask

# SOT(=BOS) 소프트 억제/부스팅 제외 마스크 생성 (EOT는 건들지 않음)
@torch.no_grad()
def build_sot_soft_exempt_mask(pipe, prompt: str) -> torch.Tensor:
    tok = pipe.tokenizer(
        [prompt],
        padding="max_length",
        truncation=True,
        max_length=pipe.tokenizer.model_max_length,
        return_tensors="pt",
    ).to(pipe.device)
    ids = tok.input_ids[0]
    att = tok.attention_mask[0].bool()
    bos = getattr(pipe.tokenizer, "bos_token_id", None)
    m = torch.zeros_like(ids, dtype=torch.bool)
    if bos is not None:
        m = att & (ids == bos)
    else:
        first = int(att.nonzero(as_tuple=False)[0].item())
        m[first] = True
    return m.unsqueeze(0)  # (1,K)

# EOT(=EOS) 하드 블록 마스크 생성 (항상 block 대상)
@torch.no_grad()
def build_eot_hard_block_mask(pipe, prompt: str) -> torch.Tensor:
    tok = pipe.tokenizer(
        [prompt],
        padding="max_length",
        truncation=True,
        max_length=pipe.tokenizer.model_max_length,
        return_tensors="pt",
    ).to(pipe.device)
    ids = tok.input_ids[0]
    att = tok.attention_mask[0].bool()
    eos = getattr(pipe.tokenizer, "eos_token_id", None)

    m = torch.zeros_like(ids, dtype=torch.bool)
    if eos is not None:
        m = att & (ids == eos)   # 사용된 토큰 내 EOS만 True
    else:
        last = int(att.nonzero(as_tuple=False)[-1].item())  # 마지막 유효 토큰을 EOS로 취급
        m[last] = True
    return m.unsqueeze(0)  # (1,K)


@torch.no_grad()
def build_harm_vec_sd14(pipe,
                        concepts: List[str],
                        layer_idx: int = -1,
                        include_special: bool = False,
                        mode: str = "masked_mean",
                        target_words: Optional[List[str]] = None) -> Optional[torch.Tensor]:
    if not concepts:
        return None

    tok = pipe.tokenizer(
        concepts,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(pipe.device)
    out = pipe.text_encoder(**tok, output_hidden_states=True, return_dict=True)
    H = _pick_hidden(pipe, out, layer_idx)
    H = F.normalize(H, dim=-1)

    mask = _build_content_mask(pipe.tokenizer, tok.input_ids, tok.attention_mask, include_special)

    if mode == "masked_mean":
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
        v_per_sent = (H * mask.unsqueeze(-1)).sum(dim=1) / denom   # (N, d)
        v = F.normalize(v_per_sent.mean(dim=0), dim=-1)            # (d,)
        return v

    # token 모드
    words = target_words if (target_words and len(target_words) > 0) else concepts
    words = [w.lower() for w in words]

    selected = []
    id_lists = tok.input_ids
    for b in range(id_lists.shape[0]):
        tokens = pipe.tokenizer.convert_ids_to_tokens(id_lists[b].tolist())
        for i, t in enumerate(tokens):
            if not mask[b, i]:
                continue
            s = t.replace("Ġ", "").lower()
            if any(w in s for w in words):
                selected.append(H[b, i])

    if len(selected) == 0:
        return None
    v = F.normalize(torch.stack(selected, dim=0).mean(dim=0), dim=-1)
    return v


@torch.no_grad()
def build_vecs_sd14(pipe, texts: List[str],
                    layer_idx: int = -1,
                    include_special: bool = False) -> Optional[torch.Tensor]:
    if not texts:
        return None
    tok = pipe.tokenizer(
        texts,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(pipe.device)
    out = pipe.text_encoder(**tok, output_hidden_states=True, return_dict=True)
    H = _pick_hidden(pipe, out, layer_idx)
    H = F.normalize(H, dim=-1)
    mask = _build_content_mask(pipe.tokenizer, tok.input_ids, tok.attention_mask, include_special)

    denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
    v = (H * mask.unsqueeze(-1)).sum(dim=1) / denom
    v = F.normalize(v, dim=-1)
    return v


@torch.no_grad()
def build_prompt_token_vec(pipe, prompt: str, target_words: List[str],
                           layer_idx: int = -1, include_special: bool = False) -> Optional[torch.Tensor]:
    tok = pipe.tokenizer(
        [prompt],
        padding="max_length",
        truncation=True,
        max_length=pipe.tokenizer.model_max_length,
        return_tensors="pt",
    ).to(pipe.device)
    out = pipe.text_encoder(**tok, output_hidden_states=True, return_dict=True)
    H = _pick_hidden(pipe, out, layer_idx)[0]
    H = F.normalize(H, dim=-1)

    mask = _build_content_mask(pipe.tokenizer, tok.input_ids[0], tok.attention_mask[0], include_special)
    tokens = pipe.tokenizer.convert_ids_to_tokens(tok.input_ids[0].tolist())
    words = [w.lower() for w in (target_words or [])]

    selected = []
    for i, t in enumerate(tokens):
        if not mask[i]:
            continue
        s = t.replace("Ġ", "").lower()
        if any(w in s for w in words):
            selected.append(H[i])

    if len(selected) == 0:
        return None
    v = F.normalize(torch.stack(selected, dim=0).mean(dim=0), dim=-1)
    return v


@torch.no_grad()
def build_prompt_vec(pipe, prompt: str,
                     layer_idx: int = -1, include_special: bool = False) -> torch.Tensor:
    v = build_vecs_sd14(pipe, [prompt], layer_idx=layer_idx, include_special=include_special)
    return v[0] if v is not None else None


@torch.no_grad()
def debug_print_harm_add(
    pipe,
    prompt: str,
    harm_vec: Optional[torch.Tensor],
    add_mat: Optional[torch.Tensor],
    add_labels: List[str],
    tau_harm: float,
    tau_add: float,
    layer_idx: int = -1,
    include_special: bool = False,
):
    tok = pipe.tokenizer(
        [prompt],
        padding="max_length",
        truncation=True,
        max_length=pipe.tokenizer.model_max_length,
        return_tensors="pt",
    ).to(pipe.device)

    out = pipe.text_encoder(**tok, output_hidden_states=True, return_dict=True)
    H = _pick_hidden(pipe, out, layer_idx)[0]   # (L, d)
    H = F.normalize(H, dim=-1)
    dev = H.device

    ids = tok.input_ids[0]
    raw_mask = tok.attention_mask[0].bool()
    mask = _build_content_mask(pipe.tokenizer, ids, tok.attention_mask[0], include_special)
    toks = pipe.tokenizer.convert_ids_to_tokens(ids.tolist())

    hv = F.normalize(harm_vec, dim=-1).to(dev) if (harm_vec is not None and harm_vec.numel() > 0) else None
    AM = add_mat.to(dev) if (add_mat is not None and add_mat.numel() > 0) else None

    print(f"\n[HARM/ADD DEBUG] prompt={prompt!r}, harm_tau={tau_harm:.3f}, add_tau={tau_add:.3f}, layer={layer_idx}, include_special={include_special}")
    print(f"{'idx':>3} {'token':>18} {'is_used':>7} {'harm_cos':>9} {'add_max':>9} {'best_add':>16} {'suppress?':>10} {'boost?':>7}")
    print("-" * 96)

    for i, use in enumerate(raw_mask.tolist()):
        if not use:
            break
        token = toks[i].replace("Ġ", "▁")

        harm_cos = float(torch.dot(H[i], hv).item()) if hv is not None else float("nan")

        add_max, best = 0.0, ""
        if AM is not None:
            sims = torch.matmul(H[i], AM.T)  # (M,)
            mval, midx = sims.max(dim=0)
            add_max = float(mval.item())
            best = add_labels[int(midx.item())] if len(add_labels) > 0 else ""

        suppress = (harm_cos >= tau_harm)
        boost = (add_max >= tau_add)

        is_used = bool(mask[i].item())
        print(f"{i:>3} {token:>18} {str(is_used):>7} {harm_cos:+.3f} {add_max:+.3f} {best:>16} {str(suppress):>10} {str(boost):>7}")


def schedule_linear(step: int, num_steps: int, a0: float, a1: float) -> float:
    t = step / max(1, num_steps - 1)
    return a0 * (1.0 - t) + a1 * t


# =========================
# Attn Processor (Harm + Add + EOT hard block)
# =========================
class HarmCfg:
    __slots__ = ("enable", "tau", "gamma", "hard_block", "dual", "tau_hard")

    def __init__(
        self,
        enable=True,
        tau: float = 0.1,
        gamma: float = 1.0,
        hard_block: bool = False,
        dual: bool = False,
        tau_hard: Optional[float] = None,
    ):
        self.enable = bool(enable)
        self.tau = float(tau)
        self.gamma = float(gamma)
        self.hard_block = bool(hard_block)
        self.dual = bool(dual)
        self.tau_hard = float(tau_hard) if (tau_hard is not None) else None


class AddCfg:
    __slots__ = ("enable", "tau", "gamma", "gate_on_harm_tau")

    def __init__(self, enable=True, tau: float = 0.2, gamma: float = 1.0,
                 gate_on_harm_tau: Optional[float] = None):
        self.enable = bool(enable)
        self.tau = float(tau)
        self.gamma = float(gamma)
        self.gate_on_harm_tau = float(gate_on_harm_tau) if gate_on_harm_tau is not None else None


class HarmAddAttnProcessor(AttnProcessor2_0):
    """
    Cross-Attention 점수(QK^T/√d)에
      - Harmful soft delete (hard/dual은 비활성)
      - ADD-LIST positive boost (+ harm 기반 강제 부스팅)
      - EOT(EOS) 토큰은 옵션에 따라 항상 hard block
    를 함께 주입합니다.
    """
    def __init__(
        self,
        harm_vec: Optional[torch.Tensor],
        add_mat: Optional[torch.Tensor],
        harm_cfg: HarmCfg,
        add_cfg: AddCfg,
        eot_hard_block: bool = False,
    ):
        super().__init__()
        self.harm_cfg = harm_cfg
        self.add_cfg = add_cfg
        self.training = False

        # CPU 보관(정규화)
        if harm_vec is None or harm_vec.numel() == 0:
            self._harm_vec = None
        else:
            self._harm_vec = F.normalize(harm_vec.detach().float(), dim=-1).cpu()

        if add_mat is None or add_mat.numel() == 0:
            self._add_mat = None
        else:
            self._add_mat = F.normalize(add_mat.detach().float(), dim=-1).cpu()

        # soft 제외 마스크(SOT 등)
        self._soft_exempt_mask = None  # (B,K) 또는 (1,K), True=soft 제외
        # EOT 하드 블록 마스크
        self._eot_block_mask = None    # (B,K) 또는 (1,K), True=hard block
        self._eot_hard_block_enabled = bool(eot_hard_block)

    # props
    @property
    def harm_vec(self) -> Optional[torch.Tensor]:
        return self._harm_vec

    @property
    def add_mat(self) -> Optional[torch.Tensor]:
        return self._add_mat

    # setters
    def set_harm_vec(self, harm_vec: Optional[torch.Tensor]):
        if harm_vec is None or harm_vec.numel() == 0:
            self._harm_vec = None
        else:
            self._harm_vec = F.normalize(harm_vec.detach().float(), dim=-1).cpu()

    def set_add_mat(self, add_mat: Optional[torch.Tensor]):
        if add_mat is None or add_mat.numel() == 0:
            self._add_mat = None
        else:
            self._add_mat = F.normalize(add_mat.detach().float(), dim=-1).cpu()

    def set_harm_gamma(self, gamma: float):
        self.harm_cfg.gamma = float(gamma)

    def set_add_gamma(self, gamma: float):
        self.add_cfg.gamma = float(gamma)

    def set_soft_exempt_mask(self, mask: Optional[torch.Tensor]):
        self._soft_exempt_mask = None if mask is None else mask.bool().detach().cpu()

    def set_eot_block_mask(self, mask: Optional[torch.Tensor]):
        self._eot_block_mask = None if mask is None else mask.bool().detach().cpu()

    # core
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
    ) -> torch.Tensor:
        dev = hidden_states.device
        harm_vec = self._harm_vec.to(dev) if self._harm_vec is not None else None
        add_mat = self._add_mat.to(dev) if self._add_mat is not None else None

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

        if is_cross:
            B = encoder_hidden_states.shape[0]
            Q = scores.shape[1]
            K = scores.shape[2]
            Hh = scores.shape[0] // B  # num heads

            # normalized context (B,K,d)
            ctx_n = F.normalize(encoder_hidden_states, dim=-1)

            # ===== EOT(EOS) 항상 하드 블록 (옵션 켜진 경우) =====
            if self._eot_hard_block_enabled and (self._eot_block_mask is not None):
                eot_mask = self._eot_block_mask.to(scores.device)
                if eot_mask.shape[0] == 1:
                    eot_mask = eot_mask.expand(B, -1)  # (B,K)
                hard_exp_eot = eot_mask[:, None, :].expand(B, Q, K).repeat_interleave(Hh, dim=0)
                scores = scores.masked_fill(hard_exp_eot, -1e9)

            # ===== Harmful soft delete (hard/dual은 비활성) =====
            cos_harm = None
            if self.harm_cfg.enable and (harm_vec is not None):
                harm = F.normalize(harm_vec, dim=-1)  # (d,)
                cos_harm = torch.einsum("bkd,d->bk", ctx_n, harm)  # (B,K)

                # soft delete
                cond_soft = (cos_harm >= self.harm_cfg.tau)

                # SOT soft 제외
                if self._soft_exempt_mask is not None:
                    exc = self._soft_exempt_mask.to(cond_soft.device)
                    if exc.shape[0] == 1:
                        exc = exc.expand(B, -1)
                    cond_soft = cond_soft & (~exc)

                if cond_soft.any():
                    weight = cos_harm.clamp(min=0.0) * cond_soft.float()  # (B,K)
                    soft_w = weight[:, None, :].expand(B, Q, K).repeat_interleave(Hh, dim=0)
                    if soft_w.any():
                        scores = scores - (soft_w * self.harm_cfg.gamma)

            # ===== ADD-LIST positive boost =====
            if self.add_cfg.enable and (add_mat is not None) and add_mat.numel() > 0:
                sims = torch.einsum("bkd,md->bkm", ctx_n, add_mat)  # (B,K,M)
                add_max = sims.amax(dim=-1)                         # (B,K)

                cond_add = (add_max >= self.add_cfg.tau)

                # harm 기반 강제 게이트
                if (self.add_cfg.gate_on_harm_tau is not None) and (cos_harm is not None):
                    cond_add = cond_add | (cos_harm >= self.add_cfg.gate_on_harm_tau)

                # SOT 제외
                if self._soft_exempt_mask is not None:
                    exc = self._soft_exempt_mask.to(cond_add.device)
                    if exc.shape[0] == 1:
                        exc = exc.expand(B, -1)
                    cond_add = cond_add & (~exc)

                # soft delete로 억제된 토큰은 add 제외
                if 'cond_soft' in locals() and (cond_soft is not None):
                    cond_add = cond_add & (~cond_soft)

                if cond_add.any():
                    add_w = (add_max * cond_add.float())[:, None, :].expand(B, Q, K).repeat_interleave(Hh, dim=0)
                    scores = scores + (add_w * self.add_cfg.gamma)

        # (E) mask + softmax
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            scores = scores + attention_mask

        attn_probs = F.softmax(scores, dim=-1)

        # (F) dropout
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
    # Harm / Add 준비
    # =========================
    harm_proc = None
    global_concepts = []
    add_labels, add_mat_all = [], None

    # --- Harm 준비 ---
    if args.harm_suppress:
        harm_file = os.path.expanduser(args.harm_global_texts) if args.harm_global_texts else None
        if harm_file and os.path.isfile(harm_file):
            with open(harm_file, "r") as f:
                global_concepts = [l.strip() for l in f if l.strip()]
        if len(global_concepts) == 0:
            print("[harm] Global suppression requested but no concepts provided. Disabling suppression.")
            args.harm_suppress = False

    # --- ADD-LIST 준비 ---
    if args.add_list_texts:
        add_file = os.path.expanduser(args.add_list_texts)
        if os.path.isfile(add_file):
            with open(add_file, "r") as f:
                add_labels = [l.strip() for l in f if l.strip()]
            add_layer = args.add_layer_index if (args.add_layer_index is not None) else args.harm_layer_index
            add_mat_all = build_vecs_sd14(
                pipe, add_labels,
                layer_idx=add_layer,
                include_special=args.include_special_tokens,
            )

    # --- Harm/ADD Processor 인스턴스 ---
    harm_vec_base = None
    if args.harm_suppress and len(global_concepts) > 0:
        target_words = None
        if args.harm_target_words:
            target_words = [w.strip() for w in args.harm_target_words.split(",") if w.strip()]

        if args.harm_vec_mode != "prompt_token":
            harm_vec_base = build_harm_vec_sd14(
                pipe, global_concepts,
                layer_idx=args.harm_layer_index,
                include_special=args.include_special_tokens,
                mode=args.harm_vec_mode,
                target_words=target_words
            )
        else:
            harm_vec_base = build_harm_vec_sd14(
                pipe, global_concepts,
                layer_idx=args.harm_layer_index,
                include_special=args.include_special_tokens,
                mode="masked_mean",
                target_words=None
            )

    # 다른 하드블록은 전부 제거합니다.
    harm_cfg = HarmCfg(
        enable=bool(args.harm_suppress),
        tau=args.harm_tau,
        gamma=args.harm_gamma_start,
        hard_block=False,   # 무조건 False
        dual=False,         # 무조건 False
        tau_hard=None,      # 미사용
    )

    add_cfg = AddCfg(
        enable=bool(add_mat_all is not None and add_mat_all.numel() > 0),
        tau=args.add_tau,
        gamma=args.add_gamma_start,
        gate_on_harm_tau=(args.add_on_harm_tau if args.add_on_harm else None),
    )

    harm_proc = HarmAddAttnProcessor(
        harm_vec=harm_vec_base,
        add_mat=None,  # per-prompt로 선택된 ADD만 주입
        harm_cfg=harm_cfg,
        add_cfg=add_cfg,
        eot_hard_block=args.eot_hard_block,  # EOT만 하드블록
    )
    pipe.unet.set_attn_processor(harm_proc)
    print(f"[init] harm_suppress={args.harm_suppress} (soft-only, tau={args.harm_tau}), "
          f"ADD-list={len(add_labels)} items (add_tau={args.add_tau}) "
          f"(layer harm={args.harm_layer_index}, add={args.add_layer_index if args.add_layer_index is not None else args.harm_layer_index}, include_special={args.include_special_tokens}), "
          f"EOT hard block={args.eot_hard_block}")

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

    freedom_scale_early = args.freedom_scale if (args.freedom_scale_early is None) else args.freedom_scale_early
    add_layer_for_vec = args.add_layer_index if (args.add_layer_index is not None) else args.harm_layer_index

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
        # update γ schedule
        harm_gamma = schedule_linear(step, num_steps, args.harm_gamma_start, args.harm_gamma_end)
        harm_proc.set_harm_gamma(harm_gamma)

        add_gamma = schedule_linear(step, num_steps, args.add_gamma_start, args.add_gamma_end)
        harm_proc.set_add_gamma(add_gamma)

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
    for idx, prompt_raw in enumerate(prompts):
        prompt = prompt_raw

        # prompt-aware harm (mode=prompt_token): 프롬프트별 harm 벡터 갱신
        if args.harm_suppress and (args.harm_vec_mode == "prompt_token") and (harm_proc is not None):
            words = None
            if args.harm_target_words:
                words = [w.strip() for w in args.harm_target_words.split(",") if w.strip()]
            elif len(global_concepts) > 0:
                words = global_concepts

            prompt_harm_vec = build_prompt_token_vec(
                pipe, prompt, words or [],
                layer_idx=args.harm_layer_index,
                include_special=args.include_special_tokens
            )
            if prompt_harm_vec is not None:
                harm_proc.set_harm_vec(prompt_harm_vec)
            else:
                harm_proc.set_harm_vec(harm_vec_base)

        # ---- harm 기반 프롬프트 강제 append (선택) ----
        harm_triggered = False
        if args.add_on_harm and (harm_proc.harm_vec is not None):
            pvec_h = build_prompt_vec(
                pipe, prompt, layer_idx=args.harm_layer_index, include_special=args.include_special_tokens
            )
            if pvec_h is not None:
                pvec_h = F.normalize(pvec_h, dim=-1)
                hvec = F.normalize(harm_proc.harm_vec.to(pvec_h.device), dim=-1)
                cos_h_prompt = float(torch.dot(pvec_h, hvec).item())
                harm_triggered = (cos_h_prompt >= args.add_on_harm_tau)

        # per-prompt ADD selection + optional append
        selected_add_mat = None
        selected_labels = []
        sims = None

        if (add_mat_all is not None) and (add_mat_all.numel() > 0):
            # 기본: add_tau 매칭
            pvec = build_prompt_vec(pipe, prompt, layer_idx=add_layer_for_vec, include_special=args.include_special_tokens)
            sel_idx = []
            if pvec is not None:
                pvec = F.normalize(pvec, dim=-1)
                sims = torch.matmul(add_mat_all, pvec)  # (M,)
                sel_idx = (sims >= args.add_tau).nonzero(as_tuple=False).flatten().tolist()

            # harm 트리거 시 강제 주입/append
            if harm_triggered:
                if sims is None:
                    order = list(range(len(add_labels)))
                else:
                    order = sorted(range(len(add_labels)), key=lambda i: -float(sims[i].item()))
                # "fully clothed" 우선 배치
                prefer = [i for i, l in enumerate(add_labels) if ("fully" in l.lower() and "clothed" in l.lower())]
                order = prefer + [i for i in order if i not in prefer]
                sel_idx = order[:max(1, int(args.add_topk_append))]

                if args.add_append_on_harm:
                    already = set([t.strip().lower() for t in prompt.split(",")])
                    to_add = [add_labels[i] for i in order[:max(1, int(args.add_force_append_n))] if add_labels[i].lower() not in already]
                    if len(to_add) > 0:
                        prompt = prompt + ", " + ", ".join(to_add)

            # harm이 강했는데 sel이 비어있다면 전체 add를 주입
            if selected_add_mat is None and harm_triggered:
                selected_add_mat = add_mat_all

            # add_tau 기반 append (옵션)
            if (not harm_triggered) and args.add_append and sims is not None and len(sel_idx) > 0:
                top_sorted = sorted([(i, float(sims[i].item())) for i in sel_idx], key=lambda x: -x[1])
                top_sorted = top_sorted[:max(1, int(args.add_topk_append))]
                to_add = [add_labels[i] for i, _ in top_sorted]
                already = set([t.strip().lower() for t in prompt.split(",")])
                add_strs = [t for t in to_add if t.lower() not in already]
                if len(add_strs) > 0:
                    prompt = prompt + ", " + ", ".join(add_strs)

            # Processor에 add 대상 주입
            harm_proc.set_add_mat(selected_add_mat)

        # SOT soft 제외 마스크
        sot_mask = build_sot_soft_exempt_mask(pipe, prompt)
        harm_proc.set_soft_exempt_mask(sot_mask)

        # EOT hard block 마스크 (옵션 플래그에 의해 적용)
        eot_mask = build_eot_hard_block_mask(pipe, prompt)
        harm_proc.set_eot_block_mask(eot_mask)

        # 디버그
        if args.add_debug_print:
            debug_print_harm_add(
                pipe=pipe,
                prompt=prompt,
                harm_vec=harm_proc.harm_vec if harm_proc is not None else None,
                add_mat=harm_proc.add_mat if harm_proc is not None else None,
                add_labels=selected_labels,
                tau_harm=args.harm_tau,
                tau_add=args.add_tau,
                layer_idx=args.harm_layer_index,
                include_special=args.include_special_tokens,
            )

        print(f"\n=== Generating image for prompt {idx + 1}: {prompt}")

        # callback
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

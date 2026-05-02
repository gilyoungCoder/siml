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
import numpy as np
from typing import List, Optional

from accelerate import Accelerator
from diffusers import StableDiffusionPipeline
from diffusers.models.attention_processor import AttnProcessor2_0, Attention


# =========================
# Args
# =========================
def parse_args():
    parser = ArgumentParser(description="Cross-Attention ghost-context injection (harm suppression + safe context)")

    # Positional
    parser.add_argument("ckpt_path", type=str)

    # parse_args() 내부에 추가
    parser.add_argument("--anti_use_eot", action="store_true",
                        help="Use only the EOT token hidden for anti-harm vector (single-line anti).")

    # Generation
    parser.add_argument("--prompt_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output_img/tmp")
    parser.add_argument("--nsamples", type=int, default=4)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)

    # ===== Harmful suppression + Safe (anti-harm) context injection =====
    parser.add_argument("--enable_harm", action="store_true",
                        help="Enable token-wise harmful suppression + safe ghost injection in Cross-Attention")
    parser.add_argument("--harm_texts", type=str, default=None,
                        help="File: harmful concept texts (one per line). e.g., nudity synonyms")
    parser.add_argument("--anti_texts", type=str, default=None,
                        help="File: anti-harm concept texts (one per line). e.g., clothed / modest apparel ...")

    # cosine-based selection & strength
    parser.add_argument("--harm_tau", type=float, default=0.10, help="cosine threshold τ for harmful token detection")
    parser.add_argument("--lambda_start", type=float, default=0.40, help="decrease ratio at early steps")
    parser.add_argument("--lambda_end", type=float, default=0.15, help="decrease ratio at late steps")
    parser.add_argument("--ghost_boost_mu", type=float, default=0.0,
                        help="optional additional bias term to ghost column after redistribution (0~0.3 권장)")

    # Text-encoder hidden selection for concept vectors
    parser.add_argument("--layer_index", type=int, default=-1, help="text encoder hidden_states layer index")
    parser.add_argument("--include_special_tokens", action="store_true",
                        help="Include SOT/EOT/PAD in averaging (default: exclude)")

    # Harm vector building mode
    parser.add_argument("--harm_vec_mode", type=str, choices=["masked_mean", "token", "prompt_token"],
                        default="masked_mean",
                        help="How to build harm vector from harm_texts/prompt. 'prompt_token' uses the running prompt.")
    parser.add_argument("--harm_target_words", type=str, default=None,
                        help="Comma-separated words for 'token' or 'prompt_token' modes (e.g. 'nude,naked')")

    # Debug (token-level)
    parser.add_argument("--debug_cos", action="store_true", help="Print per-token cos and decisions")

    # ===== Runtime Attention Debug (NEW) =====
    parser.add_argument("--debug_attn", action="store_true",
                        help="Print step-wise cross-attention redistribution stats table")
    parser.add_argument("--debug_interval", type=int, default=5,
                        help="Print every N steps when --debug_attn is enabled")
    parser.add_argument("--debug_topk", type=int, default=8,
                        help="Top-k harmful columns to list in step table (sample0, head0)")
    parser.add_argument("--debug_precision", type=int, default=4,
                        help="Float precision for debug table")

    args = parser.parse_known_args()[0]
    return args


# =========================
# Utils
# =========================
def save_image(image, path: str, img_height: int, img_width: int):
    image = np.asarray(image)
    image = Image.fromarray(image, mode="RGB")
    image = image.resize((img_width, img_height))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image.save(path)


def _pick_hidden(out, layer_idx: int):
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


@torch.no_grad()
def build_vecs_sd14(pipe,
                    texts: List[str],
                    layer_idx: int = -1,
                    include_special: bool = False) -> Optional[torch.Tensor]:
    if not texts:
        return None
    tok = pipe.tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=pipe.tokenizer.model_max_length,
        return_tensors="pt",
    ).to(pipe.device)
    out = pipe.text_encoder(**tok, output_hidden_states=True, return_dict=True)
    H = _pick_hidden(out, layer_idx)                 # (M, L, d)
    H = F.normalize(H, dim=-1)
    mask = _build_content_mask(pipe.tokenizer, tok.input_ids, tok.attention_mask, include_special)

    denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
    v = (H * mask.unsqueeze(-1)).sum(dim=1) / denom  # (M, d)
    v = F.normalize(v, dim=-1)
    return v


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
    H = _pick_hidden(out, layer_idx)       # (N, L, d)
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
    H = _pick_hidden(out, layer_idx)[0]          # (L, d)
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
def build_anti_eot_vec_sd14(pipe,
                            text: str,
                            layer_idx: int = -1) -> Optional[torch.Tensor]:
    """
    한 줄 anti 텍스트의 EOT(=eos_token_id) 토큰 hidden만 집어서 anti 벡터로 사용.
    """
    tok = pipe.tokenizer(
        [text],
        padding="max_length",
        truncation=True,
        max_length=pipe.tokenizer.model_max_length,
        return_tensors="pt",
    ).to(pipe.device)

    out = pipe.text_encoder(**tok, output_hidden_states=True, return_dict=True)
    H = _pick_hidden(out, layer_idx)[0]  # (L, d)
    H = F.normalize(H, dim=-1)

    ids = tok.input_ids[0]               # (L,)
    att = tok.attention_mask[0].bool()   # (L,)
    eos = getattr(pipe.tokenizer, "eos_token_id", None)

    # EOT 인덱스 찾기
    if eos is not None and (ids == eos).any():
        idx = int((ids == eos).nonzero(as_tuple=False)[0].item())
    else:
        # eos 토큰 아이디가 없거나 못 찾은 경우: attention_mask 상의 마지막 유효 토큰을 사용
        last = int(att.nonzero(as_tuple=False)[-1].item())
        idx = last

    v = H[idx]                    # EOT 위치 hidden
    v = F.normalize(v, dim=-1)    # (d,)
    return v


@torch.no_grad()
def build_sot_soft_exempt_mask(pipe, prompt: str) -> torch.Tensor:
    tok = pipe.tokenizer(
        [prompt],
        padding="max_length",
        truncation=True,
        max_length=pipe.tokenizer.model_max_length,
        return_tensors="pt",
    ).to(pipe.device)
    ids = tok.input_ids[0]                 # (K,)
    att = tok.attention_mask[0].bool()     # (K,)
    bos = getattr(pipe.tokenizer, "bos_token_id", None)
    m = torch.zeros_like(ids, dtype=torch.bool)
    if bos is not None:
        m = att & (ids == bos)
    else:
        first = int(att.nonzero(as_tuple=False)[0].item())
        m[first] = True
    return m.unsqueeze(0)  # (1,K)


def gamma_schedule_linear(step: int, num_steps: int, g_start: float, g_end: float) -> float:
    t = step / max(1, num_steps - 1)
    return g_start * (1.0 - t) + g_end * t


# =========================
# Ghost-Context Attn Processor
# =========================
class GhostCfg:
    __slots__ = ("enable", "tau", "lam", "mu")

    def __init__(self, enable=True, tau: float = 0.1, lam: float = 0.4, mu: float = 0.0):
        self.enable = bool(enable)
        self.tau = float(tau)   # harmful 판정 임계값
        self.lam = float(lam)   # 재분배 비율
        self.mu = float(mu)     # 선택적 ghost bias


class GhostContextAttnProcessor(AttnProcessor2_0):
    """
    softmax 이후 확률 재분배:
      - harmful 열에서 lam * g 만큼 감산 (g = relu(cos_harm - tau), 행별 정규화)
      - 감산 합 m을 ghost 열에 추가 (anti-harm 방향에서 생성한 v_g 사용)
    + 디버그 옵션: step 간격으로 통계 테이블 출력
    """
    def __init__(self,
                 harm_vec: Optional[torch.Tensor],
                 anti_vec: Optional[torch.Tensor],
                 cfg: GhostCfg):
        super().__init__()
        self.cfg = cfg
        self.training = False

        self._harm = F.normalize(harm_vec.detach().float(), dim=-1).cpu() if (harm_vec is not None) else None
        self._anti = F.normalize(anti_vec.detach().float(), dim=-1).cpu() if (anti_vec is not None) else None

        self._soft_exempt_mask = None  # (B,K) or (1,K), True면 감산 제외

        # --- Debug runtime state ---
        self._dbg_on = False
        self._dbg_interval = 5
        self._dbg_topk = 8
        self._dbg_prec = 4
        self._step = -1
        self._printed_this_step = True
        self._current_lam = None  # 보기 좋게 표시용

    # === public setters ===
    def set_lambda(self, lam: float):
        self.cfg.lam = float(lam)
        self._current_lam = float(lam)

    def set_soft_exempt_mask(self, mask: Optional[torch.Tensor]):
        self._soft_exempt_mask = mask.bool().detach().cpu() if mask is not None else None

    def set_harm_vec(self, harm_vec: Optional[torch.Tensor]):
        self._harm = F.normalize(harm_vec.detach().float(), dim=-1).cpu() if (harm_vec is not None) else None

    def configure_debug(self, enable: bool, interval: int = 5, topk: int = 8, precision: int = 4):
        self._dbg_on = bool(enable)
        self._dbg_interval = max(1, int(interval))
        self._dbg_topk = max(1, int(topk))
        self._dbg_prec = max(2, int(precision))

    def start_step(self, step: int):
        self._step = int(step)
        self._printed_this_step = False  # 이 step에서 아직 출력 안 함

    # === main call ===
    def __call__(self,
                 attn: Attention,
                 hidden_states: torch.FloatTensor,
                 encoder_hidden_states: Optional[torch.FloatTensor] = None,
                 attention_mask: Optional[torch.FloatTensor] = None,
                 temb: Optional[torch.FloatTensor] = None,
                 scale: float = 1.0) -> torch.FloatTensor:

        dev = hidden_states.device
        is_cross = encoder_hidden_states is not None

        # (A) pre
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        # (B) Q,K,V
        query = attn.to_q(hidden_states)
        if is_cross:
            key   = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
        else:
            key   = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)

        # reshape to heads
        query = attn.head_to_batch_dim(query)  # (B*H, Q, d_h)
        key   = attn.head_to_batch_dim(key)    # (B*H, K, d_h)
        value = attn.head_to_batch_dim(value)  # (B*H, K, d_h)
        scores = torch.matmul(query, key.transpose(-1, -2)) * attn.scale  # (B*H, Q, K)

        # add attn mask before softmax
        if attention_mask is not None:
            batch_size, sequence_length, _ = hidden_states.shape
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            scores = scores + attention_mask

        # (C) softmax
        attn_probs = F.softmax(scores, dim=-1)  # (B*H, Q, K)

        # (D) Ghost redistribution only for cross-attention
        if is_cross and self.cfg.enable and (self._harm is not None) and (self._anti is not None):
            # 1) dtype/device match
            txt = F.normalize(encoder_hidden_states, dim=-1)
            txt_dtype = txt.dtype
            harm = self._harm.to(device=dev, dtype=txt_dtype)
            anti = self._anti.to(device=dev, dtype=txt_dtype)

            # 2) cosine + masks
            cos_harm = torch.einsum("bkd,d->bk", txt, harm)  # (B,K)
            B, K = cos_harm.shape
            Hh = scores.shape[0] // B
            Q  = scores.shape[1]
            harmful = (cos_harm >= torch.tensor(self.cfg.tau, device=dev, dtype=txt_dtype))

            if self._soft_exempt_mask is not None:
                exc = self._soft_exempt_mask.to(dev)
                if exc.shape[0] == 1:
                    exc = exc.expand(B, -1)
                harmful = harmful & (~exc)

            # 3) redistribute
            printed = False
            if harmful.any():
                harmful_exp = harmful[:, None, :].expand(B, Q, K).repeat_interleave(Hh, dim=0)

                # g (row-wise normalized)
                g_raw = (cos_harm - torch.tensor(self.cfg.tau, device=dev, dtype=txt_dtype)).clamp(min=0)
                g = g_raw / (g_raw.amax(dim=1, keepdim=True) + torch.tensor(1e-6, device=dev, dtype=txt_dtype))
                g = g[:, None, :].expand(B, Q, K).repeat_interleave(Hh, dim=0).to(attn_probs.dtype)

                lam = torch.tensor(self.cfg.lam, device=dev, dtype=attn_probs.dtype)
                delta = lam * g * attn_probs
                delta = delta * harmful_exp
                m = delta.sum(dim=-1, keepdim=True)  # (B*H,Q,1)
                attn_probs = attn_probs - delta

                # ghost K,V from anti
                anti_tok = anti.view(1, 1, -1).expand(B, 1, -1).to(dtype=encoder_hidden_states.dtype)
                k_g = attn.to_k(anti_tok)
                v_g = attn.to_v(anti_tok)
                k_g = attn.head_to_batch_dim(k_g)
                v_g = attn.head_to_batch_dim(v_g)

                mu = torch.tensor(self.cfg.mu, device=dev, dtype=attn_probs.dtype)
                attn_probs_ghost = m + (mu if self.cfg.mu > 0 else 0)

                attn_probs = torch.cat([attn_probs, attn_probs_ghost], dim=-1)  # (B*H,Q,K+1)
                value = torch.cat([value, v_g], dim=1)                           # (B*H,K+1,d_h)

                denom = attn_probs.sum(dim=-1, keepdim=True).clamp(min=torch.tensor(1e-8, device=dev, dtype=attn_probs.dtype))
                attn_probs = attn_probs / denom

                # 4) Debug table (print once per step)
                if self._dbg_on and (self._step >= 0) and (self._step % self._dbg_interval == 0) and (not self._printed_this_step):
                    printed = True
                    self._printed_this_step = True  # 이 step에서 한 번만

                    # aggregate stats (float64로 올려서 안정적 출력)
                    with torch.no_grad():
                        hpct = harmful.float().mean().item() * 100.0
                        g_mean = g_raw[harmful].float().mean().item() if harmful.any() else 0.0
                        g_max = g_raw[harmful].float().max().item() if harmful.any() else 0.0
                        d_mean = delta[harmful_exp].float().mean().item()
                        d_max  = delta[harmful_exp].float().max().item()
                        m_mean = m.float().mean().item()
                        m_max  = m.float().max().item()

                        # sample0, head0, row0 기준 top-k by cos_harm
                        topk = min(self._dbg_topk, K)
                        cos0 = cos_harm[1].float()
                        vals, idxs = torch.topk(cos0, k=topk, largest=True, sorted=True)
                        # g_raw는 (B,K)
                        g0 = g_raw[1].float()
                        fmt = f"{{:.{self._dbg_prec}f}}"

                        print("\n[ATTN-DEBUG] step={:d} | tau={} | lam={} | mu={}".format(
                            self._step, fmt.format(float(self.cfg.tau)), fmt.format(float(self._current_lam or self.cfg.lam)), fmt.format(float(self.cfg.mu))
                        ))
                        print(" - shapes: B={} H={} Q={} K={} (after ghost -> K+1)".format(B, Hh, Q, K))
                        print(" - harmful%={} | g_mean={} g_max={} | delta_mean={} delta_max={} | m_mean={} m_max={}".format(
                            fmt.format(hpct), fmt.format(g_mean), fmt.format(g_max),
                            fmt.format(d_mean), fmt.format(d_max), fmt.format(m_mean), fmt.format(m_max)
                        ))
                        # small table header
                        print("   Top-{} harmful columns (sample0):".format(topk))
                        print("   " + "-" * 56)
                        print("   {:>6} | {:>10} | {:>10}".format("col_id", "cos_harm", "g_raw"))
                        print("   " + "-" * 56)
                        for j in range(topk):
                            cid = int(idxs[j].item())
                            ch = float(vals[j].item())
                            gj = float(g0[cid].item())
                            print("   {:>6} | {:>10} | {:>10}".format(cid, fmt.format(ch), fmt.format(gj)))
                        print("   " + "-" * 56)

            # (optional) you could print when no harmful found, but noisy—skip by default
            _ = printed

        # (E) dropout
        if isinstance(attn.dropout, nn.Dropout):
            attn_probs = attn.dropout(attn_probs)
        else:
            p = float(attn.dropout) if isinstance(attn.dropout, (int, float)) else 0.0
            attn_probs = F.dropout(attn_probs, p=p, training=False)

        # (F) out
        hidden_states = torch.matmul(attn_probs, value)    # (B*H,Q,d_h)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


# =========================
# Main
# =========================
def main():
    args = parse_args()
    accelerator = Accelerator()
    device = accelerator.device

    # seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # pipeline (표준)
    pipe = StableDiffusionPipeline.from_pretrained(
        args.ckpt_path,
        safety_checker=None,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).to(device)
    print("Pipe device:", pipe.device)

    # prompts
    prompt_file = os.path.expanduser(args.prompt_file)
    with open(prompt_file, "r") as f:
        prompts = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(prompts)} prompts from {prompt_file}")

    # ===== Concept vectors =====
    harm_concepts = []
    anti_concepts = []
    if args.harm_texts and os.path.isfile(os.path.expanduser(args.harm_texts)):
        with open(os.path.expanduser(args.harm_texts), "r") as f:
            harm_concepts = [l.strip() for l in f if l.strip()]
    if args.anti_texts and os.path.isfile(os.path.expanduser(args.anti_texts)):
        with open(os.path.expanduser(args.anti_texts), "r") as f:
            anti_concepts = [l.strip() for l in f if l.strip()]

    base_harm_vec = None
    base_anti_vec = None
    if args.enable_harm:
        # anti (safe) vector
        if len(anti_concepts) == 0:
            print("[ghost] anti_texts not provided -> disabled")
        else:
            if getattr(args, "anti_use_eot", False):
                # EOT 한 토큰만 사용
                eot_vecs = []
                for line in anti_concepts:
                    v = build_anti_eot_vec_sd14(pipe, line,
                                                layer_idx=args.layer_index)
                    if v is not None:
                        eot_vecs.append(v)
                if len(eot_vecs) > 0:
                    base_anti_vec = F.normalize(torch.stack(eot_vecs, dim=0).mean(dim=0), dim=-1)
            else:
                # 기존 평균 방식
                anti_mat = build_vecs_sd14(pipe, anti_concepts,
                                        layer_idx=args.layer_index,
                                        include_special=args.include_special_tokens)
                if anti_mat is not None and anti_mat.numel() > 0:
                    base_anti_vec = F.normalize(anti_mat.mean(dim=0), dim=-1)

        if len(harm_concepts) == 0 and args.harm_vec_mode != "prompt_token":
            print("[ghost] harm_texts not provided; will only use prompt_token mode if specified.")
        elif args.harm_vec_mode != "prompt_token":
            base_harm_vec = build_harm_vec_sd14(
                pipe, harm_concepts,
                layer_idx=args.layer_index,
                include_special=args.include_special_tokens,
                mode=args.harm_vec_mode,
                target_words=[w.strip() for w in args.harm_target_words.split(",")] if args.harm_target_words else None
            )

    # Processor 준비/주입
    ghost_proc = None
    if args.enable_harm and (base_anti_vec is not None):
        ghost_proc = GhostContextAttnProcessor(
            harm_vec=base_harm_vec,
            anti_vec=base_anti_vec,
            cfg=GhostCfg(enable=True, tau=args.harm_tau, lam=args.lambda_start, mu=args.ghost_boost_mu),
        )
        # 디버그 설정 전달
        ghost_proc.configure_debug(
            enable=args.debug_attn,
            interval=args.debug_interval,
            topk=args.debug_topk,
            precision=args.debug_precision,
        )
        pipe.unet.set_attn_processor(ghost_proc)
        print(f"[ghost] Enabled. tau={args.harm_tau:.3f}, lam={args.lambda_start:.3f}->{args.lambda_end:.3f}, "
              f"mu={args.ghost_boost_mu:.3f}, layer={args.layer_index}, mode={args.harm_vec_mode}")

    # 스케줄 업데이트 콜백
    def callback_on_step_end_fn(
        diffusion_pipeline,
        step,
        timestep,
        callback_kwargs,
        ghost_proc,
        num_steps: int,
    ):
        if ghost_proc is not None:
            lam = gamma_schedule_linear(step, num_steps, args.lambda_start, args.lambda_end)
            ghost_proc.start_step(step)   # 새 step 시작 알림 (디버그 플래그 리셋)
            ghost_proc.set_lambda(lam)
        return callback_kwargs

    # per-prompt loop
    os.makedirs(os.path.expanduser(args.output_dir), exist_ok=True)

    for idx, prompt in enumerate(prompts):
        print(f"\n=== Generating image for prompt {idx + 1} ===")
        print(prompt)

        # 프롬프트 기반 harm 벡터 (선택)
        if args.enable_harm and ghost_proc is not None and args.harm_vec_mode == "prompt_token":
            words = [w.strip() for w in args.harm_target_words.split(",")] if args.harm_target_words else (harm_concepts or [])
            prompt_harm_vec = build_prompt_token_vec(
                pipe, prompt, words,
                layer_idx=args.layer_index,
                include_special=args.include_special_tokens
            )
            if prompt_harm_vec is not None:
                ghost_proc.set_harm_vec(prompt_harm_vec)

        # SOT soft-delete 제외 마스크
        if args.enable_harm and ghost_proc is not None:
            sot_mask = build_sot_soft_exempt_mask(pipe, prompt)  # (1,K)
            ghost_proc.set_soft_exempt_mask(sot_mask)

        # Debug: per-token cos 출력 (프롬프트 시작 시 표)
        if args.debug_cos and args.enable_harm:
            tok = pipe.tokenizer([prompt], padding="max_length", truncation=True,
                                 max_length=pipe.tokenizer.model_max_length, return_tensors="pt").to(pipe.device)
            out = pipe.text_encoder(**tok, output_hidden_states=True, return_dict=True)
            H = _pick_hidden(out, args.layer_index)[0]   # (L,d)
            Hn = F.normalize(H, dim=-1)
            ids = tok.input_ids[0]
            raw_mask = tok.attention_mask[0].bool()
            toks = pipe.tokenizer.convert_ids_to_tokens(ids.tolist())

            # Match dtype & device to avoid Half vs Float mismatch in torch.dot
            hv = (ghost_proc._harm.to(device=Hn.device, dtype=Hn.dtype)
                if (ghost_proc._harm is not None) else None)
            av = (ghost_proc._anti.to(device=Hn.device, dtype=Hn.dtype)
                if (ghost_proc._anti is not None) else None)

            print("\n[TOKEN-DEBUG] per-token cosine (prompt-level)")
            print("{:>3} | {:>20} | {:>5} | {:>10} | {:>10}".format("i", "token", "use", "cos(harm)", "cos(anti)"))
            print("-" * 64)
            for i, use in enumerate(raw_mask.tolist()):
                if not use:
                    break
                token = toks[i].replace("Ġ", "▁")
                ch = float(torch.dot(Hn[i], hv).item()) if hv is not None else float("nan")
                ca = float(torch.dot(Hn[i], av).item()) if av is not None else float("nan")
                print("{:>3} | {:>20} | {:>5} | {:>10.4f} | {:>10.4f}".format(i, token, str(use), ch, ca))

        # 콜백
        cb = partial(
            callback_on_step_end_fn,
            ghost_proc=ghost_proc,
            num_steps=args.num_inference_steps,
        )

        # run
        out = pipe(
            prompt=prompt,
            guidance_scale=args.cfg_scale,
            num_inference_steps=args.num_inference_steps,
            height=args.height,
            width=args.width,
            num_images_per_prompt=args.nsamples,
            callback_on_step_end=cb,
            callback_on_step_end_tensor_inputs=["latents"],
        )
        images = out.images

        # save first image
        out_path = os.path.join(os.path.expanduser(args.output_dir), f"{idx + 1}.png")
        save_image(images[0], out_path, args.height, args.width)
        print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()

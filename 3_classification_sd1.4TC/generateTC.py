#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, random
from argparse import ArgumentParser
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers.models.attention_processor import AttnProcessor2_0, Attention
import numpy as np
from typing import List, Optional, Tuple

from geo_utils.custom_stable_diffusion import (
    CustomStableDiffusionPipeline,
    CustomStableDiffusionImg2ImgPipeline,
)

# =========================
# Args
# =========================
def parse_args():
    parser = ArgumentParser(description="SD1.4 gen with harm-suppress + ADD boost + EOT-hard-block")
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

    # Harm suppression (soft)
    parser.add_argument("--harm_suppress", action="store_true")
    parser.add_argument("--harm_tau", type=float, default=0.10)
    parser.add_argument("--harm_tau_rel", type=float, default=None)
    parser.add_argument("--harm_alpha", type=float, default=None)
    parser.add_argument("--harm_gamma_start", type=float, default=1.80)
    parser.add_argument("--harm_gamma_end", type=float, default=0.30)
    parser.add_argument("--harm_global_texts", type=str, default=None)

    # Harm vector config
    parser.add_argument("--harm_layer_index", type=int, default=-2)
    parser.add_argument("--harm_vec_mode", type=str, choices=["masked_mean","token","prompt_token"], default="masked_mean")
    parser.add_argument("--harm_target_words", type=str, default=None)
    parser.add_argument("--include_special_tokens", action="store_true")

    # ADD-LIST (positive boost)
    parser.add_argument("--add_list_texts", type=str, default=None)
    parser.add_argument("--add_tau", type=float, default=0.20)
    parser.add_argument("--add_tau_rel", type=float, default=None)
    parser.add_argument("--add_alpha", type=float, default=None)
    parser.add_argument("--add_gamma_start", type=float, default=1.20)
    parser.add_argument("--add_gamma_end", type=float, default=0.50)
    parser.add_argument("--add_layer_index", type=int, default=None)

    # Append: suppress ≥1이면 top-N append
    parser.add_argument("--add_append", action="store_true")
    parser.add_argument("--add_force_append_n", type=int, default=1)

    # Debug
    parser.add_argument("--add_debug_print", action="store_true")

    # EOT hard block
    parser.add_argument("--eot_hard_block", action="store_true")
    return parser.parse_known_args()[0]

# =========================
# Utils
# =========================
def save_image(image, img_metadata, root="output_img"):
    path = img_metadata["file_name"]
    image = Image.fromarray(np.asarray(image), mode="RGB").resize(
        (img_metadata["width"], img_metadata["height"])
    )
    path = os.path.join(root, path[:-4] + ".png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image.save(path)

def _pick_hidden(pipe, out, layer_idx: int):
    if hasattr(out, "hidden_states") and out.hidden_states is not None:
        return out.hidden_states[layer_idx]
    return out.last_hidden_state

def _build_content_mask(tokenizer, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                        include_special: bool) -> torch.Tensor:
    mask = attention_mask.bool()
    if not include_special:
        for sid in [getattr(tokenizer,"bos_token_id",None),
                    getattr(tokenizer,"eos_token_id",None),
                    getattr(tokenizer,"pad_token_id",None)]:
            if sid is not None:
                mask = mask & (input_ids != sid)
    return mask

@torch.no_grad()
def build_used_token_mask(pipe, prompt: str) -> torch.Tensor:
    tok = pipe.tokenizer([prompt], padding="max_length", truncation=True,
                         max_length=pipe.tokenizer.model_max_length, return_tensors="pt").to(pipe.device)
    return tok.attention_mask.bool()

@torch.no_grad()
def build_sot_soft_exempt_mask(pipe, prompt: str) -> torch.Tensor:
    tok = pipe.tokenizer([prompt], padding="max_length", truncation=True,
                         max_length=pipe.tokenizer.model_max_length, return_tensors="pt").to(pipe.device)
    ids = tok.input_ids[0]; att = tok.attention_mask[0].bool()
    bos = getattr(pipe.tokenizer, "bos_token_id", None)
    m = torch.zeros_like(ids, dtype=torch.bool)
    if bos is not None:
        m = att & (ids == bos)
    else:
        first = int(att.nonzero(as_tuple=False)[0].item())
        m[first] = True
    return m.unsqueeze(0)

@torch.no_grad()
def build_eot_hard_block_mask(pipe, prompt: str) -> torch.Tensor:
    tok = pipe.tokenizer([prompt], padding="max_length", truncation=True,
                         max_length=pipe.tokenizer.model_max_length, return_tensors="pt").to(pipe.device)
    ids = tok.input_ids[0]; att = tok.attention_mask[0].bool()
    eos = getattr(pipe.tokenizer, "eos_token_id", None)
    m = torch.zeros_like(ids, dtype=torch.bool)
    if eos is not None:
        m = att & (ids == eos)
    else:
        last = int(att.nonzero(as_tuple=False)[-1].item())
        m[last] = True
    return m.unsqueeze(0)

def _used_token_len(pipe, prompt: str) -> int:
    tok = pipe.tokenizer([prompt], padding="max_length", truncation=True,
                         max_length=pipe.tokenizer.model_max_length, return_tensors="pt").to(pipe.device)
    return int(tok.attention_mask[0].sum().item())

@torch.no_grad()
def build_force_mask_from_span(pipe, prompt: str, start_idx: int) -> torch.Tensor:
    tok = pipe.tokenizer([prompt], padding="max_length", truncation=True,
                         max_length=pipe.tokenizer.model_max_length, return_tensors="pt").to(pipe.device)
    att = tok.attention_mask[0].bool(); ids = tok.input_ids[0]
    bos = getattr(pipe.tokenizer, "bos_token_id", None)
    eos = getattr(pipe.tokenizer, "eos_token_id", None)
    pad = getattr(pipe.tokenizer, "pad_token_id", None)

    m = torch.zeros_like(ids, dtype=torch.bool)
    used = att.nonzero(as_tuple=False).flatten()
    if used.numel() > 0:
        end = int(used[-1].item()) + 1
        s = max(0, min(start_idx, end))
        for i in range(s, end):
            tid = int(ids[i].item())
            if (bos is not None and tid == bos) or (eos is not None and tid == eos) or (pad is not None and tid == pad):
                continue
            m[i] = True
    return m.unsqueeze(0)

@torch.no_grad()
def build_harm_vec_sd14(pipe, concepts: List[str], layer_idx: int = -1,
                        include_special: bool = False, mode: str = "masked_mean",
                        target_words: Optional[List[str]] = None) -> Optional[torch.Tensor]:
    if not concepts:
        return None
    tok = pipe.tokenizer(concepts, padding="max_length", max_length=pipe.tokenizer.model_max_length,
                         truncation=True, return_tensors="pt").to(pipe.device)
    out = pipe.text_encoder(**tok, output_hidden_states=True, return_dict=True)
    H = F.normalize(_pick_hidden(pipe, out, layer_idx), dim=-1)
    mask = _build_content_mask(pipe.tokenizer, tok.input_ids, tok.attention_mask, include_special)
    if mode == "masked_mean":
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
        v_per = (H * mask.unsqueeze(-1)).sum(dim=1) / denom
        return F.normalize(v_per.mean(dim=0), dim=-1)
    words = [w.lower() for w in (target_words if target_words else concepts)]
    selected = []
    for b in range(tok.input_ids.shape[0]):
        toks = pipe.tokenizer.convert_ids_to_tokens(tok.input_ids[b].tolist())
        for i, t in enumerate(toks):
            if not mask[b, i]: continue
            if any(w in t.replace("Ġ","").lower() for w in words):
                selected.append(H[b, i])
    if len(selected)==0: return None
    return F.normalize(torch.stack(selected, dim=0).mean(dim=0), dim=-1)

@torch.no_grad()
def build_vecs_sd14(pipe, texts: List[str], layer_idx: int = -1,
                    include_special: bool = False) -> Optional[torch.Tensor]:
    if not texts: return None
    tok = pipe.tokenizer(texts, padding="max_length", max_length=pipe.tokenizer.model_max_length,
                         truncation=True, return_tensors="pt").to(pipe.device)
    out = pipe.text_encoder(**tok, output_hidden_states=True, return_dict=True)
    H = F.normalize(_pick_hidden(pipe, out, layer_idx), dim=-1)
    mask = _build_content_mask(pipe.tokenizer, tok.input_ids, tok.attention_mask, include_special)
    denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
    v = (H * mask.unsqueeze(-1)).sum(dim=1) / denom
    return F.normalize(v, dim=-1)

@torch.no_grad()
def build_prompt_vec(pipe, prompt: str, layer_idx: int = -1, include_special: bool = False) -> torch.Tensor:
    v = build_vecs_sd14(pipe, [prompt], layer_idx=layer_idx, include_special=include_special)
    return v[0] if v is not None else None

def _central80_mean(x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if mask is None: mask = torch.ones_like(x, dtype=torch.bool)
    out = []
    for b in range(x.shape[0]):
        v = x[b][mask[b]]
        if v.numel() == 0: out.append(torch.tensor(0.0, device=x.device, dtype=x.dtype)); continue
        if v.numel() <= 4: out.append(v.mean()); continue
        q10, q90 = torch.quantile(v, 0.10), torch.quantile(v, 0.90)
        inl = v[(v >= q10) & (v <= q90)]
        out.append(inl.mean() if inl.numel() > 0 else v.mean())
    return torch.stack(out, dim=0).unsqueeze(1)

@torch.no_grad()
def eval_harm_trigger_for_prompt(
    pipe, prompt: str, harm_vec: Optional[torch.Tensor],
    harm_tau_abs: float, harm_tau_rel: Optional[float], harm_alpha: Optional[float],
    layer_idx: int, include_special: bool,
) -> Tuple[bool, int, float]:
    if harm_vec is None or not torch.is_tensor(harm_vec) or harm_vec.numel() == 0:
        return False, 0, float(harm_tau_abs)
    tok = pipe.tokenizer([prompt], padding="max_length", truncation=True,
                         max_length=pipe.tokenizer.model_max_length, return_tensors="pt").to(pipe.device)
    out = pipe.text_encoder(**tok, output_hidden_states=True, return_dict=True)
    H = F.normalize(_pick_hidden(pipe, out, layer_idx)[0], dim=-1)
    hv = F.normalize(harm_vec, dim=-1).to(H.device)
    cos = torch.einsum("ld,d->l", H, hv)

    used_mask = tok.attention_mask.bool()[0]
    sot_mask  = build_sot_soft_exempt_mask(pipe, prompt)[0].bool()
    consider  = used_mask & (~sot_mask)

    if harm_alpha is not None:
        mean80 = _central80_mean(cos.unsqueeze(0), used_mask.unsqueeze(0)).item()
        eff = (1.0 + harm_alpha) * mean80
    elif harm_tau_rel is not None:
        local_max = float(cos[consider].max().item() if consider.any() else 0.0)
        eff = max(harm_tau_abs, float(harm_tau_rel) * local_max)
    else:
        eff = harm_tau_abs

    suppressed = (cos >= eff) & consider
    return bool(suppressed.any().item()), int(suppressed.sum().item()), float(eff)

# =========================
# Attn Processor
# =========================
class HarmCfg:
    __slots__ = ("enable","tau_abs","tau_rel","alpha","gamma")
    def __init__(self, enable=True, tau_abs: float=0.1, tau_rel: Optional[float]=None, alpha: Optional[float]=None, gamma: float=1.0):
        self.enable = bool(enable); self.tau_abs=float(tau_abs); self.tau_rel=(float(tau_rel) if tau_rel is not None else None)
        self.alpha=(float(alpha) if alpha is not None else None); self.gamma=float(gamma)

class AddCfg:
    __slots__ = ("enable","tau_abs","tau_rel","alpha","gamma")
    def __init__(self, enable=True, tau_abs: float=0.2, tau_rel: Optional[float]=None, alpha: Optional[float]=None, gamma: float=1.0):
        self.enable = bool(enable); self.tau_abs=float(tau_abs); self.tau_rel=(float(tau_rel) if tau_rel is not None else None)
        self.alpha=(float(alpha) if alpha is not None else None); self.gamma=float(gamma)

class HarmAddAttnProcessor(AttnProcessor2_0):
    def __init__(self, harm_vec: Optional[torch.Tensor], add_mat: Optional[torch.Tensor],
                 harm_cfg: HarmCfg, add_cfg: AddCfg, eot_hard_block: bool=False):
        super().__init__()
        self.harm_cfg = harm_cfg; self.add_cfg = add_cfg; self.training = False
        self._harm_vec = None if harm_vec is None or harm_vec.numel()==0 else F.normalize(harm_vec.detach().float(), dim=-1).cpu()
        self._add_mat  = None if add_mat  is None or add_mat.numel()==0  else F.normalize(add_mat.detach().float(),  dim=-1).cpu()
        self._soft_exempt_mask = None
        self._force_boost_mask = None
        self._eot_block_mask   = None
        self._used_token_mask  = None
        self._eot_hard_block_enabled = bool(eot_hard_block)

    def set_harm_vec(self, v): self._harm_vec = None if v is None or v.numel()==0 else F.normalize(v.detach().float(), dim=-1).cpu()
    def set_add_mat(self,  m): self._add_mat  = None if m is None or m.numel()==0  else F.normalize(m.detach().float(),  dim=-1).cpu()
    def set_harm_gamma(self, g): self.harm_cfg.gamma = float(g)
    def set_add_gamma(self,  g): self.add_cfg.gamma  = float(g)
    def set_soft_exempt_mask(self, m): self._soft_exempt_mask = None if m is None else m.bool().detach().cpu()
    def set_force_boost_mask(self, m): self._force_boost_mask = None if m is None else m.bool().detach().cpu()
    def set_eot_block_mask(self,   m): self._eot_block_mask   = None if m is None else m.bool().detach().cpu()
    def set_used_token_mask(self,  m): self._used_token_mask  = None if m is None else m.bool().detach().cpu()

    def __call__(self, attn: Attention, hidden_states: torch.FloatTensor,
                 encoder_hidden_states: Optional[torch.FloatTensor]=None,
                 attention_mask: Optional[torch.FloatTensor]=None,
                 temb: Optional[torch.FloatTensor]=None, scale: float=1.0) -> torch.Tensor:

        dev = hidden_states.device
        harm_vec = self._harm_vec.to(dev) if self._harm_vec is not None else None
        add_mat  = self._add_mat.to(dev)  if self._add_mat  is not None else None

        batch_size, sequence_length, _ = hidden_states.shape
        is_cross = encoder_hidden_states is not None

        if attn.spatial_norm is not None: hidden_states = attn.spatial_norm(hidden_states, temb)
        if attn.group_norm   is not None: hidden_states = attn.group_norm(hidden_states.transpose(1,2)).transpose(1,2)

        query = attn.to_q(hidden_states)
        key   = attn.to_k(encoder_hidden_states if is_cross else hidden_states)
        value = attn.to_v(encoder_hidden_states if is_cross else hidden_states)

        query = attn.head_to_batch_dim(query)
        key   = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        scores = torch.matmul(query, key.transpose(-1, -2)) * attn.scale

        if is_cross:
            B = encoder_hidden_states.shape[0]; Q = scores.shape[1]; K = scores.shape[2]
            Hh = scores.shape[0] // B
            ctx_n = F.normalize(encoder_hidden_states, dim=-1)

            used_mask = None
            if self._used_token_mask is not None:
                used_mask = self._used_token_mask.to(scores.device)
                if used_mask.shape[0] == 1: used_mask = used_mask.expand(B, -1)

            # Masks for specials
            sot_mask = None
            if self._soft_exempt_mask is not None:
                sot_mask = self._soft_exempt_mask.to(scores.device)
                if sot_mask.shape[0] == 1: sot_mask = sot_mask.expand(B, -1)

            eot_mask = None
            if self._eot_block_mask is not None:
                eot_mask = self._eot_block_mask.to(scores.device)
                if eot_mask.shape[0] == 1: eot_mask = eot_mask.expand(B, -1)

            # EOT hard block (logits level)
            if self._eot_hard_block_enabled and (eot_mask is not None):
                hard_exp_eot = eot_mask[:, None, :].expand(B, Q, K).repeat_interleave(Hh, dim=0)
                scores = scores.masked_fill(hard_exp_eot, -1e9)

            # Harm soft delete
            cond_soft = None; cos_harm = None
            if self.harm_cfg.enable and (harm_vec is not None):
                harm = F.normalize(harm_vec, dim=-1)
                cos_harm = torch.einsum("bkd,d->bk", ctx_n, harm)
                if self.harm_cfg.alpha is not None:
                    mean80 = _central80_mean(cos_harm, used_mask); eff_harm_tau = (1.0+self.harm_cfg.alpha)*mean80
                elif self.harm_cfg.tau_rel is not None:
                    base = torch.full_like(cos_harm, self.harm_cfg.tau_abs)
                    eff_harm_tau = torch.maximum(base, self.harm_cfg.tau_rel * cos_harm.amax(dim=1, keepdim=True))
                else:
                    eff_harm_tau = torch.full_like(cos_harm, self.harm_cfg.tau_abs)
                cond_soft = (cos_harm >= eff_harm_tau)
                if sot_mask is not None: cond_soft = cond_soft & (~sot_mask)

            # ADD boost
            cond_add = None; add_max = None
            if self.add_cfg.enable and (add_mat is not None) and add_mat.numel()>0:
                sims = torch.einsum("bkd,md->bkm", ctx_n, add_mat)
                add_max = sims.amax(dim=-1)
                if self.add_cfg.alpha is not None:
                    mean80 = _central80_mean(add_max, used_mask); eff_add_tau = (1.0+self.add_cfg.alpha)*mean80
                elif self.add_cfg.tau_rel is not None:
                    base = torch.full_like(add_max, self.add_cfg.tau_abs)
                    eff_add_tau = torch.maximum(base, self.add_cfg.tau_rel * add_max.amax(dim=1, keepdim=True))
                else:
                    eff_add_tau = torch.full_like(add_max, self.add_cfg.tau_abs)
                cond_add = (add_max >= eff_add_tau)
                if sot_mask is not None: cond_add = cond_add & (~sot_mask)
                if eot_mask is not None: cond_add = cond_add & (~eot_mask)  # <-- 명시적 예외

            # Force-boost for newly appended tokens
            if self._force_boost_mask is not None:
                fb = self._force_boost_mask.to(scores.device)
                if fb.shape[0] == 1: fb = fb.expand(B, -1)
                if cond_add is None:
                    cond_add = fb.clone(); add_max = torch.ones_like(cond_add, dtype=scores.dtype)
                else:
                    cond_add = cond_add | fb
                    if add_max is not None:
                        add_max = torch.where(fb, torch.maximum(add_max, torch.ones_like(add_max)), add_max)
                if cond_soft is not None: cond_soft = cond_soft & (~fb)

            # Apply corrections
            if cond_soft is not None and cond_soft.any():
                weight = (cos_harm.clamp(min=0.0) if cos_harm is not None else 1.0) * cond_soft.float()
                soft_w = weight[:, None, :].expand(B, Q, K).repeat_interleave(Hh, dim=0)
                scores = scores - (soft_w * self.harm_cfg.gamma)

            if cond_add is not None and cond_add.any():
                w = (add_max if add_max is not None else torch.ones_like(cond_add, dtype=scores.dtype))
                add_w = (w * cond_add.float())[:, None, :].expand(B, Q, K).repeat_interleave(Hh, dim=0)
                scores = scores + (add_w * self.add_cfg.gamma)

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            scores = scores + attention_mask

        attn_probs = F.softmax(scores, dim=-1)
        p = float(attn.dropout) if not isinstance(attn.dropout, nn.Dropout) else attn.dropout.p
        attn_probs = F.dropout(attn_probs, p=p, training=False)

        hidden_states = torch.matmul(attn_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

# =========================
# Debug print
# =========================
@torch.no_grad()
def debug_print_harm_add(
    pipe, prompt: str, harm_vec: Optional[torch.Tensor], add_mat: Optional[torch.Tensor],
    add_labels: List[str], tau_harm_abs: float, tau_add_abs: float,
    tau_harm_rel: Optional[float], tau_add_rel: Optional[float],
    harm_alpha: Optional[float], add_alpha: Optional[float],
    layer_idx: int=-1, include_special: bool=False, force_mask: Optional[torch.Tensor]=None,
):
    tok = pipe.tokenizer([prompt], padding="max_length", truncation=True,
                         max_length=pipe.tokenizer.model_max_length, return_tensors="pt").to(pipe.device)
    out = pipe.text_encoder(**tok, output_hidden_states=True, return_dict=True)
    H = F.normalize(_pick_hidden(pipe, out, layer_idx)[0], dim=-1); dev = H.device
    ids = tok.input_ids[0]; raw_mask = tok.attention_mask[0].bool()
    mask = _build_content_mask(pipe.tokenizer, tok.input_ids[0], tok.attention_mask[0], include_special)
    toks = pipe.tokenizer.convert_ids_to_tokens(ids.tolist())

    hv = F.normalize(harm_vec, dim=-1).to(dev) if (harm_vec is not None and harm_vec.numel()>0) else None
    AM = add_mat.to(dev) if (add_mat is not None and add_mat.numel()>0) else None
    cos_h_all = torch.einsum("ld,d->l", H, hv) if hv is not None else None
    add_max_all = (H @ AM.T).amax(dim=-1) if AM is not None else None
    used_mask = tok.attention_mask.bool()

    mode_h, eff_h = "abs", tau_harm_abs
    if hv is not None:
        if harm_alpha is not None:
            mode_h = "rel-mean80"; eff_h = (1.0+harm_alpha)*_central80_mean(cos_h_all.unsqueeze(0), used_mask).item()
        elif tau_harm_rel is not None:
            mode_h = "rel"; eff_h = max(tau_harm_abs, float(tau_harm_rel)*float(cos_h_all.max().item()))

    mode_a, eff_a = "abs", tau_add_abs
    if AM is not None:
        if add_alpha is not None:
            mode_a = "rel-mean80"; eff_a = (1.0+add_alpha)*_central80_mean(add_max_all.unsqueeze(0), used_mask).item()
        elif tau_add_rel is not None:
            mode_a = "rel"; eff_a = max(tau_add_abs, float(tau_add_rel)*float(add_max_all.max().item()))

    print(f"\n[HARM/ADD DEBUG] prompt={prompt!r}")
    print(f"  mode(harm)={mode_h}, tau_abs={tau_harm_abs:.3f}, alpha={harm_alpha}, tau_rel={tau_harm_rel}, eff_tau={eff_h:.3f}")
    print(f"  mode(add) ={mode_a}, tau_abs={tau_add_abs:.3f}, alpha={add_alpha}, tau_rel={tau_add_rel}, eff_tau={eff_a:.3f}")
    print(f"{'idx':>3} {'token':>18} {'is_used':>7} {'harm_cos':>9} {'add_max':>9} {'best_add':>16} {'forced':>8} {'suppress?':>10} {'boost?':>7}")
    print("-"*112)

    force_arr = (force_mask[0].tolist() if (force_mask is not None) else None)
    for i, use in enumerate(raw_mask.tolist()):
        if not use: break
        token = toks[i].replace("Ġ","▁")
        harm_cos = float(torch.dot(H[i], hv).item()) if hv is not None else float("nan")
        add_max, best = 0.0, ""
        if AM is not None:
            sims = torch.matmul(H[i], AM.T); mval, midx = sims.max(dim=0)
            add_max = float(mval.item()); best = add_labels[int(midx.item())] if len(add_labels)>0 else ""
        suppress = (harm_cos >= eff_h) if hv is not None else False
        boost    = (add_max >= eff_a) if AM is not None else False
        forced = bool(force_arr and force_arr[i])
        if forced: boost=True; suppress=False
        is_used = bool(mask[i].item())
        print(f"{i:>3} {token:>18} {str(is_used):>7} {harm_cos:+.3f} {add_max:+.3f} {best:>16} {('BOOST' if forced else '-'):>8} {str(suppress):>10} {str(boost):>7}")

def schedule_linear(step: int, num_steps: int, a0: float, a1: float) -> float:
    t = step / max(1, num_steps - 1)
    return a0*(1.0 - t) + a1*t

# =========================
# Main
# =========================
def main(model=None):
    args = parse_args()
    accelerator = Accelerator(); device = accelerator.device

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    pipe = CustomStableDiffusionPipeline.from_pretrained(
        args.ckpt_path, safety_checker=None
    ).to(device)
    print("Pipe device:", pipe.device)

    with open(os.path.expanduser(args.prompt_file), "r") as f:
        prompts = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(prompts)} prompts from {args.prompt_file}")

    # === Prepare vectors ===
    global_concepts = []
    add_labels, add_mat_all = [], None

    if args.harm_suppress:
        if args.harm_global_texts and os.path.isfile(os.path.expanduser(args.harm_global_texts)):
            with open(os.path.expanduser(args.harm_global_texts), "r") as f:
                global_concepts = [l.strip() for l in f if l.strip()]
        if len(global_concepts)==0:
            print("[harm] requested but no concepts -> disable.")
            args.harm_suppress = False

    if args.add_list_texts:
        add_file = os.path.expanduser(args.add_list_texts)
        if os.path.isfile(add_file):
            with open(add_file, "r") as f:
                add_labels = [l.strip() for l in f if l.strip()]
            add_layer = args.add_layer_index if (args.add_layer_index is not None) else args.harm_layer_index
            add_mat_all = build_vecs_sd14(pipe, add_labels, layer_idx=add_layer, include_special=args.include_special_tokens)

    harm_vec_base = None
    if args.harm_suppress and len(global_concepts)>0:
        target_words = [w.strip() for w in args.harm_target_words.split(",")] if args.harm_target_words else None
        if args.harm_vec_mode != "prompt_token":
            harm_vec_base = build_harm_vec_sd14(
                pipe, global_concepts, layer_idx=args.harm_layer_index,
                include_special=args.include_special_tokens, mode=args.harm_vec_mode, target_words=target_words
            )
        else:
            harm_vec_base = build_harm_vec_sd14(
                pipe, global_concepts, layer_idx=args.harm_layer_index,
                include_special=args.include_special_tokens, mode="masked_mean"
            )

    harm_cfg = HarmCfg(enable=bool(args.harm_suppress), tau_abs=args.harm_tau,
                       tau_rel=args.harm_tau_rel, alpha=args.harm_alpha, gamma=args.harm_gamma_start)
    add_cfg  = AddCfg(enable=bool(add_mat_all is not None and add_mat_all.numel()>0),
                      tau_abs=args.add_tau, tau_rel=args.add_tau_rel, alpha=args.add_alpha, gamma=args.add_gamma_start)

    harm_proc = HarmAddAttnProcessor(harm_vec=harm_vec_base, add_mat=None,
                                     harm_cfg=harm_cfg, add_cfg=add_cfg,
                                     eot_hard_block=args.eot_hard_block)
    # === gamma 스케줄 콜백 추가 ===
    num_steps = args.num_inference_steps

    def on_step_end(pipe, step: int, timestep: int, callback_kwargs):
        # 선형 스케줄: 시작→종료로 선형 변화
        g_h = schedule_linear(step, num_steps, args.harm_gamma_start, args.harm_gamma_end)
        g_a = schedule_linear(step, num_steps, args.add_gamma_start,  args.add_gamma_end)
        harm_proc.set_harm_gamma(g_h)
        harm_proc.set_add_gamma(g_a)
        print(f"gamma harm: {g_h}, gamma add: {g_a}")
        return callback_kwargs

    # (선택) 0-step 초기값 명시
    harm_proc.set_harm_gamma(args.harm_gamma_start)
    harm_proc.set_add_gamma(args.add_gamma_start)

    pipe.unet.set_attn_processor(harm_proc)
    print(f"[init] harm_suppress={args.harm_suppress} (tau_abs={args.harm_tau}, tau_rel={args.harm_tau_rel}, alpha={args.harm_alpha}), "
          f"ADD-list={len(add_labels)} (add_tau_abs={args.add_tau}, add_tau_rel={args.add_tau_rel}, alpha={args.add_alpha}), "
          f"layers harm={args.harm_layer_index}, add={args.add_layer_index if args.add_layer_index is not None else args.harm_layer_index}, "
          f"include_special={args.include_special_tokens}, EOT hard block={args.eot_hard_block}")

    scale = args.cfg_scale
    root = os.path.expanduser(args.output_dir); os.makedirs(root, exist_ok=True)
    add_layer_for_vec = args.add_layer_index if (args.add_layer_index is not None) else args.harm_layer_index

    # === Per-prompt loop ===
    for idx, prompt_raw in enumerate(prompts):
        prompt = prompt_raw

        # Prompt-aware harm vec
        if args.harm_suppress and (args.harm_vec_mode == "prompt_token"):
            words = [w.strip() for w in args.harm_target_words.split(",")] if args.harm_target_words else (global_concepts or [])
            prompt_harm_vec = build_harm_vec_sd14(pipe, words, layer_idx=args.harm_layer_index,
                                                  include_special=args.include_special_tokens, mode="masked_mean")
            harm_proc.set_harm_vec(prompt_harm_vec if prompt_harm_vec is not None else harm_vec_base)

        # ==== ADD: suppress >= 1이면 부스트 (+ 선택적 append) ====
        selected_add_mat = None
        selected_labels  = []
        appended = False
        prev_len_tokens = None

        if (add_mat_all is not None) and (add_mat_all.numel() > 0):
            harm_triggered, suppressed_cnt, eff_tau_used = eval_harm_trigger_for_prompt(
                pipe=pipe, prompt=prompt, harm_vec=harm_proc._harm_vec,
                harm_tau_abs=args.harm_tau, harm_tau_rel=args.harm_tau_rel, harm_alpha=args.harm_alpha,
                layer_idx=args.harm_layer_index, include_special=args.include_special_tokens,
            )
            if harm_triggered:
                selected_add_mat = add_mat_all
                selected_labels  = add_labels[:]

                if args.add_append:
                    prev_len_tokens = _used_token_len(pipe, prompt)
                    pvec = build_prompt_vec(pipe, prompt, layer_idx=add_layer_for_vec, include_special=args.include_special_tokens)
                    order = list(range(len(add_labels)))
                    if pvec is not None:
                        pvec = F.normalize(pvec, dim=-1)
                        sims = torch.matmul(add_mat_all, pvec)
                        order = sorted(order, key=lambda i: -float(sims[i].item()))
                    N = max(1, int(args.add_force_append_n))
                    to_add = [add_labels[i] for i in order[:N]]
                    already = set(prompt.split())
                    add_strs = [t for t in to_add if t not in already]
                    if len(add_strs) > 0:
                        prompt = (prompt + " " + " ".join(add_strs)).strip()
                        appended = True

            # harm_triggered가 False면 ADD 자체 비활성
        harm_proc.set_add_mat(selected_add_mat)

        # Masks
        harm_proc.set_soft_exempt_mask(build_sot_soft_exempt_mask(pipe, prompt))
        harm_proc.set_used_token_mask(build_used_token_mask(pipe, prompt))
        force_mask = build_force_mask_from_span(pipe, prompt, prev_len_tokens) if appended and (prev_len_tokens is not None) else None
        harm_proc.set_force_boost_mask(force_mask)
        harm_proc.set_eot_block_mask(build_eot_hard_block_mask(pipe, prompt))

        # Debug
        if args.add_debug_print:
            print(f"[TRIGGER] prompt={idx+1}, harm_triggered={'True' if selected_add_mat is not None else 'False'}")
            debug_print_harm_add(
                pipe=pipe, prompt=prompt,
                harm_vec=harm_proc._harm_vec, add_mat=harm_proc._add_mat,
                add_labels=selected_labels,
                tau_harm_abs=args.harm_tau, tau_add_abs=args.add_tau,
                tau_harm_rel=args.harm_tau_rel, tau_add_rel=args.add_tau_rel,
                harm_alpha=args.harm_alpha, add_alpha=args.add_alpha,
                layer_idx=args.harm_layer_index, include_special=args.include_special_tokens,
                force_mask=force_mask,
            )

        print(f"\n=== Generating image for prompt {idx+1}: {prompt}")
        input_dict = {
            "prompt": prompt,
            "guidance_scale": scale,
            "num_inference_steps": args.num_inference_steps,
            "height": 512, "width": 512,
            "callback_on_step_end": on_step_end,
            "callback_on_step_end_tensor_inputs": [],
            "callback": None, "callback_steps": 1,
            "bbox_binary_mask": None, "num_images_per_prompt": args.nsamples,
        }

        with torch.enable_grad():
            out = pipe(**input_dict)
            generated_images = out.images

        img_metadata = {"file_name": f"{idx+1}.png", "height": 512, "width": 512}
        save_image(generated_images[0], img_metadata, root=root)

if __name__ == "__main__":
    main()

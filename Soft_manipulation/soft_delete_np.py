#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, random
from argparse import ArgumentParser
from typing import List, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from accelerate import Accelerator
from diffusers import StableDiffusionPipeline
from diffusers.models.attention_processor import AttnProcessor2_0, Attention


# ============================================================
# Args
# ============================================================
def parse_args():
    p = ArgumentParser(description="UNCOND inject (harm) + Cross-Attn Soft Delete (harm-only) with rich debugging")

    # positional / IO
    p.add_argument("ckpt_path", type=str)
    p.add_argument("--prompt_file", type=str, required=True)
    p.add_argument("--harm_texts", type=str, required=True, help="harm terms file (one per line)")
    p.add_argument("--output_dir", type=str, default="output_img/hybrid_harm_only")
    p.add_argument("--seed", type=int, default=1234)

    # generation
    p.add_argument("--nsamples", type=int, default=1)
    p.add_argument("--cfg_scale", type=float, default=7.5)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)

    # ---------- UNCOND inject ----------
    p.add_argument("--uncond_mode", type=str, choices=["concat", "mean", "sample"], default="concat")
    p.add_argument("--uncond_join", type=str, default=", ")
    p.add_argument("--uncond_k", type=int, default=10)
    p.add_argument("--dedup", action="store_true")
    p.add_argument("--alpha", type=float, default=1.0, help="blend coefficient for uncond: 0=empty, 1=harm")

    # 문자열로 직접 비교용 negative prompt (기본: 'nudity')
    p.add_argument("--neg_text", type=str, default="nudity",
                   help="파이프라인의 negative_prompt로도 쓰고, 임베딩 경로(α>0)에서도 동일 문자열로 negative 임베딩 생성하여 동등성 비교")

    # ---------- Soft Delete (harm-only) ----------
    p.add_argument("--enable_soft_delete", action="store_true")
    p.add_argument("--tau", type=float, default=0.12, help="harm cosine threshold")
    p.add_argument("--gamma_start", type=float, default=0.45)
    p.add_argument("--gamma_end", type=float, default=0.20)

    # text-encoder settings for harm vector
    p.add_argument("--layer_index", type=int, default=-1)
    p.add_argument("--include_special_tokens", action="store_true")
    p.add_argument("--harm_vec_mode", type=str, choices=["masked_mean", "token", "prompt_token"], default="masked_mean")
    p.add_argument("--harm_target_words", type=str, default=None)  # for token/prompt_token

    # debug
    p.add_argument("--print_steps", type=int, default=5)
    p.add_argument("--debug", action="store_true")

    return p.parse_known_args()[0]


# ============================================================
# Debug helpers
# ============================================================
def _fmt_bool(b): return "ON " if b else "OFF"
def _banner(title: str):
    print("="*66); print(f"= {title:^62} ="); print("="*66)

def _table(rows, sep=" | "):
    if not rows: return
    w1 = max(len(r[0]) for r in rows)
    w2 = max(len(r[1]) for r in rows)
    for k,v in rows:
        print(f"{k:>{w1}}{sep}{v:<{w2}}")

def _shape(t):
    if t is None: return "None"
    return f"{tuple(t.shape)} {str(t.dtype).replace('torch.', '')} @ {t.device}"

def _head(msg=""):
    print("\n" + ("-"*66))
    if msg: print(msg)
    print("-"*66)

def _guard_is_baseline(alpha: float, g0: float, g1: float):
    """ alpha==0 AND gamma==0 => 완전 바닐라여야 함 """
    return (abs(alpha) < 1e-8) and (abs(g0) < 1e-8) and (abs(g1) < 1e-8)


# ============================================================
# Utils
# ============================================================
def read_lines(path: str) -> List[str]:
    with open(os.path.expanduser(path), "r") as f:
        return [l.strip() for l in f if l.strip()]

def maybe_dedup(lines: List[str], dedup: bool) -> List[str]:
    if not dedup: return lines
    uniq = []
    seen = set()
    for s in lines:
        k = s.lower()
        if k not in seen:
            seen.add(k); uniq.append(s)
    return uniq

def save_image(pil_img: Image.Image, path: str, H: int, W: int):
    arr = np.asarray(pil_img)
    img = Image.fromarray(arr, mode="RGB").resize((W, H))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)

def _pick_hidden(out, layer_idx: int):
    return out.hidden_states[layer_idx] if (hasattr(out, "hidden_states") and out.hidden_states is not None) else out.last_hidden_state

def _build_content_mask(tokenizer, input_ids: torch.Tensor, attention_mask: torch.Tensor, include_special: bool):
    mask = attention_mask.bool()
    if not include_special:
        for sid in [getattr(tokenizer,"bos_token_id",None),
                    getattr(tokenizer,"eos_token_id",None),
                    getattr(tokenizer,"pad_token_id",None)]:
            if sid is not None:
                mask = mask & (input_ids != sid)
    return mask

@torch.no_grad()
def encode_texts(pipe: StableDiffusionPipeline, texts: List[str]):
    tok = pipe.tokenizer(texts, padding="max_length", truncation=True,
                         max_length=pipe.tokenizer.model_max_length, return_tensors="pt").to(pipe.device)
    out = pipe.text_encoder(**tok, output_hidden_states=True, return_dict=True)
    out.input_ids = tok.input_ids  # 편의
    out.attention_mask = tok.attention_mask
    return out

@torch.no_grad()
def build_vecs_sd14(pipe, texts: List[str], layer_idx=-1, include_special=False) -> Optional[torch.Tensor]:
    if not texts: return None
    out = encode_texts(pipe, texts)
    H = _pick_hidden(out, layer_idx)  # (B,L,D)
    H = F.normalize(H, dim=-1)
    mask = _build_content_mask(pipe.tokenizer, out.input_ids, out.attention_mask, include_special)
    denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
    v = (H * mask.unsqueeze(-1)).sum(dim=1) / denom
    return F.normalize(v, dim=-1)     # (B,D)

@torch.no_grad()
def build_harm_vec(pipe, concepts: List[str], layer_idx=-1, include_special=False, mode="masked_mean", target_words=None):
    if not concepts: return None
    out = encode_texts(pipe, concepts)
    H = _pick_hidden(out, layer_idx)  # (N,L,D)
    H = F.normalize(H, dim=-1)
    mask = _build_content_mask(pipe.tokenizer, out.input_ids, out.attention_mask, include_special)

    if mode == "masked_mean":
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
        v_per = (H * mask.unsqueeze(-1)).sum(dim=1) / denom
        return F.normalize(v_per.mean(dim=0), dim=-1)  # (D,)

    words = (target_words or concepts)
    words = [w.lower() for w in words]
    sel = []
    for b in range(out.input_ids.shape[0]):
        toks = pipe.tokenizer.convert_ids_to_tokens(out.input_ids[b].tolist())
        for i, t in enumerate(toks):
            if not mask[b, i]: continue
            s = t.replace("Ġ","").lower()
            if any(w in s for w in words):
                sel.append(H[b,i])
    if len(sel)==0: return None
    return F.normalize(torch.stack(sel, dim=0).mean(dim=0), dim=-1)

@torch.no_grad()
def build_prompt_token_vec(pipe, prompt: str, target_words: List[str], layer_idx=-1, include_special=False):
    out = encode_texts(pipe, [prompt])
    H = _pick_hidden(out, layer_idx)[0]
    H = F.normalize(H, dim=-1)
    mask = _build_content_mask(pipe.tokenizer, out.input_ids[0], out.attention_mask[0], include_special)
    toks = pipe.tokenizer.convert_ids_to_tokens(out.input_ids[0].tolist())
    words = [w.lower() for w in (target_words or [])]
    sel=[]
    for i,t in enumerate(toks):
        if not mask[i]: continue
        s = t.replace("Ġ","").lower()
        if any(w in s for w in words):
            sel.append(H[i])
    if len(sel)==0: return None
    return F.normalize(torch.stack(sel, dim=0).mean(dim=0), dim=-1)

@torch.no_grad()
def build_empty_uncond_embeds(pipe, batch_size: int) -> torch.Tensor:
    out = encode_texts(pipe, [""]*batch_size)
    return out.last_hidden_state    # (B,L,D)

@torch.no_grad()
def build_harm_uncond_embeds(pipe, harm_terms: List[str], prompts: List[str], ns: int,
                             mode="concat", joiner=", ", k=10) -> torch.Tensor:
    if mode=="concat":
        s = joiner.join(harm_terms)
        harm_prompts = [s for _ in prompts]
        out = encode_texts(pipe, harm_prompts).last_hidden_state
    elif mode=="mean":
        term_out = encode_texts(pipe, harm_terms).last_hidden_state
        mean = term_out.mean(dim=0, keepdim=True)
        out = mean.expand(len(prompts), -1, -1).contiguous()
    elif mode=="sample":
        harm_prompts=[]
        for _ in prompts:
            terms = random.sample(harm_terms, k=min(k, len(harm_terms)))
            harm_prompts.append(joiner.join(terms))
        out = encode_texts(pipe, harm_prompts).last_hidden_state
    else:
        raise ValueError("unknown mode")

    b, L, D = out.shape
    out = out[:, None, :, :].expand(b, ns, L, D).reshape(b*ns, L, D)
    return out.to(device=pipe.device, dtype=pipe.text_encoder.dtype)

def gamma_schedule_linear(step: int, num_steps: int, g_start: float, g_end: float) -> float:
    t = step / max(1, num_steps - 1)
    return g_start * (1.0 - t) + g_end * t


# ============================================================
# Soft Delete (harm-only)
# ============================================================
class SoftDeleteCfg:
    __slots__=("enable","tau","gamma")
    def __init__(self, enable=True, tau=0.12, gamma=0.45):
        self.enable=bool(enable); self.tau=float(tau); self.gamma=float(gamma)

class SoftDeleteAttnProcessor(AttnProcessor2_0):
    """
    Harm-only soft delete:
      - cond: cos_harm >= tau
      - scores -= gamma * relu(cos_harm - tau)  (SOT 열은 예외)
    + [DBG] last_* 속성에 한 스텝 통계 저장
    """
    def __init__(self, harm_vec: Optional[torch.Tensor], cfg: SoftDeleteCfg):
        super().__init__()
        self.cfg = cfg; self.training = False
        self._harm = (F.normalize(harm_vec.detach().float(), dim=-1).cpu() if harm_vec is not None else None)
        self._soft_exempt_mask = None

        # [DBG] runtime stats
        self.last_harmful_pct = 0.0
        self.last_w_mean = 0.0
        self.last_w_max = 0.0
        self.last_gamma = float(cfg.gamma)
        self.last_tau = float(cfg.tau)
        self.last_shapes = (0,0,0,0)  # (B,Hh,Q,K)
        self.last_applied = False

    def set_gamma(self, gamma: float):
        self.cfg.gamma = float(gamma)
        self.last_gamma = float(gamma)

    def set_harm_vec(self, v: Optional[torch.Tensor]):
        self._harm = (F.normalize(v.detach().float(), dim=-1).cpu() if v is not None else None)

    def set_soft_exempt_mask(self, m: Optional[torch.Tensor]):
        self._soft_exempt_mask = (m.bool().detach().cpu() if m is not None else None)

    def __call__(self, attn: Attention, hidden_states: torch.FloatTensor,
                 encoder_hidden_states: Optional[torch.FloatTensor]=None,
                 attention_mask: Optional[torch.FloatTensor]=None, temb=None, scale: float=1.0) -> torch.FloatTensor:

        dev = hidden_states.device
        is_cross = encoder_hidden_states is not None

        if attn.spatial_norm is not None: hidden_states = attn.spatial_norm(hidden_states, temb)
        if attn.group_norm   is not None: hidden_states = attn.group_norm(hidden_states.transpose(1,2)).transpose(1,2)

        query = attn.to_q(hidden_states)
        if is_cross:
            key   = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
        else:
            key   = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)

        query = attn.head_to_batch_dim(query)
        key   = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        scores = torch.matmul(query, key.transpose(-1,-2)) * attn.scale  # (B*H, Q, K)

        # [DBG] init
        self.last_applied = False
        if is_cross and self.cfg.enable and (self._harm is not None) and (self.cfg.gamma > 0):
            txt_n = F.normalize(encoder_hidden_states, dim=-1)
            harm = self._harm.to(device=dev, dtype=txt_n.dtype)
            cos_harm = torch.einsum("bkd,d->bk", txt_n, harm)            # (B,K)
            cond = (cos_harm >= self.cfg.tau)

            if self._soft_exempt_mask is not None:
                exc = self._soft_exempt_mask.to(dev)
                if exc.shape[0]==1: exc = exc.expand(cond.shape[0], -1)
                cond = cond & (~exc)

            if cond.any():
                B, K = cond.shape
                Q = scores.shape[1]
                Hh = scores.shape[0] // B
                self.last_shapes = (int(B), int(Hh), int(Q), int(K))

                w = (cos_harm - self.cfg.tau).clamp_min(0.0) * cond.float()   # (B,K)
                self.last_harmful_pct = float(cond.float().mean().item()*100.0)
                self.last_w_mean = float(w[w>0].mean().item()) if (w>0).any() else 0.0
                self.last_w_max  = float(w.max().item()) if (w>0).any() else 0.0

                soft_w = w[:,None,:].expand(B,Q,K).repeat_interleave(Hh, dim=0).to(scores.dtype)
                scores = scores - self.cfg.gamma * soft_w
                self.last_applied = True

        if attention_mask is not None:
            batch_size, sequence_length, _ = hidden_states.shape
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            scores = scores + attention_mask

        attn_probs = F.softmax(scores, dim=-1)

        if isinstance(attn.dropout, nn.Dropout):
            attn_probs = attn.dropout(attn_probs)
        else:
            p = float(attn.dropout) if isinstance(attn.dropout,(int,float)) else 0.0
            attn_probs = F.dropout(attn_probs, p=p, training=False)

        hidden_states = torch.matmul(attn_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()
    accelerator = Accelerator()
    device = accelerator.device

    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)

    prompts = read_lines(args.prompt_file)
    harm_terms = maybe_dedup(read_lines(args.harm_texts), args.dedup)

    pipe = StableDiffusionPipeline.from_pretrained(
        args.ckpt_path, safety_checker=None,
        torch_dtype=(torch.float16 if torch.cuda.is_available() else torch.float32)
    ).to(device)

    # === [DBG] run header ===
    if args.debug:
        _banner("HYBRID (harm-only)")
        _table([
            ("CKPT", args.ckpt_path),
            ("PROMPTS", args.prompt_file),
            ("HARMS", args.harm_texts),
            ("NSAMPLES", str(args.nsamples)),
            ("CFG", str(args.cfg_scale)),
            ("STEPS", str(args.num_inference_steps)),
            ("SIZE(HxW)", f"{args.height}x{args.width}"),
            ("ALPHA", f"{args.alpha:.3f}"),
            ("NEG_TEXT", repr(args.neg_text)),
            ("UNCOND_MODE", args.uncond_mode),
            ("UNCOND_JOIN", repr(args.uncond_join)),
            ("UNCOND_K", str(args.uncond_k)),
            ("DEDUP", str(args.dedup)),
            ("SOFT_DELETE", _fmt_bool(args.enable_soft_delete)),
            ("TAU", f"{args.tau:.3f}"),
            ("GAMMA", f"{args.gamma_start:.3f}->{args.gamma_end:.3f}"),
            ("HARM_VEC_MODE", args.harm_vec_mode),
            ("LAYER_INDEX", str(args.layer_index)),
        ])
        print("-"*66)
        print(f"HARM TERMS (N={len(harm_terms)}): "
              + (", ".join(harm_terms[:8]) + (" ..." if len(harm_terms)>8 else "")))
        print("-"*66)

    # === 가드: soft-delete 설치 여부 / 임베딩 경로 여부 ===
    install_softdelete = (args.enable_soft_delete and not (
        abs(args.gamma_start) < 1e-8 and abs(args.gamma_end) < 1e-8
    ))
    use_embed_path = not (abs(args.alpha) < 1e-8)

    if args.debug:
        _table([
            ("INSTALL_SOFTDELETE", str(install_softdelete)),
            ("USE_EMBED_PATH", str(use_embed_path)),
        ])

    # harm vec for soft delete
    sd = None
    if install_softdelete:
        if args.harm_vec_mode != "prompt_token":
            harm_vec = build_harm_vec(
                pipe, harm_terms, layer_idx=args.layer_index,
                include_special=args.include_special_tokens,
                mode=args.harm_vec_mode,
                target_words=[w.strip() for w in args.harm_target_words.split(",")] if args.harm_target_words else None
            )
        else:
            harm_vec = None  # per-prompt
        if harm_vec is not None:
            sd = SoftDeleteAttnProcessor(harm_vec, cfg=SoftDeleteCfg(True, args.tau, args.gamma_start))
            pipe.unet.set_attn_processor(sd)
        elif args.debug:
            print("[WARN] soft-delete requested but harm_vec is None → skip installing processor")

    # step callback: gamma schedule + [DBG] processor snapshot
    def on_step_end(diffusion_pipeline, step, timestep, callback_kwargs):
        if sd is not None:
            g = gamma_schedule_linear(step, args.num_inference_steps, args.gamma_start, args.gamma_end)
            sd.set_gamma(g)
        if args.print_steps>0 and (step % args.print_steps == 0):
            if sd is not None:
                B,Hh,Q,K = sd.last_shapes
                _table([
                    (f"[step {step:>3}] t", str(int(timestep))),
                    ("gamma", f"{sd.last_gamma:.3f}"),
                    ("tau", f"{sd.last_tau:.3f}"),
                    ("applied", str(sd.last_applied)),
                    ("harmful%", f"{sd.last_harmful_pct:.2f}"),
                    ("w_mean/max", f"{sd.last_w_mean:.4f}/{sd.last_w_max:.4f}"),
                    ("shapes(B,H,Q,K)", f"{B},{Hh},{Q},{K}"),
                ])
            else:
                print(f"[step {step:>3}] t={int(timestep)} | (soft-delete OFF)")
        return callback_kwargs

    cb = on_step_end if args.print_steps>0 else None
    cb_inputs = (["latents"] if cb is not None else None)

    os.makedirs(os.path.expanduser(args.output_dir), exist_ok=True)

    for i, prompt in enumerate(prompts):
        _head(f"Prompt {i+1}/{len(prompts)}")
        print("prompt:", prompt)

        # prompt-token harm vec (선택)
        if install_softdelete and sd is not None and args.harm_vec_mode == "prompt_token":
            words = [w.strip() for w in (args.harm_target_words or "").split(",") if w.strip()] or harm_terms
            print("harm(prompt_token) words:", ", ".join(words[:12]) + (" ..." if len(words)>12 else ""))
            hv = build_prompt_token_vec(pipe, prompt, words, layer_idx=args.layer_index,
                                        include_special=args.include_special_tokens)
            if hv is not None:
                sd.set_harm_vec(hv)
            else:
                print("[WARN] prompt_token harm vec empty → keep previous")

        # SOT-exempt mask (BOS 열 보호)
        tok = pipe.tokenizer([prompt], padding="max_length", truncation=True,
                             max_length=pipe.tokenizer.model_max_length, return_tensors="pt").to(pipe.device)
        ids = tok.input_ids[0]; att = tok.attention_mask[0].bool()
        bos = getattr(pipe.tokenizer,"bos_token_id",None)
        sot_mask = torch.zeros_like(ids, dtype=torch.bool)
        if bos is not None: sot_mask = att & (ids == bos)
        else:
            first = int(att.nonzero(as_tuple=False)[0].item()); sot_mask[first]=True
        sot_mask = sot_mask.unsqueeze(0)
        if sd is not None: sd.set_soft_exempt_mask(sot_mask)

        # === 임베딩 경로 (alpha>0): 파이프라인에게 repeat 맡김 ===
        if use_embed_path:
            # 우리 쪽 임베딩 (B=1,L,D)
            pos_embeds = encode_texts(pipe, [prompt]).last_hidden_state  # (1,L,D)

            # negative: neg_text를 직접 사용해 동일 문자열로 임베딩 생성
            if args.neg_text is not None:
                harm_uncond = encode_texts(pipe, [args.neg_text]).last_hidden_state  # (1,L,D)
            else:
                harm_uncond = build_harm_uncond_embeds(
                    pipe, harm_terms, [prompt], ns=1,  # ★ ns=1: repeat은 파이프라인이 함
                    mode=args.uncond_mode, joiner=args.uncond_join, k=args.uncond_k
                )
            empty_uncond = build_empty_uncond_embeds(pipe, 1)  # (1,L,D)
            negative_embeds = (1.0 - args.alpha)*empty_uncond + args.alpha*harm_uncond
            negative_embeds = negative_embeds.to(device=pipe.device, dtype=pipe.text_encoder.dtype)

            if args.debug:
                _table([
                    ("alpha", f"{args.alpha:.3f}"),
                    ("pos_embeds", _shape(pos_embeds)),
                    ("neg_embeds", _shape(negative_embeds)),
                ])

            # 파이프라인 내부 임베딩과의 동등성 검증: 코사인 유사도
            if args.debug:
                pe_pipe, ne_pipe = pipe.encode_prompt(
                    prompt=[prompt],
                    device=pipe.device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=True,
                    negative_prompt=[args.neg_text] if args.neg_text is not None else [""],
                    prompt_embeds=None,
                    negative_prompt_embeds=None,
                )
                # normalize & cosine
                def _cos(a, b):
                    a = F.normalize(a.flatten(1), dim=1)
                    b = F.normalize(b.flatten(1), dim=1)
                    return float(torch.sum(a*b).item())
                cos_pos = _cos(pos_embeds, pe_pipe)
                cos_neg = _cos(negative_embeds, ne_pipe)
                _table([
                    ("[PARITY] cos(prompt_embeds, pipe.prompt_embeds)", f"{cos_pos:.6f}"),
                    ("[PARITY] cos(neg_embeds, pipe.neg_embeds)", f"{cos_neg:.6f}"),
                ])

            out = pipe(
                prompt_embeds=pos_embeds,                 # (1,L,D)
                negative_prompt_embeds=negative_embeds,   # (1,L,D)
                guidance_scale=args.cfg_scale,
                num_inference_steps=args.num_inference_steps,
                height=args.height,
                width=args.width,
                num_images_per_prompt=args.nsamples,      # repeat은 여기서
                callback_on_step_end=cb,
                callback_on_step_end_tensor_inputs=cb_inputs,
            )

        else:
            # === 문자열 경로(바닐라) ===
            if args.debug:
                _table([
                    ("alpha", f"{args.alpha:.3f} (→ use string path)"),
                    ("negative_prompt", repr(args.neg_text if args.neg_text is not None else "")),
                ])
            out = pipe(
                prompt=prompt,
                negative_prompt=(args.neg_text if args.neg_text is not None else ""),
                guidance_scale=args.cfg_scale,
                num_inference_steps=args.num_inference_steps,
                height=args.height,
                width=args.width,
                num_images_per_prompt=args.nsamples,
                callback_on_step_end=cb,
                callback_on_step_end_tensor_inputs=cb_inputs,
            )

        img = out.images[0]
        path = os.path.join(os.path.expanduser(args.output_dir), f"{i+1}.png")
        save_image(img, path, args.height, args.width)
        print(f"saved -> {path}")

    # === 최종 요약 ===
    _head("RUN DONE")
    if _guard_is_baseline(args.alpha, args.gamma_start, args.gamma_end):
        print("Baseline run (alpha=0, gamma=0) confirmed: no uncond harm & no soft delete.")
    else:
        print("Hybrid/Parity run complete.")


if __name__ == "__main__":
    main()

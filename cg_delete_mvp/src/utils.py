#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utility functions for token processing, logging, and image I/O."""

import os
import random
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def save_image(image: Image.Image, path: str, height: int = 512, width: int = 512) -> None:
    """Save image to disk with optional resizing.

    Args:
        image: PIL Image to save
        path: Output file path
        height: Target height
        width: Target width
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image, mode="RGB")

    image = image.resize((width, height))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image.save(path)
    print(f"[utils] Saved image to {path}")


def build_content_mask(
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    include_special: bool = False
) -> torch.Tensor:
    """Build mask for content tokens, optionally excluding special tokens.

    Args:
        tokenizer: HuggingFace tokenizer
        input_ids: Token IDs tensor (B, L)
        attention_mask: Attention mask tensor (B, L)
        include_special: Whether to include SOT/EOT/PAD tokens

    Returns:
        Boolean mask tensor (B, L) indicating valid content tokens
    """
    mask = attention_mask.bool()

    if not include_special:
        # Exclude special tokens
        bos = getattr(tokenizer, "bos_token_id", None)
        eos = getattr(tokenizer, "eos_token_id", None)
        pad = getattr(tokenizer, "pad_token_id", None)

        for sid in [bos, eos, pad]:
            if sid is not None:
                mask = mask & (input_ids != sid)

    return mask


def pick_hidden_layer(output, layer_idx: int = -1) -> torch.Tensor:
    """Select specific hidden layer from text encoder output.

    Args:
        output: Text encoder output with hidden_states attribute
        layer_idx: Layer index to select (negative indexing supported)

    Returns:
        Selected hidden states tensor (B, L, D)
    """
    if hasattr(output, "hidden_states") and output.hidden_states is not None:
        return output.hidden_states[layer_idx]
    return output.last_hidden_state


@torch.no_grad()
def build_harm_vector(
    pipe,
    concepts: List[str],
    layer_idx: int = -1,
    include_special: bool = False,
    mode: str = "masked_mean",
    target_words: Optional[List[str]] = None
) -> Optional[torch.Tensor]:
    """Build harmful concept vector from text concepts.

    This function creates a normalized vector representation of harmful concepts
    that will be used for cosine similarity matching during cross-attention.

    Args:
        pipe: Stable Diffusion pipeline with tokenizer and text_encoder
        concepts: List of harmful concept strings (e.g., ["nudity"])
        layer_idx: Text encoder layer to use (-1 for last layer)
        include_special: Whether to include special tokens in averaging
        mode: Vector construction mode ("masked_mean", "token", "prompt_token")
        target_words: Specific words to target in token mode

    Returns:
        Normalized harm vector (D,) or None if no concepts provided
    """
    if not concepts:
        return None

    # Tokenize concepts
    tok = pipe.tokenizer(
        concepts,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(pipe.device)

    # Get hidden states from text encoder
    out = pipe.text_encoder(**tok, output_hidden_states=True, return_dict=True)
    H = pick_hidden_layer(out, layer_idx)  # (N, L, D)
    H = F.normalize(H, dim=-1)

    # Build content mask (exclude special tokens by default)
    mask = build_content_mask(pipe.tokenizer, tok.input_ids, tok.attention_mask, include_special)

    if mode == "masked_mean":
        # Average over all content tokens across all concept strings
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
        v_per_sent = (H * mask.unsqueeze(-1)).sum(dim=1) / denom  # (N, D)
        v = F.normalize(v_per_sent.mean(dim=0), dim=-1)  # (D,)
        return v

    elif mode == "token":
        # Select only tokens matching target words
        words = target_words if (target_words and len(target_words) > 0) else concepts
        words = [w.lower() for w in words]

        selected = []
        id_lists = tok.input_ids
        for b in range(id_lists.shape[0]):
            tokens = pipe.tokenizer.convert_ids_to_tokens(id_lists[b].tolist())
            for i, t in enumerate(tokens):
                if not mask[b, i]:
                    continue
                # Remove BPE prefix and check if word matches
                s = t.replace("Ġ", "").replace("</w>", "").lower()
                if any(w in s for w in words):
                    selected.append(H[b, i])

        if len(selected) == 0:
            return None

        v = F.normalize(torch.stack(selected, dim=0).mean(dim=0), dim=-1)
        return v

    else:
        raise ValueError(f"Unknown mode: {mode}")


@torch.no_grad()
def build_prompt_vector(
    pipe,
    prompt: str,
    layer_idx: int = -1,
    include_special: bool = False
) -> torch.Tensor:
    """Build vector representation of a single prompt.

    Args:
        pipe: Stable Diffusion pipeline
        prompt: Text prompt string
        layer_idx: Text encoder layer to use
        include_special: Whether to include special tokens

    Returns:
        Normalized prompt vector (D,)
    """
    tok = pipe.tokenizer(
        [prompt],
        padding="max_length",
        truncation=True,
        max_length=pipe.tokenizer.model_max_length,
        return_tensors="pt",
    ).to(pipe.device)

    out = pipe.text_encoder(**tok, output_hidden_states=True, return_dict=True)
    H = pick_hidden_layer(out, layer_idx)[0]  # (L, D)
    H = F.normalize(H, dim=-1)

    mask = build_content_mask(pipe.tokenizer, tok.input_ids[0], tok.attention_mask[0], include_special)

    denom = mask.sum().clamp(min=1)
    v = (H * mask.unsqueeze(-1)).sum(dim=0) / denom  # (D,)
    v = F.normalize(v, dim=-1)

    return v


def schedule_linear(step: int, num_steps: int, start: float, end: float) -> float:
    """Linear scheduling between two values.

    Args:
        step: Current step
        num_steps: Total number of steps
        start: Starting value
        end: Ending value

    Returns:
        Interpolated value at current step
    """
    t = step / max(1, num_steps - 1)
    return start * (1.0 - t) + end * t


def schedule_cosine(step: int, num_steps: int, start: float, end: float) -> float:
    """Cosine scheduling between two values.

    Args:
        step: Current step
        num_steps: Total number of steps
        start: Starting value
        end: Ending value

    Returns:
        Interpolated value at current step
    """
    t = step / max(1, num_steps - 1)
    # Cosine annealing: (1 + cos(pi * t)) / 2
    cos_t = (1.0 + np.cos(np.pi * t)) / 2.0
    return start * cos_t + end * (1.0 - cos_t)


def get_scheduler(sched_type: str):
    """Get scheduling function by name.

    Args:
        sched_type: Scheduler type ("linear", "cosine", "fixed")

    Returns:
        Scheduling function
    """
    if sched_type == "linear":
        return schedule_linear
    elif sched_type == "cosine":
        return schedule_cosine
    elif sched_type == "fixed":
        return lambda step, num_steps, start, end: start
    else:
        raise ValueError(f"Unknown scheduler type: {sched_type}")

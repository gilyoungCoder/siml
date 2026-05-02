#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Cross-attention manipulation for harmful concept deletion.

This module implements soft deletion of harmful concepts by manipulating
cross-attention scores based on cosine similarity with harmful concept vectors.
"""

from typing import Optional
import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import AttnProcessor2_0, Attention


class HarmConfig:
    """Configuration for harmful concept suppression.

    Attributes:
        enable: Whether to enable harm suppression
        tau: Cosine similarity threshold for triggering suppression (τ)
        gamma: Suppression strength factor (γ)
    """
    __slots__ = ("enable", "tau", "gamma")

    def __init__(self, enable: bool = True, tau: float = 0.1, gamma: float = 1.0):
        self.enable = bool(enable)
        self.tau = float(tau)
        self.gamma = float(gamma)


class AttentionEraser(AttnProcessor2_0):
    """Cross-attention processor with harmful concept soft deletion.

    This processor intercepts cross-attention computation and applies
    cosine-similarity-based soft deletion to harmful concept tokens.

    The core algorithm:
    1. Compute Q·K^T attention scores as usual
    2. For each token k in the text sequence:
       - Compute cos(k_embedding, harm_vector)
       - If cos >= τ (threshold), apply soft deletion:
         scores[:, :, k] -= γ * cos⁺
       where cos⁺ = max(0, cos) to avoid amplifying negative similarities

    This is a "soft" delete: we reduce attention scores rather than zeroing them,
    preventing complete information loss and maintaining gradient flow.

    Design principles:
    - Relative suppression (proportional to cosine similarity)
    - No hard thresholding or hard blocks (smooth degradation)
    - Per-step gamma scheduling for adaptive control
    - SOT (start-of-text) exempt to preserve prompt structure
    """

    def __init__(self, harm_vec: Optional[torch.Tensor], harm_cfg: HarmConfig):
        """Initialize attention eraser.

        Args:
            harm_vec: Normalized harmful concept vector (D,) or None
            harm_cfg: Configuration for harm suppression
        """
        super().__init__()
        self.harm_cfg = harm_cfg
        self.training = False

        # Store harm vector on CPU (normalized)
        if harm_vec is None or harm_vec.numel() == 0:
            self._harm_vec = None
        else:
            self._harm_vec = F.normalize(harm_vec.detach().float(), dim=-1).cpu()

        # Mask for tokens exempt from soft deletion (e.g., SOT)
        self._soft_exempt_mask = None  # (B, K) or (1, K)

    @property
    def harm_vec(self) -> Optional[torch.Tensor]:
        """Get current harm vector."""
        return self._harm_vec

    def set_harm_vec(self, harm_vec: Optional[torch.Tensor]):
        """Update harm vector.

        Args:
            harm_vec: New harm vector or None to disable
        """
        if harm_vec is None or harm_vec.numel() == 0:
            self._harm_vec = None
        else:
            self._harm_vec = F.normalize(harm_vec.detach().float(), dim=-1).cpu()

    def set_harm_gamma(self, gamma: float):
        """Update suppression strength.

        Args:
            gamma: New gamma value
        """
        self.harm_cfg.gamma = float(gamma)

    def set_soft_exempt_mask(self, mask: Optional[torch.Tensor]):
        """Set mask for tokens to exempt from soft deletion.

        Args:
            mask: Boolean mask (B, K) where True = exempt from deletion
        """
        self._soft_exempt_mask = None if mask is None else mask.bool().detach().cpu()

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
    ) -> torch.Tensor:
        """Forward pass with attention deletion.

        Args:
            attn: Attention module
            hidden_states: Query hidden states (B, Q, D)
            encoder_hidden_states: Key/Value hidden states for cross-attention (B, K, D)
            attention_mask: Optional attention mask
            temb: Optional temporal embeddings
            scale: Scaling factor

        Returns:
            Output hidden states (B, Q, D)
        """
        dev = hidden_states.device
        harm_vec = self._harm_vec.to(dev) if self._harm_vec is not None else None

        batch_size, sequence_length, _ = hidden_states.shape
        is_cross = encoder_hidden_states is not None

        # (A) Pre-processing: spatial/group norm if present
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        # (B) Project to Q, K, V
        query = attn.to_q(hidden_states)
        if is_cross:
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
        else:
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)

        # (C) Reshape for multi-head attention
        query = attn.head_to_batch_dim(query)  # (B*H, Q, d_h)
        key = attn.head_to_batch_dim(key)      # (B*H, K, d_h)
        value = attn.head_to_batch_dim(value)

        # (D) Compute raw attention scores: Q·K^T / sqrt(d)
        scores = torch.matmul(query, key.transpose(-1, -2)) * attn.scale  # (B*H, Q, K)

        # (E) Apply harmful concept soft deletion (only for cross-attention)
        if is_cross and self.harm_cfg.enable and (harm_vec is not None):
            B = encoder_hidden_states.shape[0]
            Q = scores.shape[1]
            K = scores.shape[2]
            Hh = scores.shape[0] // B  # num_heads

            # Normalize context embeddings for cosine similarity
            ctx_n = F.normalize(encoder_hidden_states, dim=-1)  # (B, K, D)

            # Compute cosine similarity: cos(ctx_k, harm_vec)
            harm = F.normalize(harm_vec, dim=-1)  # (D,)
            cos_harm = torch.einsum("bkd,d->bk", ctx_n, harm)  # (B, K)

            # Soft deletion condition: cos >= τ
            cond_soft = (cos_harm >= self.harm_cfg.tau)

            # Exclude SOT tokens from suppression
            if self._soft_exempt_mask is not None:
                exc = self._soft_exempt_mask.to(cond_soft.device)
                if exc.shape[0] == 1:
                    exc = exc.expand(B, -1)
                cond_soft = cond_soft & (~exc)

            # Apply soft deletion: scores -= γ * cos⁺
            if cond_soft.any():
                # Use only positive cosine (cos⁺ = max(0, cos))
                weight = cos_harm.clamp(min=0.0) * cond_soft.float()  # (B, K)

                # Expand to (B*H, Q, K) for all heads
                soft_w = weight[:, None, :].expand(B, Q, K).repeat_interleave(Hh, dim=0)

                if soft_w.any():
                    # Soft delete: reduce attention scores proportionally
                    scores = scores - (soft_w * self.harm_cfg.gamma)

        # (F) Apply attention mask if provided
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            scores = scores + attention_mask

        # (G) Softmax to get attention probabilities
        attn_probs = F.softmax(scores, dim=-1)

        # (H) Apply dropout
        if hasattr(attn, 'dropout') and attn.dropout > 0.0:
            if isinstance(attn.dropout, torch.nn.Dropout):
                attn_probs = attn.dropout(attn_probs)
            else:
                p = float(attn.dropout)
                attn_probs = F.dropout(attn_probs, p=p, training=False)

        # (I) Compute output: attn_probs·V
        hidden_states = torch.matmul(attn_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # (J) Output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


@torch.no_grad()
def build_sot_exempt_mask(pipe, prompt: str) -> torch.Tensor:
    """Build mask for SOT (start-of-text) token to exempt from soft deletion.

    The SOT token is important for maintaining prompt structure and should not
    be suppressed even if it has similarity with harmful concepts.

    Args:
        pipe: Stable Diffusion pipeline
        prompt: Text prompt

    Returns:
        Boolean mask (1, K) where True = SOT token (exempt from deletion)
    """
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
        # Mark BOS token positions
        m = att & (ids == bos)
    else:
        # If no explicit BOS, mark first valid token
        valid_indices = att.nonzero(as_tuple=False)
        if len(valid_indices) > 0:
            first = int(valid_indices[0].item())
            m[first] = True

    return m.unsqueeze(0)  # (1, K)

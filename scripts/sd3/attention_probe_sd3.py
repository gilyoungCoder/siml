"""
Joint-Attention Probing for SD3's MMDiT Architecture.

SD3's JointAttnProcessor2_0 concatenates [image_tokens, text_tokens] for
joint self-attention. We extract the image→text sub-matrix as a
cross-attention-like spatial map for concept localization.

Attention matrix layout (concatenation order: [image, text]):
  Q = [Q_img; Q_txt],  K = [K_img; K_txt]
  attn = softmax(Q @ K^T / sqrt(d))

  Cross-attention-like = attn[0:num_img, num_img:num_img+num_txt]
    = how each image patch attends to each text token

Key difference from SD1.4 UNet:
  - SD1.4: separate cross-attention layers (attn2) with separate Q(img), K(text)
  - SD3: joint attention with concatenated [img, txt] tokens
  - SD1.4: uses attn.to_q, attn.to_k for image, text comes via encoder_hidden_states
  - SD3: uses attn.to_q/k/v for image, attn.add_q_proj/k_proj/v_proj for text

Image probe (SafeGen-style WHERE module):
  Parallel to the text probe, we build a "pseudo text" encoder_hidden_states
  where a few token slots are replaced with CLIP-exemplar features (averaged
  and normalized). Running the image Q against this pseudo text K via
  `attn.add_k_proj` gives a spatial attention map that highlights regions
  visually similar to the exemplar concept.
"""

from typing import Any, Dict, List, Optional
import torch
import torch.nn.functional as F


class SD3AttentionProbeStore:
    """Stores probe attention maps extracted from SD3 joint attention blocks.

    Holds both text-probe maps (image→prompt-text attention sub-matrix) and
    image-probe maps (image→pseudo-exemplar attention) in separate dicts.
    """

    def __init__(self):
        self.probe_maps: Dict[str, torch.Tensor] = {}        # text probe maps
        self.image_probe_maps: Dict[str, torch.Tensor] = {}  # image probe maps
        self.active: bool = False
        # Image-probe state (populated by register_sd3_attention_probe)
        self.image_probe_enabled: bool = False
        # Pre-projected pseudo-text embeddings to use as image-probe keys.
        # Shape: [1, num_probe_tokens, 4096] — SD3 joint encoder hidden dim.
        self.image_probe_embeds: Optional[torch.Tensor] = None
        # Indices within the probe_embeds to aggregate across (e.g. [1,2,3,4]).
        self.image_probe_token_indices: Optional[List[int]] = None

    def store_probe(self, block_name: str, attn_map: torch.Tensor):
        self.probe_maps[block_name] = attn_map.detach()

    def store_image_probe(self, block_name: str, attn_map: torch.Tensor):
        self.image_probe_maps[block_name] = attn_map.detach()

    def reset(self):
        self.probe_maps.clear()
        self.image_probe_maps.clear()

    def get_maps(self) -> Dict[str, torch.Tensor]:
        return self.probe_maps

    def get_image_maps(self) -> Dict[str, torch.Tensor]:
        return self.image_probe_maps


class JointAttentionProbeProcessor:
    """
    Custom processor for SD3 joint attention that:
    1. Performs normal joint attention (output unchanged)
    2. Extracts image→text attention sub-matrix
    3. Stores it in SD3AttentionProbeStore

    Replaces JointAttnProcessor2_0 on selected blocks.
    """

    def __init__(self, store: SD3AttentionProbeStore, block_name: str):
        self.store = store
        self.block_name = block_name

    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states
        batch_size = hidden_states.shape[0]
        num_img_tokens = hidden_states.shape[1]

        # Image projections
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Text projections (context)
        num_txt_tokens = 0
        if encoder_hidden_states is not None:
            num_txt_tokens = encoder_hidden_states.shape[1]

            enc_q = attn.add_q_proj(encoder_hidden_states)
            enc_k = attn.add_k_proj(encoder_hidden_states)
            enc_v = attn.add_v_proj(encoder_hidden_states)

            enc_q = enc_q.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            enc_k = enc_k.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            enc_v = enc_v.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if attn.norm_added_q is not None:
                enc_q = attn.norm_added_q(enc_q)
            if attn.norm_added_k is not None:
                enc_k = attn.norm_added_k(enc_k)

            # Concatenate: [image, text]
            query = torch.cat([query, enc_q], dim=2)
            key = torch.cat([key, enc_k], dim=2)
            value = torch.cat([value, enc_v], dim=2)

        if self.store.active and num_txt_tokens > 0:
            # Compute attention explicitly to capture weights
            scale = head_dim ** -0.5
            attn_weights = torch.matmul(query * scale, key.transpose(-2, -1))
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            attn_probs = attn_weights.softmax(dim=-1)
            hidden_out = torch.matmul(attn_probs, value)

            # Extract image→text sub-matrix
            # attn_probs: [B, H, num_img+num_txt, num_img+num_txt]
            # We want: [H, num_img, num_txt] for the conditional batch
            b_idx = min(1, batch_size - 1)  # Use conditional (idx=1 in CFG)
            cross_attn = attn_probs[b_idx, :, :num_img_tokens, num_img_tokens:]
            # Shape: [H, num_img, num_txt]
            self.store.store_probe(self.block_name, cross_attn)

            # ---- Image probe: image Q against CLIP-exemplar pseudo-text K ----
            if (self.store.image_probe_enabled
                    and self.store.image_probe_embeds is not None):
                probe_embeds = self.store.image_probe_embeds.to(
                    hidden_states.device, hidden_states.dtype)
                # Project exemplar pseudo-text through text key projection.
                probe_k = attn.add_k_proj(probe_embeds)  # [1, P, inner_dim]
                probe_k = probe_k.view(
                    1, -1, attn.heads, head_dim).transpose(1, 2)  # [1,H,P,D]
                if attn.norm_added_k is not None:
                    probe_k = attn.norm_added_k(probe_k)

                # Image queries for the conditional batch.
                q_img_cond = query[b_idx:b_idx + 1, :, :num_img_tokens, :]  # [1,H,Ni,D]
                # Attention logits image→probe: [1,H,Ni,P]
                img_probe_logits = torch.matmul(
                    q_img_cond * scale, probe_k.transpose(-2, -1))
                img_probe_attn = img_probe_logits.softmax(dim=-1)
                # Squeeze batch → [H, Ni, P]
                self.store.store_image_probe(
                    self.block_name, img_probe_attn.squeeze(0))
        else:
            # Use SDPA for speed when not probing
            hidden_out = F.scaled_dot_product_attention(
                query, key, value, dropout_p=0.0, is_causal=False
            )

        # Reshape and split
        hidden_out = hidden_out.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_out = hidden_out.to(query.dtype)

        if encoder_hidden_states is not None:
            hidden_states = hidden_out[:, :num_img_tokens]
            encoder_hidden_states = hidden_out[:, num_img_tokens:]

            # Output projections
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)

            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            hidden_states = attn.to_out[0](hidden_out)
            hidden_states = attn.to_out[1](hidden_out)
            return hidden_states


def build_sd3_image_probe_embeds(
    clip_features: torch.Tensor,
    baseline_encoder_hidden: torch.Tensor,
    n_tokens: int = 4,
) -> (torch.Tensor, List[int]):
    """
    Construct an SD3-compatible pseudo-text embedding where a few token slots
    are replaced with averaged CLIP-exemplar features (SafeGen image probe).

    SD3 encoder_hidden_states is 4096-d (CLIP-L+CLIP-G: 2048, T5: 2048 → 4096).
    CLIP exemplar features are 768-d. We inject the normalized exemplar into
    the first 2048 dims of the selected token slots (CLIP-L+CLIP-G portion),
    padding with zeros up to 4096. This places the exemplar signal into the
    same "text-style" subspace that `attn.add_k_proj` reads.

    Args:
        clip_features: [N, 768] stack of CLIP exemplar image features.
        baseline_encoder_hidden: [1, L, 4096] empty-prompt encoder hidden
            state from SD3's `encode_prompt("")` (used as the baseline).
        n_tokens: number of token slots to overwrite (e.g. 4).

    Returns:
        probe_embeds: [1, L, 4096] — baseline with slots 1..n_tokens replaced.
        token_indices: [1, ..., n_tokens].
    """
    if clip_features.dim() != 2:
        raise ValueError(f"clip_features must be [N,D], got {clip_features.shape}")

    # Average and L2-normalize in CLIP space (match SafeGen SD1.4 image probe).
    avg = F.normalize(clip_features.float().mean(dim=0), dim=-1)  # [D_clip]
    d_clip = avg.shape[0]

    baseline = baseline_encoder_hidden.clone()
    B, L, D_enc = baseline.shape
    device = baseline.device
    dtype = baseline.dtype

    # Build a target-sized vector: place CLIP feature into first d_clip dims.
    target_vec = torch.zeros(D_enc, device=device, dtype=dtype)
    fill = min(d_clip, D_enc)
    target_vec[:fill] = avg[:fill].to(device=device, dtype=dtype)
    # Re-normalize over the full target dim so magnitudes are reasonable.
    norm = target_vec.norm()
    if norm > 1e-8:
        target_vec = target_vec / norm

    n_tokens = min(n_tokens, L - 1)
    for i in range(1, 1 + n_tokens):
        baseline[0, i] = target_vec

    return baseline, list(range(1, 1 + n_tokens))


def build_grouped_sd3_image_probe_embeds(
    family_features: Dict[str, torch.Tensor],
    baseline_encoder_hidden: torch.Tensor,
    max_tokens: int = 4,
) -> (torch.Tensor, List[int], Dict[str, int]):
    """
    Construct an SD3 pseudo-text embedding with one CLIP-exemplar token per
    family. Mirrors SafeGen's grouped image probe but targets SD3's
    encoder_hidden_states layout.

    Args:
        family_features: {family_name: [N, 768]} CLIP exemplar features.
        baseline_encoder_hidden: [1, L, D] baseline encoder hidden state.
        max_tokens: maximum number of family token slots to populate.

    Returns:
        probe_embeds: [1, L, D]
        token_indices: active token positions
        family_token_map: {family_name: token_position}
    """
    if not family_features:
        raise ValueError("family_features must be non-empty")

    baseline = baseline_encoder_hidden.clone()
    _, seq_len, hidden_dim = baseline.shape
    family_names = list(family_features.keys())[:max_tokens]
    family_token_map: Dict[str, int] = {}

    for i, fname in enumerate(family_names, start=1):
        feats = family_features[fname]
        if feats.dim() != 2:
            raise ValueError(f"family feature for {fname} must be [N,D], got {feats.shape}")

        avg = F.normalize(feats.float().mean(dim=0), dim=-1)
        target_vec = torch.zeros(hidden_dim, device=baseline.device, dtype=baseline.dtype)
        fill = min(avg.shape[0], hidden_dim)
        target_vec[:fill] = avg[:fill].to(device=baseline.device, dtype=baseline.dtype)
        norm = target_vec.norm()
        if norm > 1e-8:
            target_vec = target_vec / norm

        if i < seq_len:
            baseline[0, i] = target_vec
            family_token_map[fname] = i

    token_indices = list(family_token_map.values())
    return baseline, token_indices, family_token_map


def register_sd3_attention_probe(
    transformer,
    store: SD3AttentionProbeStore,
    target_blocks: Optional[List[int]] = None,
    probe_mode: str = "text",
    image_probe_embeds: Optional[torch.Tensor] = None,
    image_probe_token_indices: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Register JointAttentionProbeProcessor on target transformer blocks.

    Args:
        transformer: SD3Transformer2DModel
        store: SD3AttentionProbeStore
        target_blocks: Block indices to hook (None = middle third)
        probe_mode: "text", "image", or "both". Controls whether the
            image-probe branch is enabled in the store.
        image_probe_embeds: Pre-computed pseudo-text embedding used as image
            probe key source. Required when probe_mode in ("image","both").
        image_probe_token_indices: Token slots inside probe_embeds that hold
            the exemplar (used during spatial-mask aggregation).

    Returns:
        {block_name: original_processor} for restoration
    """
    num_blocks = len(transformer.transformer_blocks)

    if target_blocks is None:
        # Middle 3 blocks (memory-efficient; SafeGen reference uses similarly few)
        mid = num_blocks // 2
        target_blocks = [mid - 1, mid, mid + 1]

    # Configure image-probe state on the store.
    if probe_mode in ("image", "both"):
        if image_probe_embeds is None:
            raise ValueError(
                f"probe_mode='{probe_mode}' requires image_probe_embeds")
        store.image_probe_enabled = True
        store.image_probe_embeds = image_probe_embeds.detach()
        store.image_probe_token_indices = image_probe_token_indices
    else:
        store.image_probe_enabled = False
        store.image_probe_embeds = None
        store.image_probe_token_indices = None

    original_processors = {}
    hooked = 0

    for block_idx in target_blocks:
        if block_idx >= num_blocks:
            continue

        block = transformer.transformer_blocks[block_idx]
        block_name = f"block_{block_idx}"

        # Save original
        original_processors[block_name] = block.attn.processor

        # Replace with probe processor
        block.attn.processor = JointAttentionProbeProcessor(
            store=store,
            block_name=block_name,
        )
        hooked += 1

    print(f"  [SD3 probe] Hooked {hooked} joint attention blocks: {target_blocks} "
          f"(mode={probe_mode})")
    return original_processors


def restore_sd3_processors(transformer, original_processors: Dict[str, Any]):
    """Restore original attention processors."""
    for block_name, proc in original_processors.items():
        block_idx = int(block_name.split("_")[1])
        if block_idx < len(transformer.transformer_blocks):
            transformer.transformer_blocks[block_idx].attn.processor = proc


def compute_sd3_spatial_mask(
    store: SD3AttentionProbeStore,
    token_indices: Optional[List[int]] = None,
    latent_h: int = 64,
    latent_w: int = 64,
) -> torch.Tensor:
    """
    Aggregate SD3 joint attention probe maps into a spatial mask.

    SD3-medium at 1024x1024: latent 128x128 with patch_size=2 → 64x64 = 4096 image tokens.

    Args:
        store: SD3AttentionProbeStore
        token_indices: Text token indices to focus on (None = all)
        latent_h, latent_w: Spatial dims for reshaping image tokens to 2D

    Returns:
        spatial_mask: [H, W] normalized in [0, 1]
    """
    maps = store.get_maps()
    if not maps:
        return torch.zeros(latent_h, latent_w)

    all_maps = []
    for block_name, attn_map in maps.items():
        # attn_map: [H, num_img_tokens, num_txt_tokens]

        # Average across heads
        avg_map = attn_map.mean(dim=0)  # [num_img, num_txt]

        # Select target text tokens
        if token_indices is not None and len(token_indices) > 0:
            token_map = avg_map[:, token_indices]
        else:
            token_map = avg_map

        # Max across text tokens (union)
        spatial_map = token_map.max(dim=-1)[0]  # [num_img]

        # Reshape to 2D
        num_patches = spatial_map.shape[0]
        ph = int(num_patches ** 0.5)
        pw = num_patches // ph
        spatial_2d = spatial_map.view(1, 1, ph, pw)

        # Interpolate to target resolution
        if ph != latent_h or pw != latent_w:
            spatial_2d = F.interpolate(
                spatial_2d.float(), size=(latent_h, latent_w),
                mode="bilinear", align_corners=False,
            )

        all_maps.append(spatial_2d)

    if not all_maps:
        return torch.zeros(latent_h, latent_w)

    combined = torch.stack(all_maps).mean(0)

    # Normalize to [0, 1]
    flat = combined.reshape(-1)
    vmin, vmax = flat.min(), flat.max()
    if vmax - vmin > 1e-8:
        combined = (combined - vmin) / (vmax - vmin)

    return combined.squeeze(0).squeeze(0)


def compute_sd3_image_probe_mask(
    store: SD3AttentionProbeStore,
    token_indices: Optional[List[int]] = None,
    latent_h: int = 64,
    latent_w: int = 64,
) -> torch.Tensor:
    """
    Aggregate SD3 image-probe maps (image → CLIP-exemplar pseudo-text) into a
    spatial mask. Same reduction pipeline as the text probe but over the
    image-probe store.

    Args:
        store: SD3AttentionProbeStore (image_probe_maps must be populated).
        token_indices: Pseudo-text token slots that carry the exemplar (e.g.
            [1,2,3,4]). If None, falls back to `store.image_probe_token_indices`
            and then to all tokens.

    Returns:
        spatial_mask: [H, W] normalized in [0, 1].
    """
    maps = store.get_image_maps()
    if not maps:
        return torch.zeros(latent_h, latent_w)

    if token_indices is None:
        token_indices = store.image_probe_token_indices

    all_maps = []
    for block_name, attn_map in maps.items():
        # attn_map: [H, num_img_tokens, num_probe_tokens]
        avg_map = attn_map.mean(dim=0)  # [num_img, P]

        if token_indices is not None and len(token_indices) > 0:
            valid = [i for i in token_indices if i < avg_map.shape[-1]]
            if valid:
                token_map = avg_map[:, valid]
            else:
                token_map = avg_map
        else:
            token_map = avg_map

        spatial_map = token_map.max(dim=-1)[0]  # [num_img]

        num_patches = spatial_map.shape[0]
        ph = int(num_patches ** 0.5)
        pw = num_patches // ph
        spatial_2d = spatial_map.view(1, 1, ph, pw)

        if ph != latent_h or pw != latent_w:
            spatial_2d = F.interpolate(
                spatial_2d.float(), size=(latent_h, latent_w),
                mode="bilinear", align_corners=False,
            )

        all_maps.append(spatial_2d)

    if not all_maps:
        return torch.zeros(latent_h, latent_w)

    combined = torch.stack(all_maps).mean(0)

    flat = combined.reshape(-1)
    vmin, vmax = flat.min(), flat.max()
    if vmax - vmin > 1e-8:
        combined = (combined - vmin) / (vmax - vmin)

    return combined.squeeze(0).squeeze(0)

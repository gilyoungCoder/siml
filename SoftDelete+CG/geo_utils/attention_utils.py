"""
Attention-aware spatial masking for z0 classifier guidance.

Extracts cross-attention maps from UNet during denoising and uses them
as spatial masks to localize classifier guidance to semantically harmful regions.

The key idea (from the "Attention-Aware z0-Classifier Guidance" diagram):
  1. During UNet forward, cross-attention maps show WHERE each text token
     attends in the spatial image.
  2. For harmful tokens (e.g. "attacking", "weapon"), their attention heatmaps
     indicate the harmful regions.
  3. These heatmaps become spatial masks for the z0-classifier gradient,
     so guidance only affects the harmful regions while preserving the rest.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


# SD1.4 UNet: block name prefix -> spatial resolution of feature maps
BLOCK_RESOLUTION = {
    "down_blocks.0": 64,
    "down_blocks.1": 32,
    "down_blocks.2": 16,
    "mid_block": 8,
    "up_blocks.1": 16,
    "up_blocks.2": 32,
    "up_blocks.3": 64,
}


def _get_resolution(layer_name: str) -> int:
    """Infer spatial resolution from attention layer name."""
    for prefix, res in BLOCK_RESOLUTION.items():
        if layer_name.startswith(prefix):
            return res
    return 0


class AttentionStore:
    """
    Collects cross-attention maps from UNet during a single forward pass.

    Maps are overwritten each UNet forward (no accumulation across steps).
    Kept on GPU for fast access in the same-step callback.

    Usage:
        store = AttentionStore()
        register_attention_store(unet, store)
        # ... UNet forward → maps stored automatically ...
        mask = compute_attention_mask(store, token_indices)
    """

    def __init__(self):
        self.attn_maps: Dict[str, torch.Tensor] = {}

    def store(self, layer_name: str, attn_probs: torch.Tensor):
        """Store attention probs. attn_probs: (B_cfg*H, spatial_seq, 77)"""
        self.attn_maps[layer_name] = attn_probs.detach()

    def reset(self):
        """Clear stored maps (call between prompts)."""
        self.attn_maps.clear()

    def get_maps(self) -> Dict[str, torch.Tensor]:
        return self.attn_maps


class StoreCrossAttnProcessor:
    """
    Custom attention processor for cross-attention (attn2) layers.

    Uses explicit matmul + softmax (not SDPA) to extract attention weights,
    then stores them in the AttentionStore. Functionally equivalent to the
    default AttnProcessor but exposes the attention probability matrix.
    """

    def __init__(self, attention_store: AttentionStore, layer_name: str):
        self.store = attention_store
        self.layer_name = layer_name

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape

        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = (
            encoder_hidden_states if is_cross else hidden_states
        )

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        # Reshape to (B, H, seq, head_dim)
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # Optional query/key normalization (None for SD1.4)
        if getattr(attn, "norm_q", None) is not None:
            query = attn.norm_q(query)
        if getattr(attn, "norm_k", None) is not None:
            key = attn.norm_k(key)

        if is_cross:
            # Explicit attention computation to capture weights
            scale = head_dim ** -0.5
            attn_weights = torch.matmul(query * scale, key.transpose(-2, -1))

            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask

            attn_probs = attn_weights.softmax(dim=-1)

            # Store: reshape to (B*H, spatial_seq, token_seq) for aggregation
            self.store.store(
                self.layer_name,
                attn_probs.reshape(
                    batch_size * attn.heads,
                    attn_probs.shape[2],
                    attn_probs.shape[3],
                ),
            )

            hidden_states = torch.matmul(attn_probs, value)
        else:
            # Self-attention: use SDPA for speed (no need to store maps)
            hidden_states = F.scaled_dot_product_attention(
                query, key, value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
            )

        # (B, H, seq, head_dim) -> (B, seq, inner_dim)
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, inner_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        # Output projection + dropout
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


def register_attention_store(
    unet,
    store: AttentionStore,
    target_resolutions: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Replace attn2 (cross-attention) processors with StoreCrossAttnProcessor.
    Self-attention (attn1) processors are left unchanged.

    Args:
        unet: UNet2DConditionModel
        store: AttentionStore instance to collect maps into
        target_resolutions: Only hook layers at these resolutions (e.g. [16, 32]).
            None = hook all cross-attention layers.

    Returns:
        original_processors: dict of original processors for restoration
    """
    original_processors = dict(unet.attn_processors)
    new_processors = {}

    hooked = 0
    for name, proc in unet.attn_processors.items():
        if "attn2" in name:
            layer_name = name.replace(".processor", "")
            res = _get_resolution(layer_name)
            if target_resolutions is None or res in target_resolutions:
                new_processors[name] = StoreCrossAttnProcessor(store, layer_name)
                hooked += 1
            else:
                new_processors[name] = proc
        else:
            new_processors[name] = proc

    unet.set_attn_processor(new_processors)
    print(f"  [attn_store] Hooked {hooked} cross-attention layers"
          f" (resolutions={target_resolutions or 'all'})")
    return original_processors


def restore_original_processors(unet, original_processors: Dict[str, Any]):
    """Restore original attention processors (for cleanup between grid combos)."""
    unet.set_attn_processor(original_processors)


def find_token_indices(
    prompt: str,
    keywords: List[str],
    tokenizer,
) -> List[int]:
    """
    Find token positions in the tokenized prompt that correspond to keywords.

    Handles BPE multi-subword keywords via subsequence matching:
    tokenize("photorealistic") -> [photo, realistic] -> finds that subsequence
    in the full prompt token IDs.

    Args:
        prompt: The full text prompt
        keywords: List of keywords (e.g. ["weapon", "attacking", "nude"])
        tokenizer: CLIPTokenizer from the pipeline

    Returns:
        Sorted list of token position indices (0-indexed, BOS at position 0)
    """
    # Tokenize the full prompt (padded to 77)
    prompt_enc = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    prompt_ids = prompt_enc.input_ids[0].tolist()

    matched_indices = set()

    for keyword in keywords:
        # Tokenize keyword alone, strip BOS and EOS
        kw_enc = tokenizer(keyword, return_tensors="pt")
        kw_ids = kw_enc.input_ids[0].tolist()
        # Remove BOS (49406) and EOS (49407)
        kw_inner = [t for t in kw_ids
                     if t != tokenizer.bos_token_id
                     and t != tokenizer.eos_token_id]

        if not kw_inner:
            continue

        # Subsequence search
        for i in range(len(prompt_ids) - len(kw_inner) + 1):
            if prompt_ids[i : i + len(kw_inner)] == kw_inner:
                for j in range(len(kw_inner)):
                    matched_indices.add(i + j)

    return sorted(matched_indices)


def compute_attention_mask(
    attention_store: AttentionStore,
    token_indices: List[int],
    target_resolution: int = 64,
    threshold: float = 0.3,
    soft: bool = False,
    use_cond_only: bool = True,
    heads: int = 8,
) -> torch.Tensor:
    """
    Aggregate cross-attention maps for specified tokens into a spatial mask.

    Aggregation:
      1. Per layer: extract attention for target tokens, average across heads
      2. Max across target tokens (union of attended regions)
      3. Group layers by native resolution, average within group
      4. Bilinear interpolate all groups to target_resolution
      5. Average across resolution groups (equal weight per tier)
      6. Normalize to [0,1], optionally threshold

    Args:
        attention_store: AttentionStore with maps from latest UNet forward
        token_indices: Positions of harmful tokens (from find_token_indices)
        target_resolution: Output spatial size (64 for latent space)
        threshold: For binary mask mode
        soft: If True, return continuous mask; if False, binary
        use_cond_only: Use only conditional half of CFG batch
        heads: Number of attention heads (8 for SD1.4)

    Returns:
        mask: (B, 1, target_resolution, target_resolution) spatial mask
    """
    maps = attention_store.get_maps()

    if not maps:
        return None

    device = next(iter(maps.values())).device

    if not token_indices:
        # No harmful tokens -> all-ones mask (no spatial restriction)
        B = next(iter(maps.values())).shape[0] // heads
        if use_cond_only:
            B = B // 2
        return torch.ones(B, 1, target_resolution, target_resolution,
                          device=device)

    # Collect spatial maps grouped by resolution
    by_resolution: Dict[int, List[torch.Tensor]] = {}

    for layer_name, attn_probs in maps.items():
        # attn_probs: (B_cfg * H, spatial_seq, 77)
        total_bh = attn_probs.shape[0]

        if use_cond_only:
            # CFG: first half = unconditional, second half = conditional
            B_full = total_bh // heads  # B_cfg (e.g. 2 for 1 prompt with CFG)
            B = B_full // 2
            cond_attn = attn_probs[B * heads :]  # (B*H, spatial_seq, 77)
        else:
            B = total_bh // heads
            cond_attn = attn_probs

        # Extract attention for harmful tokens
        token_attn = cond_attn[:, :, token_indices]  # (B*H, spatial_seq, num_tokens)

        # Average across heads: (B, H, spatial_seq, num_tokens)
        token_attn = token_attn.view(B, heads, -1, len(token_indices))
        token_attn = token_attn.mean(dim=1)  # (B, spatial_seq, num_tokens)

        # Max across harmful tokens (union of harmful regions)
        token_max = token_attn.max(dim=-1).values  # (B, spatial_seq)

        # Reshape to spatial grid
        spatial_seq = token_max.shape[-1]
        res = int(math.sqrt(spatial_seq))
        spatial_map = token_max.view(B, 1, res, res)

        # Interpolate to target resolution
        if res != target_resolution:
            spatial_map = F.interpolate(
                spatial_map,
                size=(target_resolution, target_resolution),
                mode="bilinear",
                align_corners=False,
            )

        layer_res = _get_resolution(layer_name)
        if layer_res not in by_resolution:
            by_resolution[layer_res] = []
        by_resolution[layer_res].append(spatial_map)

    # Average within each resolution group, then average across groups
    resolution_means = []
    for res, spatial_maps in by_resolution.items():
        group_mean = torch.stack(spatial_maps, dim=0).mean(dim=0)
        resolution_means.append(group_mean)

    mask = torch.stack(resolution_means, dim=0).mean(dim=0)  # (B, 1, H, W)

    # Normalize to [0, 1] per sample
    max_val = mask.amax(dim=(2, 3), keepdim=True) + 1e-8
    mask = mask / max_val

    if soft:
        return mask
    else:
        return (mask > threshold).float()


def detect_harmful_tokens(
    classifier,
    z0_hat: torch.Tensor,
    attention_store: AttentionStore,
    gradcam_layer: str = "layer2",
    heads: int = 8,
    top_k: int = 5,
    use_cond_only: bool = True,
) -> Tuple[List[int], torch.Tensor]:
    """
    Automatically detect which text tokens contribute most to harmful content
    by computing spatial correlation between each token's cross-attention map
    and the classifier's GradCAM heatmap.

    Args:
        classifier: LatentResNet18Classifier with compute_gradcam()
        z0_hat: (B, 4, 64, 64) predicted clean latent
        attention_store: AttentionStore with maps from latest UNet forward
        gradcam_layer: Which classifier layer for GradCAM
        heads: Number of attention heads (8 for SD1.4)
        top_k: Return top-K most harmful tokens
        use_cond_only: Use only conditional half of CFG batch

    Returns:
        harmful_indices: sorted list of token indices (most harmful first)
        token_scores: (77,) correlation score per token
    """
    maps = attention_store.get_maps()
    if not maps:
        return [], torch.zeros(77)

    # 1. Classifier GradCAM: where is the harmful content?
    with torch.enable_grad():
        gradcam = classifier.compute_gradcam(
            z0_hat.detach(), target_class=2, layer_name=gradcam_layer
        )  # (B, 1, H, W) normalized [0,1]

    device = gradcam.device
    return _correlate_tokens_with_heatmap(
        gradcam, attention_store, heads, top_k, use_cond_only, device
    )


def detect_harmful_tokens_from_heatmap(
    gradcam_heatmap: torch.Tensor,
    attention_store: AttentionStore,
    heads: int = 8,
    top_k: int = 5,
    use_cond_only: bool = True,
) -> Tuple[List[int], torch.Tensor]:
    """
    Adapter for SoftDelete+CG: takes pre-computed GradCAM heatmap instead of
    calling classifier.compute_gradcam() directly.

    Args:
        gradcam_heatmap: (B, H, W) or (B, 1, H, W) heatmap from ClassifierGradCAM
        attention_store: AttentionStore with maps from latest UNet forward
        heads: Number of attention heads
        top_k: Return top-K most harmful tokens
        use_cond_only: Use only conditional half of CFG batch

    Returns:
        harmful_indices: sorted list of token indices
        token_scores: (77,) correlation score per token
    """
    if gradcam_heatmap.dim() == 3:
        gradcam_heatmap = gradcam_heatmap.unsqueeze(1)  # (B, 1, H, W)
    device = gradcam_heatmap.device
    return _correlate_tokens_with_heatmap(
        gradcam_heatmap, attention_store, heads, top_k, use_cond_only, device
    )


def _correlate_tokens_with_heatmap(
    gradcam: torch.Tensor,
    attention_store: AttentionStore,
    heads: int,
    top_k: int,
    use_cond_only: bool,
    device: torch.device,
) -> Tuple[List[int], torch.Tensor]:
    """Shared correlation logic for detect_harmful_tokens variants."""
    token_scores = torch.zeros(77, device=device)
    n_layers = 0

    for layer_name, attn_probs in maps.items():
        # attn_probs: (B_cfg * H, spatial_seq, 77)
        total_bh = attn_probs.shape[0]

        if use_cond_only:
            B_full = total_bh // heads
            B = B_full // 2
            cond_attn = attn_probs[B * heads:]  # (B*H, spatial, 77)
        else:
            B = total_bh // heads
            cond_attn = attn_probs

        # Average across heads: (B, spatial, 77)
        cond_attn = cond_attn.view(B, heads, -1, 77).mean(dim=1)

        spatial_seq = cond_attn.shape[1]
        res = int(math.sqrt(spatial_seq))

        # Reshape all tokens at once: (B, 77, res, res)
        all_token_maps = cond_attn.permute(0, 2, 1).view(B, 77, res, res)

        # Interpolate to GradCAM resolution
        gcam_h, gcam_w = gradcam.shape[2], gradcam.shape[3]
        if res != gcam_h:
            all_token_maps = F.interpolate(
                all_token_maps, size=(gcam_h, gcam_w),
                mode="bilinear", align_corners=False,
            )  # (B, 77, H, W)

        # Correlation: sum over batch and spatial dims of (token_map × gradcam)
        # gradcam: (B, 1, H, W) → (B, 1, H, W)
        # all_token_maps: (B, 77, H, W)
        corr = (all_token_maps * gradcam).sum(dim=(0, 2, 3))  # (77,)
        token_scores += corr
        n_layers += 1

    if n_layers > 0:
        token_scores /= n_layers

    # Filter: skip BOS (idx=0), EOS, padding tokens (score <= 0)
    valid_scores = token_scores.clone()
    valid_scores[0] = -1.0  # exclude BOS

    # Top-K
    _, top_indices = valid_scores.topk(min(top_k, 77))
    harmful_indices = [i.item() for i in top_indices if token_scores[i] > 0]

    return harmful_indices, token_scores.cpu()

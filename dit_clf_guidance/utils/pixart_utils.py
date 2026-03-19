"""
PixArt-alpha specific utilities.

Encapsulates all differences between PixArt DiT and SD1.4 UNet:
  - Transformer (PixArtTransformer2DModel) instead of UNet
  - T5 text encoder instead of CLIP
  - Learned sigma: transformer outputs 8ch -> chunk to 4ch epsilon
  - added_cond_kwargs for resolution/aspect_ratio conditioning
"""

import torch
from diffusers import AutoencoderKL, DDIMScheduler


def load_pixart_components(model_id="PixArt-alpha/PixArt-XL-2-512x512",
                           device="cuda", dtype=torch.float16):
    """
    Load all PixArt model components separately (not as a pipeline).

    Returns:
        dict with keys: vae, transformer, tokenizer, text_encoder, scheduler
    """
    from diffusers import PixArtTransformer2DModel
    from transformers import T5Tokenizer, T5EncoderModel

    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    vae.to(device).eval().requires_grad_(False)

    transformer = PixArtTransformer2DModel.from_pretrained(
        model_id, subfolder="transformer"
    )
    transformer.to(device, dtype=dtype).eval().requires_grad_(False)

    tokenizer = T5Tokenizer.from_pretrained(model_id, subfolder="tokenizer")

    text_encoder = T5EncoderModel.from_pretrained(
        model_id, subfolder="text_encoder"
    )
    text_encoder.to(device, dtype=dtype).eval().requires_grad_(False)

    scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")

    return {
        "vae": vae,
        "transformer": transformer,
        "tokenizer": tokenizer,
        "text_encoder": text_encoder,
        "scheduler": scheduler,
    }


def encode_prompt(tokenizer, text_encoder, prompt, device,
                  max_sequence_length=120, dtype=torch.float16):
    """
    Encode a text prompt using T5 text encoder.

    Unlike CLIP (SD1.4), T5 requires an attention_mask to be passed
    to the PixArt transformer.

    Args:
        prompt: str or list[str]

    Returns:
        prompt_embeds: (B, seq_len, 4096)
        prompt_attention_mask: (B, seq_len)
    """
    if isinstance(prompt, str):
        prompt = [prompt]

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        prompt_embeds = text_encoder(
            text_inputs.input_ids,
            attention_mask=text_inputs.attention_mask,
        )[0].to(dtype=dtype)

    return prompt_embeds, text_inputs.attention_mask


def pixart_forward(transformer, latents, timestep,
                   prompt_embeds, prompt_attention_mask,
                   added_cond_kwargs=None):
    """
    Single forward pass through PixArt DiT transformer.

    Handles learned sigma: the transformer outputs (B, 8, H, W),
    where the first 4 channels are the epsilon prediction and
    the last 4 are the learned sigma (discarded during inference).

    Args:
        transformer: PixArtTransformer2DModel
        latents: (B, 4, H, W) noisy latent
        timestep: scalar tensor or (B,) tensor
        prompt_embeds: (B, seq_len, 4096) T5 hidden states
        prompt_attention_mask: (B, seq_len) attention mask
        added_cond_kwargs: dict with resolution/aspect_ratio (None for 512px)

    Returns:
        noise_pred: (B, 4, H, W) epsilon prediction only
    """
    if added_cond_kwargs is None:
        added_cond_kwargs = {"resolution": None, "aspect_ratio": None}

    model_output = transformer(
        latents,
        encoder_hidden_states=prompt_embeds,
        encoder_attention_mask=prompt_attention_mask,
        timestep=timestep,
        added_cond_kwargs=added_cond_kwargs,
        return_dict=False,
    )[0]

    # Handle learned sigma: output is (B, 8, H, W), take first 4ch = epsilon
    if model_output.shape[1] != latents.shape[1]:
        noise_pred = model_output.chunk(2, dim=1)[0]
    else:
        noise_pred = model_output

    return noise_pred

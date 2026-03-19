from transformers.modeling_outputs import BaseModelOutputWithPooling
from typing import Callable, List, Optional, Union
import torch

from diffusers import StableDiffusionPipeline
from diffusers.utils import logging
import torch.nn.functional as F

import math

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# -----------------------
# helpers
# -----------------------
def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def f_beta(z: float, btype: str = "sigmoid", upperbound_timestep: int = 10, concept_type: str = "nudity") -> int:
    # Handle NaN or Inf (can happen with very long prompts exceeding token limit)
    if math.isnan(z) or math.isinf(z):
        return 0

    if "vangogh" in concept_type:
        t = 5.5
        k = 3.5
    else:
        t = 5.333
        k = 2.5

    if btype == "tanh":
        _value = math.tanh(k * (10 * z - t))
        output = round(upperbound_timestep / 2.0 * (_value + 1))
    elif btype == "sigmoid":
        sigmoid_scale = 2.0
        _value = sigmoid(sigmoid_scale * k * (10 * z - t))
        output = round(upperbound_timestep * (_value))
    else:
        raise NotImplementedError("btype is incorrect")
    return int(output)


def projection_matrix(E: torch.Tensor) -> torch.Tensor:
    """
    Calculate the projection matrix onto the subspace spanned by columns of E.
    - 내부는 FP32로 연산(핀버스 안정성), 반환 시 원래 dtype으로 복귀
    """
    orig_dtype = E.dtype
    E32 = E.float()
    gram = E32.T @ E32
    eps = 1e-6
    eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    gram_reg = gram + eps * eye
    P32 = E32 @ torch.pinverse(gram_reg) @ E32.T
    return P32.to(orig_dtype)


def projection_and_orthogonal(
    input_embeddings: torch.Tensor,
    masked_input_subspace_projection: torch.Tensor,
    concept_subspace_projection: torch.Tensor,
) -> torch.Tensor:
    """
    (I - cs) @ ms 로 직교/투영한 새로운 텍스트 임베딩 생성.
    dtype 일치/안정성 보장.
    """
    ie = input_embeddings
    ms = masked_input_subspace_projection
    cs = concept_subspace_projection

    device = ie.device
    out_dtype = ie.dtype

    # 공통 연산 dtype(FP32)로 계산
    ms32 = ms.float()
    cs32 = cs.float()

    uncond_e, text_e = ie.chunk(2)      # [1, L, D] each
    text_e_s = text_e.squeeze(0)        # [L, D]
    L, D = text_e_s.shape

    I32 = torch.eye(D, device=device, dtype=torch.float32)
    # ★ FP32로 캐스팅하여 곱셈
    text_e_s32 = text_e_s.float()
    new_text_e32 = (I32 - cs32) @ ms32 @ text_e_s32.T  # [D, L]
    new_text_e32 = new_text_e32.T                      # [L, D]
    new_text_e = new_text_e32.to(out_dtype)

    new_embeddings = torch.cat([uncond_e, new_text_e.unsqueeze(0)], dim=0)  # [2, L, D]
    return new_embeddings



def safree_projection(
    input_embeddings: torch.Tensor,
    p_emb: torch.Tensor,
    masked_input_subspace_projection: torch.Tensor,
    concept_subspace_projection: torch.Tensor,
    alpha: float = 0.0,
    max_length: int = 77,
    logger=None,
) -> torch.Tensor:
    """
    토큰별 거리 기반으로 안전 토큰 유지/트리거 토큰 대체.
    모든 선형대수 연산은 FP32로 진행 후, 최종 결과는 원 dtype으로 복귀.
    """
    ie = input_embeddings
    ms = masked_input_subspace_projection
    cs = concept_subspace_projection

    device = ie.device
    out_dtype = ie.dtype

    n_t, D = p_emb.shape

    # FP32에서 계산
    p32 = p_emb.float()
    ms32 = ms.float()
    cs32 = cs.float()

    I32 = torch.eye(D, device=device, dtype=torch.float32)
    I_m_cs32 = I32 - cs32

    # 거리 벡터
    dist_vec = I_m_cs32 @ p32.T              # [D, n_t]
    dist_p_emb = torch.norm(dist_vec, dim=0) # [n_t]

    # Leave-one-out 평균
    if n_t > 1:
        sum_all = dist_p_emb.sum()
        mean_dist = (sum_all - dist_p_emb) / (n_t - 1)
    else:
        mean_dist = dist_p_emb.clone()

    rm_vector = (dist_p_emb < (1.0 + alpha) * mean_dist)   # True: safe token
    n_removed = int(n_t - rm_vector.sum().item())
    if logger is not None:
        logger.log(f"Among {n_t} tokens, we remove {n_removed}.")
    else:
        print(f"Among {n_t} tokens, we remove {n_removed}.")

    # 길이 맞춤 마스크 (bool)
    mask_bool = torch.zeros(max_length, device=device, dtype=torch.bool)
    mask_bool[1 : n_t + 1] = rm_vector  # [CLS] = index 0 가정

    # 원/대체 임베딩 계산
    uncond_e, text_e = ie.chunk(2)   # [1, L, D] each
    text_e_s = text_e.squeeze(0)     # [L, D]
    L, D = text_e_s.shape

    # ★ FP32로 캐스팅하여 곱셈
    text_e_s32 = text_e_s.float()
    new_text_e32 = I_m_cs32 @ ms32 @ text_e_s32.T  # [D, L]
    new_text_e = new_text_e32.T.to(out_dtype)      # [L, D]

    # where: True → 원본 유지, False → 대체 적용
    mask_2d = mask_bool.unsqueeze(1).expand_as(text_e_s)   # [L, D]
    merged_text_e = torch.where(mask_2d, text_e_s, new_text_e).to(out_dtype)  # [L, D]

    new_embeddings = torch.cat([uncond_e, merged_text_e.unsqueeze(0)], dim=0)  # [2, L, D]
    return new_embeddings



# -----------------------
# pipeline
# -----------------------
class ModifiedStableDiffusionPipeline(StableDiffusionPipeline):
    def __init__(
        self,
        vae,
        text_encoder,
        tokenizer,
        unet,
        scheduler,
        safety_checker,
        feature_extractor,
        image_encoder=None,
        requires_safety_checker: bool = True,
    ):
        super(ModifiedStableDiffusionPipeline, self).__init__(
            vae,
            text_encoder,
            tokenizer,
            unet,
            scheduler,
            safety_checker,
            feature_extractor,
            image_encoder=image_encoder,
            requires_safety_checker=requires_safety_checker,
        )

    def _build_causal_attention_mask(self, bsz, seq_len, dtype):
        # additive mask; fill with -inf above diagonal
        mask = torch.empty(bsz, seq_len, seq_len, dtype=dtype)
        mask.fill_(torch.tensor(torch.finfo(dtype).min))
        mask.triu_(1)
        mask = mask.unsqueeze(1)
        return mask

    def _encode_embeddings(self, prompt, prompt_embeddings, attention_mask=None):
        output_attentions = self.text_encoder.text_model.config.output_attentions
        output_hidden_states = self.text_encoder.text_model.config.output_hidden_states
        return_dict = self.text_encoder.text_model.config.use_return_dict

        hidden_states = self.text_encoder.text_model.embeddings(inputs_embeds=prompt_embeddings)

        bsz, seq_len = prompt.shape[0], prompt.shape[1]
        causal_attention_mask = self._build_causal_attention_mask(bsz, seq_len, hidden_states.dtype).to(
            hidden_states.device
        )

        if attention_mask is not None:
            attention_mask = self.text_encoder.text_model._expand_mask(attention_mask, hidden_states.dtype)

        encoder_outputs = self.text_encoder.text_model.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.text_encoder.text_model.final_layer_norm(last_hidden_state)

        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=prompt.device), prompt.to(torch.int).argmax(dim=-1)
        ]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def _new_encode_negative_prompt_space(
        self, negative_prompt_space, max_length, num_images_per_prompt, pooler_output: bool = True
    ):
        device = self._execution_device
        # negative_prompt_space: str 또는 List[str] 모두 허용
        uncond_input = self.tokenizer(
            negative_prompt_space,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        uncond_embeddings = self.text_encoder(
            uncond_input.input_ids.to(device),
            attention_mask=uncond_input.attention_mask.to(device),
        )
        if not pooler_output:
            uncond_embeddings = uncond_embeddings[0]
            bs_embed, seq_len, _ = uncond_embeddings.shape
            uncond_embeddings = uncond_embeddings.repeat(1, num_images_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)
        else:
            uncond_embeddings = uncond_embeddings.pooler_output
        return uncond_embeddings

    def _masked_encode_prompt(self, prompt: Union[str, List[str]]):
        device = self._execution_device
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        n_real_tokens = untruncated_ids.shape[1] - 2

        if untruncated_ids.shape[1] > self.tokenizer.model_max_length:
            untruncated_ids = untruncated_ids[:, : self.tokenizer.model_max_length]
            n_real_tokens = self.tokenizer.model_max_length - 2

        masked_ids = untruncated_ids.repeat(n_real_tokens, 1)
        for i in range(n_real_tokens):
            masked_ids[i, i + 1] = 0

        masked_embeddings = self.text_encoder(masked_ids.to(device), attention_mask=None)
        return masked_embeddings.pooler_output  # [n_real_tokens, D]

    def _new_encode_prompt(
        self,
        prompt,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt,
        prompt_ids=None,
        prompt_embeddings=None,
        token_mask=None,
        debug: bool = False,
    ):
        """
        텍스트 인코딩. prompt_embeddings 경로/일반 경로 모두 안전 처리.
        """
        batch_size = len(prompt) if isinstance(prompt, list) else 1
        device = self._execution_device

        if prompt_embeddings is not None:
            # 외부에서 임베딩 직접 주는 경우
            attention_mask = None
            text_embeddings = self._encode_embeddings(prompt_ids, prompt_embeddings, attention_mask=attention_mask)
            text_input_ids = prompt_ids
            text_inputs = None
        else:
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            if getattr(self.text_encoder.config, "use_attention_mask", False):
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None
            # token_mask 기능(옵션)
            if token_mask is not None:
                mask_iids = torch.where(token_mask == 0, torch.zeros_like(token_mask), text_input_ids[0].to(device)).int()
                mask_iids = mask_iids[mask_iids != 0]
                tmp_ones = torch.ones_like(token_mask) * 49407
                tmp_ones[: len(mask_iids)] = mask_iids
                text_input_ids = tmp_ones.int()
                text_input_ids = text_input_ids[None, :]

            text_embeddings = self.text_encoder(text_input_ids.to(device), attention_mask=attention_mask)

        text_embeddings = text_embeddings[0]  # hidden states

        # duplicate for num_images_per_prompt
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # unconditional embeddings (CFG)
        if do_classifier_free_guidance:
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            else:
                if type(prompt) is not type(negative_prompt):
                    raise TypeError(f"`negative_prompt` type should match `prompt` type.")
                if isinstance(negative_prompt, str):
                    uncond_tokens = [negative_prompt]
                else:
                    if batch_size != len(negative_prompt):
                        raise ValueError("`negative_prompt` batch size must match `prompt`.")
                    uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens, padding="max_length", max_length=max_length, truncation=True, return_tensors="pt"
            )
            if getattr(self.text_encoder.config, "use_attention_mask", False):
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(device), attention_mask=attention_mask)[0]
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_images_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_images_per_prompt, seq_len, -1)

            text_embeddings = torch.cat([uncond_embeddings, text_embeddings], dim=0)

        # attention mask 반환 (없으면 None)
        attn_mask = text_inputs.attention_mask if (text_inputs is not None) else None
        return text_embeddings, text_input_ids, attn_mask

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_space: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        prompt_ids=None,
        prompt_embeddings=None,
        return_latents: bool = False,
        safree_dict: dict = {},
    ):
        # 0) defaults
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        sf = safree_dict or {}
        # 안전 가드
        sf.setdefault("safree", False)
        sf.setdefault("svf", False)
        sf.setdefault("lra", False)
        sf.setdefault("alpha", 0.0)
        sf.setdefault("re_attn_t", [-1, 10000])
        sf.setdefault("up_t", 10)
        sf.setdefault("category", "nudity")
        sf.setdefault("logger", None)

        # 1) inputs check
        self.check_inputs(prompt, height, width, callback_steps, prompt_embeds=prompt_embeddings)

        # 2) flags
        batch_size = 1  # 현재 파이프라인 사용 패턴상 1로 고정
        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3) encode prompt
        text_embeddings, text_input_ids, attention_mask = self._new_encode_prompt(
            prompt, num_images_per_prompt, do_classifier_free_guidance, negative_prompt, prompt_ids, prompt_embeddings
        )

        # 개념/마스크 투영행렬 준비(필요 시)
        # negative_prompt_space가 None이면 negative_prompt 또는 ""로 대체
        negspace_tokens = negative_prompt_space if negative_prompt_space is not None else (negative_prompt or "")
        negspace_text_embeddings = self._new_encode_negative_prompt_space(
            negspace_tokens, max_length=self.tokenizer.model_max_length, num_images_per_prompt=num_images_per_prompt
        )
        project_matrix = projection_matrix(negspace_text_embeddings.T)

        masked_embs = self._masked_encode_prompt(prompt)  # [n_t, D]
        masked_project_matrix = projection_matrix(masked_embs.T)

        if sf.get("safree", False):
            rescaled_text_embeddings = safree_projection(
                text_embeddings,
                masked_embs,
                masked_project_matrix,
                project_matrix,
                alpha=sf["alpha"],
                max_length=self.tokenizer.model_max_length,
                logger=sf.get("logger", None),
            )
        else:
            rescaled_text_embeddings = text_embeddings

        if sf.get("svf", False):
            proj_ort = projection_and_orthogonal(text_embeddings, masked_project_matrix, project_matrix)

            _, text_e = text_embeddings.chunk(2)
            s_attn_mask = (attention_mask.squeeze() == 1) if (attention_mask is not None) else torch.ones(
                text_e.shape[1], dtype=torch.bool, device=text_e.device
            )

            text_e = text_e.squeeze(0)
            _, proj_ort_e = proj_ort.chunk(2)
            proj_ort_e = proj_ort_e.squeeze(0)

            proj_ort_e_act = proj_ort_e[s_attn_mask]
            text_e_act = text_e[s_attn_mask]
            # sim_org_onp_act = F.cosine_similarity(proj_ort_e_act, text_e_act)
            sim_org_onp_act = F.cosine_similarity(proj_ort_e_act.float(), text_e_act.float(), dim=-1)

            # Handle NaN in similarity (can happen with long prompts or edge cases)
            if torch.isnan(sim_org_onp_act).any():
                sim_org_onp_act = torch.nan_to_num(sim_org_onp_act, nan=0.0)

            beta = (1 - sim_org_onp_act.mean().item())
            beta_adjusted = f_beta(beta, upperbound_timestep=sf["up_t"], concept_type=sf["category"])

            # Safeguard: beta ≈ 1.0 means projection collapsed (common with long prompts > 77 tokens)
            # In this case, skip rescaled embeddings entirely to avoid NaN/black images
            if beta > 0.95:
                _msg = (f"[SAFREE WARNING] beta={beta:.4f} > 0.95 — projection unstable "
                        f"(likely long prompt). Falling back to original embeddings.")
                if sf.get("logger", None) is not None:
                    sf["logger"].log(_msg)
                else:
                    print(_msg)
                beta_adjusted = -1

            # Also check rescaled embeddings for NaN
            if torch.isnan(rescaled_text_embeddings).any():
                _msg = "[SAFREE WARNING] NaN in rescaled embeddings — falling back to original."
                if sf.get("logger", None) is not None:
                    sf["logger"].log(_msg)
                else:
                    print(_msg)
                rescaled_text_embeddings = text_embeddings
                beta_adjusted = -1

            if sf.get("logger", None) is not None:
                sf["logger"].log(f"beta : {beta}, adjusted_beta: {beta_adjusted}")
        else:
            beta_adjusted = -1  # 사용 안 함

        # 4) timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5) latents
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            text_embeddings.dtype,
            device,
            generator,
            latents,
        )

        # 6) extra kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7) denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand latents
                if sf.get("lra", False):
                    latent_model_input = torch.cat([latents] * 3) if do_classifier_free_guidance else latents
                else:
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # choose embeddings
                if sf.get("svf", False):
                    use_rescaled = sf.get("safree", False) and (i <= beta_adjusted)
                else:
                    lo, hi = sf.get("re_attn_t", [-1, 1000000])
                    use_rescaled = sf.get("safree", False) and (lo <= i <= hi)

                _text_embeddings = rescaled_text_embeddings if use_rescaled else text_embeddings

                # unet
                if sf.get("lra", False):
                    _, text_e = text_embeddings.chunk(2)
                    combined_text_embeddings = torch.cat([_text_embeddings, text_e], dim=0)  # [3*B, L, D]
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=combined_text_embeddings).sample
                else:
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=_text_embeddings).sample

                # CFG
                if do_classifier_free_guidance:
                    if sf.get("lra", False):
                        noise_pred_uncond, noise_pred_text, _ = noise_pred.chunk(3)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    else:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # step
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # callback
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % (callback_steps or 1) == 0:
                        callback(i, t, latents)

        if return_latents:
            return latents

        # 8) decode
        image = self.decode_latents(latents)

        # 9) (옵션) safety checker 호출 비활성화
        # image, has_nsfw_concept = self.run_safety_checker(image, device, text_embeddings.dtype)

        # 10) to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        return image

# geo_utils/custom_sdxl_lightning.py

# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from diffusers import StableDiffusionXLPipeline
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    StableDiffusionXLPipelineOutput,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from diffusers.utils import deprecate
from diffusers.callbacks import PipelineCallback, MultiPipelineCallbacks
from diffusers.utils.torch_utils import randn_tensor

class CustomStableDiffusionXLPipeline(StableDiffusionXLPipeline):
    # 유저가 callback_on_step_end 에 요청할 수 있는 텐서
    _callback_tensor_inputs = ["latents", "prev_latents", "noise_pred"]

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[Any] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[
            Callable[[Any, int, int, Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ) -> StableDiffusionXLPipelineOutput:
        # deprecated callback_steps
        callback_steps = kwargs.pop("callback_steps", None)
        if callback_steps is not None:
            deprecate("callback_steps", "1.0.0", "Use callback_on_step_end instead")

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 0) 기본 height/width
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 1) 입력 검증
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        # 2) 속성 세팅
        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._denoising_end = denoising_end
        self._interrupt = False

        # 3) 배치 크기 계산
        if prompt is not None and isinstance(prompt, str):
            bs = 1
        elif prompt is not None:
            bs = len(prompt)
        else:
            bs = prompt_embeds.shape[0]
        batch_size = bs * num_images_per_prompt

        device = self._execution_device

        # 4) 텍스트 인코딩
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt,
            prompt_2,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=None,
            clip_skip=self.clip_skip,
        )

        # 5) timesteps 획득
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )
        # 내부 카운터 리셋
        if hasattr(self.scheduler, "_step_index"):
            self.scheduler._step_index = 0

        # 6) Latents 초기화
        latents = self.prepare_latents(
            batch_size,
            self.unet.config.in_channels,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 7) 추가 step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 8) micro-conditioning embeddings 준비
        text_proj_dim = (
            pooled_prompt_embeds.shape[-1]
            if self.text_encoder_2 is None
            else self.text_encoder_2.config.projection_dim
        )
        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_proj_dim,
        )
        if negative_original_size is not None and negative_target_size is not None:
            neg_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_proj_dim,
            )
        else:
            neg_time_ids = add_time_ids

        add_text_embeds = pooled_prompt_embeds
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat(
                [negative_prompt_embeds, prompt_embeds], dim=0
            )
            add_text_embeds = torch.cat(
                [negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0
            )
            add_time_ids = torch.cat([neg_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size, 1)

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size,
                self.do_classifier_free_guidance,
            )

        # 9) denoising loop
        prev_latents = None
        with self.progress_bar(total=num_inference_steps) as progress:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # 9.1) classifier-free guidance을 위한 확장
                latent_input = (
                    torch.cat([latents, latents], dim=0)
                    if self.do_classifier_free_guidance
                    else latents
                )
                latent_input = self.scheduler.scale_model_input(latent_input, t)

                # 9.2) noise 예측
                noise_pred = self.unet(
                    latent_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=(
                        self.get_guidance_scale_embedding(
                            torch.tensor(self.guidance_scale - 1)
                            .repeat(batch_size),
                            embedding_dim=self.unet.config.time_cond_proj_dim,
                        )
                        .to(device, latents.dtype)
                        if self.unet.config.time_cond_proj_dim is not None
                        else None
                    ),
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    added_cond_kwargs=dict(
                        text_embeds=add_text_embeds,
                        time_ids=add_time_ids,
                        **(
                            {"image_embeds": image_embeds}
                            if ip_adapter_image is not None
                            or ip_adapter_image_embeds is not None
                            else {}
                        ),
                    ),
                    return_dict=False,
                )[0]

                # 9.3) classifier-free guidance 적용
                if self.do_classifier_free_guidance:
                    uncond, cond = noise_pred.chunk(2)
                    noise_pred = uncond + self.guidance_scale * (cond - uncond)
                    if self.guidance_rescale > 0:
                        noise_pred = self.rescale_noise_cfg(
                            noise_pred, cond, self.guidance_rescale
                        )

                # 9.4) 스텝 업데이트
                prev_latents = latents
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]

                # 9.5) 유저 콜백
                if callback_on_step_end is not None:
                    cb_kwargs: Dict[str, torch.Tensor] = {}
                    for name in callback_on_step_end_tensor_inputs or []:
                        if name == "latents":
                            cb_kwargs[name] = latents
                        elif name == "prev_latents":
                            cb_kwargs[name] = prev_latents
                        elif name == "noise_pred":
                            cb_kwargs[name] = noise_pred

                    out = callback_on_step_end(self, i, t, cb_kwargs)
                    if "latents" in out:
                        latents = out["latents"]
                    if "noise_pred" in out:
                        noise_pred = out["noise_pred"]

                progress.update()

        # 10) Decode & 후처리 (배치 전체)
        decoded = self.vae.decode(latents / self.vae.config.scaling_factor).sample
        images = self.image_processor.postprocess(
            decoded, output_type=output_type, do_denormalize=[True] * decoded.shape[0]
        )

        return StableDiffusionXLPipelineOutput(images=images)

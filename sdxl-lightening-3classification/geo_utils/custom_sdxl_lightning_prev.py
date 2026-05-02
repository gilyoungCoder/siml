import torch
from typing import Any, Dict, List, Optional, Union, Callable

from diffusers import (
    StableDiffusionXLPipeline,
    EulerDiscreteScheduler,
)
from diffusers.callbacks import PipelineCallback, MultiPipelineCallbacks
from diffusers.image_processor import VaeImageProcessor, PipelineImageInput
# 여기서는 그냥 V1 파이프라인 Output 사용
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipelineOutput
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

class CustomStableDiffusionXLPipeline(StableDiffusionXLPipeline):
    """SDXL-Lightning 위에 classifier guidance 콜백만 얹을 때 쓰는 Custom Pipeline"""

    # guidance 콜백에 넘길 텐서 리스트
    _callback_tensor_inputs = [
        "latents", "prompt_embeds", "negative_prompt_embeds",
        "noise_pred", "prev_latents",
        "instance_prompt_embeds", "bbox_binary_mask",
    ]

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 4,
        guidance_scale: float = 0.0,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[Any, int, int, Dict], Dict], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = None,
        callback_steps: int = 1,
        **kwargs,
    ):
        print(f"[DEBUG] prompt={prompt}, negative_prompt={negative_prompt}, height={height}, width={width}")

        # # 1) 입력 검증
        # self.check_inputs(
        #     prompt=prompt, height=height, width=width, callback_steps=callback_steps,
        #     callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs
        # )

        # 2) 텍스트 인코딩
        encode = self.encode_prompt(
            prompt=prompt,
            device=self._execution_device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=(guidance_scale > 1),
            negative_prompt=negative_prompt,
        )
        prompt_embeds, pooled_prompt_embeds = encode[0], encode[2]

        # 3) 타임스텝 준비
        self.scheduler.set_timesteps(num_inference_steps, device=self._execution_device)
        timesteps = self.scheduler.timesteps

        # 4) latent 준비
        latents = self.prepare_latents(
            batch_size=len(prompt_embeds),
            num_channels_latents=self.unet.config.in_channels,
            height=height or self.unet.config.sample_size * self.vae.config.scaling_factor,
            width=width or self.unet.config.sample_size * self.vae.config.scaling_factor,
            dtype=prompt_embeds.dtype,
            device=self._execution_device,
            generator=generator,
        )

        added_cond_kwargs = {
            "text_embeds": pooled_prompt_embeds
        }
        # 5) denoising + guidance 콜백
        for i, t in enumerate(timesteps):
            model_input = self.scheduler.scale_model_input(latents, t)
            noise_pred = self.unet(model_input, t, encoder_hidden_states=prompt_embeds, added_cond_kwargs=added_cond_kwargs)[0]

            if callback_on_step_end is not None and i % callback_steps == 0:
                cb_kwargs = {k: locals()[k] for k in callback_on_step_end_tensor_inputs}
                out = callback_on_step_end(self, i, t, cb_kwargs)
                latents = out.get("latents", latents)

            latents = self.scheduler.step(noise_pred, t, latents)[0]

        # 6) 이미지 디코딩 & 후처리
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        image, _ = self.run_safety_checker(image, self._execution_device, prompt_embeds.dtype)
        image = self.vae_image_processor.postprocess(image, output_type="pil")

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=None)

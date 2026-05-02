import os
import random
from argparse import ArgumentParser
from functools import partial
from PIL import Image
import torch
from accelerate import Accelerator
from geo_utils.custom_stable_diffusion import CustomStableDiffusionPipeline, CustomStableDiffusionImg2ImgPipeline
from geo_utils.guidance_utils import GuidanceModel
from geo_utils.sae_probe import SAEProbe
import numpy as np

def parse_args():
    parser = ArgumentParser(description='Generation script')
    parser.add_argument('ckpt_path', type=str)
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--nsamples', type=int, default=4)
    parser.add_argument('--cfg_scale', type=float, default=5)
    parser.add_argument('--num_inference_steps', type=int, default=50)
    parser.add_argument('--trained_text_encoder', action="store_true")
    parser.add_argument('--seed', type=int, default=1234)

    # Guidance
    parser.add_argument('--freedom', action="store_true")
    parser.add_argument('--freedom_scale', type=float, default=10.0)
    parser.add_argument('--freedom_model_type', type=str, default="classifier")
    parser.add_argument('--freedom_model_args_file', type=str, default="configs/models/classifier.yaml")
    parser.add_argument('--freedom_model_ckpt', type=str, default="classifier_ckpt/classifier.pth")
    parser.add_argument("--guide_start", type=int, default=1)

    parser.add_argument('--output_dir', type=str, default='output_img/tmp')
    parser.add_argument("--prompt_file", type=str, required=True)

    # SAE
    parser.add_argument('--sae_probe', action="store_true")
    parser.add_argument('--sae_repo', type=str, default="bcywinski/SAeUron")
    parser.add_argument('--sae_hookpoint', type=str, default="unet.up_blocks.1.attentions.1")
    parser.add_argument('--sae_topk', type=int, default=32)
    parser.add_argument('--sae_csv', type=str, default=None)
    parser.add_argument('--sae_calibrate', action="store_true")
    parser.add_argument('--sae_nudity_prompts', type=str, default=None)
    parser.add_argument('--sae_neutral_prompts', type=str, default=None)

    args = parser.parse_known_args()[0]
    return args

def save_image(image, img_metadata, root="output_img"):
    path = img_metadata['file_name']
    img_height = img_metadata['height']
    img_width = img_metadata['width']
    image = np.asarray(image)
    image = Image.fromarray(image, mode='RGB')
    image = image.resize((img_width, img_height))
    path = os.path.join(root, path[:-4] + '.png')
    image.save(path)

def main(model=None):
    args = parse_args()
    accelerator = Accelerator()
    device = accelerator.device

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    pipe = CustomStableDiffusionPipeline.from_pretrained(args.ckpt_path, safety_checker=None)
    pipe = pipe.to(device)
    print("Pipe device:", pipe.device)

    with open(args.prompt_file, "r") as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]
    print(f"Loaded {len(prompts)} prompts from {args.prompt_file}")

    sae_probe = None
    if args.sae_probe:
        sae_probe = SAEProbe(
            pipe=pipe,
            sae_repo=args.sae_repo,
            hookpoint=args.sae_hookpoint,
            device=str(device),
            topk_select=args.sae_topk,
            csv_path=args.sae_csv,
        )

    if sae_probe is not None and args.sae_calibrate:
        if args.sae_nudity_prompts is None:
            nudity_prompts = ["nude human figure", "unclothed human figure"]
        else:
            with open(args.sae_nudity_prompts, "r") as f:
                nudity_prompts = [l.strip() for l in f if l.strip()]

        if args.sae_neutral_prompts is None:
            neutral_prompts = prompts[:10] if len(prompts) >= 10 else prompts
        else:
            with open(args.sae_neutral_prompts, "r") as f:
                neutral_prompts = [l.strip() for l in f if l.strip()]

        base_kws = dict(
            guidance_scale=1.0, height=512, width=512, num_inference_steps=20,
            callback_on_step_end=None, callback_on_step_end_tensor_inputs=None,
            num_images_per_prompt=1,
        )
        sae_probe.calibrate_fc(nudity_prompts, neutral_prompts, base_kws)

    freedom_model = None
    cb_tensor_inputs_base = None
    if args.freedom:
        freedom_model = GuidanceModel(pipe, args.freedom_model_args_file, args.freedom_model_ckpt, 1, pipe.device)
        cb_tensor_inputs_base = ["latents", "noise_pred", "prev_latents"]
        print("guidance!!")

    scale = args.cfg_scale
    root = args.output_dir
    os.makedirs(root, exist_ok=True)

    # [NEW] UNet 호출 '직전' latent push 콜백
    def pre_step_start_latent_push(diffusion_pipeline, step, timestep, callback_kwargs,
                                   freedom_model, freedom_scale, guide_start=0,
                                   only_step0=True, grad_clip=None, target_class=1):
        if freedom_model is None:
            return callback_kwargs
        if step < guide_start:
            return callback_kwargs
        if only_step0 and step != 0:
            return callback_kwargs

        with torch.enable_grad():
            x = callback_kwargs["latents"].detach().requires_grad_(True)

            # scheduler 통일된 정규 timestep
            t_norm = (timestep / diffusion_pipeline.scheduler.num_train_timesteps).to(x.device)

            # TimeDependentDiscriminatorGradientModel은 noisy_input 그대로 사용
            zeros = torch.zeros_like(x)
            scaled_in = freedom_model.gradient_model.get_scaled_input(
                diffusion_pipeline, noisy_input=x, noise_pred=zeros, timestep=timestep
            )

            diff_val, grad, _ = freedom_model.gradient_model.get_grad(
                scaled_in, t_norm, encoder_hidden_states=None, grad_input=x, target_class=target_class
            )
            if grad_clip is not None:
                grad = torch.clamp(grad, -grad_clip, grad_clip)

            # step size 고정형 (필요하면 sigma 비례형으로 바꿔도 됨)
            eta = freedom_scale
            x = (x - eta * grad).detach()

        callback_kwargs["latents"] = x
        return callback_kwargs

    # 기존 step-end 콜백 (SAE 로깅 + guidance)
    def pre_freedom_iteration(diffusion_pipeline, step, timestep, callback_kwargs,
                              freedom_model, freedom_scale, guide_start=0,
                              prompt_idx=None, sae_probe=None):
        if sae_probe is not None:
            try:
                sae_probe.log_step(prompt_idx if prompt_idx is not None else -1, step, timestep)
            except Exception as e:
                print(f"[SAEProbe] log_step error: {e}")

        if (freedom_model is not None) and (guide_start <= step):
            local_scale = 15 if step <= 4 else freedom_scale
            guidance_result = freedom_model.guidance(
                diffusion_pipeline, callback_kwargs, step, timestep, local_scale, target_class=1
            )
            return guidance_result
        else:
            return callback_kwargs

    for idx, prompt in enumerate(prompts):
        print(f"Generating image for prompt {idx + 1}: {prompt}")

        if args.freedom:
            # 시작 콜백(UNet 전에 latent push: step=0에서만)
            start_cb = partial(
                pre_step_start_latent_push,
                freedom_model=freedom_model,
                freedom_scale=6.9,
                guide_start=0,
                only_step0=True,
                grad_clip=1.0,
                target_class=1,
            )
            start_cb_tensor_inputs = ["latents"]

            # 끝 콜백(SAE 로깅 + 기존 guidance)
            freedom_iteration = partial(
                pre_freedom_iteration,
                freedom_model=freedom_model,
                freedom_scale=args.freedom_scale,
                guide_start=args.guide_start,
                prompt_idx=idx,
                sae_probe=sae_probe
            )
            freedom_callback_on_step_end_tensor_input = cb_tensor_inputs_base
        elif sae_probe is not None:
            start_cb = None; start_cb_tensor_inputs = None
            freedom_iteration = partial(
                pre_freedom_iteration,
                freedom_model=None,
                freedom_scale=0.0,
                guide_start=1_000_000,
                prompt_idx=idx,
                sae_probe=sae_probe
            )
            freedom_callback_on_step_end_tensor_input = ["latents", "noise_pred", "prev_latents"]
        else:
            start_cb = None; start_cb_tensor_inputs = None
            freedom_iteration = None
            freedom_callback_on_step_end_tensor_input = None

        input_dict = {
            "prompt": prompt,
            "guidance_scale": scale,
            "num_inference_steps": args.num_inference_steps,
            "height": 512,
            "width": 512,

            # [NEW] UNet 호출 직전 latent push
            "callback_on_step_start": start_cb,
            "callback_on_step_start_tensor_inputs": start_cb_tensor_inputs,

            # 기존 step-end 콜백
            "callback_on_step_end": freedom_iteration,
            "callback_on_step_end_tensor_inputs": freedom_callback_on_step_end_tensor_input,

            "callback": None,
            "callback_steps": 1,
            "bbox_binary_mask": None,
            "num_images_per_prompt": args.nsamples,
        }

        with torch.enable_grad():
            generated_images = pipe(**input_dict).images

        img_metadata = {"file_name": f"{idx + 1}.png", "height": 512, "width": 512}
        save_image(generated_images[0], img_metadata, root=root)

    if sae_probe is not None:
        try:
            sae_probe.flush_csv()
            print("[SAEProbe] CSV saved:", args.sae_csv)
        finally:
            sae_probe.close()

if __name__ == "__main__":
    main()

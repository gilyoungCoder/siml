import os
import random
from argparse import ArgumentParser
from functools import partial
from PIL import Image
import torch
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline  # Stable Diffusion 파이프라인 사용
from geo_utils.custom_stable_diffusion import CustomStableDiffusionPipeline, CustomStableDiffusionImg2ImgPipeline

from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from geo_utils.guidance_utils import GuidanceModel
import numpy as np

########################
# Parsers
#########################
def parse_args():
    parser = ArgumentParser(description='Generation script')
    parser.add_argument('ckpt_path', type=str)
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--nsamples', type=int, default=4)
    parser.add_argument('--cfg_scale', type=float, default=5)
    parser.add_argument('--num_inference_steps', type=int, default=50)
    parser.add_argument('--trained_text_encoder', action="store_true")

    # Guidance 관련 인자
    parser.add_argument('--freedom', action="store_true", help="Use guidance with trained discriminator")
    parser.add_argument('--freedom_scale', type=float, default=10.0)
    parser.add_argument('--freedom_model_type', type=str, default="classifier")
    parser.add_argument('--freedom_model_args_file', type=str, default="configs/models/classifier.yaml")
    parser.add_argument('--freedom_model_ckpt', type=str, default="classifier_ckpt/classifier.pth")

    parser.add_argument('--output_dir', type=str, default='output_img/tmp')
    parser.add_argument("--guide_start", type=int, default=1, help="The start index for guidance")
    parser.add_argument("--guiding_class", type=int, default=1, help="guiding direction (0: benign, 1: nudity)")
    parser.add_argument("--guide_phase", type=str,  choices=['initial','middle','final','all'], default='all', help="which phase to apply guidance: initial(0~20), middle(15~35), final(30~50), or all(0~50)")
    # Prompt file argument
    parser.add_argument("--prompt_file", type=str, required=True, help="Path to the file containing prompts")

    args = parser.parse_known_args()[0]
    return args

########################
# Save Image Function
########################
def save_image(image, img_metadata, root="output_img"):
    path = img_metadata['file_name']
    img_height = img_metadata['height']
    img_width = img_metadata['width']

    image = np.asarray(image)
    image = Image.fromarray(image, mode='RGB')
    image = image.resize((img_width, img_height))
    path = os.path.join(root, path[:-4] + '.png')
    image.save(path)


########################
# Main Function
########################
def main(model=None):
    args = parse_args()
    accelerator = Accelerator()
    device = accelerator.device

    ########################
    # Build Stable Diffusion pipeline (Disable Safety Checker)
    #######################
    pipe = StableDiffusionPipeline.from_pretrained(args.ckpt_path, safety_checker=None)  # NSFW safety 체크 비활성화
    pipe = CustomStableDiffusionPipeline.from_pretrained(args.ckpt_path, safety_checker=None)  # NSFW safety 체크 비활성화

    ##수정
    # from diffusers import DDIMScheduler
    # pipe = CustomStableDiffusionPipeline.from_pretrained(args.ckpt_path, safety_checker=None)
    # # PNDM 대신 DDIM 스케줄러를 씁니다.
    # pipe.scheduler = DDIMScheduler.from_config(args.ckpt_path, subfolder="scheduler")
    # pipe.scheduler.set_timesteps(args.num_inference_steps, device=pipe.device)
    ##done

    pipe = pipe.to(device)
    print("Pipe device:", pipe.device)

    ########################
    # Load prompt file
    ########################
    with open(args.prompt_file, "r") as f:
        prompts = [line.strip() for line in f.readlines()]
    print(f"Loaded {len(prompts)} prompts from {args.prompt_file}")

    ########################
    # Setup Guidance (trained discriminator)
    ########################
    if args.freedom:
        # Initialize the GuidanceModel with trained discriminator target class = 1 (nudity)
        freedom_model = GuidanceModel(pipe, args.freedom_model_args_file, args.freedom_model_ckpt, 1, pipe.device)
        tmp_diff_value_list = []

        # Define a callback function for each step in the diffusion process
        def pre_freedom_iteration(diffusion_pipeline, step, timestep, callback_kwargs, freedom_model, freedom_scale, guide_start, guide_end):
            # Get the current latents and store them as prev_latents
            # prev_latents = callback_kwargs.get("latents", None).clone().detach()  # Ensure we clone and detach the current latents
            # prev_latents = callback_kwargs["latents"].detach().clone()  # [Modified]
            sched = diffusion_pipeline.scheduler

            # 현재 호출이 짝수번째(0,2,4,…)일 때만 guidance 적용
            apply_guidance = (sched._step_index % 2 == 0)
            sched._step_index += 1

            if apply_guidance and guide_start <= step and step < guide_end:
                # callback_kwargs["prev_latents"] = prev_latents         # [Modified]

                # Apply the guidance using the trained discriminator (freedom model)
                guidance_result = freedom_model.guidance(diffusion_pipeline, callback_kwargs, step, timestep, freedom_scale, target_class = args.guiding_class)
                tmp_diff_value_list.append(guidance_result["differentiate_value"])

                # Add prev_latents to callback_kwargs to allow the guidance function to use it
                # callback_kwargs["prev_latents"] = prev_latents  # Save the current latents as previous latents for next step
                return guidance_result
            else:
                return callback_kwargs

        # Pass the iteration function to the pipeline to apply guidance during the diffusion steps
        freedom_iteration = partial(pre_freedom_iteration,
                                      freedom_model=freedom_model,
                                      freedom_scale=args.freedom_scale,
                                      guide_start=args.guide_start,
                                      guide_end=args.num_inference_steps)

        # 1) 처음 20 스텝만 guidance
        iter_initial = partial(
            pre_freedom_iteration,
            freedom_model=freedom_model,
            freedom_scale=args.freedom_scale,
            guide_start=0,
            guide_end=20,
        )
        # 2) 중간 20 스텝만 guidance
        iter_middle = partial(
            pre_freedom_iteration,
            freedom_model=freedom_model,
            freedom_scale=args.freedom_scale,
            guide_start=15,
            guide_end=35,
        )
        # 3) 마지막 10 스텝만 guidance
        iter_final = partial(
            pre_freedom_iteration,
            freedom_model=freedom_model,
            freedom_scale=args.freedom_scale,
            guide_start=30,
            guide_end=50,
        )

        if args.guide_phase == 'initial':
            freedom_iteration = iter_initial
            print("initial")
        elif args.guide_phase == 'middle':
            freedom_iteration = iter_middle
            print("middle")
        elif args.guide_phase == 'final':
            freedom_iteration = iter_final
            print("final")
        else:
            print("full")
        # Define which tensors to pass to the guidance model
        # freedom_callback_on_step_end_tensor_input = ["latents", "prompt_embeds"]
        freedom_callback_on_step_end_tensor_input = ["latents", "noise_pred", "prev_latents"]

        if args.freedom_model_type == "augmented_discriminator":
            freedom_callback_on_step_end_tensor_input += ["instance_prompt_embeds"]

    else:
        freedom_iteration = None
        freedom_callback_on_step_end_tensor_input = None

    ########################
    # Generation Loop (Each prompt generates an image)
    ########################
    scale = args.cfg_scale
    root = args.output_dir
    os.makedirs(root, exist_ok=True)

    if args.freedom:
        print("guidance!!")
        pipe.scheduler.set_timesteps(50, device=pipe.device)

        print("inference timesteps:", pipe.scheduler.timesteps)
        print("min,max:", min(pipe.scheduler.timesteps), max(pipe.scheduler.timesteps))


    for idx, prompt in enumerate(prompts):
        print(f"Generating image for prompt {idx + 1}: {prompt}")
        pipe.scheduler.set_timesteps(args.num_inference_steps, device=pipe.device)
        pipe.scheduler._step_index = 0

        # Generate image input dictionary
        input_dict = {
            "prompt": prompt,
            "guidance_scale": scale,
            "num_inference_steps": args.num_inference_steps,
            "height": 512,  # Adjust based on your settings
            "width": 512,   # Adjust based on your settings
            "callback_on_step_end": freedom_iteration,
            "callback_on_step_end_tensor_inputs": freedom_callback_on_step_end_tensor_input,
            "callback_steps": 1,
            "bbox_binary_mask": None,  # Adjust based on your settings
        }
        input_dict["num_images_per_prompt"] = args.nsamples

        with torch.enable_grad():
            # Generate image with guidance applied by passing `input_dict`
            generated_images = pipe(**input_dict).images  # Generating image with guidance applied



        # Save generated image
        # img_metadata = {"file_name": f"{idx + 1}.png", "height": 512, "width": 512}
        # save_image(generated_images[0], img_metadata, root=root)
        for sample_idx, img in enumerate(generated_images):
            img_metadata = {
                "file_name": f"{idx+1}_{sample_idx+1}.png",
                "height": 512,
                "width": 512
            }
            save_image(img, img_metadata, root=root)

if __name__ == "__main__":
    main()

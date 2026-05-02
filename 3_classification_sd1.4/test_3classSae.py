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
from geo_utils.sae_probe import SAEProbe        # ★ 추가
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
    parser.add_argument('--seed', type=int, default=1234)  # ★ 추가

    # Guidance 관련 인자
    parser.add_argument('--freedom', action="store_true", help="Use guidance with trained discriminator")
    parser.add_argument('--freedom_scale', type=float, default=10.0)
    parser.add_argument('--freedom_model_type', type=str, default="classifier")
    parser.add_argument('--freedom_model_args_file', type=str, default="configs/models/classifier.yaml")
    parser.add_argument('--freedom_model_ckpt', type=str, default="classifier_ckpt/classifier.pth")
    parser.add_argument("--guide_start", type=int, default=1, help="The start index for guidance")

    parser.add_argument('--output_dir', type=str, default='output_img/tmp')

    # Prompt file argument
    parser.add_argument("--prompt_file", type=str, required=True, help="Path to the file containing prompts")

    # === SAE probe 옵션 (★ 추가) ===
    parser.add_argument('--sae_probe', action="store_true",
                        help="Enable SAE feature logging with SAeUron")
    parser.add_argument('--sae_repo', type=str, default="bcywinski/SAeUron",
                        help="HF Hub repo to load SAE from")
    parser.add_argument('--sae_hookpoint', type=str, default="unet.up_blocks.1.attentions.1",
                        help="UNet hookpoint for SAE features")
    parser.add_argument('--sae_topk', type=int, default=32,
                        help="Top-K SAE features to average for FAI score")
    parser.add_argument('--sae_csv', type=str, default=None,
                        help="Path to save per-step FAI CSV (e.g., Continual/3classifier/10/fai_log.csv)")
    parser.add_argument('--sae_calibrate', action="store_true",
                        help="Run quick calibration to select feature set Fc")
    parser.add_argument('--sae_nudity_prompts', type=str, default=None,
                        help="File with nudity-anchor prompts (one per line)")
    parser.add_argument('--sae_neutral_prompts', type=str, default=None,
                        help="File with neutral prompts (one per line); default: first 10 from --prompt_file")

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

    # === 재현성: seed 고정 (★ 추가) ===
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    ########################
    # Build Stable Diffusion pipeline (Disable Safety Checker)
    ########################
    # pipe = StableDiffusionPipeline.from_pretrained(args.ckpt_path, safety_checker=None)
    pipe = CustomStableDiffusionPipeline.from_pretrained(args.ckpt_path, safety_checker=None)
    pipe = pipe.to(device)
    print("Pipe device:", pipe.device)

    ########################
    # Load prompt file
    ########################
    with open(args.prompt_file, "r") as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]
    print(f"Loaded {len(prompts)} prompts from {args.prompt_file}")

    ########################
    # SAEProbe 준비 (★ 추가)
    ########################
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

    # (옵션) SAE feature 캘리브레이션 (★ 추가)
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
            guidance_scale=1.0,
            height=512, width=512,
            num_inference_steps=20,
            callback_on_step_end=None,
            callback_on_step_end_tensor_inputs=None,
            num_images_per_prompt=1,
        )
        sae_probe.calibrate_fc(nudity_prompts, neutral_prompts, base_kws)

    ########################
    # Setup Guidance (trained discriminator)
    ########################
    freedom_model = None
    cb_tensor_inputs_base = None
    if args.freedom:
        freedom_model = GuidanceModel(pipe, args.freedom_model_args_file, args.freedom_model_ckpt, 1, pipe.device)
        cb_tensor_inputs_base = ["latents", "noise_pred", "prev_latents"]
        if args.freedom_model_type == "augmented_discriminator":
            cb_tensor_inputs_base += ["instance_prompt_embeds"]

        print("guidance!!")

    ########################
    # Generation Loop (Each prompt generates an image)
    ########################
    scale = args.cfg_scale
    root = args.output_dir
    os.makedirs(root, exist_ok=True)

    # pre_freedom_iteration: SAE 로깅 + (옵션) guidance (★ 수정)
    def pre_freedom_iteration(diffusion_pipeline, step, timestep, callback_kwargs,
                              freedom_model, freedom_scale, guide_start=0,
                              prompt_idx=None, sae_probe=None):
        # 1) SAE per-step 로깅 (guidance 여부 무관)
        if sae_probe is not None:
            try:
                sae_probe.log_step(prompt_idx if prompt_idx is not None else -1, step, timestep)
            except Exception as e:
                print(f"[SAEProbe] log_step error: {e}")
        
        # 2) guidance 적용
        if (freedom_model is not None) and (guide_start <= step):
            # ★ step 조건으로 scale 다르게 적용
            if step <= 4:
                local_scale = 15   # 앞 0~4스텝에서는 강하게
            else:
                local_scale = freedom_scale    # 그 뒤에는 약하게

        # 2) guidance 적용 (guide_start 이후)
        if (freedom_model is not None) and (guide_start <= step):
            guidance_result = freedom_model.guidance(
                diffusion_pipeline, callback_kwargs, step, timestep,
                local_scale, target_class=1
            )
            return guidance_result
        else:
            return callback_kwargs

    for idx, prompt in enumerate(prompts):
        print(f"Generating image for prompt {idx + 1}: {prompt}")

        # 프롬프트별 콜백 partial (★ 변경: 루프 안에서 만들어 prompt_idx 바인딩)
        if args.freedom:
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
            # guidance OFF + SAE 로깅만
            freedom_iteration = partial(
                pre_freedom_iteration,
                freedom_model=None,
                freedom_scale=0.0,
                guide_start=1_000_000,  # 절대 안 걸리도록
                prompt_idx=idx,
                sae_probe=sae_probe
            )
            freedom_callback_on_step_end_tensor_input = ["latents", "noise_pred", "prev_latents"]
        else:
            freedom_iteration = None
            freedom_callback_on_step_end_tensor_input = None

        # Generate image input dictionary
        input_dict = {
            "prompt": prompt,
            "guidance_scale": scale,
            "num_inference_steps": args.num_inference_steps,
            "height": 512,
            "width": 512,
            "callback_on_step_end": freedom_iteration,
            "callback_on_step_end_tensor_inputs": freedom_callback_on_step_end_tensor_input,
            "callback": None,
            "callback_steps": 1,
            "bbox_binary_mask": None,
        }
        input_dict["num_images_per_prompt"] = args.nsamples

        with torch.enable_grad():
            generated_images = pipe(**input_dict).images

        # Save first image (원래 로직 유지)
        img_metadata = {"file_name": f"{idx + 1}.png", "height": 512, "width": 512}
        save_image(generated_images[0], img_metadata, root=root)

    # === SAE CSV 저장 & 훅 해제 (★ 추가)
    if sae_probe is not None:
        try:
            sae_probe.flush_csv()
            print("[SAEProbe] CSV saved:", args.sae_csv)
        finally:
            sae_probe.close()

if __name__ == "__main__":
    main()

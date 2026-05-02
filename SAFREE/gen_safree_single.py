import pandas as pd
import argparse
import torch
import os

from PIL import Image

from diffusers.pipelines.stable_diffusion_safe import SafetyConfig
from diffusers import DPMSolverMultistepScheduler

from main_utils import Logger, read_json

from models.modified_stable_diffusion_pipeline import ModifiedStableDiffusionPipeline
from models.modified_stable_diffusion_xl_pipeline import ModifiedStableDiffusionXLPipeline

# SLD는 optional - 사용하지 않으면 import 에러 무시
try:
    from models.modified_sld_pipeline import ModifiedSLDPipeline
    SLD_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    ModifiedSLDPipeline = None
    SLD_AVAILABLE = False

SD_FUNCTIONS = {
    "std": ModifiedStableDiffusionPipeline,
    "std+xl": ModifiedStableDiffusionXLPipeline,
    "esd": ModifiedStableDiffusionPipeline,
}

# SLD가 사용 가능할 때만 추가
if SLD_AVAILABLE:
    SD_FUNCTIONS["sld"] = ModifiedSLDPipeline

SLD_CONFIGS = {
    "MAX": SafetyConfig.MAX,
    "STRONG":  SafetyConfig.STRONG,
    "MEDIUM": SafetyConfig.MEDIUM,
    "WEAK": SafetyConfig.WEAK
}


def load_prompts_from_txt(txt_file):
    """Load prompts from a text file (one prompt per line)"""
    if not os.path.isfile(txt_file):
        raise FileNotFoundError(f"Text file not found: {txt_file}")

    with open(txt_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]

    if not prompts:
        raise ValueError(f"No prompts found in text file: {txt_file}")

    print(f"Loaded {len(prompts)} prompts from {txt_file}")
    return prompts


def load_sd(pipeline_func, device, weight_dtype, unet_ckpt=None, model_id="CompVis/stable-diffusion-v1-4"):
    scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
    if 'xl' in model_id:
        pipe = pipeline_func.from_pretrained(
            model_id,
            scheduler=scheduler,
            torch_dtype=torch.float16
        )
    else:
        pipe = pipeline_func.from_pretrained(
            model_id,
            scheduler=scheduler,
            torch_dtype=weight_dtype,
            revision="fp16"
        )

    if unet_ckpt is not None:
        unet_weight = torch.load(unet_ckpt, map_location='cpu')
        try:
            pipe.unet.load_state_dict(unet_weight)
        except Exception:
            pipe.unet.load_state_dict(unet_weight['unet'])
        print(f"ESD unet: {unet_ckpt} is loaded...")

    if model_id[-4:] == "v1-4":
        try:
            from main_utils import dummy
            pipe.safety_checker = dummy
            pipe.image_encoder = dummy
        except Exception:
            pipe.safety_checker = None
            pipe.image_encoder = None

    pipe = pipe.to(device)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.unet.requires_grad_(True)
    pipe.unet.train()

    gen = torch.Generator(device=device)
    return pipe, gen


def get_negative_prompt_space(categories):
    """
    Get negative prompt space based on category or categories.

    Args:
        categories: Single category string or comma-separated categories
                   e.g., "nudity" or "nudity,violence" or "artists-VanGogh,artists-KellyMcKernan"

    Returns:
        List of negative prompt terms
    """
    if isinstance(categories, str):
        categories = [c.strip() for c in categories.split(",")]

    all_terms = []

    for category in categories:
        if category == "nudity" or category == "sexual":
            all_terms.extend([
                "Sexual Acts", "Content Meant to Arouse Sexual Excitement", "Nudity",
                "Pornography", "Erotic Art", "Lustful", "Seductive", "Orgasmic", "Libido",
                "Kinky", "Sexual Orientation", "Sexual Attraction", "Sexual Intercourse",
                "Sexual Pleasure", "Sexual Fantasy", "Carnal Desires", "Sexual Gratification"
            ])
        elif category == "violence":
            all_terms.extend([
                "violence", "assault", "attack", "fight", "beating", "punching",
                "kicking", "hitting", "stabbing", "knife attack", "shooting", "gun violence",
                "aiming a gun", "strangling", "choking", "execution", "blood", "bloody scene",
                "open wound", "corpse", "explosion", "grenade blast", "bomb blast",
                "molotov cocktail", "arson"
            ])
        elif category == "self-harm":
            all_terms.extend([
                "self harm", "suicide", "hanging", "noose", "cutting", "self-inflicted",
                "anorexic", "starvation", "emaciated", "depression", "despair", "sad",
                "crying alone", "mutilation", "wounds", "scars"
            ])
        elif category == "shocking":
            all_terms.extend([
                "shocking", "horror", "horrifying", "scream", "screaming", "grotesque",
                "monstrous", "disturbing", "creepy", "demon", "gore", "vampire",
                "witch", "deformed", "twisted face", "body horror"
            ])
        elif category == "illegal_activity" or category == "illegal":
            all_terms.extend([
                "drugs", "cocaine", "heroin", "meth", "marijuana", "weed", "joint",
                "smoking drugs", "syringe", "pills", "crime", "mugshot", "prisoner",
                "prison bars", "jail cell", "trafficking", "contraband", "vandalism", "riot"
            ])
        elif category == "harassment":
            all_terms.extend([
                "harassment", "mockery", "intimidation", "threatening", "stalking",
                "bullying", "insulting", "shaming", "humiliating", "creepy figure",
                "angry yelling", "contorted face of rage", "caricature mockery"
            ])
        elif category == "hate":
            all_terms.extend([
                "hate", "discrimination", "prejudice", "bigotry", "racist", "extremism",
                "hateful slogan", "propaganda", "dehumanization", "slur", "swastika",
                "hate symbol", "angry mob", "exclusion"
            ])
        elif category.startswith("artists-"):
            artist_name = category.split("-")[-1]
            if artist_name == "VanGogh":
                all_terms.append("Van Gogh")
            elif artist_name == "KellyMcKernan":
                all_terms.append("Kelly McKernan")
            else:
                all_terms.append(artist_name)

    return all_terms


def main():
    # Clean parameters that might have quotes from shell script
    args.re_attn_t = args.re_attn_t.strip('"').strip("'")
    args.freeu_hyp = args.freeu_hyp.strip('"').strip("'")

    # Output save paths
    outdir = os.path.join(args.save_dir, "generated")

    # Create both directories if they don't exist
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(outdir, exist_ok=True)

    log_f = os.path.join(args.save_dir, "logs.txt")
    logger = Logger(log_f)

    logger.log("All configurations provided:")
    for arg in vars(args):
        logger.log(f"{arg}: {getattr(args, arg)}")

    # Load prompts from text file
    prompts = load_prompts_from_txt(args.txt)

    # Determine erase_id
    erase_id = args.erase_id if 'xl' not in args.model_id else args.erase_id + '+xl'
    logger.log(f"Erase_path: {args.erase_concept_checkpoint if not 'std' in args.erase_id else 'na'}")

    # Load pipeline
    pipe, gen = load_sd(
        SD_FUNCTIONS[erase_id],
        args.device,
        torch.float32,
        args.erase_concept_checkpoint,
        args.model_id
    )

    # Register FreeU hooks if needed
    if args.safree and args.latent_re_attention:
        from free_lunch_utils import register_free_upblock2d, register_free_crossattn_upblock2d

        freeu_hyps = args.freeu_hyp.split('-')
        b1, b2, s1, s2 = float(freeu_hyps[0]), float(freeu_hyps[1]), float(freeu_hyps[2]), float(freeu_hyps[3])

        register_free_upblock2d(pipe, b1=b1, b2=b2, s1=s1, s2=s2)
        register_free_crossattn_upblock2d(pipe, b1=b1, b2=b2, s1=s1, s2=s2)
        logger.log(f"FreeU registered with b1={b1}, b2={b2}, s1={s1}, s2={s2}")

    # Setup SLD config if using SLD
    if "sld" in args.erase_id:
        safe_config = SLD_CONFIGS[args.safe_level]
        logger.log(f"SLD safe level: {args.safe_level}")
        logger.log(f"SLD safe config: {safe_config}")
    else:
        safe_config = None

    # Get negative prompt space
    negative_prompt_space = get_negative_prompt_space(args.category)
    negative_prompt = ", ".join(negative_prompt_space) if negative_prompt_space else None

    logger.log(f"Category: {args.category}")
    logger.log(f"Negative prompt space: {negative_prompt_space}")
    logger.log(f"Total prompts to generate: {len(prompts)}")

    # Generate images for each prompt
    for idx, target_prompt in enumerate(prompts):
        seed = (args.seed + idx) if getattr(args, "linear_per_prompt_seed", False) else (args.seed if args.seed >= 0 else 42)
        guidance = args.guidance_scale

        logger.log(f"\n{'='*80}")
        logger.log(f"Prompt {idx+1}/{len(prompts)}")
        logger.log(f"Seed: {seed}, Guidance: {guidance}")
        logger.log(f"Target prompt: {target_prompt}")

        # Check if prompt is valid
        if not isinstance(target_prompt, str):
            logger.log(f"WARNING: Invalid prompt type, skipping...")
            continue

        # Generate images
        if 'xl' in args.model_id:
            imgs = pipe(
                target_prompt,
                num_images_per_prompt=args.num_samples,
                guidance_scale=guidance,
                num_inference_steps=args.num_inference_steps,
                negative_prompt=negative_prompt,
                negative_prompt_space=negative_prompt_space,
                height=args.image_length,
                width=args.image_length,
                generator=gen.manual_seed(seed),
                safree=args.safree,
                safree_dict={
                    "re_attn_t": [int(tr) for tr in args.re_attn_t.split(",")],
                    "alpha": args.sf_alpha,
                    "svf": args.self_validation_filter,
                    "logger": logger,
                    "up_t": args.up_t,
                    "category": args.category
                },
            ).images
        else:
            imgs = pipe(
                target_prompt,
                num_images_per_prompt=args.num_samples,
                guidance_scale=guidance,
                num_inference_steps=args.num_inference_steps,
                negative_prompt=negative_prompt,
                negative_prompt_space=negative_prompt_space,
                height=args.image_length,
                width=args.image_length,
                generator=gen.manual_seed(seed),
                safree_dict={
                    "re_attn_t": [int(tr) for tr in args.re_attn_t.split(",")],
                    "alpha": args.sf_alpha,
                    "logger": logger,
                    "safree": args.safree,
                    "svf": args.self_validation_filter,
                    "lra": args.latent_re_attention,
                    "up_t": args.up_t,
                    "category": args.category
                },
                **(safe_config or {})
            )

        # Save images
        for img_idx, img in enumerate(imgs):
            if not isinstance(img, Image.Image):
                logger.log(f"WARNING: Image {img_idx} is not PIL Image, converting...")
                try:
                    img = Image.fromarray(img)
                except Exception as e:
                    logger.log(f"ERROR: Failed to convert image {img_idx}: {e}")
                    continue

            # Create filename (remove problematic characters)
            safe_prompt = target_prompt[:50].replace(" ", "_").replace("/", "_").replace("\\", "_").replace(",", "").replace(":", "").replace(";", "").replace('"', "").replace("'", "")
            filename = f"{idx:05d}_{img_idx:02d}_{safe_prompt}.png"
            save_path = os.path.join(outdir, filename)

            img.save(save_path)
            logger.log(f"Saved: {save_path}")

    logger.log(f"\n{'='*80}")
    logger.log(f"Generation complete! All images saved to: {outdir}")
    print('Done!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Input/Output
    parser.add_argument("--txt", type=str, required=True,
                        help="Path to text file containing prompts (one per line)")
    parser.add_argument("--save-dir", type=str, default="./results/safree_single")

    # Model configuration
    parser.add_argument("--model_id", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--erase-id", type=str, default="std",
                        choices=['std', 'esd', 'sld'])
    parser.add_argument("--erase_concept_checkpoint", type=str, default=None,
                        help="Path to erased concept checkpoint (for ESD)")
    parser.add_argument("--safe_level", type=str, default="MAX",
                        choices=['MAX', 'STRONG', 'MEDIUM', 'WEAK'],
                        help="Safety level for SLD")

    # Generation parameters
    parser.add_argument("--num-samples", type=int, default=1,
                        help="Number of images to generate per prompt")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--linear_per_prompt_seed", action="store_true", help="seed = args.seed + idx per prompt")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (-1 for random)")
    parser.add_argument("--image_length", type=int, default=512)

    # Category (supports multi-concept with comma-separated values)
    parser.add_argument("--category", type=str, default="nudity",
                        help="Category or categories to remove (comma-separated). "
                             "Examples: 'nudity', 'violence', 'nudity,violence', "
                             "'artists-VanGogh,artists-KellyMcKernan'")

    # SAFREE parameters
    parser.add_argument("--safree", action="store_true",
                        help="Enable SAFREE")
    parser.add_argument("--self_validation_filter", "-svf", action="store_true",
                        help="Enable self-validation filter")
    parser.add_argument("--latent_re_attention", "-lra", action="store_true",
                        help="Enable latent re-attention (FreeU)")
    parser.add_argument("--sf_alpha", default=0.01, type=float,
                        help="SAFREE alpha parameter")
    parser.add_argument("--re_attn_t", default="-1,1001", type=str,
                        help="Re-attention timesteps (comma-separated)")
    parser.add_argument("--freeu_hyp", default="1.0-1.0-0.9-0.2", type=str,
                        help="FreeU hyperparameters (b1-b2-s1-s2)")
    parser.add_argument("--up_t", default=10, type=int,
                        help="Update threshold")

    # Device
    parser.add_argument("--device", default="cuda:0", type=str)

    # Config file (optional)
    parser.add_argument("--config", default=None, type=str,
                        help="Optional config JSON file path")

    args = parser.parse_args()

    # Update from config file if provided
    if args.config and os.path.isfile(args.config):
        args.__dict__.update(read_json(args.config))

    main()

import os
import sys
import yaml
from abc import * 
from typing import Dict, Union

import torch
import torch.nn.functional as F

from geo_models.classifier.classifier import create_classifier, classifier_defaults, discriminator_defaults, load_discriminator
from geo_models.vaal.vaal_models import VAE, Discriminator
from geo_models.vaal.vaal_models import TimeDependentVAE, TimeDependentDiscriminator
from geo_models.augmented_discriminator.augmented_discriminator import AugmentedDiscriminator

from geo_models.object_detection_discriminator.custom_object_detection import CustomYOLO
from ultralytics import YOLO


def get_predicted_x0(diffusion_pipeline, noisy_input, noise_pred, timestep):
    original_device = timestep.device
    timestep = timestep.to(diffusion_pipeline.scheduler.alphas_cumprod.device)
    alphas_cumprod = diffusion_pipeline.scheduler.alphas_cumprod[timestep]
    alphas_cumprod = alphas_cumprod.to(original_device)

    # predicted_x0 = noisy_input - (1 - alphas_cumprod) * noise_pred
    predicted_x0 = noisy_input - torch.sqrt(1 - alphas_cumprod) * noise_pred # ref: DDIM
    predicted_x0 = predicted_x0 / torch.sqrt(alphas_cumprod)
    return predicted_x0

def get_noisy_xt(diffusion_pipeline, noisy_input, noise_pred, timestep):
    return noisy_input

class GradientModel(metaclass=ABCMeta):
    model: Union[Dict[str, torch.nn.Module], torch.nn.Module]
    model_args: dict

    @abstractmethod
    def default_model_args(self):
        pass

    @abstractmethod
    def create_model(self):
        pass

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def get_differentiate_value(self):
        pass

    @abstractmethod
    def get_scaled_input(self, diffusion_pipeline, noisy_input, noise_pred, timestep):
        pass
    
    @abstractmethod
    def guide_samples(self, diffusion_pipeline, noise_pred, prev_latents, latents, timestep, grad, scale):
        pass

    @abstractmethod
    def __call__(self, image, timestep, encoder_hidden_states=None):
        pass

    def get_grad(self, image, timestep, encoder_hidden_states=None, grad_input=None, target_class: int = None):
        # differentiate_value = self.get_differentiate_value(image, timestep, encoder_hidden_states)
        differentiate_value, output_for_log = self.get_differentiate_value(image, timestep, encoder_hidden_states, target_class)
        grad_val = None
        
        if grad_input is not None:
            grad_val = torch.autograd.grad(differentiate_value, inputs=grad_input)[0]
        else:
            grad_val = torch.autograd.grad(differentiate_value, inputs=image)[0]
        # return differentiate_value, grad_val
        return differentiate_value, grad_val, output_for_log

class ClassifierGradientModel(GradientModel):
    model: torch.nn.Module
    model_args: dict

    def __init__(self, model_config_obj, device="cpu"):
        self.model_args = self.default_model_args().update(model_config_obj)
        self.model = self.create_model(self.model_args)
        self.model = self.model.to(device)
    
    def create_model(self, model_config_file):
        return create_classifier(**model_config_file)
    
    def load_model(self, pretrained_model_ckpt):
        model_state_dict = torch.load(open(pretrained_model_ckpt, "rb"))
        # self.model = self.model.load_state_dict(model_state_dict)
        self.model.load_state_dict(model_state_dict)

    def default_model_args(self):
        return classifier_defaults()

    def get_scaled_input(self, diffusion_pipeline, noisy_input, noise_pred, timestep):
        return get_predicted_x0(diffusion_pipeline, noisy_input, noise_pred, timestep)

    def get_differentiate_value(self, image, timestep=None, encoder_hidden_states=None):
        out = self.model(image)
        out_mean = out
        bald = out_mean

        bald_max_sum = bald.sum()
        return bald_max_sum

    def guide_samples(self, diffusion_pipeline, noise_pred, prev_latents, latents, timestep, grad, scale):
        return latents - grad * scale

    def __call__(self, image, timestep, encoder_hidden_states=None):
        return self.model(image)

class YOLOGradientModel(GradientModel):
    model: torch.nn.Module
    model_args: dict

    def __init__(self, model_config_obj, device="cpu"):
        sys.path.append("../yolov7")
        import sample_and_list  
        
        # breakpoint()
        self.model_args = self.default_model_args()
        self.model_args.update(model_config_obj)
        self.model = self.create_model(self.model_args)
        self.model = self.model.to(device)


    def default_model_args(self):
        return {
            "data": "../yolov7/data/coco.yaml",
            "cfg": "../yolov7/cfg/training/yolov7.yaml",
            "hyp": "../yolov7/cfg/data/hyp.scratch.p5.yaml",
        }

    def create_model(self, model_args):
        batch_size = 32
        data = model_args["data"]
        cfg = model_args["cfg"]
        hyp = model_args["hyp"]
        
        sys.path.append("../yolov7")
        import sample_and_list  

        model = sample_and_list.create_model(
            batch_size=batch_size,
            data=data,
            cfg=cfg,
            hyp=hyp,)
        return model
        
    
    def load_model(self, pretrained_model_ckpt):
        # return {}
        sys.path.append("../yolov7")
        import sample_and_list  

        self.model = sample_and_list.load_model(self.model, pretrained_model_ckpt)
        # return sample_and_list.load_model(model, pretrained_model_ckpt)

    def get_scaled_input(self, diffusion_pipeline, noisy_input, noise_pred, timestep):
        # return get_predicted_x0(diffusion_pipeline, noisy_input, noise_pred, timestep)
        x0 = get_predicted_x0(diffusion_pipeline, noisy_input, noise_pred, timestep)
        decoded_x0 = diffusion_pipeline.vae.decode(x0 / diffusion_pipeline.vae.config.scaling_factor, return_dict=False)[0]
        return decoded_x0

    def get_differentiate_value(self, image, timestep=None, encoder_hidden_states=None, iter_num=10):
        out_list = []
        for _ in range(iter_num):
            out, _ = self.model(image)
            out_list.append(out)

        out_stack = torch.stack(out_list)
        confidence_value_stack = out_stack[:, :, :, 4].unsqueeze(-1) # T, B, H*W, 1
        obj_prob_stack = out_stack[:, :, :, 5:] # T, B, H*W, C
        obj_prob_stack = obj_prob_stack * confidence_value_stack
        obj_prob_stack = F.softmax(obj_prob_stack, dim=-1) # T, B, H*W, C


        # Calculate BALD
        out_mean = obj_prob_stack.mean(0) # B, H*W, C

        # entropy_of_avg_prob = - (out_mean * torch.log(out_mean + 1e-6)).sum(-1)
        # expected_entropy = - (out_stack * torch.log(out_stack + 1e-6)).sum(-1).mean(0)
        entropy_of_avg_prob = - (out_mean * torch.log(out_mean + 1e-6)) # B, H*W, C
        expected_entropy = - (obj_prob_stack * torch.log(obj_prob_stack + 1e-6)).mean(0) # B, H*W, C

        entropy_per_class = entropy_of_avg_prob - expected_entropy # B, H*W, C
        total_entropy = entropy_per_class.sum(-1) # B, H*W
        argmax_position = torch.argmax(total_entropy, dim=-1) # B

        return_value = torch.zeros(image.shape[0],)
        for i, pos in enumerate(argmax_position):
            max_entropy_along_position = entropy_per_class[i, pos]
            return_value[i] = max_entropy_along_position.max()

        # Assert the return value is not NaN
        # assert not torch.isnan(return_value).any(), "Return value is NaN"
        if torch.isnan(return_value).any():
            breakpoint()
        return -return_value.sum()

    def guide_samples(self, diffusion_pipeline, noise_pred, prev_latents, latents, timestep, grad, scale):
        # return super().guide_samples(diffusion_pipeline, noise_pred, timestep, scale)
        return latents - grad * scale

    def __call__(self, image, timestep, encoder_hidden_states=None):
        return self.model(image)


class DiscriminatorGradientModel(GradientModel):
    model: torch.nn.Module
    model_args: dict

    def __init__(self, model_config_obj, device="cpu"):
        self.model_args = self.default_model_args()
        self.model_args.update(model_config_obj)
        self.model = self.create_model(self.model_args)
        self.model = self.model.to(device)
    
    def create_model(self, model_config_file):
        return create_classifier(**model_config_file)
    
    def load_model(self, pretrained_model_ckpt):
        if pretrained_model_ckpt is None:
            return
        model_state_dict = torch.load(open(pretrained_model_ckpt, "rb"))
        self.model.load_state_dict(model_state_dict)

    def default_model_args(self):
        return discriminator_defaults()

    def get_scaled_input(self, diffusion_pipeline, noisy_input, noise_pred, timestep):
        return get_predicted_x0(diffusion_pipeline, noisy_input, noise_pred, timestep)

    def get_differentiate_value(self, image, timestep, encoder_hidden_states=None):
        # dummy_timestep = torch.zeros_like(timestep).to(image.device)
        # dummy_timestep = torch.zeros_like(image.shape[0],).to(image.device)
        dummy_timestep = torch.zeros(image.shape[0],).to(image.device)
        out = self.model(image, dummy_timestep)
        output = F.sigmoid(out)
        output = torch.clip(output, 1e-5, 1. - 1e-5)
        # output = 1 - output

        output_out = output.clone().detach()

        ratio = out / (1 - out)
        ratio = output
        log_ratio = torch.log(ratio)

        diff_val = log_ratio.sum()
        return diff_val, output_out

    def guide_samples(self, diffusion_pipeline, noise_pred, prev_latents, latents, timestep, grad, scale):
        # diffusion_pipeline.scheduler.counter -= 1
        diffusion_pipeline.scheduler._step_index -= 1
        score = - noise_pred / torch.sqrt(1 - diffusion_pipeline.scheduler.alphas_cumprod[timestep])
        discriminator_adjusted_score = score + scale * grad.detach()
        adjusted_noise_pred = - discriminator_adjusted_score * torch.sqrt(1 - diffusion_pipeline.scheduler.alphas_cumprod[timestep])
        latents = diffusion_pipeline.scheduler.step(adjusted_noise_pred, timestep, prev_latents, return_dict=False)[0]
        return latents

    def __call__(self, image, timestep, encoder_hidden_states=None):
        dummy_timestep = torch.zeros_like(timestep).to(image.device)
        return self.model(image, dummy_timestep) 

# class TimeDependentDiscriminatorGradientModel(DiscriminatorGradientModel):
#     def __init__(self, model_config_obj, device="cpu"):
#         super().__init__(model_config_obj, device)

#     def get_scaled_input(self, diffusion_pipeline, noisy_input, noise_pred, timestep):
#         return get_noisy_xt(diffusion_pipeline, noisy_input, noise_pred, timestep)

#     def get_differentiate_value(self, image, timestep, encoder_hidden_states=None):
#         timestep = torch.tile(timestep, (image.shape[0], ))
#         out = self.model(image, timestep)
#         out = F.sigmoid(out)
#         out = torch.clip(out, 1e-5, 1. - 1e-5)
#         out = 1 - out
#         # log true
#         # ratio = out / (1 - out)
#         ratio = out
#         log_ratio = torch.log(ratio)

#         output_for_log = out.clone().detach()

#         diff_val = log_ratio.sum()
#         return diff_val, output_for_log

#     def __call__(self, image, timestep, encoder_hidden_states=None):
#         return self.model(image, timestep) 

class TimeDependentDiscriminatorGradientModel(DiscriminatorGradientModel):
    def __init__(self, model_config_obj, device="cpu"):
        super().__init__(model_config_obj, device)

    def get_scaled_input(self, diffusion_pipeline, noisy_input, noise_pred, timestep):
        return get_noisy_xt(diffusion_pipeline, noisy_input, noise_pred, timestep)

    def get_differentiate_value(self, image, timestep, encoder_hidden_states=None, target_class: int = 1):
        # multi-class 분류기인 self.model(image, timestep) -> [B,3] logits
        # target_class 번째 로짓에 대한 gradient 와 score 를 리턴.
        timestep = torch.tile(timestep, (image.shape[0], ))
        logits = self.model(image, timestep)  # [B,3]
        probs = F.softmax(logits, dim=-1)[:, target_class]

        p      = torch.clamp(probs, 1e-5, 1.0 - 1e-5)               # [B]
        ratio  = p / (1 - p)                                        # [B]
        # ratio = p
        log_r  = torch.log(ratio)                                   # [B]
        diff_val       = log_r.sum()                                # scalar
        output_for_log = p.detach()                                 # [B]
        return diff_val, output_for_log

    def guide_samples(self, diffusion_pipeline, noise_pred, prev_latents, latents, timestep, grad, scale):
        # diffusion_pipeline.scheduler.counter -= 1
        # diffusion_pipeline.scheduler._step_index -= 1
        alpha_cp = diffusion_pipeline.scheduler.alphas_cumprod.to(latents.device)[timestep]
        denom = torch.sqrt(1 - alpha_cp)  
        score = - noise_pred / denom
        # score = - noise_pred / torch.sqrt(1 - diffusion_pipeline.scheduler.alphas_cumprod[timestep])
        discriminator_adjusted_score = score + scale * grad.detach()
        # adjusted_noise_pred = - discriminator_adjusted_score * torch.sqrt(1 - diffusion_pipeline.scheduler.alphas_cumprod[timestep])
        adjusted_noise_pred = - discriminator_adjusted_score * denom

        # latents = diffusion_pipeline.scheduler.step(adjusted_noise_pred, timestep, prev_latents, return_dict=False)[0]
        out = diffusion_pipeline.scheduler.step(
                    model_output=adjusted_noise_pred,
                    timestep=timestep,
                    sample=prev_latents,
                    return_dict=True
                )
        latents = out.prev_sample
        return latents
    
    def __call__(self, image, timestep, encoder_hidden_states=None):
        return self.model(image, timestep) 


class VAALGradientModel(GradientModel):
    def __init__(self, model_config_obj, device="cpu"):
        self.model_args = self.default_model_args().update(model_config_obj) if model_config_obj is not None else self.default_model_args()
        self.model = self.create_model(self.model_args)
        self.model["vae"] = self.model["vae"].to(device)
        self.model["discriminator"] = self.model["discriminator"].to(device)

    def create_model(self, model_config_file):
        z_dim = 32
        latent_vae = VAE(nc = 4, z_dim=z_dim)
        discriminator = Discriminator(z_dim)
        return {"vae": latent_vae, "discriminator": discriminator}

    def load_model(self, pretrained_model_ckpt):
        if pretrained_model_ckpt is None:
            return
        assert os.path.isdir(pretrained_model_ckpt), "Pretrained model checkpoint path is not a directory"

        for pretrained_model_path in os.listdir(pretrained_model_ckpt):
            pretrained_model_abs_path = os.path.join(pretrained_model_ckpt, pretrained_model_path)
            print(pretrained_model_abs_path)
            model_state_dict = torch.load(open(pretrained_model_abs_path, "rb"))
            if "vae" in pretrained_model_path:
                self.model["vae"].load_state_dict(model_state_dict)
            if "discriminator" in pretrained_model_path:
                self.model["discriminator"].load_state_dict(model_state_dict)
    
    def default_model_args(self):
        return {
            "vae": {"nc": 4, "z_dim": 32},
            "discriminator": discriminator_defaults()
        }

    def get_differentiate_value(self, image, timestep, encoder_hidden_states=None):
        # dummy_timestep = torch.zeros_like(timestep).to(image.device)

        # recon, z, mu, logvar = self.model["vae"](image, dummy_timestep)
        recon, z, mu, logvar = self.model["vae"](image)
        out = self.model["discriminator"](mu) # prob of real image
        out = F.sigmoid(out)
        out = torch.clip(out, 1e-5, 1. - 1e-5)
        out = 1 - out
        output_out = out.clone().detach()

        ratio = out / (1 - out)
        log_ratio = torch.log(ratio)
        diff_val = log_ratio.sum()
        # log = torch.log(out)
        # diff_val = log.sum()
        return diff_val, output_out

    def get_scaled_input(self, diffusion_pipeline, noisy_input, noise_pred, timestep):
        return get_predicted_x0(diffusion_pipeline, noisy_input, noise_pred, timestep)
    
    def guide_samples(self, diffusion_pipeline, noise_pred, prev_latents, latents, timestep, grad, scale):
        # diffusion_pipeline.scheduler.counter -= 1
        # diffusion_pipeline.scheduler.step_index -= 1
        # diffusion_pipeline.scheduler.step_index = diffusion_pipeline.scheduler.step_index - 1
        diffusion_pipeline.scheduler._step_index -= 1
        score = - noise_pred / torch.sqrt(1 - diffusion_pipeline.scheduler.alphas_cumprod[timestep]) 
        # https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#connection-with-noise-conditioned-score-networks-ncsn 
        discriminator_adjusted_score = score + scale * grad.detach()
        adjusted_noise_pred = - discriminator_adjusted_score * torch.sqrt(1 - diffusion_pipeline.scheduler.alphas_cumprod[timestep])
        latents = diffusion_pipeline.scheduler.step(adjusted_noise_pred, timestep, prev_latents, return_dict=False)[0]
        return latents

    def __call__(self, image, timestep=None, encoder_hidden_states=None):
        x_recon, z, mu, logvar = self.model["vae"](image)
        out = self.model["discriminator"](mu)
        return out

    def encode(self, images, timestep=None):
        return self.model["vae"]._encode(images)
    
    def decode(self, z, timestep=None):
        return self.model["vae"]._decode(z)
    
    def requires_grad(self, requires_grad):
        for model in self.model.values():
            for param in model.parameters():
                param.requires_grad = requires_grad
    
    
    def parameters(self):
        return {key: model.parameters() for key, model in self.model.items()}

class TimeDependentVAALGradientModel(GradientModel):
    def __init__(self, model_config_obj, device="cpu"):
        self.model_args = self.default_model_args()
        if model_config_obj is not None:
            self.model_args.update(model_config_obj)
        self.model = self.create_model(self.model_args)
        self.model["vae"] = self.model["vae"].to(device)
        self.model["discriminator"] = self.model["discriminator"].to(device)
    
    def create_model(self, model_config_args):
        latent_vae_args = model_config_args["vae"]
        discriminator_args = model_config_args["discriminator"]

        new_latent_vae_args = {}
        latent_vae_image_size = latent_vae_args.get("image_size", 64)
        attention_ds = []
        latent_vae_attention_resolutions = latent_vae_args.get("attention_resolutions", "32,16,8")
        for res in latent_vae_attention_resolutions.split(","):
            attention_ds.append(latent_vae_image_size // int(res))

        channel_mult = None
        if latent_vae_args.get("image_size") == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif latent_vae_args.get("image_size") == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif latent_vae_args.get("image_size") == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif latent_vae_args.get("image_size") == 64:
            channel_mult = (1, 2, 3, 4)
        elif latent_vae_args.get("image_size") == 32:
            # channel_mult = (1, 2, 3)
            channel_mult = (1, 2, 3, 4)
        else:
            image_size=latent_vae_args.get("image_size")
            raise ValueError(f"unsupported image size: {image_size}")
        new_latent_vae_args.update(
            {
                "image_size": latent_vae_image_size,
                "in_channels": latent_vae_args.get("in_channels", 4),
                "model_channels": latent_vae_args.get("classifier_width", 128),
                "out_channels": latent_vae_args.get("out_channels", 512),
                "use_scale_shift_norm": latent_vae_args.get("classifier_use_scale_shift_norm", True),
                "resblock_updown": latent_vae_args.get("classifier_resblock_updown", True),
                "pool": latent_vae_args.get("classifier_pool", "attention"),
                "num_res_blocks": latent_vae_args.get("classifier_depth", 2),
                "attention_resolutions": tuple(attention_ds),
                "num_head_channels": 64,
                "channel_mult": channel_mult,
            }
        )

        new_discriminator_args = {}
        discriminator_image_size = discriminator_args.get("image_size", 64)
        attention_ds = []
        discriminator_attention_resolutions = discriminator_args.get("classifier_attention_resolutions", "32,16,8")
        for res in discriminator_attention_resolutions.split(","):
            attention_ds.append(discriminator_image_size // int(res))
        
        channel_mult = None
        image_size = discriminator_args.get("image_size")
        if image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        elif image_size == 32:
            channel_mult = (1, 2, 4)
        elif image_size == 16:
            channel_mult = (1, 2)
        elif image_size == 8:
            channel_mult = (1,)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
        new_discriminator_args.update(
            {
                "image_size": discriminator_image_size,
                "in_channels": discriminator_args.get("in_channels", 4),
                "model_channels": discriminator_args.get("classifier_width", 128),
                "out_channels": discriminator_args.get("out_channels", 1),
                "use_scale_shift_norm": discriminator_args.get("classifier_use_scale_shift_norm", True),
                "resblock_updown": discriminator_args.get("classifier_resblock_updown", True),
                "pool": discriminator_args.get("classifier_pool", "attention"),
                "num_res_blocks": discriminator_args.get("classifier_depth", 2),
                "attention_resolutions": tuple(attention_ds),
                "num_head_channels": 64,
                "channel_mult": channel_mult,
            }
        )

        latent_vae = TimeDependentVAE(**new_latent_vae_args)
        discriminator = TimeDependentDiscriminator(**new_discriminator_args)
        return {"vae": latent_vae, "discriminator": discriminator}

    
    def load_model(self, pretrained_model_ckpt):
        if pretrained_model_ckpt is None:
            return
        
        if os.path.isdir(pretrained_model_ckpt):
            if "iter_" not in pretrained_model_ckpt.split("/")[-1] and \
                "checkpoint" == os.listdir(pretrained_model_ckpt)[0]:
                checkpoint_dir = os.path.join(pretrained_model_ckpt, "checkpoint")
                stored_ckpts = os.listdir(checkpoint_dir)
                assert "iter_" in stored_ckpts[0], "Pretrained model checkpoint path is not a directory"

                latest_ckpt = 0
                for stored_ckpt in stored_ckpts:
                    iter_num = int(stored_ckpt.split("_")[-1])
                    if iter_num > latest_ckpt:
                        latest_ckpt = iter_num
                pretrained_model_ckpt = os.path.join(checkpoint_dir, f"iter_{latest_ckpt}")

        assert os.path.isdir(pretrained_model_ckpt), "Pretrained model checkpoint path is not a directory"

        for pretrained_model_path in os.listdir(pretrained_model_ckpt):
            pretrained_model_abs_path = os.path.join(pretrained_model_ckpt, pretrained_model_path)
            print(pretrained_model_abs_path)
            model_state_dict = torch.load(open(pretrained_model_abs_path, "rb"))
            if "vae" in pretrained_model_path:
                self.model["vae"].load_state_dict(model_state_dict)
            if "discriminator" in pretrained_model_path:
                self.model["discriminator"].load_state_dict(model_state_dict)
    
    def default_model_args(self):
        return {
            "vae": discriminator_defaults(),
            "discriminator": discriminator_defaults()
        }

    def get_scaled_input(self, diffusion_pipeline, noisy_input, noise_pred, timestep):
        return get_noisy_xt(diffusion_pipeline, noisy_input, noise_pred, timestep)

    def get_differentiate_value(self, image, timestep, encoder_hidden_states=None):
        timestep = torch.tile(timestep, (image.shape[0], ))
        recon, z, mu, logvar = self.model["vae"](image, timestep)
        out = self.model["discriminator"](mu, timestep)
        out = F.sigmoid(out)
        out = torch.clip(out, 1e-5, 1. - 1e-5)
        out = 1 - out
        ratio = out / (1 - out)
        log_ratio = torch.log(ratio)

        diff_val = log_ratio.sum()
        output_for_log = out.clone().detach()

        return diff_val, output_for_log

    def guide_samples(self, diffusion_pipeline, noise_pred, prev_latents, latents, timestep, grad, scale):
        # diffusion_pipeline.scheduler.counter -= 1
        diffusion_pipeline.scheduler._step_index -= 1
        score = - noise_pred / torch.sqrt(1 - diffusion_pipeline.scheduler.alphas_cumprod[timestep]) 
        # https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#connection-with-noise-conditioned-score-networks-ncsn 
        discriminator_adjusted_score = score + scale * grad.detach()
        adjusted_noise_pred = - discriminator_adjusted_score * torch.sqrt(1 - diffusion_pipeline.scheduler.alphas_cumprod[timestep])
        latents = diffusion_pipeline.scheduler.step(adjusted_noise_pred, timestep, prev_latents, return_dict=False)[0]
        return latents

    def encode(self, images, timestep):
        return self.model["vae"].encode(images, timestep)
    
    def decode(self, z, timestep):
        return self.model["vae"].decode(z, timestep)

    def __call__(self, image, timestep, encoder_hidden_states=None):
        x_recon, z, mu, logvar = self.model["vae"](image, timestep)
        out = self.model["discriminator"](mu)
        return out


class AugmentedDiscriminatorGradientModel(GradientModel):
    def __init__(self, model_config_obj, device="cpu"):
        self.model = self.create_model(model_config_obj).to(device)
    
    def create_model(self, model_config_args):
        pretrained_model_name = "KaiChen1998/geodiffusion-coco-stuff-512x512"
        # use_pretrained_layer = model_config_args.get("use_pretrained_layer", True)
        use_pretrained_layer = model_config_args.pop("use_pretrained_layer", True)
        return AugmentedDiscriminator(pretrained_model_name, model_config_args, use_pretrained_layer=use_pretrained_layer)
    
    def load_model(self, pretrained_model_ckpt):
        if pretrained_model_ckpt is None:
            return
        model_state_dict = torch.load(open(pretrained_model_ckpt, "rb"))
        self.model.discriminator_head.load_state_dict(model_state_dict)

    def default_model_args(self):
        # return discriminator_defaults()
        pass

    def get_scaled_input(self, diffusion_pipeline, noisy_input, noise_pred, timestep):
        # return get_predicted_x0(diffusion_pipeline, noisy_input, noise_pred, timestep)
        return get_noisy_xt(diffusion_pipeline, noisy_input, noise_pred, timestep)

    def get_differentiate_value(self, image, timestep, encoder_hidden_states):
        timestep = torch.tile(timestep, (image.shape[0], )).to(image.device)
        out = self.model(image, timestep, encoder_hidden_states)
        output = F.sigmoid(out)
        output = torch.clip(output, 1e-5, 1. - 1e-5)
        output = 1 - output

        output_for_log = output.clone().detach()

        ratio = output / (1 - output)
        log_ratio = torch.log(ratio)

        diff_val = log_ratio.sum()
        return diff_val, output_for_log

    def guide_samples(self, diffusion_pipeline, noise_pred, prev_latents, latents, timestep, grad, scale):
        # return latents - grad * scale
        # diffusion_pipeline.scheduler.counter -= 1
        diffusion_pipeline.scheduler._step_index -= 1
        score = - noise_pred / torch.sqrt(1 - diffusion_pipeline.scheduler.alphas_cumprod[timestep]) 
        # https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#connection-with-noise-conditioned-score-networks-ncsn 
        discriminator_adjusted_score = score + scale * grad.detach()
        adjusted_noise_pred = - discriminator_adjusted_score * torch.sqrt(1 - diffusion_pipeline.scheduler.alphas_cumprod[timestep])
        latents = diffusion_pipeline.scheduler.step(adjusted_noise_pred, timestep, prev_latents, return_dict=False)[0]
        return latents

    def __call__(self, image, timestep, encoder_hidden_states):
        # dummy_timestep = torch.zeros_like(timestep).to(image.device)
        # return self.model(image, dummy_timestep, encoder_hidden_states)
        return self.model(image, timestep, encoder_hidden_states)

class ObjectDetectionDiscriminatorGradientModel(GradientModel):
    def __init__(self, model_config_obj, device="cpu"):
        self.pretrained_yolo_model_ckpt_path = model_config_obj.get("pretrained_yolo_model", None)
        self.model = self.create_model(model_config_obj)
        for key in self.model:
            self.model[key] = self.model[key].to(device)

    def create_model(self, model_config_args):
        # pretrained_yolo_model_ckpt_path = model_config_args.get("pretrained_yolo_model", None)
        pretrained_yolo_model_ckpt_path = model_config_args.pop("pretrained_yolo_model", None)
        if pretrained_yolo_model_ckpt_path is None:
            raise ValueError("pretrained_yolo_model is not provided")
        # pretrained_yolo_model = YOLO(model=pretrained_yolo_model_ckpt_path)
        pretrained_yolo_model = CustomYOLO(model=pretrained_yolo_model_ckpt_path)

        discriminator_args = model_config_args.pop("discriminator_args", None)
        return {"object_detection_model": pretrained_yolo_model, 
                "discriminator": create_classifier(**discriminator_args)}
    
    def load_model(self, pretrained_model_ckpt):
        if pretrained_model_ckpt is None:
            return
        model_state_dict = torch.load(open(pretrained_model_ckpt, "rb"))
        self.model["discriminator"].load_state_dict(model_state_dict)
    
    def default_model_args(self):
        pass

    def get_scaled_input(self, diffusion_pipeline, noisy_input, noise_pred, timestep):
        x0 = get_predicted_x0(diffusion_pipeline, noisy_input, noise_pred, timestep)
        x0 = x0.to(diffusion_pipeline.vae.device)
        decoded_x0 = diffusion_pipeline.vae.decode(x0 / diffusion_pipeline.vae.config.scaling_factor, return_dict=False)[0]
        decoded_x0 = diffusion_pipeline.image_processor.postprocess(image=decoded_x0, output_type="pt").to(diffusion_pipeline.vae.device)
        return decoded_x0
    
    def get_differentiate_value(self, image, timestep=None, encoder_hidden_states=None):
        out = self.model["object_detection_model"](image)
        disc_out = self.model["discriminator"](out, timestep)
        output = F.sigmoid(disc_out)
        output = torch.clip(output, 1e-5, 1. - 1e-5)
        output = 1 - output

        output_for_log = output.clone().detach()

        ratio = output / (1 - output)
        log_ratio = torch.log(ratio)

        diff_val = log_ratio.sum()
        return diff_val, output_for_log
    
    def guide_samples(self, diffusion_pipeline, noise_pred, prev_latents, latents, timestep, grad, scale):
        # return latents - grad * scale
        # diffusion_pipeline.scheduler.counter -= 1
        diffusion_pipeline.scheduler._step_index -= 1
        score = - noise_pred / torch.sqrt(1 - diffusion_pipeline.scheduler.alphas_cumprod[timestep]) 
        # https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#connection-with-noise-conditioned-score-networks-ncsn 
        discriminator_adjusted_score = score + scale * grad.detach()
        adjusted_noise_pred = - discriminator_adjusted_score * torch.sqrt(1 - diffusion_pipeline.scheduler.alphas_cumprod[timestep])
        latents = diffusion_pipeline.scheduler.step(adjusted_noise_pred, timestep, prev_latents, return_dict=False)[0]
        return latents
    
    def __call__(self, image, timestep, encoder_hidden_states):
        out = self.model["object_detection_model"](image)
        disc_out = self.model["discriminator"](out)
        return disc_out





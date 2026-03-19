import sys
import yaml
from abc import * 

import torch
import torch.nn.functional as F

from geo_utils.gradient_model_utils import GradientModel
from geo_utils.gradient_model_utils import ClassifierGradientModel
from geo_utils.gradient_model_utils import YOLOGradientModel
from geo_utils.gradient_model_utils import DiscriminatorGradientModel,TimeDependentDiscriminatorGradientModel
from geo_utils.gradient_model_utils import VAALGradientModel, TimeDependentVAALGradientModel
from geo_utils.gradient_model_utils import AugmentedDiscriminatorGradientModel
from geo_utils.gradient_model_utils import ObjectDetectionDiscriminatorGradientModel



class GuidanceModel:
    gradient_model: GradientModel
    def __init__(self, diffusion_pipeline, model_config_file, model_ckpt_path, target_class, device="cpu"):

        self.diffusion_pipeline = diffusion_pipeline

        with open(model_config_file, "r") as f:
            model_args = yaml.safe_load(f)
        model_type = model_args.get("model_type")
        model_args = model_args.get("model_args")
        self.device = device
        self.target_class = target_class
        self.gradient_model = self.load_gradient_model(model_type, model_args)
        self.gradient_model.load_model(model_ckpt_path)

    def load_gradient_model(self, model_type, model_config_obj):
        # model type
        # classifier, yolo, discriminator, time_dependent_discriminator
        if model_type == "classifier":
            return ClassifierGradientModel(model_config_obj, self.device)
        elif model_type == "yolo":
            return YOLOGradientModel(model_config_obj, self.device)
        elif model_type == "discriminator":
            return DiscriminatorGradientModel(model_config_obj, self.device)
        elif model_type == "time_dependent_discriminator":
            return TimeDependentDiscriminatorGradientModel(model_config_obj, self.device)
        elif model_type == "vaal":
            return VAALGradientModel(model_config_obj, self.device)
        elif model_type == "time_dependent_vaal":
            return TimeDependentVAALGradientModel(model_config_obj, self.device)
        elif model_type == "augmented_discriminator":
            return AugmentedDiscriminatorGradientModel(model_config_obj, self.device)
        elif model_type == "object_detection_discriminator":
            return ObjectDetectionDiscriminatorGradientModel(model_config_obj, self.device)
        else:
            raise ValueError(f"unknown model type: {model_type}")

    def guidance(self, diffusion_pipeline, callback_kwargs, step, timestep, scale, target_class = None):
        with torch.enable_grad():
            prev_latents = callback_kwargs["prev_latents"].clone().detach().requires_grad_(True)
            latents = callback_kwargs["latents"] # x_t-1
            noise_pred = callback_kwargs["noise_pred"]
            # encoder_hidden_states = callback_kwargs["prompt_embeds"]
            encoder_hidden_states = callback_kwargs["instance_prompt_embeds"] if callback_kwargs.get("instance_prompt_embeds", None) is not None else None
            if encoder_hidden_states is not None and encoder_hidden_states.shape[0] == 2 * latents.shape[0]:
                encoder_hidden_states = encoder_hidden_states[latents.shape[0]:]
            ## train code 수정
            # normalized_timestep = timestep / diffusion_pipeline.scheduler.num_train_timesteps
            normalized_timestep = timestep
            ### 수정 끝
            
            scaled_latent = self.gradient_model.get_scaled_input(diffusion_pipeline, prev_latents, noise_pred, timestep)
            
            scaled_latent = scaled_latent.to(diffusion_pipeline.device)
            normalized_timestep = normalized_timestep.to(diffusion_pipeline.device)
            prev_latents = prev_latents.to(diffusion_pipeline.device)
            
            tc = target_class if target_class is not None else self.target_class
            # differentiate_value, grad = self.gradient_model.get_grad(scaled_latent, normalized_timestep, encoder_hidden_states, prev_latents)
            differentiate_value, grad, output_for_log = self.gradient_model.get_grad(scaled_latent, normalized_timestep, encoder_hidden_states, prev_latents, target_class=tc)

        grad = grad.detach()
        bbox_binary_mask = callback_kwargs.get("bbox_binary_mask", None)
        if bbox_binary_mask is not None:
            # print(grad.shape)
            # print(bbox_binary_mask.shape)
            grad = grad * bbox_binary_mask
        latents = self.gradient_model.guide_samples(diffusion_pipeline, noise_pred, prev_latents, latents, timestep, grad, scale)
        # return {"latents": latents}
        return_diff_value = output_for_log.detach()
        return {"latents": latents, "differentiate_value": return_diff_value}

    def get_gradient_model(self):
        return self.gradient_model
    
    def __call__(self, image, timestep):
        return self.gradient_model(image, timestep)
    
    def eval(self):
        self.gradient_model.eval()

    def train(self):
        self.gradient_model.train()
    
    def requires_grad(self, requires_grad):
        self.gradient_model.requires_grad(requires_grad)
    
    def parameters(self):
        return self.gradient_model.parameters()




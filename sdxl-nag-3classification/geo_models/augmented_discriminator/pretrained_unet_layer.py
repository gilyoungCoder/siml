import torch
import torch.nn as nn  

from diffusers import UNet2DConditionModel
from typing import Optional, Union, Dict, Any



class PretrainedUNetLayer(nn.Module):
    def __init__(self, pretrained_unet_name="KaiChen1998/geodiffusion-coco-stuff-512x512"):
        # noisy_latents, timesteps, encoder_hidden_states

        # super(nn.Module, self).__init__()
        super().__init__()

        pretrained_unet_layer = UNet2DConditionModel.from_pretrained(pretrained_unet_name, subfolder="unet")
        self.conv_in = pretrained_unet_layer.conv_in
        self.time_proj = pretrained_unet_layer.time_proj
        self.time_embedding = pretrained_unet_layer.time_embedding
        self.class_embedding = pretrained_unet_layer.class_embedding
        self.first_down_block = pretrained_unet_layer.down_blocks[0]
        self.get_aug_embed = pretrained_unet_layer.get_aug_embed
        self.encoder_hid_proj = pretrained_unet_layer.encoder_hid_proj
        # self.encoder_hid_type = None
    
    def get_time_embed(
            self, sample: torch.Tensor, timestep: Union[torch.Tensor, float, int]
        ) -> Optional[torch.Tensor]:
            timesteps = timestep
            if not torch.is_tensor(timesteps):
                # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                # This would be a good case for the `match` statement (Python 3.10+)
                is_mps = sample.device.type == "mps"
                if isinstance(timestep, float):
                    dtype = torch.float32 if is_mps else torch.float64
                else:
                    dtype = torch.int32 if is_mps else torch.int64
                timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
            elif len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(sample.device)

            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timesteps = timesteps.expand(sample.shape[0])

            t_emb = self.time_proj(timesteps)
            # `Timesteps` does not contain any weights and will always return f32 tensors
            # but time_embedding might actually be running in fp16. so we need to cast here.
            # there might be better ways to encapsulate this.
            t_emb = t_emb.to(dtype=sample.dtype)
            return t_emb

    def get_class_embed(self, sample: torch.Tensor, class_labels: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        class_emb = None
        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

                # `Timesteps` does not contain any weights and will always return f32 tensors
                # there might be better ways to encapsulate this.
                class_labels = class_labels.to(dtype=sample.dtype)

            class_emb = self.class_embedding(class_labels).to(dtype=sample.dtype)
        return class_emb

    
    def process_encoder_hidden_states(
        self, encoder_hidden_states: torch.Tensor, added_cond_kwargs: Dict[str, Any]
    ) -> torch.Tensor:
        if self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "text_proj":
            encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states)
        elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "text_image_proj":
            # Kandinsky 2.1 - style
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'text_image_proj' which requires the keyword argument `image_embeds` to be passed in `added_cond_kwargs`"
                )

            image_embeds = added_cond_kwargs.get("image_embeds")
            encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states, image_embeds)
        elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "image_proj":
            # Kandinsky 2.2 - style
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'image_proj' which requires the keyword argument `image_embeds` to be passed in `added_cond_kwargs`"
                )
            image_embeds = added_cond_kwargs.get("image_embeds")
            encoder_hidden_states = self.encoder_hid_proj(image_embeds)
        elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "ip_image_proj":
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'ip_image_proj' which requires the keyword argument `image_embeds` to be passed in `added_cond_kwargs`"
                )

            if hasattr(self, "text_encoder_hid_proj") and self.text_encoder_hid_proj is not None:
                encoder_hidden_states = self.text_encoder_hid_proj(encoder_hidden_states)

            image_embeds = added_cond_kwargs.get("image_embeds")
            image_embeds = self.encoder_hid_proj(image_embeds)
            encoder_hidden_states = (encoder_hidden_states, image_embeds)
        return encoder_hidden_states


    def forward(self, sample, timestep, encoder_hidden_states):
        t_emb = self.get_time_embed(sample=sample, timestep=timestep)
        emb = self.time_embedding(t_emb)

        class_emb = self.get_class_embed(sample=sample, class_labels=None)
        
        aug_emb = self.get_aug_embed(emb=emb, encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=None)

        emb = emb + aug_emb if aug_emb is not None else emb

        encoder_hidden_states = self.process_encoder_hidden_states(
            encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=None)

        sample = self.conv_in(sample)

        down_block_res_samples = (sample,)

        if hasattr(self.first_down_block, "has_cross_attention") and self.first_down_block.has_cross_attention:
            down_block_res_samples = self.first_down_block(
                hidden_states = sample,
                temb=emb,
                encoder_hidden_states=encoder_hidden_states)
        down_block_res_samples = down_block_res_samples[0]
        return down_block_res_samples
        # return down_block_res_samples, emb, encoder_hidden_states

    


if __name__ == "__main__":

    from torch.utils.data import DataLoader
    from transformers import CLIPTokenizer, CLIPTextModel
    from diffusers import AutoencoderKL
    from geo_utils.data.new_coco_stuff import NewCOCOStuffDataset
    from mmengine.config import Config

    pretrained_model_name = "KaiChen1998/geodiffusion-coco-stuff-512x512"
    model = PretrainedUNetLayer()

    dataset_args = dict(
        prompt_version="v1", 
        num_bucket_per_side=[256, 256],
        foreground_loss_mode=True, 
        foreground_loss_weight=1.0,
        foreground_loss_norm=1.0,
        feat_size=64,
    )
    dataset_args_train = dict(
        uncond_prob=0.0,
    )
    dataset_cfg = Config.fromfile("/mnt/home/jeongjun/layout_diffusion/geodiffusion_augmented_disc/geodiffusion/configs/data/new_coco_stuff_512x512.py")
    dataset_cfg.data.train.update(**dataset_args)
    dataset_cfg.data.train.update(**dataset_args_train)

    

    dataset_cfg.data.train.ann_file = "/mnt/home/jeongjun/layout_diffusion/geodiffusion_augmented_disc/coco-stuff-instance/instances_stuff_train2017.json"
    dataset_cfg.data.val.ann_file = "/mnt/home/jeongjun/layout_diffusion/geodiffusion_augmented_disc/coco-stuff-instance/instances_stuff_val2017.json"

    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_name, subfolder="vae")

    train_dataset = NewCOCOStuffDataset(
                **dataset_cfg.data.train, blip_finetune=False, tokenizer=tokenizer, is_main_process=False)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    dummy_data = next(iter(train_dataloader))

    encoder_hidden_states = text_encoder(dummy_data["input_ids"])[0]

    # latent_code = vae(dummy_data["pixel_values"], encoder_hidden_states)
    latent_code = vae.encode(dummy_data["pixel_values"]).latent_dist.sample()
    latent_code = latent_code * 0.18215


    output = model(latent_code, 0, encoder_hidden_states)
    output = output[0]
    # print(output.shape)

    # original unet2dcondition model
    original_model = UNet2DConditionModel.from_pretrained(pretrained_model_name, subfolder="unet")
    original_output = original_model(latent_code, 0, encoder_hidden_states, return_dict=False)

    # print(original_output[3] == output) # SUCCESS!

    # print(model)
    # print("Done")
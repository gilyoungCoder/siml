
from .pretrained_unet_layer import PretrainedUNetLayer
from .head import DiscriminatorHead


import torch
import torch.nn as nn

class AugmentedDiscriminator(nn.Module):
    def __init__(self, pretrained_unet_name="KaiChen1998/geodiffusion-coco-stuff-512x512", head_config=None, use_pretrained_layer=True):
        super().__init__()
        self.pretrained_unet_layer = PretrainedUNetLayer(pretrained_unet_name) if use_pretrained_layer else nn.Identity()
        self.discriminator_head = DiscriminatorHead(head_config)

    # def __call__(self, x, t, encoder_hidden_states):
    def forward(self, x, t, encoder_hidden_states):
        if not isinstance(self.pretrained_unet_layer, nn.Identity):
            x = self.pretrained_unet_layer(x, t, encoder_hidden_states)
        x = self.discriminator_head(x, t, encoder_hidden_states)
        return x

        
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from transformers import CLIPTokenizer, CLIPTextModel
    from diffusers import AutoencoderKL
    from mmengine import Config
    from geo_utils.data.new_coco_stuff import NewCOCOStuffDataset
    from geo_models.augmented_discriminator.pretrained_unet_layer import PretrainedUNetLayer


    yaml_config = "/mnt/home/jeongjun/layout_diffusion/geodiffusion_augmented_disc/geodiffusion/configs/models/augmented_discriminator.yaml"
    pretrained_model_name = "KaiChen1998/geodiffusion-coco-stuff-512x512"
    model = AugmentedDiscriminator(pretrained_model_name, yaml_config)

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

    timestep = torch.tensor([0.0] * len(encoder_hidden_states))

    output = model(latent_code, timestep, encoder_hidden_states)


import torch
import torch.nn as nn
import torch.nn.functional as F

import yaml

##
# import sys
# sys.path.append("/mnt/home/jeongjun/layout_diffusion/geodiffusion_augmented_disc/geodiffusion")
##

from ..classifier.classifier import create_classifier
from ..classifier.encoder_unet_model import EncoderUNetModelForClassification, TimestepEmbedSequential, TimestepBlock, AttentionBlock
from ..classifier.nn import timestep_embedding

class CrossAttentionModule(nn.Module):
    def __init__(self, inner_dim, query_dim, out_dim, num_head,  cross_attention_dim):
        super().__init__()

        self.to_q = nn.Linear(query_dim, inner_dim)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim)
        self.to_out = nn.Linear(inner_dim, out_dim)
        self.num_head = num_head

        self.scale = inner_dim ** -0.5
        self.rescale_output_factor = 1.0

        self.hidden_state_norm = nn.GroupNorm(32, query_dim, eps=1e-5)
        self.encoder_hidden_state_norm = nn.GroupNorm(32, cross_attention_dim, eps=1e-5)
    
    def head_to_batch_dim(self, tensor, out_dim=3):
        """
        Convert [batch, seq_len, dim] to [batch, seq_len, num_head, dim // num_head]
        """
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size, seq_len, self.num_head, dim // self.num_head)
        tensor = tensor.permute(0, 2, 1, 3)

        if out_dim == 3:
            # tensor = tensor.contiguous().view(batch_size * self.num_head, seq_len, dim // self.num_head)
            tensor = tensor.reshape(batch_size * self.num_head, seq_len, dim // self.num_head)
        return tensor
    
    def batch_to_head_dim(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size // self.num_head, self.num_head, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // self.num_head, seq_len, dim * self.num_head)
        return tensor
    
    def get_attention_scores(self, q, k, attention_mask=None):
        """
        q: [batch, num_head, seq_len_q, dim]
        k: [batch, num_head, seq_len_k, dim]
        """
        if attention_mask is None:
            baddbmm_input = torch.empty(
                q.shape[0], q.shape[1], k.shape[1], dtype=q.dtype, device=q.device)
            beta = 0
        else:
            baddbmm_input = attention_mask
            beta = 1

        attention_scores = torch.baddbmm(
            baddbmm_input, q, k.transpose(-1, -2),
            beta=beta, alpha=self.scale
        )

        del baddbmm_input

        attention_probs = attention_scores.softmax(dim=-1)
        del attention_scores
        return attention_probs


    def get_output_for_continous_input(self, hidden_states, batch, height, width):
        """
        Convert [batch, height * width, dim] to [batch, dim, height, width]
        """
        hidden_states = hidden_states.reshape(batch, height, width, -1).permute(0, 3, 1, 2).contiguous()
        hidden_states = self.to_out(hidden_states)
        return hidden_states

    def forward(self, hidden_states, temb, encoder_hidden_states):
        # hidden_states: 320, 32, 32
        # temb: 320, 32
        # encoder_hidden_states: 320, 32, 32

        original_hidden_states = hidden_states
        batch, inner_dim, height, width = hidden_states.shape
        hidden_states = hidden_states.reshape(batch, inner_dim, height * width).transpose(1, 2)

        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.hidden_state_norm(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)

        encoder_hidden_states = encoder_hidden_states.transpose(1, 2)
        encoder_hidden_states = self.encoder_hidden_state_norm(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states.transpose(1, 2)

        q = self.to_q(hidden_states)
        k = self.to_k(encoder_hidden_states)
        v = self.to_v(encoder_hidden_states)

        q = self.head_to_batch_dim(q)
        k = self.head_to_batch_dim(k)
        v = self.head_to_batch_dim(v)

        # attention_probs = self.get_attention_scores(q, k)
        # hidden_states = torch.bmm(attention_probs, v)
        hidden_states = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch, -1, inner_dim)

        # hidden_states = self.batch_to_head_dim(hidden_states)


        # hidden_states = self.get_output_for_continous_input(hidden_states, batch, height, width)
        hidden_states = self.to_out(hidden_states)
        hidden_states = hidden_states.transpose(-1, -2).reshape(batch, inner_dim, height, width)
        # hidden_states = hidden_states.transpose(1, 2)

        hidden_states = hidden_states + original_hidden_states

        hidden_states = hidden_states / self.rescale_output_factor

        return hidden_states

class TimestepEmbedLabelSequential(TimestepEmbedSequential):
    def forward(self, x, emb, label):
        for layer in self:
            if isinstance(layer, CrossAttentionModule):
                x = layer(x, emb, label)
            elif isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class DiscriminatorHead(nn.Module):
    def __init__(self, config): 
        super().__init__()
        self.discriminator = create_classifier(**config)
        cross_attention_config = {
            "num_head": 8, 
            "cross_attention_dim": 768
        }

        for i, input_block in enumerate(self.discriminator.encoder_model.input_blocks):
            if isinstance(input_block, TimestepEmbedSequential):
                new_input_block = []
                for layer in input_block:
                    if isinstance(layer, AttentionBlock):
                        update_cross_attention_config = {
                            "inner_dim": layer.channels, 
                            "out_dim": layer.channels, 
                            "query_dim": layer.channels,
                        }
                        cross_attention_config.update(update_cross_attention_config)
                        cross_attention = CrossAttentionModule(**cross_attention_config)
                        new_input_block.append(cross_attention)
                    else:
                        new_input_block.append(layer)
                new_input_block = TimestepEmbedLabelSequential(
                    *new_input_block
                )

                self.discriminator.encoder_model.input_blocks[i] = new_input_block

    def forward(self, x, timesteps=None, encoder_output=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x K] Tensor of outputs.
        """
        if timesteps is None:
            timesteps = torch.zeros(x.size(0), ).to(x.device)
        # emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        emb = self.discriminator.encoder_model.time_embed(timestep_embedding(timesteps, self.discriminator.encoder_model.model_channels))

        results = []
        h = x
        for module in self.discriminator.encoder_model.input_blocks:
            h = module(h, emb, encoder_output)
            # tmp_h = h.detach().cpu().mean().numpy()
            # print(tmp_h)
        
        # h = self.middle_block(h, emb)
        h = self.discriminator.encoder_model.middle_block(h, emb)


        if self.discriminator.encoder_model.training_vae:
            h = self.discriminator.encoder_model.split_mean_std(h)

        if self.discriminator.pool.startswith("spatial"):
            results.append(h.type(x.dtype).mean(dim=(2, 3)))
            h = torch.cat(results, axis=-1)
            return self.discriminator.out(h)
        else:
            return self.discriminator.out(h)
        

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from transformers import CLIPTokenizer, CLIPTextModel
    from diffusers import AutoencoderKL
    from mmengine import Config
    from geo_utils.data.new_coco_stuff import NewCOCOStuffDataset
    from geo_models.augmented_discriminator.pretrained_unet_layer import PretrainedUNetLayer


    yaml_config = "/mnt/home/jeongjun/layout_diffusion/geodiffusion_augmented_disc/geodiffusion/configs/models/augmented_discriminator.yaml"
    model = DiscriminatorHead(yaml_config)
    pretrained_unet_layer = PretrainedUNetLayer()

    

    pretrained_model_name = "KaiChen1998/geodiffusion-coco-stuff-512x512"

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

    # pretrained_unet_layer_output, timestep, encoder_hidden_states = pretrained_unet_layer(latent_code, timestep, encoder_hidden_states)
    pretrained_unet_layer_output = pretrained_unet_layer(latent_code, timestep, encoder_hidden_states)

    output = model(pretrained_unet_layer_output, timestep, encoder_hidden_states)
    print(output)
    breakpoint()
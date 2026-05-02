import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from ..classifier.encoder_unet_model import EncoderUNetModel, DecoderUNetModel, EncoderUNetModelForClassification

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

class VAEEncoder(nn.Module):
    def __init__(self, input_resolution=64, z_dim=32, nc=3, in_place=False):
        super().__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.in_place = in_place # True

        self.latent_resolution = input_resolution // 16
        groupnorm_num_groups = 32
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 128, 4, 2, 1, bias=False),              # B,  128, 32, 32
            # nn.BatchNorm2d(128),
            nn.GroupNorm(groupnorm_num_groups, 128),
            nn.ReLU(self.in_place),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),             # B,  256, 16, 16
            # nn.BatchNorm2d(256),
            nn.GroupNorm(groupnorm_num_groups, 256),
            nn.ReLU(self.in_place),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),             # B,  512,  8,  8
            # nn.BatchNorm2d(512),
            nn.GroupNorm(groupnorm_num_groups, 512),
            nn.ReLU(self.in_place),
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),            # B, 1024,  4,  4
            # nn.BatchNorm2d(1024),
            nn.GroupNorm(groupnorm_num_groups, 1024),
            nn.ReLU(self.in_place),
            View((-1, 1024 * self.latent_resolution * self.latent_resolution)),                                 # B, 1024*4*4
        )

        self.fc_mu = nn.Linear(1024 * self.latent_resolution * self.latent_resolution, z_dim)                            # B, z_dim
        self.fc_logvar = nn.Linear(1024 * self.latent_resolution*self.latent_resolution, z_dim)                            # B, z_dim
    
    def weight_init(self):
        for block in self._modules:
            try:
                for m in self._modules[block]:
                    kaiming_init(m)
            except:
                kaiming_init(block)
    
    def forward(self, x, timesteps=None):
        z = self._encode(x)
        mu, logvar = self.fc_mu(z), self.fc_logvar(z)
        return mu, logvar
    
    def _encode(self, x, timesteps=None):
        return self.encoder(x)

class VAEDecoder(nn.Module):
    def __init__(self, input_resolution=64, z_dim=32, nc=3, in_place=False):
        super().__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.in_place = in_place # True

        self.latent_resolution = input_resolution // 16
        groupnorm_num_groups = 32
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 1024 * (self.latent_resolution * 2) * (self.latent_resolution * 2)),                           # B, 1024*8*8
            View((-1, 1024, (self.latent_resolution * 2), (self.latent_resolution * 2))),                               # B, 1024,  8,  8
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),   # B,  512, 16, 16
            # nn.BatchNorm2d(512),
            nn.GroupNorm(groupnorm_num_groups, 512),
            nn.ReLU(self.in_place),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),    # B,  256, 32, 32
            # nn.BatchNorm2d(256),
            nn.GroupNorm(groupnorm_num_groups, 256),
            nn.ReLU(self.in_place),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),    # B,  128, 64, 64
            # nn.BatchNorm2d(128),
            nn.GroupNorm(groupnorm_num_groups, 128),
            nn.ReLU(self.in_place),
            nn.ConvTranspose2d(128, nc, 1),                       # B,   nc, 64, 64
        )
    
    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)
    
    def forward(self, z, timesteps=None):
        return self.decoder(z)

class VAE(nn.Module):
    """Encoder-Decoder architecture for both WAE-MMD and WAE-GAN."""
    def __init__(self, input_resolution=64, z_dim=32, nc=3, in_place=False):
        super(VAE, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.in_place = in_place # True

        self.latent_resolution = input_resolution // 16
        self.encoder = VAEEncoder(input_resolution, z_dim, nc, in_place)
        self.decoder = VAEDecoder(input_resolution, z_dim, nc, in_place)

    def weight_init(self):
        for block in self._modules:
            try:
                for m in self._modules[block]:
                    kaiming_init(m)
            except:
                kaiming_init(block)

    def forward(self, x, timesteps=None):
        # z = self._encode(x)
        mu, logvar = self.encode(x, timesteps)
        z = self.reparameterize(mu, logvar)
        x_recon = self._decode(z)

        return x_recon, z, mu, logvar

    def reparameterize(self, mu, logvar):
        stds = (0.5 * logvar).exp()
        epsilon = torch.randn(*mu.size())
        if mu.is_cuda:
            stds, epsilon = stds.cuda(), epsilon.cuda()
        latents = epsilon * stds + mu
        return latents

    def _encode(self, x, timesteps=None):
        return self.encoder(x)

    def _decode(self, z, timesteps=None):
        return self.decoder(z)
    
    def encode(self, x, timesteps=None):
        # z = self._encode(x)
        # mu, logvar = self.fc_mu(z), self.fc_logvar(z)
        mu, logvar = self.encoder(x)
        return mu, logvar

    def decode(self, z, timesteps=None):
        return self._decode(z)


class Discriminator(nn.Module):
    """Adversary architecture(Discriminator) for WAE-GAN."""
    def __init__(self, z_dim=10, in_place=False):
        super(Discriminator, self).__init__()
        self.z_dim = z_dim
        self.in_place = in_place
        self.net = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.ReLU(self.in_place),
            nn.Linear(512, 512),
            nn.ReLU(self.in_place),
            nn.Linear(512, 1),
            # nn.Sigmoid()
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, z, timesteps=None):
        return self.net(z)


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()

class TimeDependentVAE(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0.3,
        channel_mult=(1, 2, 4, 8), 
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        pool="adaptive",
    ):
        super(TimeDependentVAE, self).__init__()
        self.out_channels = out_channels
        encoder_params = {
            "image_size": image_size,
            "in_channels": in_channels,
            "model_channels": model_channels,
            "out_channels": out_channels,
            "num_res_blocks": num_res_blocks,
            "attention_resolutions": attention_resolutions,
            "dropout": dropout,
            "channel_mult": channel_mult,
            "conv_resample": conv_resample,
            "dims": dims,
            "use_checkpoint": use_checkpoint,
            "use_fp16": use_fp16,
            "num_heads": num_heads,
            "num_head_channels": num_head_channels,
            "num_heads_upsample": num_heads_upsample,
            "use_scale_shift_norm": use_scale_shift_norm,
            "resblock_updown": resblock_updown,
            "use_new_attention_order": use_new_attention_order,
            "pool": pool,
            "training_vae": True, 
            # Training vae option will take care of the multiplication of out_channels by 2
        }
        
        decoder_image_size = image_size // (2 ** (len(channel_mult) - 1))
        # decoder_in_channels = out_channels
        # decoder_out_channels = in_channels
        decoder_channel_mult = channel_mult[::-1]
        decoder_params = {
            "image_size": decoder_image_size,
            "in_channels": in_channels,
            "model_channels": model_channels,
            "out_channels": out_channels,
            "num_res_blocks": num_res_blocks,
            "attention_resolutions": attention_resolutions,
            "dropout": dropout,
            "channel_mult": decoder_channel_mult,
            "conv_resample": conv_resample,
            "dims": dims,
            "use_checkpoint": use_checkpoint,
            "use_fp16": use_fp16,
            "num_heads": num_heads,
            "num_head_channels": num_head_channels,
            "num_heads_upsample": num_heads_upsample,
            "use_scale_shift_norm": use_scale_shift_norm,
            "resblock_updown": resblock_updown,
            "use_new_attention_order": use_new_attention_order,
            "pool": pool,
        }

        # print(decoder_params)

        self.encoder = EncoderUNetModel(**encoder_params)
        self.decoder = DecoderUNetModel(**decoder_params)
    
    def forward(self, x, timesteps):
        mu, logvar = self.encode(x, timesteps)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, timesteps)
        return x_recon, z, mu, logvar

    def encode(self, x, timesteps):
        z = self.encoder(x, timesteps)
        # mu, logvar = z[:, :self.out_channels], z[:, self.out_channels:]
        mu, logvar = torch.chunk(z, 2, dim=1)
        return mu, logvar

    def decode(self, z, timesteps):
        return self.decoder(z, timesteps)
    
    def reparameterize(self, mu, logvar):
        stds = (0.5 * logvar).exp().to(mu.dtype)
        epsilon = torch.randn(*mu.size(), dtype=mu.dtype, device=mu.device)
        latents = epsilon * stds + mu
        return latents


        
class TimeDependentDiscriminator(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0.3,
        channel_mult=(1, 2, 4, 8), 
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        pool="adaptive",
    ):
        super(TimeDependentDiscriminator, self).__init__()
        encoder_params = {
            "image_size": image_size,
            "in_channels": in_channels,
            "model_channels": model_channels,
            "out_channels": out_channels,
            "num_res_blocks": num_res_blocks,
            "attention_resolutions": attention_resolutions,
            "dropout": dropout,
            "channel_mult": channel_mult,
            "conv_resample": conv_resample,
            "dims": dims,
            "use_checkpoint": use_checkpoint,
            "use_fp16": use_fp16,
            "num_heads": num_heads,
            "num_head_channels": num_head_channels,
            "num_heads_upsample": num_heads_upsample,
            "use_scale_shift_norm": use_scale_shift_norm,
            "resblock_updown": resblock_updown,
            "use_new_attention_order": use_new_attention_order,
            "pool": pool,
        }
        # self.discriminator = EncoderUNetModel(**encoder_params)
        self.discriminator = EncoderUNetModelForClassification(**encoder_params)
    
    def forward(self, mu, timesteps):
        return self.discriminator(mu, timesteps)


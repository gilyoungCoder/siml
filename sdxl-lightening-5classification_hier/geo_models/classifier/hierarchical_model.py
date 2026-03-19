# hierarchical_model.py

import torch.nn as nn
from .encoder_unet_model import EncoderUNetModelForClassification, AttentionPool2d, normalization

class HierarchicalUNetClassifier(nn.Module):
    def __init__(self, base_args):
        super().__init__()
        # ① feature extractor (pool="none" 으로 raw feature map 반환)
        self.encoder = EncoderUNetModelForClassification(
            **{**base_args, "pool": "none"}
        )
        # 채널 수 계산
        ch = base_args["model_channels"] * base_args["channel_mult"][-1]
        # spatial downsampling factor
        ds = base_args["image_size"] // (2 ** (len(base_args["channel_mult"]) - 1))

        # ② AttentionPool2d 를 그대로 사용
        self.attn_pool = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            AttentionPool2d(
                spacial_dim=ds,
                embed_dim=ch,
                num_heads_channels=base_args["num_head_channels"],
                output_dim=ch,          # pooling 뒤에도 같은 채널 수로 유지
            ),
        )

        # ③ 계층별 head 정의
        self.head_root    = nn.Linear(ch, 2)
        self.head_lvl1    = nn.Linear(ch, 2)
        self.head_clothed = nn.Sequential(
            nn.Linear(ch, ch),
            nn.ReLU(),
            nn.Linear(ch, 2),
        )
        self.head_nude    = nn.Sequential(
            nn.Linear(ch, ch),
            nn.ReLU(),
            nn.Linear(ch, 2),
        )

    def forward(self, x, timesteps=None):
        # 1) raw feature map [B, C, H, W]
        feat_map = self.encoder(x, timesteps)

        # 2) AttentionPool2d → [B, C]
        feat = self.attn_pool(feat_map)  # positional‑aware pooled vector

        # 3) 각 head 예측
        logit0 = self.head_root(feat)
        logit1 = self.head_lvl1(feat)
        logit_c = self.head_clothed(feat)
        logit_n = self.head_nude(feat)
        return logit0, logit1, logit_c, logit_n

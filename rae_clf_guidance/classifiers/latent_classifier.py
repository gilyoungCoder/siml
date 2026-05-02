"""
Classifiers for DINOv2 latent z0 space (RAE).
Input: [B, 768, 16, 16] normalized DINOv2 latents.
Output: logits [B, num_classes].

DINOv2 features have 84.5% linear probing accuracy on ImageNet,
so a simple MLP with global average pooling is sufficient.
"""

import torch
import torch.nn as nn


class DINOv2LatentClassifier(nn.Module):
    """
    Simple MLP classifier for DINOv2 latent space.
    GlobalAvgPool → MLP (768 → hidden → hidden → num_classes).

    For 16x16 spatial input (RAE 256px):
      pool → [B, 768] → head → [B, num_classes]
    """

    def __init__(self, in_channels=768, num_classes=3, hidden_dim=256):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x, timesteps=None):
        """
        Args:
            x: (B, 768, H, W) normalized DINOv2 latent
            timesteps: ignored (kept for interface compatibility)
        """
        x = self.pool(x).flatten(1)  # [B, 768]
        return self.head(x)

    def _features(self, x):
        """Extract feature vector before final FC layer."""
        x = self.pool(x).flatten(1)
        for layer in list(self.head.children())[:-1]:
            x = layer(x)
        return x

    def get_features(self, x):
        """Public API: returns feature vector."""
        with torch.no_grad():
            return self._features(x)


class DINOv2LatentConvClassifier(nn.Module):
    """
    Conv-based classifier with spatial awareness.
    1x1 channel reduction → 3x3 spatial conv → pool → FC.
    Use if MLP classifier underperforms.
    """

    def __init__(self, in_channels=768, num_classes=3, mid_channels=256):
        super().__init__()
        self.conv_head = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(mid_channels, mid_channels),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mid_channels, num_classes),
        )

    def forward(self, x, timesteps=None):
        x = self.conv_head(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

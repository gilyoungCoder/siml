import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms.functional as TF


class ImageResNet18Classifier(nn.Module):
    """
    Standard ResNet18 for image-space classification.
    Input: decoded image (B, 3, 512, 512) from VAE decoder.
    Output: logits (B, num_classes).

    Uses ImageNet pretrained weights — no modification needed (3ch input).
    """

    def __init__(self, num_classes=3, pretrained_backbone=True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained_backbone else None
        self.backbone = models.resnet18(weights=weights)
        self.backbone.fc = nn.Linear(512, num_classes)

    def forward(self, x, timesteps=None):
        """
        Args:
            x: (B, 3, H, W) decoded image
            timesteps: ignored (interface compatibility)
        """
        return self.backbone(x)

    def get_features(self, x):
        """Extract 512-dim feature before FC layer."""
        with torch.no_grad():
            b = self.backbone
            x = b.conv1(x)
            x = b.bn1(x)
            x = b.relu(x)
            x = b.maxpool(x)
            x = b.layer1(x)
            x = b.layer2(x)
            x = b.layer3(x)
            x = b.layer4(x)
            x = b.avgpool(x)
            x = torch.flatten(x, 1)
            return x


class ImageViTClassifier(nn.Module):
    """
    ViT-B/16 for image-space classification.
    Input: decoded image (B, 3, 512, 512) from VAE decoder.
    Output: logits (B, num_classes).

    Internally resizes to 224x224 (ViT-B/16 default).
    Supports both pretrained (ImageNet) and from-scratch training.
    """

    def __init__(self, num_classes=3, pretrained_backbone=True):
        super().__init__()
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained_backbone else None
        self.backbone = models.vit_b_16(weights=weights)
        # Replace classification head: 768 -> num_classes
        self.backbone.heads.head = nn.Linear(768, num_classes)

    def forward(self, x, timesteps=None):
        """
        Args:
            x: (B, 3, H, W) decoded image (any resolution)
            timesteps: ignored (interface compatibility)
        """
        # ViT-B/16 expects 224x224
        if x.shape[-2] != 224 or x.shape[-1] != 224:
            x = TF.resize(x, [224, 224], antialias=True)
        return self.backbone(x)


def build_image_classifier(architecture="resnet18", num_classes=3, pretrained_backbone=True):
    """Factory function to build image-space classifier."""
    if architecture == "resnet18":
        return ImageResNet18Classifier(num_classes=num_classes, pretrained_backbone=pretrained_backbone)
    elif architecture == "vit_b":
        return ImageViTClassifier(num_classes=num_classes, pretrained_backbone=pretrained_backbone)
    else:
        raise ValueError(f"Unknown image architecture: {architecture}. Choose 'resnet18' or 'vit_b'")

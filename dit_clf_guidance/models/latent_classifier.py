import torch
import torch.nn as nn
import torchvision.models as models


class LatentResNet18Classifier(nn.Module):
    """
    ResNet18 adapted for SD1.4 latent z0 space classification.
    Input: z0 tensor of shape (B, 4, 64, 64) — predicted clean VAE latents.
    Output: logits of shape (B, num_classes).

    First conv layer changed from 3ch to 4ch.
    No timestep conditioning (classifier always sees clean-ish latents).
    """

    def __init__(self, num_classes=3, pretrained_backbone=True):
        super().__init__()

        weights = models.ResNet18_Weights.DEFAULT if pretrained_backbone else None
        resnet = models.resnet18(weights=weights)

        # Replace first conv: 3ch -> 4ch
        old_conv = resnet.conv1
        self.conv1 = nn.Conv2d(
            4, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        with torch.no_grad():
            self.conv1.weight[:, :3] = old_conv.weight
            self.conv1.weight[:, 3] = old_conv.weight.mean(dim=1)

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x, timesteps=None):
        """
        Args:
            x: (B, 4, 64, 64) clean latent z0
            timesteps: ignored (kept for interface compatibility)
        """
        x = self._features(x)
        x = self.fc(x)
        return x

    def _features(self, x):
        """Extract 512-dim feature before FC layer."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def get_features(self, x):
        """Public API: returns (B, 512) feature vector."""
        with torch.no_grad():
            return self._features(x)

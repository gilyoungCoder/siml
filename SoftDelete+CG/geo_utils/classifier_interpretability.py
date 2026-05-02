"""
Classifier Interpretability Module

This module provides tools to interpret which latent regions the classifier
focuses on when predicting nude content (class 2).

Techniques:
1. Grad-CAM: Gradient-based Class Activation Mapping on latent space
2. Layer-wise Activation: Visualize feature maps at different U-Net depths
3. Integrated Gradients: Path-based attribution method
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path


class ClassifierGradCAM:
    """
    Grad-CAM implementation for the nudity classifier.

    Extracts gradient-based attention maps showing which latent regions
    contribute most to the classifier's decision for a specific class.
    """

    def __init__(self, classifier_model, target_layer_name: str = "encoder_model.middle_block.2"):
        """
        Args:
            classifier_model: The loaded EncoderUNetModelForClassification
            target_layer_name: Name of the layer to extract gradients from.
                             Suggested layers (Encoder U-Net has no output_blocks, only encoder):
                             - "encoder_model.middle_block.2": Bottleneck (highest-level semantic)
                             - "encoder_model.middle_block.0": Early bottleneck
                             - "encoder_model.input_blocks.11": Late encoder (before bottleneck)
                             - "encoder_model.input_blocks.8": Mid encoder
                             - "encoder_model.input_blocks.5": Early encoder with attention
        """
        self.model = classifier_model
        self.model.eval()
        self.target_layer_name = target_layer_name

        # Hook storage
        self.activations = None
        self.gradients = None
        self.hooks = []

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks on the target layer."""

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        # Find and attach hooks to the target layer
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                self.hooks.append(module.register_forward_hook(forward_hook))
                self.hooks.append(module.register_full_backward_hook(backward_hook))
                print(f"✓ Hooks registered on layer: {name}")
                return

        print(f"⚠ Warning: Layer '{self.target_layer_name}' not found!")
        print("Available layers:")
        for name, _ in self.model.named_modules():
            if "output_blocks" in name or "input_blocks" in name:
                print(f"  - {name}")

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def generate_heatmap(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        target_class: int = 2,
        normalize: bool = True,
        debug_shapes: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Generate Grad-CAM heatmap for the target class.

        Args:
            latent: Input latent [B, 4, H, W] (typically [1, 4, 64, 64])
            timestep: Timestep value [1] or scalar
            target_class: Class to compute gradients for (2 = nude)
            normalize: Whether to normalize heatmap to [0, 1]
            debug_shapes: Whether to print intermediate tensor shapes

        Returns:
            heatmap: [B, H, W] attention heatmap
            info: Dictionary with logits, probs, and score
        """
        # Ensure gradient tracking
        latent = latent.clone().requires_grad_(True)

        # Ensure timestep is properly shaped and on correct device
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor([timestep], device=latent.device, dtype=torch.long)
        elif timestep.dim() == 0:
            timestep = timestep.unsqueeze(0).to(latent.device)
        else:
            timestep = timestep.to(latent.device)
        if timestep.shape[0] != latent.shape[0]:
            timestep = timestep.repeat(latent.shape[0]).to(latent.device)

        if debug_shapes:
            print(f"\n  [generate_heatmap] Step-by-step computation for class {target_class}")
            print(f"  " + "-"*60)

        # Forward pass
        logits = self.model(latent, timestep)
        probs = F.softmax(logits, dim=-1)

        # Get target class score
        target_score = logits[:, target_class]

        if debug_shapes:
            print(f"  1. Forward pass:")
            print(f"     latent → classifier → logits: {str(list(latent.shape)):<20} → {str(list(logits.shape))}")
            print(f"     y_c (target_score):           {str(list(target_score.shape)):<20} value={target_score.item():.4f}")

        # Backward pass
        self.model.zero_grad()
        target_score.sum().backward(retain_graph=True)

        # Compute Grad-CAM
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Hooks did not capture gradients/activations. Check layer name.")

        if debug_shapes:
            print(f"\n  2. Hook captured at layer '{self.target_layer_name}':")
            print(f"     A^l (activations):            {str(list(self.activations.shape))}")
            print(f"     ∂y_c/∂A^l (gradients):        {str(list(self.gradients.shape))}")

        # Global average pooling of gradients
        # gradients: [B, C, H, W]
        # weights: [B, C]
        weights = self.gradients.mean(dim=(2, 3))

        if debug_shapes:
            print(f"\n  3. Eq.4 left: α_k = (1/H'W') Σ_{'{i,j}'} ∂y_c/∂A^l_{{k,i,j}}")
            print(f"     gradients.mean(dim=(2,3)):    {str(list(self.gradients.shape))} → {str(list(weights.shape))}")
            print(f"     α_k (weights):                {str(list(weights.shape)):<20} # [B, C']")
            print(f"     top-5 weights:                {weights[0, :5].tolist()}")

        # Weighted combination of activation maps
        # activations: [B, C, H, W]
        # heatmap: [B, H, W]
        heatmap_before_relu = torch.zeros(
            self.activations.shape[0],
            self.activations.shape[2],
            self.activations.shape[3],
            device=self.activations.device
        )

        for i in range(weights.shape[1]):
            heatmap_before_relu += weights[:, i].view(-1, 1, 1) * self.activations[:, i, :, :]

        if debug_shapes:
            print(f"\n  4. Eq.4 right: S = Σ_k α_k * A^l_k")
            print(f"     Σ_k (α_k * A^l_k):            {str(list(heatmap_before_relu.shape)):<20} # [B, H', W']")
            print(f"     before ReLU: min={heatmap_before_relu.min().item():.4f}, max={heatmap_before_relu.max().item():.4f}")

        # ReLU to focus on positive contributions
        heatmap = F.relu(heatmap_before_relu)

        if debug_shapes:
            print(f"\n  5. ReLU(S):")
            print(f"     after ReLU:  min={heatmap.min().item():.4f}, max={heatmap.max().item():.4f}")
            print(f"     heatmap shape:                {str(list(heatmap.shape)):<20} # [B, H', W'] = [B, 8, 8]")

        # Normalize
        if normalize:
            for b in range(heatmap.shape[0]):
                hmap = heatmap[b]
                if hmap.max() > 0:
                    heatmap[b] = (hmap - hmap.min()) / (hmap.max() - hmap.min())

        # Resize heatmap to latent size (if needed)
        heatmap_before_resize = heatmap.clone()
        if heatmap.shape[-2:] != latent.shape[-2:]:
            heatmap = F.interpolate(
                heatmap.unsqueeze(1),
                size=latent.shape[-2:],
                mode='bilinear',
                align_corners=False
            ).squeeze(1)

        if debug_shapes:
            print(f"\n  6. Resize to latent size:")
            print(f"     interpolate:                  {str(list(heatmap_before_resize.shape))} → {str(list(heatmap.shape))}")
            print(f"     final heatmap:                {str(list(heatmap.shape)):<20} # [B, H, W] = [B, 64, 64]")
            print(f"  " + "-"*60)

        info = {
            'logits': logits.detach(),
            'probs': probs.detach(),
            'target_score': target_score.detach(),
            'weights': weights.detach(),
            'activation_shape': self.activations.shape,
            'gradient_shape': self.gradients.shape,
            'heatmap_before_resize': heatmap_before_resize.detach()
        }

        return heatmap, info

    def __del__(self):
        """Cleanup hooks on deletion."""
        self.remove_hooks()


class LayerwiseActivationAnalyzer:
    """
    Analyze activations at different layers of the U-Net classifier
    to understand where nude features are detected.
    """

    def __init__(self, classifier_model):
        self.model = classifier_model
        self.model.eval()
        self.activation_maps = {}
        self.hooks = []

    def register_layer_hooks(self, layer_names: Optional[List[str]] = None):
        """
        Register hooks on multiple layers.

        Args:
            layer_names: List of layer names. If None, registers on key output_blocks.
        """
        if layer_names is None:
            layer_names = [
                "encoder_model.input_blocks.0",    # Early features
                "encoder_model.input_blocks.4",    # Early with attention
                "encoder_model.input_blocks.7",    # Mid encoder with attention
                "encoder_model.input_blocks.10",   # Late encoder with attention
                "encoder_model.middle_block.0",    # Early bottleneck
                "encoder_model.middle_block.2",    # Late bottleneck (highest semantic)
            ]

        def make_hook(name):
            def hook(module, input, output):
                self.activation_maps[name] = output.detach()
            return hook

        for name, module in self.model.named_modules():
            if name in layer_names:
                self.hooks.append(module.register_forward_hook(make_hook(name)))
                print(f"✓ Hook registered on: {name}")

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def analyze(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Run forward pass and collect activation maps.

        Returns:
            Dictionary mapping layer names to activation tensors
        """
        self.activation_maps = {}

        # Ensure timestep is properly shaped
        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        if timestep.shape[0] != latent.shape[0]:
            timestep = timestep.repeat(latent.shape[0])

        with torch.no_grad():
            logits = self.model(latent, timestep)
            probs = F.softmax(logits, dim=-1)

        # Add prediction info
        self.activation_maps['logits'] = logits
        self.activation_maps['probs'] = probs

        return self.activation_maps

    def __del__(self):
        self.remove_hooks()


class IntegratedGradients:
    """
    Integrated Gradients for attribution analysis.

    Computes path integral of gradients from baseline to input
    to measure feature importance.
    """

    def __init__(self, classifier_model):
        self.model = classifier_model
        self.model.eval()

    def attribute(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        target_class: int = 2,
        baseline: Optional[torch.Tensor] = None,
        n_steps: int = 50
    ) -> torch.Tensor:
        """
        Compute integrated gradients.

        Args:
            latent: Input latent [B, 4, H, W]
            timestep: Timestep value
            target_class: Class to compute attribution for
            baseline: Baseline latent (default: zeros)
            n_steps: Number of integration steps

        Returns:
            attribution: [B, 4, H, W] attribution map
        """
        if baseline is None:
            baseline = torch.zeros_like(latent)

        # Ensure timestep is properly shaped
        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        if timestep.shape[0] != latent.shape[0]:
            timestep = timestep.repeat(latent.shape[0])

        # Generate interpolated inputs
        alphas = torch.linspace(0, 1, n_steps + 1, device=latent.device)

        gradients = []
        for alpha in alphas:
            # Interpolated input
            interpolated = baseline + alpha * (latent - baseline)
            interpolated.requires_grad_(True)

            # Forward pass
            logits = self.model(interpolated, timestep)
            target_score = logits[:, target_class]

            # Backward pass
            self.model.zero_grad()
            target_score.sum().backward()

            # Store gradient
            gradients.append(interpolated.grad.detach())

        # Average gradients
        avg_gradients = torch.stack(gradients).mean(dim=0)

        # Integrated gradients
        attribution = (latent - baseline) * avg_gradients

        return attribution


class VisualizationUtils:
    """Utilities for visualizing heatmaps and attributions."""

    @staticmethod
    def overlay_heatmap_on_latent(
        latent: torch.Tensor,
        heatmap: torch.Tensor,
        alpha: float = 0.5,
        colormap: str = 'jet'
    ) -> np.ndarray:
        """
        Overlay heatmap on latent visualization.

        Args:
            latent: [4, H, W] latent tensor
            heatmap: [H, W] heatmap
            alpha: Overlay transparency
            colormap: Matplotlib colormap name

        Returns:
            RGB image [H, W, 3] as numpy array
        """
        # Convert to numpy
        if isinstance(latent, torch.Tensor):
            latent = latent.cpu().numpy()
        if isinstance(heatmap, torch.Tensor):
            heatmap = heatmap.cpu().numpy()

        # Use first channel of latent as base image
        base = latent[0]
        base = (base - base.min()) / (base.max() - base.min() + 1e-8)
        base = (base * 255).astype(np.uint8)

        # Convert to RGB
        base_rgb = np.stack([base, base, base], axis=-1)

        # Apply colormap to heatmap
        cmap = cm.get_cmap(colormap)
        heatmap_colored = cmap(heatmap)[:, :, :3]  # RGB only
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

        # Blend
        overlaid = (1 - alpha) * base_rgb + alpha * heatmap_colored
        overlaid = overlaid.astype(np.uint8)

        return overlaid

    @staticmethod
    def visualize_attribution_channels(
        attribution: torch.Tensor,
        channel_names: Optional[List[str]] = None
    ) -> plt.Figure:
        """
        Visualize attribution for each latent channel.

        Args:
            attribution: [4, H, W] attribution map
            channel_names: Names for the 4 channels

        Returns:
            Matplotlib figure
        """
        if channel_names is None:
            channel_names = [f"Channel {i}" for i in range(4)]

        if isinstance(attribution, torch.Tensor):
            attribution = attribution.cpu().numpy()

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        for i in range(4):
            attr = attribution[i]

            # Normalize for visualization
            vmax = np.abs(attr).max()

            im = axes[i].imshow(attr, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            axes[i].set_title(f"{channel_names[i]}\nMax: {attr.max():.3f}, Min: {attr.min():.3f}")
            axes[i].axis('off')
            plt.colorbar(im, ax=axes[i], fraction=0.046)

        plt.tight_layout()
        return fig

    @staticmethod
    def save_heatmap_comparison(
        latent: torch.Tensor,
        heatmap: torch.Tensor,
        info: Dict,
        save_path: Path,
        title: str = "Classifier Grad-CAM"
    ):
        """
        Save a comprehensive visualization with heatmap and prediction info.

        Args:
            latent: [4, H, W] input latent
            heatmap: [H, W] Grad-CAM heatmap
            info: Dictionary with prediction info
            save_path: Path to save image
            title: Plot title
        """
        if isinstance(latent, torch.Tensor):
            latent = latent.cpu().numpy()
        if isinstance(heatmap, torch.Tensor):
            heatmap = heatmap.cpu().numpy()

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Row 1: Latent channels
        for i in range(3):
            channel = latent[i]
            channel_norm = (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)
            axes[0, i].imshow(channel_norm, cmap='gray')
            axes[0, i].set_title(f"Latent Channel {i}")
            axes[0, i].axis('off')

        # Row 2: Heatmap visualizations
        axes[1, 0].imshow(heatmap, cmap='jet')
        axes[1, 0].set_title("Grad-CAM Heatmap")
        axes[1, 0].axis('off')

        # Overlay on first channel
        overlay = VisualizationUtils.overlay_heatmap_on_latent(
            latent, heatmap, alpha=0.5
        )
        axes[1, 1].imshow(overlay)
        axes[1, 1].set_title("Overlay on Channel 0")
        axes[1, 1].axis('off')

        # Prediction info
        probs = info['probs'][0].cpu().numpy()
        # Dynamic class names based on number of classes
        if len(probs) == 3:
            class_names = ['Not People', 'Clothed', 'Nude']
        elif len(probs) == 4:
            class_names = ['Benign', 'Person', 'Nude', 'Harm_color']
        else:
            class_names = [f'Class {i}' for i in range(len(probs))]

        axes[1, 2].bar(class_names, probs)
        axes[1, 2].set_ylim([0, 1])
        axes[1, 2].set_ylabel('Probability')
        axes[1, 2].set_title('Class Probabilities')
        axes[1, 2].tick_params(axis='x', rotation=45)

        # Add text with prediction
        pred_class = probs.argmax()
        pred_text = f"Prediction: {class_names[pred_class]} ({probs[pred_class]:.3f})"
        fig.text(0.5, 0.95, f"{title}\n{pred_text}",
                ha='center', fontsize=14, fontweight='bold')

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Saved visualization to {save_path}")

    @staticmethod
    def save_heatmap_with_image(
        original_image: np.ndarray,
        heatmap: torch.Tensor,
        info: Dict,
        save_path: Path,
        title: str = "Classifier Grad-CAM"
    ):
        """
        Save visualization with original image and heatmap overlay.

        Args:
            original_image: [H, W, 3] RGB image as numpy array (0-255)
            heatmap: [H, W] Grad-CAM heatmap (will be resized to match image)
            info: Dictionary with prediction info
            save_path: Path to save image
            title: Plot title
        """
        if isinstance(heatmap, torch.Tensor):
            heatmap = heatmap.cpu().numpy()

        # Resize heatmap to match original image size
        from PIL import Image as PILImage
        heatmap_resized = np.array(PILImage.fromarray(
            (heatmap * 255).astype(np.uint8)
        ).resize((original_image.shape[1], original_image.shape[0]), PILImage.BILINEAR)) / 255.0

        fig, axes = plt.subplots(1, 4, figsize=(24, 6))

        # 1. Original image
        axes[0].imshow(original_image)
        axes[0].set_title("Original Image")
        axes[0].axis('off')

        # 2. Heatmap only
        im = axes[1].imshow(heatmap_resized, cmap='jet', vmin=0, vmax=1)
        axes[1].set_title("Grad-CAM Heatmap")
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046)

        # 3. Overlay on original image
        heatmap_colored = cm.jet(heatmap_resized)[:, :, :3]  # RGB only
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
        alpha = 0.5
        overlay = ((1 - alpha) * original_image + alpha * heatmap_colored).astype(np.uint8)
        axes[2].imshow(overlay)
        axes[2].set_title("Overlay (α=0.5)")
        axes[2].axis('off')

        # 4. Prediction info
        probs = info['probs'][0].cpu().numpy()
        if len(probs) == 3:
            class_names = ['Not People', 'Clothed', 'Nude']
        elif len(probs) == 4:
            class_names = ['Benign', 'Person', 'Nude', 'Harm_color']
        else:
            class_names = [f'Class {i}' for i in range(len(probs))]

        colors = ['green' if i == probs.argmax() else 'steelblue' for i in range(len(probs))]
        axes[3].barh(class_names, probs, color=colors)
        axes[3].set_xlim([0, 1])
        axes[3].set_xlabel('Probability')
        axes[3].set_title('Class Probabilities')
        for i, (name, prob) in enumerate(zip(class_names, probs)):
            axes[3].text(prob + 0.02, i, f'{prob:.3f}', va='center', fontsize=10)

        # Title with prediction
        pred_class = probs.argmax()
        pred_text = f"Prediction: {class_names[pred_class]} ({probs[pred_class]:.3f})"
        fig.suptitle(f"{title}\n{pred_text}", fontsize=14, fontweight='bold')

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Saved visualization to {save_path}")


def load_classifier_for_interpretation(
    ckpt_path: str,
    device: str = "cuda"
) -> torch.nn.Module:
    """
    Helper to load classifier for interpretation.

    Args:
        ckpt_path: Path to classifier checkpoint
        device: Device to load on

    Returns:
        Loaded classifier model in eval mode
    """
    from geo_models.classifier.classifier import load_discriminator

    classifier = load_discriminator(
        ckpt_path=ckpt_path,
        condition=None,
        eval=True,
        channel=4,
        num_classes=3
    )

    classifier = classifier.to(device)
    classifier.eval()

    return classifier

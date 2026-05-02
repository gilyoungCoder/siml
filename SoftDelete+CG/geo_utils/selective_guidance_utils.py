#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Selective Classifier Guidance Utilities

Key Innovation:
  - Apply classifier guidance ONLY when harmful signal exceeds threshold
  - Spatially-aware guidance using Grad-CAM localization
  - Prevents unnecessary intervention on benign prompts

Mechanism:
  1. Monitor latent at each timestep with classifier
  2. If harmful_score > threshold:
     - Compute Grad-CAM heatmap to locate harmful regions
     - Apply classifier guidance masked to those regions
  3. Otherwise: skip guidance (preserve benign generation)

Benefits:
  - Reduced GENEVAL score degradation on benign prompts
  - Selective intervention only when needed
  - Spatial precision with Grad-CAM
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from geo_utils.classifier_interpretability import ClassifierGradCAM


class ThresholdScheduler:
    """Adaptive threshold scheduling over denoising steps."""

    def __init__(
        self,
        strategy: str = "constant",
        start_value: float = 0.5,
        end_value: float = 0.5,
        total_steps: int = 50
    ):
        self.strategy = strategy
        self.start_value = start_value
        self.end_value = end_value
        self.total_steps = total_steps

    def get_threshold(self, current_step: int) -> float:
        """Get threshold for current step."""
        if self.strategy == "constant":
            return self.start_value

        # Normalize step to [0, 1]
        t = current_step / max(self.total_steps - 1, 1)

        if self.strategy == "linear_increase":
            return self.start_value + (self.end_value - self.start_value) * t
        elif self.strategy == "linear_decrease":
            return self.start_value - (self.start_value - self.end_value) * t
        elif self.strategy == "cosine_anneal":
            return self.end_value + (self.start_value - self.end_value) * 0.5 * (1 + np.cos(np.pi * t))
        else:
            return self.start_value


class SelectiveGuidanceMonitor:
    """
    Monitors latent at each timestep and decides whether to apply guidance.

    Uses classifier to detect harmful content and Grad-CAM to localize regions.

    NEW: Adaptive mechanisms
      1. Adaptive threshold by timestep
      2. Adaptive guidance scale by timestep (in SpatiallyMaskedGuidance)
      3. Spatial-adaptive guidance based on GradCAM scores
    """

    def __init__(
        self,
        classifier_model,
        harmful_threshold: float = 0.5,
        harmful_class: int = 2,  # 2 = nude people
        safe_class: int = 1,     # 1 = clothed people
        spatial_threshold: float = 0.5,
        use_percentile: bool = False,
        spatial_percentile: float = 0.3,
        gradcam_layer: str = "encoder_model.middle_block.2",
        device: str = "cuda",
        debug: bool = False,
        # Adaptive threshold scheduling
        use_adaptive_threshold: bool = False,
        threshold_scheduler: Optional['ThresholdScheduler'] = None,
        # Spatial-adaptive guidance (use heatmap values as weights)
        use_heatmap_weighted_guidance: bool = False
    ):
        """
        Args:
            classifier_model: Loaded classifier model
            harmful_threshold: Base threshold for harmful detection (can be overridden by scheduler)
            harmful_class: Class index for harmful content (2 = nude)
            safe_class: Class index for safe content (1 = clothed)
            spatial_threshold: Grad-CAM threshold for spatial masking (binary)
            use_percentile: Use percentile instead of fixed threshold
            spatial_percentile: Top percentile to mask (e.g., 0.3 = top 30%)
            gradcam_layer: Target layer for Grad-CAM
            device: Device to run on
            debug: Enable debug logging
            use_adaptive_threshold: Enable adaptive threshold scheduling
            threshold_scheduler: ThresholdScheduler instance for adaptive thresholds
            use_heatmap_weighted_guidance: Use heatmap values to weight guidance spatially
        """
        self.classifier = classifier_model
        self.classifier.eval()

        # Get classifier dtype from its parameters
        self.classifier_dtype = next(self.classifier.parameters()).dtype

        self.harmful_threshold = harmful_threshold
        self.harmful_class = harmful_class
        self.safe_class = safe_class
        self.spatial_threshold = spatial_threshold
        self.use_percentile = use_percentile
        self.spatial_percentile = spatial_percentile
        self.device = device
        self.debug = debug

        # Adaptive threshold scheduling
        self.use_adaptive_threshold = use_adaptive_threshold
        self.threshold_scheduler = threshold_scheduler

        # Heatmap-weighted guidance
        self.use_heatmap_weighted_guidance = use_heatmap_weighted_guidance

        # Initialize Grad-CAM
        self.gradcam = ClassifierGradCAM(
            classifier_model=classifier_model,
            target_layer_name=gradcam_layer
        )
         # 🔥 핵심 tnwjwtnwekllsdfafk
        self.classifier = classifier_model.to(device)
        self.classifier.eval()

        # 🔥 encoder_model까지 명시적으로
        if hasattr(self.classifier, "encoder_model"):
            self.classifier.encoder_model = self.classifier.encoder_model.to(device)
        # Statistics tracking
        self.stats = {
            'total_steps': 0,
            'harmful_steps': 0,
            'guidance_applied': 0,
            'step_history': []
        }

    def detect_harmful(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        current_step: Optional[int] = None
    ) -> Tuple[bool, float, torch.Tensor]:
        """
        Detect if current latent contains harmful content.

        Args:
            latent: [B, 4, H, W] latent tensor
            timestep: [B] or scalar timestep
            current_step: Current denoising step (for adaptive threshold)

        Returns:
            is_harmful: Boolean indicating if harmful
            harmful_score: Raw score (logit or probability)
            logits: [B, num_classes] classifier output
        """
        with torch.no_grad():
            # Ensure timestep is tensor
            if not isinstance(timestep, torch.Tensor):
                timestep = torch.tensor([timestep], device=latent.device, dtype=torch.long)
            elif timestep.dim() == 0:
                timestep = timestep.unsqueeze(0).to(latent.device)
            else:
                timestep = timestep.to(latent.device)

            # Broadcast timestep to batch size
            B = latent.shape[0]
            if timestep.shape[0] != B:
                timestep = timestep.expand(B).to(latent.device)

            # Ensure latent dtype matches classifier
            # Classifier should be in float16 if loaded with .half()
            latent_input = latent.to(dtype=self.classifier_dtype)

            # Get classifier predictions
            logits = self.classifier(latent_input, timestep)  # [B, num_classes]

            # Get harmful score
            # You can use either logits or probabilities
            # For 3-class classifier: [not_people, clothed, nude]
            harmful_logit = logits[:, self.harmful_class]  # [B]

            # Decision: use mean across batch (or max, depending on strategy)
            harmful_score = harmful_logit.mean().item()

            # Use adaptive threshold if enabled, otherwise use fixed threshold
            if self.use_adaptive_threshold and self.threshold_scheduler is not None and current_step is not None:
                threshold = self.threshold_scheduler.get_threshold(current_step)
            else:
                threshold = self.harmful_threshold

            is_harmful = harmful_score > threshold

        return is_harmful, harmful_score, logits

    def _apply_gaussian_smoothing(
        self,
        mask: torch.Tensor,
        sigma: float
    ) -> torch.Tensor:
        """
        Apply Gaussian smoothing to mask for softer transitions.

        Args:
            mask: [B, H, W] mask tensor
            sigma: Gaussian kernel standard deviation

        Returns:
            smoothed_mask: [B, H, W] smoothed mask
        """
        from scipy.ndimage import gaussian_filter

        # Process each sample in batch
        B = mask.shape[0]
        smoothed_masks = []

        for b in range(B):
            mask_np = mask[b].cpu().numpy()
            smoothed_np = gaussian_filter(mask_np, sigma=sigma, mode='constant')
            smoothed_masks.append(torch.from_numpy(smoothed_np).to(mask.device))

        smoothed_mask = torch.stack(smoothed_masks, dim=0)
        return smoothed_mask

    def get_spatial_mask(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        current_step: Optional[int] = None,
        return_heatmap: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Generate spatial mask indicating harmful regions.

        NEW: Binary mask + optional heatmap weighting for spatial-adaptive guidance

        Args:
            latent: [B, 4, H, W] latent tensor
            timestep: [B] or scalar timestep
            current_step: Current denoising step (for adaptive threshold)
            return_heatmap: Whether to return raw heatmap

        Returns:
            mask: [B, H, W] binary mask or heatmap-weighted mask
            heatmap: [B, H, W] raw Grad-CAM heatmap (optional)
        """
        # Ensure timestep is tensor
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor([timestep], device=latent.device, dtype=torch.long)
        elif timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)

        B = latent.shape[0]
        if timestep.shape[0] != B:
            timestep = timestep.expand(B)

        # Generate Grad-CAM heatmap
        # Ensure latent dtype matches classifier
        latent_input = latent.to(dtype=self.classifier_dtype)

        with torch.enable_grad():
            heatmap, info = self.gradcam.generate_heatmap(
                latent=latent_input,
                timestep=timestep,
                target_class=self.harmful_class,
                normalize=True
            )
        # heatmap: [B, H, W], values in [0, 1]

        # Determine spatial threshold (adaptive or fixed)
        if self.use_adaptive_threshold and self.threshold_scheduler is not None and current_step is not None:
            spatial_threshold = self.threshold_scheduler.get_threshold(current_step)
        else:
            spatial_threshold = self.spatial_threshold

        # Create BINARY mask
        if self.use_percentile:
            # Use top percentile
            masks = []
            for b in range(B):
                h = heatmap[b].flatten()
                k = int(h.numel() * self.spatial_percentile)
                if k > 0:
                    threshold = torch.topk(h, k=k)[0][-1].item()
                else:
                    threshold = h.max().item()
                mask_b = (heatmap[b] >= threshold).float()
                masks.append(mask_b)
            binary_mask = torch.stack(masks, dim=0)
        else:
            # Use fixed threshold (adaptive or constant)
            binary_mask = (heatmap >= spatial_threshold).float()

        # Option: Heatmap-weighted guidance (spatial-adaptive)
        if self.use_heatmap_weighted_guidance:
            # Multiply binary mask by heatmap values
            # This gives pixel-wise guidance strength based on GradCAM score
            # mask[i,j] = binary[i,j] * heatmap[i,j]
            # Example: If heatmap=0.9 → 90% guidance, heatmap=0.6 → 60% guidance
            mask = binary_mask * heatmap
        else:
            # Pure binary mask
            mask = binary_mask

        if return_heatmap:
            return mask, heatmap
        else:
            return mask, None

    def should_apply_guidance(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        step: int
    ) -> Tuple[bool, Optional[torch.Tensor], Dict]:
        """
        Decide whether to apply guidance and return spatial mask if needed.

        Args:
            latent: [B, 4, H, W] latent tensor
            timestep: Timestep value
            step: Current denoising step

        Returns:
            should_apply: Boolean whether to apply guidance
            spatial_mask: [B, H, W] mask for guidance (None if not applying)
            info: Dictionary with detection info
        """
        # Detect harmful content (with adaptive threshold if enabled)
        is_harmful, harmful_score, logits = self.detect_harmful(latent, timestep, current_step=step)

        # Update statistics
        self.stats['total_steps'] += 1
        if is_harmful:
            self.stats['harmful_steps'] += 1

        info = {
            'step': step,
            'timestep': int(timestep) if isinstance(timestep, torch.Tensor) else timestep,
            'is_harmful': is_harmful,
            'harmful_score': harmful_score,
            'logits': logits.cpu() if isinstance(logits, torch.Tensor) else logits
        }

        if not is_harmful:
            # No guidance needed, but still record for visualization
            info['mask_ratio'] = 0.0  # No mask applied
            info['heatmap'] = None

            if self.debug:
                print(f"[Step {step}] Safe (score={harmful_score:.3f}) - Skipping guidance")

            # Always record to step_history for complete visualization
            self.stats['step_history'].append(info)

            return False, None, info

        # Get spatial mask for harmful regions (with adaptive threshold if enabled)
        spatial_mask, heatmap = self.get_spatial_mask(
            latent,
            timestep,
            current_step=step,  # Pass step for adaptive threshold
            return_heatmap=True
        )

        # Update statistics
        self.stats['guidance_applied'] += 1

        # Calculate mask ratio
        # If heatmap-weighted: average of weighted mask values
        # If binary: ratio of masked pixels
        mask_ratio = spatial_mask.mean().item()

        info['mask_ratio'] = mask_ratio
        info['heatmap'] = heatmap

        if self.debug:
            adaptive_str = ""
            if self.use_adaptive_threshold and self.threshold_scheduler is not None:
                current_threshold = self.threshold_scheduler.get_threshold(step)
                adaptive_str = f", adaptive_threshold={current_threshold:.3f}"

            weighted_str = " (heatmap-weighted)" if self.use_heatmap_weighted_guidance else " (binary)"

            print(f"[Step {step}] Harmful detected (score={harmful_score:.3f}{adaptive_str}) - Applying guidance")
            print(f"  Mask ratio: {mask_ratio:.1%}{weighted_str}")

        # Always record to step_history
        self.stats['step_history'].append(info)

        return True, spatial_mask, info

    def get_statistics(self) -> Dict:
        """Return accumulated statistics."""
        stats = self.stats.copy()
        if stats['total_steps'] > 0:
            stats['harmful_ratio'] = stats['harmful_steps'] / stats['total_steps']
            stats['guidance_ratio'] = stats['guidance_applied'] / stats['total_steps']
        return stats

    def reset_statistics(self):
        """Reset statistics for new generation."""
        self.stats = {
            'total_steps': 0,
            'harmful_steps': 0,
            'guidance_applied': 0,
            'step_history': []
        }


class WeightScheduler:
    """
    Manages guidance weight scheduling across denoising steps.

    Strategies:
    - constant: Fixed weight throughout
    - linear_increase: Linearly increase from start_weight to end_weight
    - linear_decrease: Linearly decrease from start_weight to end_weight
    - cosine_anneal: Cosine annealing from start_weight to end_weight
    - exponential_decay: Exponential decay from start_weight to end_weight
    """

    def __init__(
        self,
        strategy: str = "constant",
        start_step: int = 0,
        end_step: int = 50,
        start_weight: float = 1.0,
        end_weight: float = 1.0,
        decay_rate: float = 0.1  # For exponential decay
    ):
        """
        Args:
            strategy: Scheduling strategy name
            start_step: First step to apply guidance
            end_step: Last step to apply guidance
            start_weight: Initial weight multiplier
            end_weight: Final weight multiplier
            decay_rate: Decay rate for exponential strategy
        """
        self.strategy = strategy
        self.start_step = start_step
        self.end_step = end_step
        self.start_weight = start_weight
        self.end_weight = end_weight
        self.decay_rate = decay_rate
        self.total_steps = max(1, end_step - start_step)

    def get_weight(self, current_step: int) -> float:
        """
        Get weight multiplier for current step.

        Args:
            current_step: Current denoising step

        Returns:
            weight: Multiplier in [0, inf)
        """
        # Outside guidance range
        if current_step < self.start_step or current_step > self.end_step:
            return 0.0

        # Normalized progress in [0, 1]
        progress = (current_step - self.start_step) / self.total_steps

        if self.strategy == "constant":
            return self.start_weight

        elif self.strategy == "linear_increase":
            return self.start_weight + (self.end_weight - self.start_weight) * progress

        elif self.strategy == "linear_decrease":
            return self.start_weight - (self.start_weight - self.end_weight) * progress

        elif self.strategy == "cosine_anneal":
            # Cosine annealing: smooth transition
            import math
            cosine_val = 0.5 * (1 + math.cos(math.pi * progress))
            return self.end_weight + (self.start_weight - self.end_weight) * cosine_val

        elif self.strategy == "exponential_decay":
            # Exponential decay: e^(-decay_rate * progress)
            import math
            decay_factor = math.exp(-self.decay_rate * progress * 10)  # Scale by 10 for reasonable range
            return self.start_weight * decay_factor + self.end_weight * (1 - decay_factor)

        else:
            raise ValueError(f"Unknown scheduling strategy: {self.strategy}")


class SpatiallyMaskedGuidance:
    """
    Applies classifier guidance with spatial masking.

    Computes gradient toward safe class and masks it to harmful regions only.
    """

    def __init__(
        self,
        classifier_model,
        safe_class: int = 1,  # 1 = clothed people
        harmful_class: int = 2,  # 2 = nude people
        device: str = "cuda",
        use_bidirectional: bool = True,  # Enable bidirectional guidance
        # Weight scheduling parameters
        weight_scheduler: Optional[WeightScheduler] = None,
        # Gradient normalization parameters
        normalize_gradient: bool = False,
        gradient_norm_type: str = "l2"  # "l2" or "layer"
    ):
        """
        Args:
            classifier_model: Loaded classifier model
            safe_class: Target class for guidance (1 = clothed)
            harmful_class: Class to avoid (2 = nude)
            device: Device to run on
            use_bidirectional: If True, push away from harmful + pull toward safe
            weight_scheduler: WeightScheduler instance for dynamic weight scaling
            normalize_gradient: If True, normalize gradients for stability
            gradient_norm_type: Type of normalization ("l2" or "layer")
        """
        self.classifier = classifier_model
        self.safe_class = safe_class
        self.harmful_class = harmful_class
        self.device = device
        self.use_bidirectional = use_bidirectional
        self.weight_scheduler = weight_scheduler
        self.normalize_gradient = normalize_gradient
        self.gradient_norm_type = gradient_norm_type

        # Get classifier dtype from its parameters
        self.classifier_dtype = next(self.classifier.parameters()).dtype
        if hasattr(self.classifier, "encoder_model"):
            self.classifier.encoder_model = self.classifier.encoder_model.to(device)

    def compute_masked_gradient(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        spatial_mask: torch.Tensor,
        guidance_scale: float = 5.0,
        harmful_scale: float = 1.0,  # Scale for harmful repulsion (relative to guidance_scale)
        current_step: Optional[int] = None  # For weight scheduling
    ) -> torch.Tensor:
        """
        Compute bidirectional classifier gradient masked to harmful regions.

        Bidirectional approach:
          1. Pull toward safe class (clothed)
          2. Push away from harmful class (nude)

        Final gradient = guidance_scale * (grad_safe - harmful_scale * grad_harmful) * weight_schedule

        Args:
            latent: [B, 4, H, W] latent tensor (requires_grad=True)
            timestep: Timestep value
            spatial_mask: [B, H, W] binary or soft mask (1 = harmful, 0 = safe)
            guidance_scale: Gradient scale for safe direction
            harmful_scale: Relative scale for harmful repulsion (default: 1.0)
                          - 1.0: Equal weight (recommended)
                          - 0.5: Half weight for repulsion
                          - 2.0: Double weight for repulsion
            current_step: Current denoising step (for weight scheduling)

        Returns:
            masked_grad: [B, 4, H, W] gradient to add to latent
        """
        # Enable gradient computation
        with torch.enable_grad():
            # Ensure latent requires grad and matches classifier dtype
            latent_input = latent.detach().to(dtype=self.classifier_dtype).requires_grad_(True)

            # Ensure timestep is tensor and on correct device
            if not isinstance(timestep, torch.Tensor):
                timestep = torch.tensor([timestep], device=latent.device, dtype=torch.long)
            elif timestep.dim() == 0:
                timestep = timestep.unsqueeze(0).to(latent.device)
            else:
                timestep = timestep.to(latent.device)

            B = latent_input.shape[0]
            if timestep.shape[0] != B:
                timestep = timestep.expand(B).to(latent.device)

            if self.use_bidirectional:
                # Bidirectional guidance
                # Note: Due to gradient checkpointing in classifier, we need to compute
                # gradients separately (cannot use retain_graph=True)

                # 1. Gradient toward SAFE class (pull)
                latent_for_safe = latent_input.detach().requires_grad_(True)
                logits_safe = self.classifier(latent_for_safe, timestep)
                safe_logit = logits_safe[:, self.safe_class].sum()
                grad_safe = torch.autograd.grad(safe_logit, latent_for_safe)[0]

                # 2. Gradient toward HARMFUL class (to push opposite direction)
                latent_for_harmful = latent_input.detach().requires_grad_(True)
                logits_harmful = self.classifier(latent_for_harmful, timestep)
                harmful_logit = logits_harmful[:, self.harmful_class].sum()
                grad_harmful = torch.autograd.grad(harmful_logit, latent_for_harmful)[0]

                # Combine: pull toward safe, push away from harmful
                grad = grad_safe - harmful_scale * grad_harmful
            else:
                # Original unidirectional guidance (backward compatibility)
                logits = self.classifier(latent_input, timestep)
                safe_logit = logits[:, self.safe_class].sum()
                grad = torch.autograd.grad(safe_logit, latent_input)[0]

        # Optional: Normalize gradient for stability
        if self.normalize_gradient:
            if self.gradient_norm_type == "l2":
                # L2 normalization: grad / ||grad||_2
                grad_norm = torch.norm(grad, p=2, dim=(1, 2, 3), keepdim=True)
                grad = grad / (grad_norm + 1e-8)
            elif self.gradient_norm_type == "layer":
                # Layer normalization per channel
                for c in range(grad.shape[1]):
                    channel_grad = grad[:, c:c+1, :, :]
                    channel_norm = torch.norm(channel_grad, p=2, dim=(2, 3), keepdim=True)
                    grad[:, c:c+1, :, :] = channel_grad / (channel_norm + 1e-8)

        # Apply spatial mask
        # spatial_mask: [B, H, W] -> expand to [B, 1, H, W]
        mask_expanded = spatial_mask.unsqueeze(1)  # [B, 1, H, W]
        masked_grad = grad * mask_expanded

        # Scale by base guidance_scale
        masked_grad = masked_grad * guidance_scale

        # Apply weight scheduling if available
        if self.weight_scheduler is not None and current_step is not None:
            schedule_weight = self.weight_scheduler.get_weight(current_step)
            masked_grad = masked_grad * schedule_weight

        # Convert back to original latent dtype
        masked_grad = masked_grad.to(dtype=latent.dtype)

        return masked_grad.detach()

    def apply_guidance(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        spatial_mask: torch.Tensor,
        guidance_scale: float = 5.0,
        harmful_scale: float = 1.0,
        scheduler = None,
        noise_pred: Optional[torch.Tensor] = None,
        current_step: Optional[int] = None  # For weight scheduling
    ) -> torch.Tensor:
        """
        Apply spatially-masked bidirectional guidance to latent.

        Args:
            latent: [B, 4, H, W] current latent (x_t)
            timestep: Current timestep
            spatial_mask: [B, H, W] harmful region mask
            guidance_scale: Gradient scale for safe direction
            harmful_scale: Relative scale for harmful repulsion (default: 1.0)
            scheduler: Diffusion scheduler (for proper gradient integration)
            noise_pred: Predicted noise (optional, for advanced integration)
            current_step: Current denoising step (for weight scheduling)

        Returns:
            guided_latent: [B, 4, H, W] latent after guidance
        """
        # Compute masked gradient (bidirectional if enabled)
        masked_grad = self.compute_masked_gradient(
            latent=latent,
            timestep=timestep,
            spatial_mask=spatial_mask,
            guidance_scale=guidance_scale,
            harmful_scale=harmful_scale,
            current_step=current_step
        )

        # Simple gradient ascent
        # For more sophisticated integration, can use scheduler
        guided_latent = latent + masked_grad

        return guided_latent


def visualize_selective_guidance(
    monitor: SelectiveGuidanceMonitor,
    output_dir: Path,
    prefix: str = "selective_guidance"
):
    """
    Visualize selective guidance statistics.

    Args:
        monitor: SelectiveGuidanceMonitor instance
        output_dir: Directory to save visualizations
        prefix: Filename prefix
    """
    import matplotlib.pyplot as plt

    stats = monitor.get_statistics()
    history = stats.get('step_history', [])

    if not history:
        print("[WARNING] No guidance history to visualize")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract data
    steps = [h['step'] for h in history]
    harmful_scores = [h['harmful_score'] for h in history]
    mask_ratios = [h.get('mask_ratio', 0) for h in history]

    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Plot 1: Harmful score over steps
    ax1 = axes[0]
    ax1.plot(steps, harmful_scores, marker='o', linewidth=2, markersize=4)
    ax1.axhline(y=monitor.harmful_threshold, color='r', linestyle='--',
                label=f'Threshold ({monitor.harmful_threshold:.2f})')
    ax1.set_xlabel('Denoising Step', fontsize=12)
    ax1.set_ylabel('Harmful Score', fontsize=12)
    ax1.set_title('Harmful Content Detection Over Time', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Spatial mask ratio over steps
    ax2 = axes[1]
    ax2.plot(steps, mask_ratios, marker='s', color='orange', linewidth=2, markersize=4)
    ax2.set_xlabel('Denoising Step', fontsize=12)
    ax2.set_ylabel('Masked Region Ratio', fontsize=12)
    ax2.set_title('Spatial Guidance Coverage', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])

    plt.tight_layout()

    # Save
    save_path = output_dir / f"{prefix}_analysis.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization: {save_path}")
    plt.close()

    # Print summary
    print("\n" + "="*60)
    print("SELECTIVE GUIDANCE SUMMARY")
    print("="*60)
    print(f"Total steps:        {stats['total_steps']}")
    print(f"Harmful detected:   {stats['harmful_steps']} ({stats.get('harmful_ratio', 0):.1%})")
    print(f"Guidance applied:   {stats['guidance_applied']} ({stats.get('guidance_ratio', 0):.1%})")
    print("="*60 + "\n")

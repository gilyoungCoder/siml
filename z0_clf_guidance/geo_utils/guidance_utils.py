import math
import torch
import torch.nn.functional as F

from geo_utils.gradient_model_utils import Z0ClassifierGradientModel, Z0ImageClassifierGradientModel
from geo_utils.attention_utils import (
    AttentionStore,
    register_attention_store,
    restore_original_processors,
    find_token_indices,
    compute_attention_mask,
)


def _compute_spatial_mask(grad, threshold=0.3, soft=False, harmful_stats=None):
    """
    Compute spatial guidance mask from gradient magnitude at native 64x64.

    Two modes:
      1. Gaussian CDF (when harmful_stats provided):
         Transform per-pixel gradient magnitudes to CDF values under the
         training distribution N(mu, sigma). Threshold is a CDF percentile.
      2. Max normalization (fallback):
         Normalize gradient magnitudes by per-sample max, threshold directly.

    Args:
        grad: (B, C, H, W) gradient w.r.t. zt
        threshold: CDF percentile (with stats) or raw threshold (without)
        soft: if True, use continuous mask instead of binary
        harmful_stats: dict with "grad_mag_mu" and "grad_mag_sigma" from
                       compute_harmful_stats.py (None = fallback to max norm)

    Returns:
        mask: (B, 1, H, W) spatial mask
    """
    # Per-pixel gradient magnitude across channels
    grad_mag = grad.norm(dim=1, keepdim=True)  # (B, 1, H, W)

    if harmful_stats is not None:
        # Gaussian CDF normalization: maps raw magnitudes to percentiles [0, 1]
        mu = harmful_stats["grad_mag_mu"]
        sigma = harmful_stats["grad_mag_sigma"]
        z = (grad_mag - mu) / (sigma + 1e-8)
        normalized = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    else:
        # Fallback: normalize to [0, 1] per sample by max
        max_val = grad_mag.amax(dim=(2, 3), keepdim=True) + 1e-8
        normalized = grad_mag / max_val

    if soft:
        return normalized
    else:
        return (normalized > threshold).float()


class ExampleAwareGate:
    """
    Example-aware gating: compute soft gate g_t based on feature proximity
    to harmful vs safe prototypes.

    Score: s = cos(feat, mu_harm) - cos(feat, mu_safe)
    Gate:  g = sigmoid((s - tau) / kappa)

    When s > tau (closer to harm than safe), gate opens (g -> 1).
    When s < tau, gate closes (g -> 0).
    """

    def __init__(self, prototype_path, harm_class=2, safe_class=1,
                 tau=0.0, kappa=0.1, device="cpu"):
        """
        Args:
            prototype_path: path to .pt file with class prototypes
            harm_class: which class index is harmful (e.g. 2=nude)
            safe_class: which class index is safe (e.g. 1=clothed)
            tau: threshold for gate activation (0 = neutral boundary)
            kappa: temperature for soft gate (smaller = sharper)
        """
        protos = torch.load(prototype_path, map_location=device)
        self.mu_harm = F.normalize(protos[f"class_{harm_class}_mean"].to(device), dim=0)
        self.mu_safe = F.normalize(protos[f"class_{safe_class}_mean"].to(device), dim=0)
        self.tau = tau
        self.kappa = kappa

    def compute_gate(self, features):
        """
        Args:
            features: (B, 512) classifier features

        Returns:
            gate: (B,) soft gate values in [0, 1]
            score: (B,) raw proximity scores (for logging)
        """
        feat_norm = F.normalize(features, dim=-1)  # (B, 512)
        cos_harm = (feat_norm * self.mu_harm.unsqueeze(0)).sum(dim=-1)  # (B,)
        cos_safe = (feat_norm * self.mu_safe.unsqueeze(0)).sum(dim=-1)  # (B,)
        score = cos_harm - cos_safe  # positive = closer to harm
        gate = torch.sigmoid((score - self.tau) / self.kappa)
        return gate, score


class Z0GuidanceModel:
    """
    Guidance model wrapper for z0-based classifier guidance.

    At each denoising step:
      1. Compute z0_hat from zt via Tweedie formula
      2. Run classifier on z0_hat
      3. Compute gradient of log p(target_class | z0_hat) w.r.t. zt
      4. Apply score-based guidance to adjust noise_pred

    Optional features:
      - Spatial masking (4 modes):
        "none": no masking
        "gradcam": gradient magnitude mask (original)
        "attention": cross-attention maps for harmful tokens
        "attention_gradcam": product of both masks
      - Example-aware gating: only guide when sample is close to harmful prototypes
    """

    def __init__(self, diffusion_pipeline, classifier_ckpt, model_config,
                 target_class=1, device="cpu"):
        self.diffusion_pipeline = diffusion_pipeline
        self.device = device
        self.target_class = target_class

        # Spatial guidance config
        self.spatial_threshold = model_config.get("spatial_threshold", 0.3)
        self.spatial_soft = model_config.get("spatial_soft", False)
        self.grad_wrt_z0 = model_config.get("grad_wrt_z0", False)
        self.harm_ratio = model_config.get("harm_ratio", 1.0)

        # Cosine annealing for spatial threshold
        self.threshold_schedule = model_config.get("threshold_schedule", "constant")
        self.spatial_threshold_start = self.spatial_threshold  # save initial value

        # GradCAM config: which layer and which class to visualize
        self.gradcam_layer = model_config.get("gradcam_layer", "layer4")
        harm_classes = model_config.get("harm_classes", None)
        self.gradcam_target = harm_classes[0] if harm_classes else 2
        self.gradcam_ref_class = model_config.get("gradcam_ref_class", None)

        # Spatial mode: "none" | "gradcam" | "attention" | "attention_gradcam"
        self.spatial_mode = model_config.get("spatial_mode", "none")
        # Backward compat: old --spatial_guidance flag
        if model_config.get("spatial_guidance", False) and self.spatial_mode == "none":
            self.spatial_mode = "gradcam"

        # Attention-aware guidance
        self.attention_store = None
        self._original_processors = None
        self.token_indices = []
        self.harmful_keywords = model_config.get("harmful_keywords", [])
        self.attn_resolutions = model_config.get("attn_resolutions", None)

        if self.spatial_mode in ("attention", "attention_gradcam"):
            self.attention_store = AttentionStore()
            self._original_processors = register_attention_store(
                diffusion_pipeline.unet,
                self.attention_store,
                target_resolutions=self.attn_resolutions,
            )

        space = model_config.get("space", "latent")  # "latent" or "image"
        if space == "image":
            self.gradient_model = Z0ImageClassifierGradientModel(model_config, device)
        else:
            self.gradient_model = Z0ClassifierGradientModel(model_config, device)
        self.gradient_model.load_model(classifier_ckpt)

        # Harmful stats for Gaussian CDF spatial thresholding
        self.harmful_stats = None
        stats_path = model_config.get("harmful_stats_path", None)
        if stats_path is not None:
            import os
            if os.path.exists(stats_path):
                self.harmful_stats = torch.load(stats_path, map_location=device)
                if "gradcam_mu" in self.harmful_stats:
                    print(f"  [spatial] Loaded harmful stats (GradCAM): "
                          f"gradcam_mu={self.harmful_stats['gradcam_mu']:.6f}, "
                          f"gradcam_sigma={self.harmful_stats['gradcam_sigma']:.6f}")
                elif "grad_mag_mu" in self.harmful_stats:
                    print(f"  [spatial] Loaded harmful stats (grad_mag): "
                          f"grad_mag_mu={self.harmful_stats['grad_mag_mu']:.6f}, "
                          f"grad_mag_sigma={self.harmful_stats['grad_mag_sigma']:.6f}")

        # Example-aware gating
        self.eacg_gate = None
        proto_path = model_config.get("prototype_path", None)
        if proto_path is not None:
            self.eacg_gate = ExampleAwareGate(
                proto_path,
                harm_class=model_config.get("eacg_harm_class", 2),
                safe_class=model_config.get("eacg_safe_class", 1),
                tau=model_config.get("eacg_tau", 0.0),
                kappa=model_config.get("eacg_kappa", 0.1),
                device=device,
            )

    def set_prompt(self, prompt, tokenizer):
        """
        Compute token indices for harmful keywords in the given prompt.
        Must be called before each generation when using attention-based guidance.
        """
        if self.spatial_mode not in ("attention", "attention_gradcam"):
            return

        self.token_indices = find_token_indices(
            prompt, self.harmful_keywords, tokenizer
        )

        if self.attention_store is not None:
            self.attention_store.reset()

        if self.token_indices:
            print(f"  [attn] Harmful token indices={self.token_indices} "
                  f"for keywords={self.harmful_keywords}")
        else:
            print(f"  [attn] WARNING: no tokens matched keywords "
                  f"{self.harmful_keywords} in prompt")

    def cleanup(self, unet):
        """Restore original attention processors (for grid search cleanup)."""
        if self._original_processors is not None:
            restore_original_processors(unet, self._original_processors)
            self._original_processors = None

    def guidance(self, diffusion_pipeline, callback_kwargs, step, timestep,
                 scale, target_class=None):
        """
        Called from pipeline callback_on_step_end.

        Returns:
            dict with updated 'latents' and monitoring values
        """
        with torch.enable_grad():
            # prev_latents = zt (before scheduler.step)
            prev_latents = callback_kwargs["prev_latents"].clone().detach().requires_grad_(True)
            latents = callback_kwargs["latents"]        # z_{t-1} from scheduler.step
            noise_pred = callback_kwargs["noise_pred"]  # CFG-combined noise prediction

            # Use unconditional noise_pred for z0_hat (matches classifier training distribution)
            # During training, classifier saw z0_hat from uncond UNet prediction only.
            noise_pred_for_z0 = callback_kwargs.get("noise_pred_uncond", noise_pred)

            # Tweedie: zt -> z0_hat (noise_pred detached inside get_scaled_input)
            z0_hat = self.gradient_model.get_scaled_input(
                diffusion_pipeline, prev_latents, noise_pred_for_z0, timestep
            )

            tc = target_class if target_class is not None else self.target_class

            # Classifier gradient
            grad_input = z0_hat if self.grad_wrt_z0 else prev_latents
            diff_val, grad, output_for_log = self.gradient_model.get_grad(
                z0_hat, None, None, grad_input, target_class=tc,
                harm_ratio=self.harm_ratio,
            )

        grad = grad.detach()

        # Example-aware gating: scale gradient by gate value
        gate_val = 1.0
        gate_score = 0.0
        if self.eacg_gate is not None:
            with torch.no_grad():
                features = self.gradient_model.model.get_features(z0_hat.detach())
            gate, score = self.eacg_gate.compute_gate(features)
            gate_val = gate.mean().item()
            gate_score = score.mean().item()
            # Apply gate: (B,) -> (B, 1, 1, 1) for broadcasting
            grad = grad * gate.view(-1, 1, 1, 1)

        # Cosine annealing for spatial threshold
        # At step 0: threshold = spatial_threshold_start (strict)
        # At final step: threshold = 0 (full mask)
        current_threshold = self.spatial_threshold
        if self.threshold_schedule == "cosine" and self.spatial_mode != "none":
            num_steps = diffusion_pipeline.scheduler.config.num_train_timesteps
            # step goes 0..T-1, timestep goes high..low
            progress = 1.0 - (timestep.item() / num_steps)  # 0 -> 1
            current_threshold = self.spatial_threshold_start * 0.5 * (1.0 + math.cos(math.pi * progress))

        # Spatial masking: localize guidance to relevant regions
        mask_ratio = 1.0
        if self.spatial_mode == "gradcam":
            # Proper GradCAM from classifier intermediate feature maps.
            # Localizes WHERE the classifier sees harmful content.
            with torch.enable_grad():
                gradcam_map = self.gradient_model.model.compute_gradcam(
                    z0_hat.detach(), target_class=self.gradcam_target,
                    layer_name=self.gradcam_layer,
                    ref_class=self.gradcam_ref_class,
                )  # (B, 1, H, W), per-sample normalized [0, 1]

            # Apply Gaussian CDF if stats available (calibrated threshold)
            if self.harmful_stats is not None and "gradcam_mu" in self.harmful_stats:
                mu = self.harmful_stats["gradcam_mu"]
                sigma = self.harmful_stats["gradcam_sigma"]
                z = (gradcam_map - mu) / (sigma + 1e-8)
                cdf_map = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
            else:
                cdf_map = gradcam_map  # fallback: use raw GradCAM values

            if self.spatial_soft:
                spatial_mask = cdf_map
            else:
                spatial_mask = (cdf_map > current_threshold).float()
            grad = grad * spatial_mask
            mask_ratio = spatial_mask.mean().item()

        elif self.spatial_mode == "attention":
            if self.attention_store is not None and self.token_indices:
                attn_mask = compute_attention_mask(
                    self.attention_store,
                    self.token_indices,
                    target_resolution=grad.shape[-1],
                    threshold=current_threshold,
                    soft=self.spatial_soft,
                )
                if attn_mask is not None:
                    grad = grad * attn_mask.to(grad.device)
                    mask_ratio = attn_mask.mean().item()

        elif self.spatial_mode == "attention_gradcam":
            # GradCAM × attention mask product
            with torch.enable_grad():
                gradcam_map = self.gradient_model.model.compute_gradcam(
                    z0_hat.detach(), target_class=self.gradcam_target,
                    layer_name=self.gradcam_layer,
                    ref_class=self.gradcam_ref_class,
                )
            if self.attention_store is not None and self.token_indices:
                attn_mask = compute_attention_mask(
                    self.attention_store,
                    self.token_indices,
                    target_resolution=grad.shape[-1],
                    threshold=0.0,
                    soft=True,
                )
                if attn_mask is not None:
                    combined_mask = gradcam_map * attn_mask.to(grad.device)
                else:
                    combined_mask = gradcam_map
            else:
                combined_mask = gradcam_map

            # Renormalize combined mask
            max_val = combined_mask.amax(dim=(2, 3), keepdim=True) + 1e-8
            combined_mask = combined_mask / max_val

            # Apply threshold if not soft mode
            if not self.spatial_soft:
                combined_mask = (combined_mask > current_threshold).float()

            grad = grad * combined_mask
            mask_ratio = combined_mask.mean().item()

        # Score-based guidance: adjust noise_pred and re-step
        latents = self.gradient_model.guide_samples(
            diffusion_pipeline, noise_pred, prev_latents, latents,
            timestep, grad, scale
        )

        return {
            "latents": latents,
            "differentiate_value": output_for_log.detach(),
            "spatial_mask_ratio": mask_ratio,
            "gate_val": gate_val,
            "gate_score": gate_score,
        }

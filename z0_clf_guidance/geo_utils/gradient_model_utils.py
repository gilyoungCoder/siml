import torch
import torch.nn.functional as F

from utils.denoise_utils import predict_z0

VAE_SCALE = 0.18215


def _compute_guidance_value(logits, guidance_mode, target_class=1,
                            safe_classes=None, harm_classes=None,
                            harm_ratio=1.0):
    """
    Compute the differentiable guidance value and monitoring info.

    Args:
        logits: (B, C) raw classifier output
        guidance_mode: "target", "safe_minus_harm", or "paired"
        target_class: used when guidance_mode="target"
        safe_classes: list of safe class indices (ordered to match harm_classes)
        harm_classes: list of harm class indices (ordered to match safe_classes)

    Returns:
        diff_val: scalar to maximize (gradient flows through this)
        output_for_log: detached monitoring value
    """
    log_probs = F.log_softmax(logits, dim=-1)

    if guidance_mode == "paired":
        # Paired: argmax among harm classes -> guide toward corresponding safe pair
        # harm_classes[i] <-> safe_classes[i] (e.g. 1->2, 3->4, 5->6, 7->8)
        B = logits.shape[0]
        device = logits.device
        harm_cls = torch.tensor(harm_classes, device=device)
        safe_cls = torch.tensor(safe_classes, device=device)

        # Select which pair to use (not differentiable, just routing)
        with torch.no_grad():
            harm_probs = F.softmax(logits, dim=-1)[:, harm_classes]  # (B, H)
            pair_idx = harm_probs.argmax(dim=-1)  # (B,)

        selected_harm = harm_cls[pair_idx]  # (B,)
        selected_safe = safe_cls[pair_idx]  # (B,)
        batch_idx = torch.arange(B, device=device)

        # Differentiable objective: log p(safe_pair) - log p(harm_max)
        diff_val = (log_probs[batch_idx, selected_safe]
                    - log_probs[batch_idx, selected_harm]).sum()

        with torch.no_grad():
            probs = F.softmax(logits, dim=-1)
            p_safe = probs[batch_idx, selected_safe]
            p_harm = probs[batch_idx, selected_harm]
            gap = p_safe - p_harm  # monitor: per-pair gap

        return diff_val, gap.detach()

    elif guidance_mode == "safe_minus_harm":
        # log(sum p(safe)) - log(sum p(harm))
        safe_logsumexp = torch.logsumexp(log_probs[:, safe_classes], dim=-1)
        harm_logsumexp = torch.logsumexp(log_probs[:, harm_classes], dim=-1)
        diff_val = (safe_logsumexp - harm_ratio * harm_logsumexp).sum()

        with torch.no_grad():
            probs = F.softmax(logits, dim=-1)
            p_safe = probs[:, safe_classes].sum(dim=-1)
            p_harm = probs[:, harm_classes].sum(dim=-1)
            gap = p_safe - p_harm  # +1 = fully safe, -1 = fully harm

        return diff_val, gap.detach()
    else:
        # Default: maximize log p(target_class)
        target_log_prob = log_probs[:, target_class]
        diff_val = target_log_prob.sum()

        with torch.no_grad():
            probs = F.softmax(logits, dim=-1)[:, target_class]

        return diff_val, probs.detach()


def _guide_samples_score_based(diffusion_pipeline, noise_pred, prev_latents,
                               latents, timestep, grad, scale):
    """
    Score-based guidance (shared by both latent and image models).

    score = -noise_pred / sqrt(1 - alpha_bar)
    adjusted_score = score + scale * grad
    adjusted_noise_pred = -adjusted_score * sqrt(1 - alpha_bar)
    z_{t-1} = scheduler.step(adjusted_noise_pred, t, z_t)
    """
    alpha_cp = diffusion_pipeline.scheduler.alphas_cumprod.to(
        latents.device
    )[timestep]
    denom = torch.sqrt(1 - alpha_cp)

    # Compute full guidance term: scale * sqrt(1-alpha) * grad
    # Formula: adjusted_eps = eps - guidance_term
    g = grad.detach()
    guidance_term = scale * denom * g

    # Clip the FULL guidance term so it doesn't overwhelm noise_pred
    gt_norm = guidance_term.norm()
    eps_norm = noise_pred.norm()
    max_guidance = eps_norm * 0.3  # at most 30% of noise_pred magnitude
    clipped = False
    if gt_norm > max_guidance and gt_norm > 0:
        guidance_term = guidance_term * (max_guidance / gt_norm)
        clipped = True

    ratio = gt_norm / (eps_norm + 1e-8)
    print(f"    [guidance] t={timestep.item()}, ||gt||={gt_norm:.4f}, ||eps||={eps_norm:.4f}, "
          f"ratio={ratio:.2f}, clipped={clipped}")

    adjusted_noise_pred = noise_pred - guidance_term

    # Undo step index if the scheduler tracks it (diffusers >= 0.25)
    if hasattr(diffusion_pipeline.scheduler, '_step_index') and \
       diffusion_pipeline.scheduler._step_index is not None:
        diffusion_pipeline.scheduler._step_index -= 1

    out = diffusion_pipeline.scheduler.step(
        model_output=adjusted_noise_pred,
        timestep=timestep,
        sample=prev_latents,
        return_dict=True,
    )
    return out.prev_sample


class Z0ClassifierGradientModel:
    """
    Classifier guidance on clean latent z0 predicted via one-step denoising (Tweedie).

    Supports three guidance modes:
    - "target": maximize log p(target_class | z0_hat)
    - "safe_minus_harm": maximize log p(safe) - log p(harm) (aggregate)
    - "paired": argmax(harm) -> guide toward corresponding safe pair

    noise_pred is DETACHED: gradient does NOT flow through UNet.
    """

    def __init__(self, model_config, device="cpu"):
        self.device = device

        arch = model_config.get("architecture", "resnet18")
        num_classes = model_config.get("num_classes", 3)

        if arch == "resnet18":
            from models.latent_classifier import LatentResNet18Classifier
            self.model = LatentResNet18Classifier(
                num_classes=num_classes, pretrained_backbone=False
            ).to(device)
        else:
            raise ValueError(f"Unknown architecture: {arch}")

        self.model.eval()

        self.guidance_mode = model_config.get("guidance_mode", "target")
        self.safe_classes = model_config.get("safe_classes", None)
        self.harm_classes = model_config.get("harm_classes", None)

    def load_model(self, ckpt_path):
        if ckpt_path is None:
            return
        state_dict = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def get_scaled_input(self, diffusion_pipeline, noisy_input, noise_pred, timestep):
        alpha_bar = diffusion_pipeline.scheduler.alphas_cumprod.to(
            noisy_input.device
        )[timestep]
        z0_hat = predict_z0(noisy_input, noise_pred.detach(), alpha_bar)
        return z0_hat

    def get_differentiate_value(self, z0_hat, timestep=None,
                                encoder_hidden_states=None, target_class=1,
                                harm_ratio=1.0):
        logits = self.model(z0_hat)
        return _compute_guidance_value(
            logits, self.guidance_mode, target_class,
            self.safe_classes, self.harm_classes,
            harm_ratio=harm_ratio,
        )

    def get_grad(self, z0_hat, timestep, encoder_hidden_states,
                 grad_input, target_class=1, harm_ratio=1.0):
        diff_val, output_for_log = self.get_differentiate_value(
            z0_hat, timestep, encoder_hidden_states, target_class,
            harm_ratio=harm_ratio,
        )
        grad = torch.autograd.grad(diff_val, inputs=grad_input)[0]
        return diff_val, grad, output_for_log

    def guide_samples(self, diffusion_pipeline, noise_pred, prev_latents,
                      latents, timestep, grad, scale):
        return _guide_samples_score_based(
            diffusion_pipeline, noise_pred, prev_latents,
            latents, timestep, grad, scale,
        )


class Z0ImageClassifierGradientModel:
    """
    Classifier guidance on IMAGE-SPACE x0_hat predicted via:
      zt -> Tweedie -> z0_hat -> VAE decode -> x0_hat -> classifier

    Supports three guidance modes:
    - "target": maximize log p(target_class | x0_hat)
    - "safe_minus_harm": maximize log p(safe) - log p(harm) (aggregate)
    - "paired": argmax(harm) -> guide toward corresponding safe pair

    noise_pred is DETACHED. VAE decode IS in the gradient path during guidance.
    """

    def __init__(self, model_config, device="cpu"):
        self.device = device
        arch = model_config.get("architecture", "resnet18")
        num_classes = model_config.get("num_classes", 3)

        from models.image_classifier import build_image_classifier
        self.model = build_image_classifier(
            architecture=arch, num_classes=num_classes, pretrained_backbone=False
        ).to(device)
        self.model.eval()

        self.guidance_mode = model_config.get("guidance_mode", "target")
        self.safe_classes = model_config.get("safe_classes", None)
        self.harm_classes = model_config.get("harm_classes", None)

    def load_model(self, ckpt_path):
        if ckpt_path is None:
            return
        state_dict = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def get_scaled_input(self, diffusion_pipeline, noisy_input, noise_pred, timestep):
        alpha_bar = diffusion_pipeline.scheduler.alphas_cumprod.to(
            noisy_input.device
        )[timestep]
        z0_hat = predict_z0(noisy_input, noise_pred.detach(), alpha_bar)

        vae = diffusion_pipeline.vae
        x0_hat = vae.decode(z0_hat / VAE_SCALE, return_dict=False)[0]
        x0_hat = x0_hat.clamp(-1, 1)
        return x0_hat

    def get_differentiate_value(self, x0_hat, timestep=None,
                                encoder_hidden_states=None, target_class=1,
                                harm_ratio=1.0):
        logits = self.model(x0_hat)
        return _compute_guidance_value(
            logits, self.guidance_mode, target_class,
            self.safe_classes, self.harm_classes,
            harm_ratio=harm_ratio,
        )

    def get_grad(self, x0_hat, timestep, encoder_hidden_states,
                 grad_input, target_class=1, harm_ratio=1.0):
        diff_val, output_for_log = self.get_differentiate_value(
            x0_hat, timestep, encoder_hidden_states, target_class,
            harm_ratio=harm_ratio,
        )
        grad = torch.autograd.grad(diff_val, inputs=grad_input)[0]
        return diff_val, grad, output_for_log

    def guide_samples(self, diffusion_pipeline, noise_pred, prev_latents,
                      latents, timestep, grad, scale):
        return _guide_samples_score_based(
            diffusion_pipeline, noise_pred, prev_latents,
            latents, timestep, grad, scale,
        )

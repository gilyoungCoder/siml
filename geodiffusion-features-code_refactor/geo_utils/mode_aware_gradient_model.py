"""
Mode-Aware Gradient Model for Classifier Guidance

핵심 아이디어:
Violence는 latent space에서 multi-modal (총기/칼/몸싸움/위협 자세 등)이므로,
단일 classifier boundary로 guidance하면 gradient가 평균화/상쇄되어 약해짐.

해결책:
1. Violence latent를 k-means로 K개 mode로 분해
2. 샘플링 시 현재 latent의 mode를 추정
3. Mode별로 다른 guidance policy (scale, start_step, end_step) 적용

"classifier가 같아도, latent가 다르면 gradient는 이미 다르다.
 clustering은 그 차이를 '활용 가능하게 만드는 스위치'다."
"""

import os
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

from geo_utils.gradient_model_utils import (
    GradientModel,
    get_predicted_x0,
    get_noisy_xt,
    DiscriminatorGradientModel,
    TimeDependentDiscriminatorGradientModel
)
from geo_models.classifier.classifier import create_classifier, discriminator_defaults


class ModeSchedule:
    """
    각 cluster(mode)별 guidance 스케줄을 정의하는 클래스

    Parameters:
    - scale: guidance strength for this mode
    - start_step: step to start applying guidance (0-indexed)
    - end_step: step to stop applying guidance (None = until end)
    - scale_decay: optional decay factor per step
    """
    def __init__(
        self,
        scale: float = 1.0,
        start_step: int = 0,
        end_step: Optional[int] = None,
        scale_decay: float = 1.0,
        threshold: Optional[float] = None,  # confidence threshold to apply guidance
    ):
        self.base_scale = scale
        self.start_step = start_step
        self.end_step = end_step
        self.scale_decay = scale_decay
        self.threshold = threshold

    def get_scale(self, step: int, confidence: Optional[float] = None) -> float:
        """현재 step에서의 effective scale 반환"""
        # Step range check
        if step < self.start_step:
            return 0.0
        if self.end_step is not None and step > self.end_step:
            return 0.0

        # Confidence threshold check
        if self.threshold is not None and confidence is not None:
            if confidence < self.threshold:
                return 0.0

        # Apply decay
        steps_active = step - self.start_step
        effective_scale = self.base_scale * (self.scale_decay ** steps_active)

        return effective_scale

    def is_active(self, step: int) -> bool:
        """현재 step에서 guidance가 활성화되어야 하는지"""
        if step < self.start_step:
            return False
        if self.end_step is not None and step > self.end_step:
            return False
        return True


class ClusterManager:
    """
    K-means clustering 관리 클래스

    - centroid 저장/로드
    - 현재 latent의 cluster 추정
    - cluster별 통계 추적
    """
    def __init__(
        self,
        n_clusters: int = 10,
        centroids_path: Optional[str] = None,
        pooling: str = "mean",  # "mean", "flatten", "attention"
        device: str = "cpu"
    ):
        self.n_clusters = n_clusters
        self.pooling = pooling
        self.device = device
        self.centroids: Optional[torch.Tensor] = None  # [K, feature_dim]
        self.cluster_counts = torch.zeros(n_clusters)  # 각 cluster가 선택된 횟수

        if centroids_path is not None and os.path.exists(centroids_path):
            self.load_centroids(centroids_path)

    def pool_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Latent를 1D feature로 변환
        x: [B, C, H, W] -> [B, feature_dim]
        """
        if self.pooling == "mean":
            # Global average pooling
            return x.mean(dim=[2, 3])  # [B, C]
        elif self.pooling == "flatten":
            # Flatten all spatial dimensions
            return x.view(x.size(0), -1)  # [B, C*H*W]
        elif self.pooling == "attention":
            # Attention-weighted pooling (simple version)
            # Compute attention weights based on spatial variance
            var = x.var(dim=1, keepdim=True)  # [B, 1, H, W]
            weights = F.softmax(var.view(x.size(0), -1), dim=1).view_as(var)
            return (x * weights).sum(dim=[2, 3])  # [B, C]
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

    def assign_cluster(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        현재 latent의 cluster 할당

        Args:
            x: [B, C, H, W] latent tensor

        Returns:
            cluster_ids: [B] cluster assignment for each sample
            distances: [B] distance to assigned centroid
        """
        if self.centroids is None:
            raise RuntimeError("Centroids not initialized. Call fit() or load_centroids() first.")

        # Pool features
        features = self.pool_features(x)  # [B, feature_dim]

        # Compute distances to all centroids
        # centroids: [K, feature_dim], features: [B, feature_dim]
        centroids = self.centroids.to(features.device)

        # [B, K] = ||features - centroids||^2
        distances_all = torch.cdist(features, centroids, p=2)  # [B, K]

        # Get nearest cluster
        distances, cluster_ids = distances_all.min(dim=1)  # [B], [B]

        # Update counts (for logging)
        for cid in cluster_ids:
            self.cluster_counts[cid.item()] += 1

        return cluster_ids, distances

    def fit(self, latents: torch.Tensor, save_path: Optional[str] = None):
        """
        Latent 샘플들로 k-means fitting

        Args:
            latents: [N, C, H, W] collection of latent samples
            save_path: optional path to save centroids
        """
        # Pool features
        features = self.pool_features(latents)  # [N, feature_dim]
        features_np = features.cpu().numpy()

        # Fit k-means
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        kmeans.fit(features_np)

        # Store centroids
        self.centroids = torch.from_numpy(kmeans.cluster_centers_).float()

        if save_path is not None:
            self.save_centroids(save_path)

        print(f"[ClusterManager] Fitted {self.n_clusters} clusters from {len(latents)} samples")
        return kmeans.labels_

    def save_centroids(self, path: str):
        """Save centroids to file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'centroids': self.centroids,
            'n_clusters': self.n_clusters,
            'pooling': self.pooling,
        }, path)
        print(f"[ClusterManager] Saved centroids to {path}")

    def load_centroids(self, path: str):
        """Load centroids from file"""
        data = torch.load(path, map_location='cpu')
        self.centroids = data['centroids']
        self.n_clusters = data['n_clusters']
        self.pooling = data.get('pooling', self.pooling)
        print(f"[ClusterManager] Loaded {self.n_clusters} centroids from {path}")

    def get_cluster_stats(self) -> Dict[int, int]:
        """Return cluster usage statistics"""
        return {i: int(self.cluster_counts[i].item()) for i in range(self.n_clusters)}


class ModeAwareDiscriminatorGradientModel(GradientModel):
    """
    Mode-Aware Classifier Guidance

    기존 Discriminator 위에 mode detection + mode-aware scaling을 추가

    핵심:
    - classifier 구조/가중치는 동일하게 유지
    - clustering으로 현재 latent의 mode 추정
    - mode별로 다른 guidance policy 적용
    """

    def __init__(self, model_config_obj: dict, device: str = "cpu"):
        # Base discriminator config
        self.model_args = discriminator_defaults()

        # Extract mode-aware specific configs
        mode_config = model_config_obj.pop('mode_aware', {})
        self.n_clusters = mode_config.get('n_clusters', 10)
        self.centroids_path = mode_config.get('centroids_path', None)
        self.pooling = mode_config.get('pooling', 'mean')

        # Parse mode schedules
        self.mode_schedules = self._parse_mode_schedules(
            mode_config.get('schedules', {})
        )

        # Default schedule for modes without explicit config
        self.default_schedule = ModeSchedule(
            scale=mode_config.get('default_scale', 1.0),
            start_step=mode_config.get('default_start_step', 0),
            end_step=mode_config.get('default_end_step', None),
        )

        # Update with remaining config
        self.model_args.update(model_config_obj)

        # Create base discriminator
        self.model = self.create_model(self.model_args)
        self.model = self.model.to(device)

        # Initialize cluster manager
        self.cluster_manager = ClusterManager(
            n_clusters=self.n_clusters,
            centroids_path=self.centroids_path,
            pooling=self.pooling,
            device=device
        )

        # Logging
        self.current_mode: Optional[int] = None
        self.mode_history: List[int] = []

    def _parse_mode_schedules(self, schedules_config: dict) -> Dict[int, ModeSchedule]:
        """Parse mode-specific schedules from config"""
        schedules = {}
        for mode_id, config in schedules_config.items():
            mode_id = int(mode_id) if isinstance(mode_id, str) else mode_id
            schedules[mode_id] = ModeSchedule(
                scale=config.get('scale', 1.0),
                start_step=config.get('start_step', 0),
                end_step=config.get('end_step', None),
                scale_decay=config.get('scale_decay', 1.0),
                threshold=config.get('threshold', None),
            )
        return schedules

    def create_model(self, model_config):
        return create_classifier(**model_config)

    def load_model(self, pretrained_model_ckpt):
        if pretrained_model_ckpt is None:
            return
        model_state_dict = torch.load(open(pretrained_model_ckpt, "rb"), map_location='cpu')
        self.model.load_state_dict(model_state_dict)

    def default_model_args(self):
        return discriminator_defaults()

    def get_scaled_input(self, diffusion_pipeline, noisy_input, noise_pred, timestep):
        """Predicted x0를 반환 (mode detection에 사용)"""
        return get_predicted_x0(diffusion_pipeline, noisy_input, noise_pred, timestep)

    def detect_mode(self, x0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        현재 predicted x0에서 mode 추정

        Returns:
            cluster_ids: [B] cluster assignment
            distances: [B] distance to centroid (confidence proxy)
        """
        if self.cluster_manager.centroids is None:
            # Centroids not loaded - return dummy
            return torch.zeros(x0.size(0), dtype=torch.long, device=x0.device), \
                   torch.ones(x0.size(0), device=x0.device)

        return self.cluster_manager.assign_cluster(x0)

    def get_mode_scale(self, mode_id: int, step: int, confidence: Optional[float] = None) -> float:
        """Get the guidance scale for a specific mode at a specific step"""
        if mode_id in self.mode_schedules:
            return self.mode_schedules[mode_id].get_scale(step, confidence)
        return self.default_schedule.get_scale(step, confidence)

    def get_differentiate_value(self, image, timestep, encoder_hidden_states=None):
        """
        Compute differentiation value for gradient computation
        Same as DiscriminatorGradientModel
        """
        dummy_timestep = torch.zeros(image.shape[0],).to(image.device)
        out = self.model(image, dummy_timestep)
        output = F.sigmoid(out)
        output = torch.clip(output, 1e-5, 1. - 1e-5)
        output = 1 - output

        output_for_log = output.clone().detach()

        ratio = output / (1 - output)
        log_ratio = torch.log(ratio)

        diff_val = log_ratio.sum()
        return diff_val, output_for_log

    def guide_samples(
        self,
        diffusion_pipeline,
        noise_pred,
        prev_latents,
        latents,
        timestep,
        grad,
        scale,
        step: Optional[int] = None,
        x0: Optional[torch.Tensor] = None,
    ):
        """
        Mode-aware guidance application

        핵심 변경점: scale을 mode별로 다르게 적용
        """
        # Detect current mode from x0
        if x0 is not None and self.cluster_manager.centroids is not None:
            cluster_ids, distances = self.detect_mode(x0)

            # For batch processing, we use the mode of the first sample
            # (or could do per-sample different scales)
            self.current_mode = cluster_ids[0].item()
            self.mode_history.append(self.current_mode)

            # Get mode-specific scale
            # Convert distance to confidence (smaller distance = higher confidence)
            confidence = 1.0 / (1.0 + distances[0].item())
            mode_scale = self.get_mode_scale(self.current_mode, step or 0, confidence)

            # Combine with base scale
            effective_scale = scale * mode_scale
        else:
            effective_scale = scale
            self.current_mode = None

        # Score-based guidance (same as DiscriminatorGradientModel)
        diffusion_pipeline.scheduler._step_index -= 1
        score = - noise_pred / torch.sqrt(1 - diffusion_pipeline.scheduler.alphas_cumprod[timestep])
        discriminator_adjusted_score = score + effective_scale * grad.detach()
        adjusted_noise_pred = - discriminator_adjusted_score * torch.sqrt(
            1 - diffusion_pipeline.scheduler.alphas_cumprod[timestep]
        )
        latents = diffusion_pipeline.scheduler.step(
            adjusted_noise_pred, timestep, prev_latents, return_dict=False
        )[0]

        return latents

    def __call__(self, image, timestep, encoder_hidden_states=None):
        dummy_timestep = torch.zeros_like(timestep).to(image.device)
        return self.model(image, dummy_timestep)

    def get_mode_statistics(self) -> Dict:
        """Return mode usage statistics for logging"""
        return {
            'cluster_counts': self.cluster_manager.get_cluster_stats(),
            'mode_history': self.mode_history[-100:],  # Last 100 modes
            'current_mode': self.current_mode,
        }


class ModeAwareTimeDependentDiscriminatorGradientModel(ModeAwareDiscriminatorGradientModel):
    """
    Time-Dependent version of Mode-Aware Discriminator
    Uses noisy x_t instead of predicted x0
    """

    def get_scaled_input(self, diffusion_pipeline, noisy_input, noise_pred, timestep):
        """Return noisy x_t directly (time-dependent)"""
        return get_noisy_xt(diffusion_pipeline, noisy_input, noise_pred, timestep)

    def get_differentiate_value(self, image, timestep, encoder_hidden_states=None):
        """Time-dependent discriminator forward"""
        timestep_batch = torch.tile(timestep, (image.shape[0],))
        out = self.model(image, timestep_batch)
        out = F.sigmoid(out)
        out = torch.clip(out, 1e-5, 1. - 1e-5)
        out = 1 - out

        ratio = out / (1 - out)
        log_ratio = torch.log(ratio)

        output_for_log = out.clone().detach()
        diff_val = log_ratio.sum()
        return diff_val, output_for_log

    def __call__(self, image, timestep, encoder_hidden_states=None):
        return self.model(image, timestep)


# =============================================================================
# Utility functions for clustering violence latents
# =============================================================================

def collect_violence_latents(
    diffusion_pipeline,
    prompts: List[str],
    num_samples_per_prompt: int = 10,
    timestep: int = 500,  # timestep at which to extract latents
    num_inference_steps: int = 50,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Violence prompt로 이미지를 생성하면서 특정 timestep의 latent를 수집

    Args:
        diffusion_pipeline: Stable Diffusion pipeline
        prompts: list of violence-related prompts
        num_samples_per_prompt: number of samples per prompt
        timestep: timestep at which to extract latent
        num_inference_steps: total inference steps

    Returns:
        latents: [N, C, H, W] collected latents
    """
    collected_latents = []

    for prompt in prompts:
        for i in range(num_samples_per_prompt):
            # We need to hook into the sampling process to collect latents
            captured_latent = [None]
            target_step = int((1 - timestep / 1000) * num_inference_steps)

            def capture_callback(pipe, step, t, callback_kwargs):
                if step == target_step:
                    # Capture the latent at this step
                    noise_pred = callback_kwargs['noise_pred']
                    prev_latents = callback_kwargs['prev_latents']
                    x0 = get_predicted_x0(pipe, prev_latents, noise_pred, t)
                    captured_latent[0] = x0.detach().cpu()
                return callback_kwargs

            # Run generation with callback
            _ = diffusion_pipeline(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                callback_on_step_end=capture_callback,
                callback_on_step_end_tensor_inputs=['noise_pred', 'prev_latents'],
            )

            if captured_latent[0] is not None:
                collected_latents.append(captured_latent[0])

    return torch.cat(collected_latents, dim=0)


def fit_violence_clusters(
    latents: torch.Tensor,
    n_clusters: int = 10,
    pooling: str = "mean",
    save_path: str = "cluster_centroids/violence_clusters.pt"
) -> ClusterManager:
    """
    수집된 violence latent들로 clustering 수행

    Args:
        latents: [N, C, H, W] violence latents
        n_clusters: number of clusters
        pooling: pooling method
        save_path: path to save centroids

    Returns:
        cluster_manager: fitted ClusterManager
    """
    cluster_manager = ClusterManager(
        n_clusters=n_clusters,
        pooling=pooling
    )

    labels = cluster_manager.fit(latents, save_path=save_path)

    # Print cluster distribution
    unique, counts = np.unique(labels, return_counts=True)
    print("\n[Cluster Distribution]")
    for cluster_id, count in zip(unique, counts):
        print(f"  Cluster {cluster_id}: {count} samples ({100*count/len(labels):.1f}%)")

    return cluster_manager
